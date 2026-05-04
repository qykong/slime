import json
import os
import re

import torch


_HF_CONFIG_CACHE: dict[str, dict] = {}


def _get_hf_config(args) -> dict:
    """Read text_config + vision_config + GDN dims from args.hf_checkpoint/config.json (cached)."""
    ckpt = getattr(args, "hf_checkpoint", None)
    if ckpt is None:
        raise ValueError("args.hf_checkpoint is required to convert Qwen3.5 params")
    if ckpt in _HF_CONFIG_CACHE:
        return _HF_CONFIG_CACHE[ckpt]
    with open(os.path.join(ckpt, "config.json")) as f:
        cfg = json.load(f)
    text = cfg["text_config"]
    vc = cfg["vision_config"]
    out = {
        "vision": {
            "num_heads": vc["num_heads"],
            "hidden_size": vc["hidden_size"],
            "head_dim": vc["hidden_size"] // vc["num_heads"],
        },
        "gdn": {
            "hidden_size": text["hidden_size"],
            "linear_key_head_dim": text["linear_key_head_dim"],
            "linear_value_head_dim": text["linear_value_head_dim"],
            "linear_num_key_heads": text["linear_num_key_heads"],
            "linear_num_value_heads": text["linear_num_value_heads"],
        },
    }
    _HF_CONFIG_CACHE[ckpt] = out
    return out


def _get_vision_config(args) -> dict:
    return _get_hf_config(args)["vision"]


def _gdn_dims(args) -> dict:
    return _get_hf_config(args)["gdn"]


def _tp_size(args) -> int:
    return int(getattr(args, "tensor_model_parallel_size", 1) or 1)


def _split_bridge_in_proj(param: torch.Tensor, gdn: dict, tp: int):
    """Split Megatron's bridge-style fused GDN ``in_proj.weight`` into the four
    HF tensors ``in_proj_qkv``, ``in_proj_z``, ``in_proj_b``, ``in_proj_a``.

    The Megatron tensor is the post-all-gather result; rows are laid out
    rank-major as ``[r0_q, r0_k, r0_v, r0_z, r0_b, r0_a, r1_q, ...]`` where each
    component has ``size/tp`` rows on each rank.
    """
    h = gdn["hidden_size"]
    qk_head_dim = gdn["linear_key_head_dim"]
    v_head_dim = gdn["linear_value_head_dim"]
    n_qk = gdn["linear_num_key_heads"]
    n_v = gdn["linear_num_value_heads"]
    qk_dim = qk_head_dim * n_qk           # 2048
    v_dim = v_head_dim * n_v              # 4096
    qk_local = qk_dim // tp
    v_local = v_dim // tp
    n_v_local = n_v // tp

    # [tp, 2*qk_local + 2*v_local + 2*n_v_local, h]
    view = param.reshape(tp, -1, h)
    q_l, k_l, v_l, z_l, b_l, a_l = torch.split(
        view, [qk_local, qk_local, v_local, v_local, n_v_local, n_v_local], dim=1
    )
    # Bridge groups by num_qk_heads, but since we always end up flattening per
    # component, an equivalent simpler form: gather per rank then reshape.
    qkv = torch.cat(
        [q_l.reshape(qk_dim, h), k_l.reshape(qk_dim, h), v_l.reshape(v_dim, h)],
        dim=0,
    )
    z = z_l.reshape(v_dim, h)
    b = b_l.reshape(n_v, h)
    a = a_l.reshape(n_v, h)
    return qkv, z, b, a


def _split_bridge_conv1d(param: torch.Tensor, gdn: dict, tp: int) -> torch.Tensor:
    """Reorder Megatron's bridge-style fused GDN ``conv1d.weight`` into HF
    layout ``[all_q ; all_k ; all_v]``.

    Two post-all-gather layouts are observed depending on slime's
    ``all_gather_param`` and the param's ``partition_dim`` attribute:

    1. correct channel-major: ``(channels_full, 1, K)`` with ``channels_full =
       2*qk_dim + v_dim``.  Each rank's slab is contiguous in dim-0.
    2. kernel-spliced: ``(channels_full // tp, 1, K * tp)``.  This happens when
       the bridge's GDN conv1d has ``partition_dim == -1`` (which slime
       interprets as "concat along last dim" for the all-gather), so each rank's
       per-channel kernel ends up tiled along the trailing axis.  We recover by
       splitting that axis into ``(tp, K)``, moving ``tp`` to dim-0, and
       flattening.
    """
    qk_dim = gdn["linear_key_head_dim"] * gdn["linear_num_key_heads"]
    v_dim = gdn["linear_value_head_dim"] * gdn["linear_num_value_heads"]
    full_channels = 2 * qk_dim + v_dim

    if param.shape[0] == full_channels:
        # Layout 1: already channel-major full tensor.
        full = param
    elif tp > 1 and param.shape[0] == full_channels // tp and param.shape[-1] % tp == 0:
        # Layout 2: kernel-spliced -- (C/tp, 1, K*tp).
        per_rank_channels = full_channels // tp
        kernel = param.shape[-1] // tp
        # (C/tp, 1, K*tp) -> (C/tp, 1, tp, K) -> (tp, C/tp, 1, K) -> (C, 1, K)
        full = (
            param.reshape(per_rank_channels, 1, tp, kernel)
            .permute(2, 0, 1, 3)
            .reshape(full_channels, 1, kernel)
        )
    else:
        raise ValueError(
            f"unexpected conv1d weight shape {tuple(param.shape)} for tp={tp}, "
            f"qk_dim={qk_dim}, v_dim={v_dim}"
        )

    qk_local = qk_dim // tp
    v_local = v_dim // tp
    extra = full.shape[1:]
    view = full.reshape(tp, 2 * qk_local + v_local, *extra)
    q_l = view[:, :qk_local].reshape(qk_dim, *extra)
    k_l = view[:, qk_local:2 * qk_local].reshape(qk_dim, *extra)
    v_l = view[:, 2 * qk_local:].reshape(v_dim, *extra)
    return torch.cat([q_l, k_l, v_l], dim=0)


def _split_vision_linear_qkv(param: torch.Tensor, vc: dict) -> torch.Tensor:
    """Permute Megatron interleaved per-group QKV into HF concatenated [Q;K;V].

    Vision tower uses plain MHA (no GQA, no output gate), so each Megatron
    "group" carries exactly [q_head, k_head, v_head].  HF stores the same
    rows but reordered to [all_q ; all_k ; all_v].
    """
    n = vc["num_heads"]
    d = vc["head_dim"]
    h = vc["hidden_size"]

    is_bias = param.ndim == 1
    if is_bias:
        # [n*3*d] -> [n, 3, d] -> three [n*d] tensors -> concat
        view = param.view(n, 3, d)
        q = view[:, 0, :].reshape(-1)
        k = view[:, 1, :].reshape(-1)
        v = view[:, 2, :].reshape(-1)
        return torch.cat([q, k, v], dim=0)
    else:
        # [n*3*d, h] -> [n, 3, d, h] -> three [n*d, h] -> concat along dim 0
        view = param.view(n, 3, d, h)
        q = view[:, 0, :, :].reshape(-1, h)
        k = view[:, 1, :, :].reshape(-1, h)
        v = view[:, 2, :, :].reshape(-1, h)
        return torch.cat([q, k, v], dim=0)


def _convert_mtp_layer(args, name, param, layer_idx):
    """Convert MTP layer parameters from Megatron to HuggingFace format."""
    if "enorm.weight" in name:
        return [("mtp.pre_fc_norm_embedding.weight", param)]
    if "hnorm.weight" in name:
        return [("mtp.pre_fc_norm_hidden.weight", param)]
    if "final_layernorm.weight" in name:
        return [("mtp.norm.weight", param)]
    if "eh_proj.weight" in name:
        return [("mtp.fc.weight", param)]

    if "transformer_layer" in name:
        proxy_name = name.replace(f"mtp.layers.{layer_idx}.transformer_layer", f"decoder.layers.{layer_idx}")
        mapped_params = convert_qwen3_5_to_hf(args, proxy_name, param)

        final_params = []
        for hf_name, tensor in mapped_params:
            target_prefix = f"mtp.layers.{layer_idx}"
            if f"model.language_model.layers.{layer_idx}" in hf_name:
                new_hf_name = hf_name.replace(f"model.language_model.layers.{layer_idx}", target_prefix)
                final_params.append((new_hf_name, tensor))
            else:
                final_params.append((hf_name, tensor))
        return final_params

    return None


# Megatron-to-HF name mapping for vision encoder layers
_VISION_LAYER_MAP = {
    "self_attention.linear_qkv.weight": "attn.qkv.weight",
    "self_attention.linear_qkv.bias": "attn.qkv.bias",
    "self_attention.linear_proj.weight": "attn.proj.weight",
    "self_attention.linear_proj.bias": "attn.proj.bias",
    "self_attention.linear_qkv.layer_norm_weight": "norm1.weight",
    "self_attention.linear_qkv.layer_norm_bias": "norm1.bias",
    "mlp.linear_fc1.weight": "mlp.linear_fc1.weight",
    "mlp.linear_fc1.bias": "mlp.linear_fc1.bias",
    "mlp.linear_fc2.weight": "mlp.linear_fc2.weight",
    "mlp.linear_fc2.bias": "mlp.linear_fc2.bias",
    "mlp.linear_fc1.layer_norm_weight": "norm2.weight",
    "mlp.linear_fc1.layer_norm_bias": "norm2.bias",
}


def _convert_vision_to_hf(args, rest, param):
    """Convert vision model params from Megatron naming to HF naming.

    Maps decoder.layers.N.X -> model.visual.blocks.N.Y with proper name translation.
    Patch embed, pos embed, and merger params use simple prefix swap.

    Megatron's ``self_attention.linear_qkv`` stores QKV in TE's interleaved
    per-group layout, while HF's ``attn.qkv`` is a plain ``[Q;K;V]`` concat
    along dim 0.  We must permute, not just rename.
    """
    m = re.match(r"decoder\.layers\.(\d+)\.(.+)", rest)
    if m:
        idx, sub = m.groups()
        if sub == "self_attention.linear_qkv.weight" or sub == "self_attention.linear_qkv.bias":
            vc = _get_vision_config(args)
            permuted = _split_vision_linear_qkv(param, vc)
            hf_sub = _VISION_LAYER_MAP[sub]
            return [(f"model.visual.blocks.{idx}.{hf_sub}", permuted)]
        hf_sub = _VISION_LAYER_MAP.get(sub, sub)
        return [(f"model.visual.blocks.{idx}.{hf_sub}", param)]

    # Merger: patch_norm -> norm
    if rest.startswith("merger.patch_norm."):
        return [("model.visual.merger.norm." + rest[len("merger.patch_norm."):], param)]

    # Everything else (patch_embed, pos_embed, merger.linear_fc*, etc)
    return [("model.visual." + rest, param)]


def convert_qwen3_5_to_hf(args, name, param):
    """Convert Qwen3.5 model parameters from Megatron to HuggingFace format.

    Qwen3.5 uses model.language_model.layers prefix and has separate
    in_proj_qkv, in_proj_z, in_proj_b, in_proj_a for linear attention.
    """
    # Handle VL model: strip language_model prefix so LLM params match existing patterns
    if name.startswith("module.module.language_model."):
        name = "module.module." + name[len("module.module.language_model."):]
    while name.startswith("module.module.module."):
        name = name.replace("module.module.module.", "module.module.", 1)

    # Handle vision model parameters: convert Megatron naming to HF naming
    if name.startswith("module.module.vision_model."):
        return _convert_vision_to_hf(args, name[len("module.module.vision_model."):], param)

    # Handle MTP layers
    if "mtp.layers" in name:
        parts = name.split(".")
        try:
            layer_idx_loc = parts.index("layers") + 1
            layer_idx = parts[layer_idx_loc]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid MTP layer name format: {name}") from e

        result = _convert_mtp_layer(args, name, param, layer_idx)
        if result is not None:
            return result

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        prefix = f"model.language_model.layers.{layer_idx}"

        # experts (grouped gemm - fused format)
        if rest == "mlp.experts.linear_fc1":
            return [(f"{prefix}.mlp.experts.gate_up_proj", param)]
        elif rest == "mlp.experts.linear_fc2":
            return [(f"{prefix}.mlp.experts.down_proj", param)]

        # experts (ungrouped - individual expert format)
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2":
                return [(f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight", param)]
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"{prefix}.mlp.shared_expert.gate_proj.weight", gate_weight),
                    (f"{prefix}.mlp.shared_expert.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"{prefix}.mlp.shared_expert.down_proj.weight", param)]
            elif rest == "gate_weight":
                return [(f"{prefix}.mlp.shared_expert_gate.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"{prefix}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[2 * value_num_per_group, 1, 1], dim=1
            )
            q_param = (
                q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
                .transpose(1, 2)
                .reshape(-1, args.hidden_size)
            )
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"{prefix}.self_attn.q_proj.weight", q_param),
                (f"{prefix}.self_attn.k_proj.weight", k_param),
                (f"{prefix}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"{prefix}.self_attn.q_proj.bias", q_bias),
                (f"{prefix}.self_attn.k_proj.bias", k_bias),
                (f"{prefix}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{prefix}.mlp.gate_proj.weight", gate_weight),
                (f"{prefix}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"{prefix}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{prefix}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{prefix}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"{prefix}.post_attention_layernorm.weight", param)]
        elif rest == "mlp.router.weight":
            return [(f"{prefix}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"{prefix}.mlp.gate.e_score_correction_bias", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"{prefix}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"{prefix}.self_attn.k_norm.weight", param)]

        # ----- bridge-style GDN (linear-attention) layer params -----
        # These appear when the megatron model is built via
        # AutoBridge.to_megatron_provider() (i.e. --megatron-to-hf-mode bridge).
        elif rest == "self_attention.A_log":
            return [(f"{prefix}.linear_attn.A_log", param)]
        elif rest == "self_attention.dt_bias":
            return [(f"{prefix}.linear_attn.dt_bias", param)]
        elif rest == "self_attention.out_proj.weight":
            return [(f"{prefix}.linear_attn.out_proj.weight", param)]
        elif rest == "self_attention.in_proj.layer_norm_weight":
            # Pre-attention layernorm is fused into the in_proj module by TE.
            return [(f"{prefix}.input_layernorm.weight", param)]
        elif rest == "self_attention.out_norm.weight":
            # Megatron stores zero-centered (FLA's FusedRMSNormGated convention);
            # HF expects standard RMSNorm (gamma initialised to 1).
            return [(f"{prefix}.linear_attn.norm.weight", param + 1)]
        elif rest == "self_attention.conv1d.weight":
            gdn = _gdn_dims(args)
            return [(
                f"{prefix}.linear_attn.conv1d.weight",
                _split_bridge_conv1d(param, gdn, _tp_size(args)),
            )]
        elif rest == "self_attention.in_proj.weight":
            gdn = _gdn_dims(args)
            qkv, z, b, a = _split_bridge_in_proj(param, gdn, _tp_size(args))
            return [
                (f"{prefix}.linear_attn.in_proj_qkv.weight", qkv),
                (f"{prefix}.linear_attn.in_proj_z.weight", z),
                (f"{prefix}.linear_attn.in_proj_b.weight", b),
                (f"{prefix}.linear_attn.in_proj_a.weight", a),
            ]

        elif rest.startswith("self_attention.") and rest[len("self_attention.") :] in [
            "input_layernorm.weight",
            # linear attn (Qwen3.5 uses separate in_proj_b/in_proj_a)
            "linear_attn.A_log",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            # gated attn (full attention layers)
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ]:
            rest = rest[len("self_attention.") :]
            return [(f"{prefix}.{rest}", param)]

    raise ValueError(f"Unknown parameter name: {name}")
