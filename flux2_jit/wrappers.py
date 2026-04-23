from __future__ import annotations

from typing import Any, Dict

import torch

from .interpolation import compute_indices_digest, irregular_interpolation
from .runtime import JiTRuntime
from .utils import build_txt_ids, is_flux2_model, log_info, unpack_tokens_to_image


JIT_CONFIG_KEY = "flux2_jit"
JIT_RUNTIME_KEY = "flux2_jit_runtime"


def flux2_jit_diffusion_model_wrapper(executor, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None, transformer_options=None, **kwargs):
    if transformer_options is None:
        transformer_options = {}
    runtime: JiTRuntime | None = transformer_options.get(JIT_RUNTIME_KEY)
    config = transformer_options.get(JIT_CONFIG_KEY)
    diffusion_model = executor.class_obj

    if runtime is None or config is None:
        return executor(x, timestep, context, y, guidance, ref_latents, control, transformer_options, **kwargs)
    if control is not None or not hasattr(diffusion_model, "forward_orig"):
        if runtime is not None and config.verbose and not runtime.wrapper_fallback_logged:
            reasons = []
            if control is not None:
                reasons.append("control")
            if not hasattr(diffusion_model, "forward_orig"):
                reasons.append("missing forward_orig")
            log_info(config.verbose, f"Wrapper fell back to dense path ({', '.join(reasons)})")
            runtime.wrapper_fallback_logged = True
        return executor(x, timestep, context, y, guidance, ref_latents, control, transformer_options, **kwargs)
    if runtime.current_indices is None or runtime.total_tokens is None:
        runtime.initialize(diffusion_model, x)
    if runtime.current_indices is None or runtime.total_tokens is None:
        return executor(x, timestep, context, y, guidance, ref_latents, control, transformer_options, **kwargs)
    if runtime.current_indices.numel() >= runtime.total_tokens:
        if config.verbose and not runtime.wrapper_dense_logged:
            log_info(config.verbose, f"Wrapper using dense path ({runtime.total_tokens}/{runtime.total_tokens} active tokens)")
            runtime.wrapper_dense_logged = True
        return executor(x, timestep, context, y, guidance, ref_latents, control, transformer_options, **kwargs)

    patch_size = diffusion_model.patch_size
    _, _, h_orig, w_orig = x.shape
    h_len = (h_orig + (patch_size // 2)) // patch_size
    w_len = (w_orig + (patch_size // 2)) // patch_size

    img_tokens, img_ids = diffusion_model.process_img(x, transformer_options=transformer_options)
    active_indices = runtime.current_indices.to(img_tokens.device)
    active_indices_digest = runtime.current_indices_digest
    if active_indices_digest is None:
        active_indices_digest = compute_indices_digest(active_indices)
        runtime.current_indices_digest = active_indices_digest

    if config.verbose and not runtime.wrapper_sparse_logged:
        mode = "sparse path with reference latents" if ref_latents is not None else "sparse path"
        log_info(config.verbose, f"Wrapper using {mode} ({active_indices.numel()}/{runtime.total_tokens} active tokens)")
        runtime.wrapper_sparse_logged = True

    img_active = img_tokens[:, active_indices, :]
    img_ids_active = img_ids[:, active_indices, :]
    sparse_tokens = img_active
    sparse_ids = img_ids_active
    sparse_timestep = timestep
    sparse_transformer_options = transformer_options
    timestep_zero_index = None

    if ref_latents is not None:
        ref_num_tokens = []
        h = 0
        w = 0
        index = 0
        ref_latents_method = kwargs.get("ref_latents_method", diffusion_model.params.default_ref_method)
        timestep_zero = ref_latents_method == "index_timestep_zero"

        for ref in ref_latents:
            if ref_latents_method in ("index", "index_timestep_zero"):
                index += diffusion_model.params.ref_index_scale
                h_offset = 0
                w_offset = 0
            elif ref_latents_method == "uxo":
                index = 0
                h_offset = h_len * patch_size + h
                w_offset = w_len * patch_size + w
                h += ref.shape[-2]
                w += ref.shape[-1]
            else:
                index = 1
                h_offset = 0
                w_offset = 0
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

            ref_tokens, ref_ids = diffusion_model.process_img(
                ref,
                index=index,
                h_offset=h_offset,
                w_offset=w_offset,
                transformer_options=transformer_options,
            )
            if ref_tokens.shape[0] != sparse_tokens.shape[0] or ref_ids.shape[0] != sparse_ids.shape[0]:
                raise ValueError(
                    "ref_latents batch mismatch before concat: "
                    f"sparse_tokens={tuple(sparse_tokens.shape)}, ref_tokens={tuple(ref_tokens.shape)}, "
                    f"sparse_ids={tuple(sparse_ids.shape)}, ref_ids={tuple(ref_ids.shape)}, "
                    f"index={index}, h_offset={h_offset}, w_offset={w_offset}"
                )
            sparse_tokens = torch.cat([sparse_tokens, ref_tokens], dim=1)
            sparse_ids = torch.cat([sparse_ids, ref_ids], dim=1)
            ref_num_tokens.append(ref_tokens.shape[1])

        if timestep_zero and ref_num_tokens:
            sparse_timestep = torch.cat([timestep, timestep * 0], dim=0)
            timestep_zero_index = [[img_active.shape[1], sparse_ids.shape[1]]]

        sparse_transformer_options = transformer_options.copy()
        sparse_transformer_options["reference_image_num_tokens"] = ref_num_tokens

    txt_ids = build_txt_ids(diffusion_model, batch_size=context.shape[0], context_len=context.shape[1], device=x.device)

    sparse_output_tokens = diffusion_model.forward_orig(
        sparse_tokens,
        sparse_ids,
        context,
        txt_ids,
        sparse_timestep,
        y,
        guidance,
        control,
        timestep_zero_index=timestep_zero_index,
        transformer_options=sparse_transformer_options,
        attn_mask=kwargs.get("attention_mask"),
    )
    sparse_output_tokens = sparse_output_tokens[:, : img_active.shape[1], :]

    full_output_tokens = irregular_interpolation(
        sparse_output_tokens,
        active_indices,
        active_indices_digest,
        runtime.total_tokens,
        runtime.token_dim,
        runtime.grid_h,
        runtime.grid_w,
        runtime.config.blur_scale,
        runtime.coord_cache,
    )
    return unpack_tokens_to_image(full_output_tokens, patch_size, h_len, w_len, h_orig, w_orig)
