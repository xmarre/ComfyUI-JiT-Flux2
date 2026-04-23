from __future__ import annotations

from typing import Any, Dict

import torch

from .interpolation import irregular_interpolation
from .runtime import JiTRuntime
from .utils import build_txt_ids, is_flux2_model, unpack_tokens_to_image


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
    if ref_latents is not None or control is not None or not hasattr(diffusion_model, "forward_orig"):
        return executor(x, timestep, context, y, guidance, ref_latents, control, transformer_options, **kwargs)
    if runtime.current_indices is None or runtime.total_tokens is None:
        runtime.initialize(diffusion_model, x)
    if runtime.current_indices is None or runtime.current_indices.numel() >= runtime.total_tokens:
        return executor(x, timestep, context, y, guidance, ref_latents, control, transformer_options, **kwargs)

    patch_size = diffusion_model.patch_size
    _, _, h_orig, w_orig = x.shape
    h_len = (h_orig + (patch_size // 2)) // patch_size
    w_len = (w_orig + (patch_size // 2)) // patch_size

    img_tokens, img_ids = diffusion_model.process_img(x, transformer_options=transformer_options)
    active_indices = runtime.current_indices.to(img_tokens.device)
    img_active = img_tokens[:, active_indices, :]
    img_ids_active = img_ids[:, active_indices, :]
    txt_ids = build_txt_ids(diffusion_model, batch_size=context.shape[0], context_len=context.shape[1], device=x.device)

    sparse_output_tokens = diffusion_model.forward_orig(
        img_active,
        img_ids_active,
        context,
        txt_ids,
        timestep,
        y,
        guidance,
        control,
        transformer_options=transformer_options,
        attn_mask=kwargs.get("attention_mask"),
    )

    full_output_tokens = irregular_interpolation(
        sparse_output_tokens,
        active_indices,
        runtime.total_tokens,
        runtime.token_dim,
        runtime.grid_h,
        runtime.grid_w,
        runtime.config.blur_scale,
        runtime.coord_cache,
    )
    return unpack_tokens_to_image(full_output_tokens, patch_size, h_len, w_len, h_orig, w_orig)
