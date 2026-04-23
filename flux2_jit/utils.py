from __future__ import annotations

import math
import sys
from typing import Tuple

import numpy as np
import torch
from einops import rearrange


def is_flux2_model(model_patcher) -> bool:
    try:
        image_model = model_patcher.model.model_config.unet_config.get("image_model")
        return image_model == "flux2"
    except (AttributeError, KeyError, TypeError):
        return False


def create_sparse_grid(
    grid_h: int,
    grid_w: int,
    sparsity_ratio: float,
    device: torch.device,
    use_checkerboard: bool,
) -> torch.Tensor:
    total_tokens = grid_h * grid_w
    target_count = max(1, int(total_tokens * sparsity_ratio))

    if use_checkerboard:
        i_coords = torch.arange(grid_h, device=device)
        j_coords = torch.arange(grid_w, device=device)
        ii, jj = torch.meshgrid(i_coords, j_coords, indexing="ij")
        all_indices = torch.arange(total_tokens, device=device)
        mask_core = (ii % 2 == 0) & (jj % 2 == 0)
        mask_boundary = (ii == 0) | (ii == grid_h - 1) | (jj == 0) | (jj == grid_w - 1)
        indices = all_indices[(mask_core | mask_boundary).reshape(-1)]
    else:
        stride = max(1, int(np.sqrt(1.0 / sparsity_ratio)))
        grid_y = torch.arange(0, grid_h, stride, device=device)
        grid_x = torch.arange(0, grid_w, stride, device=device)
        if grid_y.numel() == 0 or grid_y[-1] != grid_h - 1:
            grid_y = torch.cat([grid_y, torch.tensor([grid_h - 1], device=device)])
        if grid_x.numel() == 0 or grid_x[-1] != grid_w - 1:
            grid_x = torch.cat([grid_x, torch.tensor([grid_w - 1], device=device)])
        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        indices = mesh_y.reshape(-1) * grid_w + mesh_x.reshape(-1)

    if indices.numel() < target_count:
        mask = torch.ones(total_tokens, dtype=torch.bool, device=device)
        mask[indices] = False
        available = torch.arange(total_tokens, device=device)[mask]
        count = min(target_count - indices.numel(), available.numel())
        if count > 0:
            supplement = available[torch.randperm(available.numel(), device=device)[:count]]
            indices = torch.cat([indices, supplement])
    elif indices.numel() > target_count:
        keep = torch.randperm(indices.numel(), device=device)[:target_count]
        indices = indices[keep]

    return indices.sort().values.long()


def build_txt_ids(diffusion_model, batch_size: int, context_len: int, device: torch.device) -> torch.Tensor:
    txt_ids = torch.zeros((batch_size, context_len, len(diffusion_model.params.axes_dim)), device=device, dtype=torch.float32)
    if len(diffusion_model.params.txt_ids_dims) > 0:
        lin = torch.linspace(0, context_len - 1, steps=context_len, device=device, dtype=torch.float32)
        for idx in diffusion_model.params.txt_ids_dims:
            txt_ids[:, :, idx] = lin
    return txt_ids


def unpack_tokens_to_image(tokens: torch.Tensor, patch_size: int, h_len: int, w_len: int, h_orig: int, w_orig: int) -> torch.Tensor:
    return rearrange(
        tokens,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h_len,
        w=w_len,
        ph=patch_size,
        pw=patch_size,
    )[:, :, :h_orig, :w_orig]


def sigma_to_velocity(x: torch.Tensor, denoised: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    while sigma.ndim < x.ndim:
        sigma = sigma.unsqueeze(-1)
    return (x - denoised) / sigma


def log_info(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[Flux2-JiT] {message}", file=sys.stderr, flush=True)
