from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class CoordCache:
    coords_full: Optional[torch.Tensor] = None
    height: Optional[int] = None
    width: Optional[int] = None

    def get(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        if self.coords_full is None or self.height != height or self.width != width or self.coords_full.device != device:
            coords_y, coords_x = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing="ij",
            )
            self.coords_full = torch.stack([coords_y.reshape(-1), coords_x.reshape(-1)], dim=-1)
            self.height = height
            self.width = width
        return self.coords_full


def gaussian_blur_2d(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 1 or sigma <= 0.0:
        return img
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=img.device, dtype=img.dtype)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    x_kernel = pdf / pdf.sum()
    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[1], 1, kernel2d.shape[0], kernel2d.shape[1])
    img = F.pad(img, [kernel_size // 2] * 4, mode="reflect")
    return F.conv2d(img, kernel2d, groups=img.shape[1])


def calculate_blur_params(sparsity_ratio: float, blur_scale: float) -> tuple[int, float]:
    if sparsity_ratio <= 0.0 or sparsity_ratio >= 1.0:
        return 3, 1.0
    characteristic_distance = 1.0 / math.sqrt(sparsity_ratio)
    sigma = max(1.0, min(10.0, blur_scale * characteristic_distance))
    kernel_size = 2 * math.ceil(3.0 * sigma) + 1
    return kernel_size, sigma


def irregular_interpolation(
    y_active: torch.Tensor,
    active_indices: torch.Tensor,
    total_tokens: int,
    token_dim: int,
    grid_h: int,
    grid_w: int,
    blur_scale: float,
    coord_cache: CoordCache,
) -> torch.Tensor:
    if active_indices.numel() == 0:
        return torch.zeros(y_active.shape[0], total_tokens, token_dim, device=y_active.device, dtype=y_active.dtype)

    coords_full = coord_cache.get(grid_h, grid_w, y_active.device)
    coords_active = coords_full[active_indices]
    dist = torch.cdist(coords_full.float(), coords_active.float(), p=2)
    nearest_idx = dist.argmin(dim=-1)

    y_active_expanded = y_active.permute(1, 0, 2)
    gathered = y_active_expanded[nearest_idx]
    y_full_nearest = gathered.permute(1, 0, 2).contiguous()

    y_full_2d_nearest = y_full_nearest.reshape(y_active.shape[0], grid_h, grid_w, token_dim).permute(0, 3, 1, 2)
    sparsity_ratio = float(active_indices.numel()) / float(total_tokens)
    kernel_size, sigma = calculate_blur_params(sparsity_ratio, blur_scale)
    y_full_2d_blur = gaussian_blur_2d(y_full_2d_nearest, kernel_size=kernel_size, sigma=sigma)

    active_mask = torch.zeros(total_tokens, device=y_active.device, dtype=y_active.dtype)
    active_mask[active_indices] = 1.0
    active_mask_2d = active_mask.reshape(1, 1, grid_h, grid_w)
    inactive_mask_2d = 1.0 - active_mask_2d
    y_full_2d = y_full_2d_nearest * active_mask_2d + y_full_2d_blur * inactive_mask_2d
    return y_full_2d.permute(0, 2, 3, 1).reshape(y_active.shape[0], total_tokens, token_dim)
