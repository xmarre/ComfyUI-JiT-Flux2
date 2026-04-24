from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class InterpolationPlan:
    nearest_idx: torch.Tensor
    active_mask_2d: torch.Tensor
    kernel_size: int
    sigma: float
    blur_mix: float


@dataclass
class CoordCache:
    coords_full: Optional[torch.Tensor] = None
    height: Optional[int] = None
    width: Optional[int] = None
    interpolation_plans: dict[tuple[bytes, int, int, int, int, str], InterpolationPlan] = field(default_factory=dict)

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
            self.interpolation_plans.clear()
        return self.coords_full

    def get_interpolation_plan(
        self,
        active_indices: torch.Tensor,
        active_indices_digest: bytes,
        total_tokens: int,
        grid_h: int,
        grid_w: int,
        blur_scale: float,
    ) -> InterpolationPlan:
        key = (
            active_indices_digest,
            total_tokens,
            grid_h,
            grid_w,
            round(float(blur_scale) * 1_000_000.0),
            str(active_indices.device),
        )
        cached = self.interpolation_plans.get(key)
        if cached is not None:
            return cached

        coords_full = self.get(grid_h, grid_w, active_indices.device)
        coords_active = coords_full[active_indices]
        dist = torch.cdist(coords_full.float(), coords_active.float(), p=2)
        nearest_idx = dist.argmin(dim=-1)

        sparsity_ratio = float(active_indices.numel()) / float(total_tokens)
        kernel_size, sigma, blur_mix = calculate_blur_params(sparsity_ratio, blur_scale)
        max_kernel = 2 * max(0, min(grid_h, grid_w) - 1) + 1
        kernel_size = min(kernel_size, max_kernel)

        active_mask = torch.zeros(total_tokens, device=active_indices.device, dtype=torch.float32)
        active_mask[active_indices] = 1.0
        active_mask_2d = active_mask.reshape(1, 1, grid_h, grid_w)

        plan = InterpolationPlan(
            nearest_idx=nearest_idx,
            active_mask_2d=active_mask_2d,
            kernel_size=kernel_size,
            sigma=sigma,
            blur_mix=blur_mix,
        )
        self.interpolation_plans[key] = plan
        return plan


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


def calculate_blur_params(sparsity_ratio: float, blur_scale: float) -> tuple[int, float, float]:
    if sparsity_ratio <= 0.0 or sparsity_ratio >= 1.0 or blur_scale <= 0.0:
        return 1, 0.0, 0.0

    characteristic_distance = 1.0 / math.sqrt(sparsity_ratio)

    # The JiT paper's blur operator is applied to the approximated velocity
    # field. In ComfyUI's Flux wrapper this lifter is applied at the diffusion
    # model boundary, so replacing inactive tokens with a fully blurred field is
    # visibly over-smoothing on Flux.2. Keep blur as a low-pass correction to
    # nearest-neighbor interpolation instead of making it the whole inactive
    # value, and allow sub-pixel sigmas for dense 40-70% anchor stages.
    sigma = max(0.0, min(10.0, blur_scale * characteristic_distance))
    if sigma < 0.25:
        return 1, 0.0, 0.0

    kernel_size = 2 * math.ceil(3.0 * sigma) + 1
    inactive_ratio = max(0.0, min(1.0, 1.0 - sparsity_ratio))
    blur_mix = max(0.0, min(1.0, blur_scale * inactive_ratio))
    return kernel_size, sigma, blur_mix


def compute_indices_digest(active_indices: torch.Tensor) -> bytes:
    return hashlib.blake2b(
        active_indices.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy().tobytes(),
        digest_size=16,
    ).digest()


def irregular_interpolation(
    y_active: torch.Tensor,
    active_indices: torch.Tensor,
    active_indices_digest: bytes,
    total_tokens: int,
    token_dim: int,
    grid_h: int,
    grid_w: int,
    blur_scale: float,
    coord_cache: CoordCache,
) -> torch.Tensor:
    if active_indices.numel() == 0:
        return torch.zeros(y_active.shape[0], total_tokens, token_dim, device=y_active.device, dtype=y_active.dtype)

    plan = coord_cache.get_interpolation_plan(
        active_indices,
        active_indices_digest,
        total_tokens,
        grid_h,
        grid_w,
        blur_scale,
    )

    y_active_expanded = y_active.permute(1, 0, 2)
    gathered = y_active_expanded[plan.nearest_idx]
    y_full_nearest = gathered.permute(1, 0, 2).contiguous()

    y_full_2d_nearest = y_full_nearest.reshape(y_active.shape[0], grid_h, grid_w, token_dim).permute(0, 3, 1, 2)
    y_full_2d_blur = gaussian_blur_2d(y_full_2d_nearest, kernel_size=plan.kernel_size, sigma=plan.sigma)

    active_mask_2d = plan.active_mask_2d.to(dtype=y_active.dtype)
    inactive_mask_2d = 1.0 - active_mask_2d
    blur_mix = y_active.new_tensor(plan.blur_mix)
    y_full_2d_inactive = y_full_2d_nearest + (y_full_2d_blur - y_full_2d_nearest) * blur_mix
    y_full_2d = y_full_2d_nearest * active_mask_2d + y_full_2d_inactive * inactive_mask_2d
    return y_full_2d.permute(0, 2, 3, 1).reshape(y_active.shape[0], total_tokens, token_dim)
