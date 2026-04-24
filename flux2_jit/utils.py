from __future__ import annotations

import math

import torch
from einops import rearrange


def is_flux2_model(model_patcher) -> bool:
    try:
        image_model = model_patcher.model.model_config.unet_config.get("image_model")
        return image_model == "flux2"
    except (AttributeError, KeyError, TypeError):
        return False


def _deterministic_hash_order(indices: torch.Tensor, grid_w: int) -> torch.Tensor:
    indices64 = indices.to(dtype=torch.int64)
    y = torch.div(indices64, grid_w, rounding_mode="floor")
    x = indices64.remainder(grid_w)
    key = torch.remainder(
        indices64 * 1103515245 + y * 12345 + x * 2654435761,
        2147483647,
    )
    return torch.argsort(key)


def _take_deterministic(indices: torch.Tensor, count: int, grid_w: int) -> torch.Tensor:
    if count <= 0 or indices.numel() == 0:
        return indices[:0]
    if count >= indices.numel():
        return indices
    return indices[_deterministic_hash_order(indices, grid_w)[:count]]


def _fill_sparse_budget(
    boundary_indices: torch.Tensor,
    phase_indices: list[torch.Tensor],
    target_count: int,
    total_tokens: int,
    grid_w: int,
    device: torch.device,
) -> torch.Tensor:
    if boundary_indices.numel() >= target_count:
        return _take_deterministic(boundary_indices, target_count, grid_w).sort().values.long()

    kept = [boundary_indices]
    kept_count = int(boundary_indices.numel())

    for phase in phase_indices:
        if phase.numel() == 0:
            continue
        remaining = target_count - kept_count
        if remaining <= 0:
            break
        selected_phase = _take_deterministic(phase, remaining, grid_w)
        kept.append(selected_phase)
        kept_count += int(selected_phase.numel())
        if kept_count >= target_count:
            break

    indices = torch.cat(kept).unique(sorted=True).long()
    if indices.numel() < target_count:
        selected = torch.zeros(total_tokens, dtype=torch.bool, device=device)
        selected[indices] = True
        available = torch.arange(total_tokens, device=device, dtype=torch.long)[~selected]
        supplement = _take_deterministic(available, target_count - int(indices.numel()), grid_w)
        indices = torch.cat([indices, supplement]).unique(sorted=True).long()

    if indices.numel() > target_count:
        indices = _take_deterministic(indices, target_count, grid_w)
    return indices.sort().values.long()


def create_sparse_grid(
    grid_h: int,
    grid_w: int,
    sparsity_ratio: float,
    device: torch.device,
    use_checkerboard: bool,
) -> torch.Tensor:
    total_tokens = grid_h * grid_w
    target_count = max(1, min(total_tokens, int(round(total_tokens * sparsity_ratio))))
    if target_count >= total_tokens:
        return torch.arange(total_tokens, device=device, dtype=torch.long)

    y_coords = torch.arange(grid_h, device=device)
    x_coords = torch.arange(grid_w, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    all_indices = torch.arange(total_tokens, device=device, dtype=torch.long)
    boundary_mask = ((yy == 0) | (yy == grid_h - 1) | (xx == 0) | (xx == grid_w - 1)).reshape(-1)
    boundary_indices = all_indices[boundary_mask]

    if use_checkerboard:
        core_mask = ((yy % 2 == 0) & (xx % 2 == 0)).reshape(-1) & ~boundary_mask
        return _fill_sparse_budget(
            boundary_indices=boundary_indices,
            phase_indices=[all_indices[core_mask]],
            target_count=target_count,
            total_tokens=total_tokens,
            grid_w=grid_w,
            device=device,
        )

    stride = max(1, int(math.ceil(math.sqrt(1.0 / max(float(sparsity_ratio), 1e-12)))))
    offsets = [(oy, ox) for oy in range(stride) for ox in range(stride)]
    offsets.sort(key=lambda item: (item[0] * item[0] + item[1] * item[1], item[0] + item[1], item[0], item[1]))

    phases = []
    for offset_y, offset_x in offsets:
        phase_mask = ((yy % stride == offset_y) & (xx % stride == offset_x)).reshape(-1) & ~boundary_mask
        phases.append(all_indices[phase_mask])

    return _fill_sparse_budget(
        boundary_indices=boundary_indices,
        phase_indices=phases,
        target_count=target_count,
        total_tokens=total_tokens,
        grid_w=grid_w,
        device=device,
    )


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
        print(f"[Flux2-JiT] {message}", flush=True)
