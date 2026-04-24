from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from .config import JiTConfig
from .interpolation import CoordCache, compute_indices_digest, irregular_interpolation
from .utils import create_sparse_grid, log_info, unpack_tokens_to_image


@dataclass
class JiTRuntime:
    config: JiTConfig
    total_steps: int
    sigmas: torch.Tensor
    global_noise_image: torch.Tensor
    current_stage: Optional[int] = None
    current_indices: Optional[torch.Tensor] = None
    current_indices_digest: Optional[bytes] = None
    last_x_image: Optional[torch.Tensor] = None
    last_velocity_image: Optional[torch.Tensor] = None
    last_sigma: Optional[torch.Tensor] = None
    grid_h: Optional[int] = None
    grid_w: Optional[int] = None
    total_tokens: Optional[int] = None
    token_dim: Optional[int] = None
    patch_size: Optional[int] = None
    coord_cache: CoordCache = field(default_factory=CoordCache)
    pending_target_stage: Optional[int] = None
    pending_newly_activated: Optional[torch.Tensor] = None
    pending_target_tokens: Optional[torch.Tensor] = None
    pending_relax_remaining: int = 0
    wrapper_call_count: int = 0
    wrapper_sparse_call_count: int = 0
    wrapper_dense_call_count: int = 0
    wrapper_fallback_call_count: int = 0
    wrapper_last_mode: Optional[str] = None
    wrapper_last_fallback_reasons: Optional[str] = None
    wrapper_sparse_logged: bool = False
    wrapper_dense_logged: bool = False
    wrapper_fallback_logged: bool = False

    def _set_current_indices(self, indices: torch.Tensor) -> None:
        self.current_indices = indices
        self.current_indices_digest = compute_indices_digest(indices)

    def initialize(self, diffusion_model, x: torch.Tensor) -> None:
        if self.current_stage is not None:
            return
        _, _, h_orig, w_orig = x.shape
        patch_size = diffusion_model.patch_size
        h_len = (h_orig + (patch_size // 2)) // patch_size
        w_len = (w_orig + (patch_size // 2)) // patch_size
        full_tokens, _ = diffusion_model.process_img(x)
        self.grid_h = h_len
        self.grid_w = w_len
        self.total_tokens = full_tokens.shape[1]
        self.token_dim = full_tokens.shape[2]
        self.patch_size = patch_size
        self.current_stage = self.config.num_stages - 1
        initial_ratio = self.config.ratio_of_stage(self.current_stage)
        self._set_current_indices(create_sparse_grid(
            grid_h=h_len,
            grid_w=w_len,
            sparsity_ratio=initial_ratio,
            device=x.device,
            use_checkerboard=self.config.use_checkerboard_init,
        ))
        log_info(self.config.verbose, f"Initialized stage {self.current_stage} with {self.current_indices.numel()}/{self.total_tokens} active tokens")

    def stage_steps(self) -> tuple[int, ...]:
        return self.config.stage_steps_for_total(self.total_steps)

    def target_stage_for_step(self, step_index: int) -> int:
        for stage_idx, stage_step in enumerate(self.stage_steps()):
            if step_index < stage_step:
                return self.config.num_stages - 1 - stage_idx
        return 0

    def _compute_importance_map(self, velocity_tokens: torch.Tensor) -> torch.Tensor:
        v_2d = velocity_tokens.reshape(velocity_tokens.shape[0], self.grid_h, self.grid_w, self.token_dim).permute(0, 3, 1, 2)
        kernel_size = 3
        mean = F.avg_pool2d(v_2d, kernel_size, stride=1, padding=kernel_size // 2)
        var = F.avg_pool2d(v_2d.square(), kernel_size, stride=1, padding=kernel_size // 2) - mean.square()
        return var.mean(dim=1).mean(dim=0)

    def _adaptive_densify(self, target_count: int, importance_map: torch.Tensor) -> torch.Tensor:
        current_indices = self.current_indices
        importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)
        importance_flat = importance_map.reshape(-1)
        mask = torch.ones(self.total_tokens, dtype=torch.bool, device=importance_map.device)
        mask[current_indices] = False
        candidate_indices = torch.arange(self.total_tokens, device=importance_map.device)[mask]
        candidate_importance = importance_flat[candidate_indices]
        num_to_add = target_count - current_indices.numel()
        if num_to_add <= 0:
            return current_indices
        if num_to_add >= candidate_indices.numel():
            return torch.cat([current_indices, candidate_indices]).sort().values.long()
        _, topk = torch.topk(candidate_importance, num_to_add)
        selected = candidate_indices[topk]
        return torch.cat([current_indices, selected]).sort().values.long()

    def _microflow_bridge(
        self,
        current_tokens: torch.Tensor,
        new_indices: torch.Tensor,
        target_tokens: torch.Tensor,
        steps_remaining: int,
    ) -> torch.Tensor:
        if new_indices.numel() == 0:
            return current_tokens
        if steps_remaining <= 0:
            current_tokens[:, new_indices, :] = target_tokens
            return current_tokens
        current = current_tokens[:, new_indices, :].clone()
        weight = 1.0 / float(steps_remaining)
        current_tokens[:, new_indices, :] = (1.0 - weight) * current + weight * target_tokens
        return current_tokens

    def _clear_pending_microflow(self) -> None:
        self.pending_target_stage = None
        self.pending_newly_activated = None
        self.pending_target_tokens = None
        self.pending_relax_remaining = 0

    def _apply_pending_microflow(self, diffusion_model, x: torch.Tensor) -> torch.Tensor:
        if self.pending_relax_remaining <= 0:
            return x
        assert self.pending_newly_activated is not None
        assert self.pending_target_tokens is not None

        current_tokens, _ = diffusion_model.process_img(x)
        current_tokens = self._microflow_bridge(
            current_tokens,
            self.pending_newly_activated,
            self.pending_target_tokens,
            self.pending_relax_remaining,
        )
        x = unpack_tokens_to_image(current_tokens, self.patch_size, self.grid_h, self.grid_w, x.shape[2], x.shape[3])
        self.pending_relax_remaining -= 1

        if self.pending_relax_remaining == 0:
            assert self.pending_target_stage is not None
            self.current_stage = self.pending_target_stage
            log_info(
                self.config.verbose,
                f"Completed microflow relaxation; stage committed to {self.current_stage}",
            )
            self._clear_pending_microflow()
        return x

    def maybe_apply_stage_transition(self, diffusion_model, x: torch.Tensor, step_index: int, sigma: torch.Tensor) -> torch.Tensor:
        self.initialize(diffusion_model, x)
        assert self.current_stage is not None
        target_stage = self.target_stage_for_step(step_index)
        if target_stage >= self.current_stage:
            return x
        if self.last_x_image is None or self.last_velocity_image is None or self.last_sigma is None:
            return x

        target_count = max(1, int(self.total_tokens * self.config.ratio_of_stage(target_stage)))
        last_velocity_tokens, _ = diffusion_model.process_img(self.last_velocity_image)

        if self.config.use_adaptive:
            importance_map = self._compute_importance_map(last_velocity_tokens)
            new_indices = self._adaptive_densify(target_count, importance_map)
            log_info(self.config.verbose, f"Step {step_index}: adaptive transition {self.current_stage}->{target_stage}")
        else:
            staged_indices = create_sparse_grid(
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                sparsity_ratio=self.config.ratio_of_stage(target_stage),
                device=x.device,
                use_checkerboard=self.config.use_checkerboard_init,
            )
            new_indices = torch.cat([self.current_indices, staged_indices]).unique(sorted=True)
            log_info(self.config.verbose, f"Step {step_index}: fixed-grid transition {self.current_stage}->{target_stage}")

        new_mask = ~torch.isin(new_indices, self.current_indices)
        newly_activated = new_indices[new_mask]
        if newly_activated.numel() == 0:
            self._set_current_indices(new_indices)
            self.current_stage = target_stage
            return x

        last_x0_image = self.last_x_image - self.last_velocity_image * self.last_sigma.view(-1, 1, 1, 1)
        x0_tokens, _ = diffusion_model.process_img(last_x0_image)
        assert self.current_indices_digest is not None
        x0_interpolated = irregular_interpolation(
            x0_tokens[:, self.current_indices, :],
            self.current_indices,
            self.current_indices_digest,
            self.total_tokens,
            self.token_dim,
            self.grid_h,
            self.grid_w,
            self.config.blur_scale,
            self.coord_cache,
        )
        noise_tokens, _ = diffusion_model.process_img(self.global_noise_image)
        sigma_scalar = sigma.flatten()[0].to(x.dtype)
        target_tokens = x0_interpolated[:, newly_activated, :] * (1.0 - sigma_scalar) + noise_tokens[:, newly_activated, :] * sigma_scalar

        current_tokens, _ = diffusion_model.process_img(x)
        steps = int(self.config.microflow_relax_steps)

        # The sparse wrapper consumes every token in current_indices on the very
        # next model call. Therefore newly activated anchors must reach their
        # DMF target before current_indices is expanded. Spreading this bridge
        # across real denoising steps feeds partially relaxed, wrong-noise-level
        # anchors into the transformer and leaves the DMF target stale as sigma
        # advances.
        if steps <= 0:
            current_tokens = self._microflow_bridge(current_tokens, newly_activated, target_tokens, 0)
        else:
            for steps_remaining in range(steps, 0, -1):
                current_tokens = self._microflow_bridge(
                    current_tokens,
                    newly_activated,
                    target_tokens,
                    steps_remaining,
                )

        x = unpack_tokens_to_image(current_tokens, self.patch_size, self.grid_h, self.grid_w, x.shape[2], x.shape[3])
        self._set_current_indices(new_indices)
        self.current_stage = target_stage
        self._clear_pending_microflow()
        log_info(
            self.config.verbose,
            f"Activated {newly_activated.numel()} new tokens; stage committed to {target_stage}",
        )
        return x
