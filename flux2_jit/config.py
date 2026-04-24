from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


DEFAULT_4X_STEPS = 18
DEFAULT_7X_STEPS = 11
DEFAULT_STAGE_RATIOS = (0.4, 0.65, 1.0)
DEFAULT_4X_SPARSITY = (0.45, 0.70, 1.0)
DEFAULT_7X_SPARSITY = (0.40, 0.66, 1.0)


@dataclass(frozen=True)
class JiTConfig:
    expected_total_steps: int
    stage_ratios: Tuple[float, ...]
    sparsity_ratios: Tuple[float, ...]
    use_checkerboard_init: bool = True
    use_adaptive: bool = True
    microflow_relax_steps: int = 3
    blur_scale: float = 0.4
    verbose: bool = False

    @property
    def num_stages(self) -> int:
        return len(self.sparsity_ratios)

    def ratio_of_stage(self, stage_k: int) -> float:
        ratios = self.sparsity_ratios
        if ratios[0] <= ratios[-1]:
            return ratios[self.num_stages - 1 - stage_k]
        return ratios[stage_k]

    def stage_steps_for_total(self, total_steps: int) -> Tuple[int, ...]:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        raw = [int(total_steps * ratio) for ratio in self.stage_ratios]
        stage_steps = []
        prev = 0
        for value in raw:
            bounded = max(prev + 1, min(total_steps, value))
            stage_steps.append(bounded)
            prev = bounded
        stage_steps[-1] = total_steps
        return tuple(stage_steps)


def _parse_csv_floats(value: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in value.split(",") if x.strip())


def config_from_inputs(
    preset: str,
    expected_total_steps: int,
    stage_ratios_csv: str,
    sparsity_ratios_csv: str,
    use_checkerboard_init: bool,
    use_adaptive: bool,
    microflow_relax_steps: int,
    blur_scale: float,
    verbose: bool,
) -> JiTConfig:
    if preset == "default_4x":
        expected_total_steps = DEFAULT_4X_STEPS
        stage_ratios = DEFAULT_STAGE_RATIOS
        sparsity_ratios = DEFAULT_4X_SPARSITY
    elif preset == "default_7x":
        expected_total_steps = DEFAULT_7X_STEPS
        stage_ratios = DEFAULT_STAGE_RATIOS
        sparsity_ratios = DEFAULT_7X_SPARSITY
    else:
        stage_ratios = _parse_csv_floats(stage_ratios_csv)
        sparsity_ratios = _parse_csv_floats(sparsity_ratios_csv)

    if len(stage_ratios) != len(sparsity_ratios):
        raise ValueError(f"stage_ratios and sparsity_ratios must have the same length; got {len(stage_ratios)} and {len(sparsity_ratios)}")
    if len(stage_ratios) < 2:
        raise ValueError("At least two JiT stages are required")
    if any(r <= 0.0 or r > 1.0 for r in sparsity_ratios):
        raise ValueError("All sparsity ratios must be in (0, 1]")
    if any(stage_ratios[i] <= 0.0 or stage_ratios[i] > 1.0 for i in range(len(stage_ratios))):
        raise ValueError("All stage ratios must be in (0, 1]")
    if any(stage_ratios[i] <= stage_ratios[i - 1] for i in range(1, len(stage_ratios))):
        raise ValueError("stage_ratios must be strictly increasing")
    if abs(stage_ratios[-1] - 1.0) > 1e-6:
        raise ValueError("The last stage ratio must be 1.0")

    return JiTConfig(
        expected_total_steps=int(expected_total_steps),
        stage_ratios=tuple(stage_ratios),
        sparsity_ratios=tuple(sparsity_ratios),
        use_checkerboard_init=bool(use_checkerboard_init),
        use_adaptive=bool(use_adaptive),
        microflow_relax_steps=int(microflow_relax_steps),
        blur_scale=float(blur_scale),
        verbose=bool(verbose),
    )
