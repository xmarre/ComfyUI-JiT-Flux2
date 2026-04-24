from __future__ import annotations

import math

import torch


def generalized_time_snr_shift(t: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def _approx_beta_ppf(probabilities: torch.Tensor, alpha: float, beta: float, resolution: int) -> torch.Tensor:
    resolution = max(512, int(resolution))
    alpha = float(alpha)
    beta = float(beta)
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("beta_alpha and beta_beta must be positive")

    eps = 1e-5
    grid = torch.linspace(eps, 1.0 - eps, resolution, dtype=torch.float64)
    log_pdf = (alpha - 1.0) * torch.log(grid) + (beta - 1.0) * torch.log1p(-grid)
    pdf = torch.exp(log_pdf - torch.max(log_pdf))
    cdf = torch.cumsum(pdf, dim=0)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])

    probs = probabilities.to(dtype=torch.float64).clamp(0.0, 1.0)
    flat = probs.reshape(-1)
    clamped = flat.clamp(float(cdf[1]), float(cdf[-2]))
    upper = torch.searchsorted(cdf, clamped, right=False).clamp(1, resolution - 1)
    lower = upper - 1
    cdf_lower = cdf[lower]
    cdf_upper = cdf[upper]
    denom = (cdf_upper - cdf_lower).clamp_min(1e-12)
    weight = (clamped - cdf_lower) / denom
    values = grid[lower] + weight * (grid[upper] - grid[lower])
    values = values.reshape_as(probs)
    values = torch.where(probs <= 0.0, torch.zeros_like(values), values)
    values = torch.where(probs >= 1.0, torch.ones_like(values), values)
    return values.to(dtype=probabilities.dtype)


def get_flux2_jit_sigmas(
    steps: int,
    width: int,
    height: int,
    schedule: str = "flux2",
    beta_alpha: float = 1.4,
    beta_beta: float = 0.42,
    beta_resolution: int = 8192,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    image_seq_len = round(width * height / (16 * 16))
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=steps)
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)

    if schedule == "flux2":
        timesteps = base
    elif schedule == "jit_beta":
        timesteps = _approx_beta_ppf(base, beta_alpha, beta_beta, beta_resolution).to(dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported Flux2 JiT scheduler mode: {schedule}")

    return generalized_time_snr_shift(timesteps, mu, 1.0).to(dtype=torch.float32)
