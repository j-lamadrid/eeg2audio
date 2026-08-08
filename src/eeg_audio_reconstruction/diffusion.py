"""Minimal DDPM utilities for latent diffusion."""

from __future__ import annotations

import torch
from torch import nn


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        target_alpha_bar: float | None = None,
    ) -> None:
        super().__init__()
        self.timesteps = int(timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.target_alpha_bar = target_alpha_bar

        betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        if target_alpha_bar is not None:
            betas = self._rescale_betas_to_alpha_bar(betas, float(target_alpha_bar))
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars).clamp_min(1e-12)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))

    def q_sample(self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean)
        sqrt_ab = self._extract(self.sqrt_alpha_bars, timesteps, clean.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, clean.shape)
        return sqrt_ab * clean + sqrt_omab * noise

    @torch.no_grad()
    def p_sample(
        self,
        denoiser: nn.Module,
        noisy: torch.Tensor,
        eeg: torch.Tensor,
        timestep: int,
        stochastic: bool = True,
    ) -> torch.Tensor:
        t = torch.full((noisy.shape[0],), timestep, device=noisy.device, dtype=torch.long)
        pred_noise = denoiser(noisy, t, eeg)
        beta = self._extract(self.betas, t, noisy.shape)
        alpha = self._extract(self.alphas, t, noisy.shape)
        alpha_bar = self._extract(self.alpha_bars, t, noisy.shape)

        mean = (noisy - beta * pred_noise / torch.sqrt((1.0 - alpha_bar).clamp_min(1e-12))) / torch.sqrt(alpha)
        if timestep == 0 or not stochastic:
            return mean
        variance = self._extract(self.posterior_variance, t, noisy.shape)
        return mean + torch.sqrt(variance) * torch.randn_like(noisy)

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        eeg: torch.Tensor,
        shape: tuple[int, ...],
        device: torch.device | str,
        stochastic: bool = True,
    ) -> torch.Tensor:
        noisy = torch.randn(shape, device=device)
        for timestep in range(self.timesteps - 1, -1, -1):
            noisy = self.p_sample(denoiser, noisy, eeg, timestep, stochastic=stochastic)
        return noisy

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
        gathered = values.gather(0, timesteps)
        return gathered.view(timesteps.shape[0], *([1] * (len(shape) - 1)))

    def _rescale_betas_to_alpha_bar(self, betas: torch.Tensor, target_alpha_bar: float) -> torch.Tensor:
        if not 0.0 < target_alpha_bar < 1.0:
            raise ValueError("target_alpha_bar must be between 0 and 1.")

        def final_alpha_bar(scale: float) -> torch.Tensor:
            scaled = (betas * scale).clamp(max=0.999)
            return torch.cumprod(1.0 - scaled, dim=0)[-1]

        low = 0.0
        high = 1.0
        while float(final_alpha_bar(high)) > target_alpha_bar and high < 1e6:
            high *= 2.0

        for _ in range(64):
            mid = (low + high) * 0.5
            if float(final_alpha_bar(mid)) > target_alpha_bar:
                low = mid
            else:
                high = mid
        return (betas * high).clamp(min=1e-8, max=0.999)
