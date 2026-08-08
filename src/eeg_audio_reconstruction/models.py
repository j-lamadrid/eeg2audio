"""Neural network modules for baseline and latent-diffusion models."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as torch_f


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = torch_f.pad(emb, (0, 1))
    return emb


class EEGEncoder(nn.Module):
    def __init__(self, eeg_channels: int = 20, embedding_dim: int = 256, base_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(eeg_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.SiLU(),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_channels * 4),
            nn.SiLU(),
            nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.dim() != 3:
            raise ValueError(f"Expected EEG tensor (B,C,T); got {tuple(eeg.shape)}")
        return self.proj(self.net(eeg))


class SpectrogramAutoencoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        frames: int = 64,
        latent_channels: int = 8,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if n_mels % 8 or frames % 8:
            raise ValueError("n_mels and frames must be divisible by 8.")
        self.n_mels = n_mels
        self.frames = frames
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, latent_channels, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.latent_channels, self.n_mels // 8, self.frames // 8)

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encoder(mel)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(mel))


class ConditionalResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.cond = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch_f.silu(self.norm1(x)))
        scale, shift = self.cond(cond).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(torch_f.silu(h))
        return x + h


class ConditionalLatentDenoiser(nn.Module):
    def __init__(
        self,
        latent_channels: int = 8,
        eeg_channels: int = 20,
        model_channels: int = 96,
        cond_dim: int = 256,
        num_res_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.eeg_encoder = EEGEncoder(eeg_channels=eeg_channels, embedding_dim=cond_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_fuse = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.input = nn.Conv2d(latent_channels, model_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ConditionalResBlock(model_channels, cond_dim) for _ in range(num_res_blocks)])
        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(model_channels), model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, latent_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, noisy_latent: torch.Tensor, timesteps: torch.Tensor, eeg: torch.Tensor) -> torch.Tensor:
        eeg_emb = self.eeg_encoder(eeg)
        time_emb = self.time_mlp(timestep_embedding(timesteps, eeg_emb.shape[1]))
        cond = self.cond_fuse(torch.cat([eeg_emb, time_emb], dim=1))
        h = self.input(noisy_latent)
        for block in self.blocks:
            h = block(h, cond)
        return self.output(h)


class DirectMelRegressor(nn.Module):
    """Baseline EEG-to-mel model with no diffusion."""

    def __init__(
        self,
        eeg_channels: int = 20,
        n_mels: int = 128,
        frames: int = 64,
        embedding_dim: int = 256,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        if n_mels % 8 or frames % 8:
            raise ValueError("n_mels and frames must be divisible by 8.")
        self.n_mels = n_mels
        self.frames = frames
        self.base_h = n_mels // 8
        self.base_w = frames // 8
        self.eeg_encoder = EEGEncoder(eeg_channels=eeg_channels, embedding_dim=embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, base_channels * 4 * self.base_h * self.base_w),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.SiLU(),
            nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        emb = self.eeg_encoder(eeg)
        h = self.fc(emb).view(eeg.shape[0], -1, self.base_h, self.base_w)
        return self.decoder(h)

