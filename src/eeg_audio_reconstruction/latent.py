"""Latent normalization helpers for deterministic autoencoder latents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


LatentStats = Mapping[str, Any] | None


def normalize_latent(latent: torch.Tensor, stats: LatentStats) -> torch.Tensor:
    """Normalize autoencoder latents with checkpointed channel statistics."""

    if stats is None:
        return latent
    mean, std = latent_stats_tensors(stats, latent)
    return (latent - mean) / std


def denormalize_latent(latent: torch.Tensor, stats: LatentStats) -> torch.Tensor:
    """Restore normalized diffusion latents before autoencoder decoding."""

    if stats is None:
        return latent
    mean, std = latent_stats_tensors(stats, latent)
    return latent * std + mean


def latent_stats_tensors(stats: Mapping[str, Any], reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(stats["mean"], dtype=reference.dtype, device=reference.device).view(1, -1, 1, 1)
    std = torch.as_tensor(stats["std"], dtype=reference.dtype, device=reference.device).view(1, -1, 1, 1)
    return mean, std.clamp_min(float(stats.get("eps", 1e-6)))


@torch.no_grad()
def estimate_latent_stats(
    autoencoder: nn.Module,
    extractor: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str,
    max_batches: int | None = None,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Estimate per-channel latent mean/std over a dataset loader."""

    total: int | None = None
    sums: torch.Tensor | None = None
    sumsq: torch.Tensor | None = None

    autoencoder.eval()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        audio = batch["audio"].to(device)
        mel = extractor(audio, batch["audio_sample_rate"]).to(device)
        latent = autoencoder.encode(mel).detach()
        reduce_dims = (0, 2, 3)
        batch_count = latent.shape[0] * latent.shape[2] * latent.shape[3]
        batch_sum = latent.sum(dim=reduce_dims)
        batch_sumsq = latent.square().sum(dim=reduce_dims)

        if sums is None:
            sums = batch_sum
            sumsq = batch_sumsq
            total = batch_count
        else:
            sums += batch_sum
            sumsq = sumsq + batch_sumsq
            total = int(total or 0) + batch_count

    if sums is None or sumsq is None or total is None or total <= 0:
        raise ValueError("Cannot estimate latent stats from an empty loader.")

    mean = sums / total
    variance = (sumsq / total - mean.square()).clamp_min(eps * eps)
    std = variance.sqrt()
    return {
        "mean": mean.detach().cpu().tolist(),
        "std": std.detach().cpu().tolist(),
        "eps": float(eps),
        "count": int(total),
    }
