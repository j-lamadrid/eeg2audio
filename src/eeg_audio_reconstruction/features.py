"""Audio feature transforms for mel-domain training and reconstruction."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as torch_f
import torchaudio
import torchaudio.transforms as T

from .config import FeatureConfig


class LogMelExtractor(nn.Module):
    """Convert waveform windows to normalized log-mel images.

    Output values are clamped to ``[-1, 1]`` where ``1`` is the local spectral
    peak and ``-1`` is ``top_db`` below that peak.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        super().__init__()
        self.config = config or FeatureConfig()
        f_max = self.config.f_max or float(self.config.sample_rate // 2)
        self.mel = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=f_max,
            power=2.0,
            center=True,
        )
        self.inverse_mel = T.InverseMelScale(
            n_stft=self.config.n_fft // 2 + 1,
            n_mels=self.config.n_mels,
            sample_rate=self.config.sample_rate,
            f_min=self.config.f_min,
            f_max=f_max,
        )
        self.griffin_lim = T.GriffinLim(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            power=2.0,
            n_iter=32,
        )

    def forward(self, waveform: torch.Tensor, source_sample_rate: int | torch.Tensor | None = None) -> torch.Tensor:
        waveform = self._prepare_waveform(waveform, source_sample_rate)
        mel = self.mel(waveform)
        db = 10.0 * torch.log10(mel.clamp_min(1e-10))
        peak = db.amax(dim=(-2, -1), keepdim=True)
        db = (db - peak).clamp(min=-self.config.top_db, max=0.0)
        normalized = (db + self.config.top_db) / self.config.top_db
        normalized = normalized.mul(2.0).sub(1.0)
        normalized = self._fix_frames(normalized)
        return normalized.unsqueeze(1)

    def denormalize_mel(self, normalized: torch.Tensor) -> torch.Tensor:
        if normalized.dim() == 4:
            normalized = normalized.squeeze(1)
        db = (normalized.clamp(-1.0, 1.0) + 1.0) * 0.5 * self.config.top_db
        db = db - self.config.top_db
        return torch.pow(10.0, db / 10.0)

    @torch.no_grad()
    def mel_to_waveform(self, normalized: torch.Tensor) -> torch.Tensor:
        mel_power = self.denormalize_mel(normalized).clamp_min(1e-10)
        linear = self.inverse_mel(mel_power)
        waveform = self.griffin_lim(linear).unsqueeze(1)
        peak = waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return waveform / peak

    def _prepare_waveform(
        self,
        waveform: torch.Tensor,
        source_sample_rate: int | torch.Tensor | None,
    ) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)
        if waveform.dim() != 2:
            raise ValueError(f"Expected waveform with shape (B,T) or (B,C,T); got {tuple(waveform.shape)}")

        source_sr = self._coerce_sample_rate(source_sample_rate)
        if source_sr != self.config.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_sr, self.config.sample_rate)

        peak = waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return waveform / peak

    def _fix_frames(self, mel: torch.Tensor) -> torch.Tensor:
        frames = mel.shape[-1]
        target = self.config.frames
        if frames == target:
            return mel
        if frames < target:
            return torch_f.pad(mel, (0, target - frames), value=-1.0)
        return mel[..., :target]

    def _coerce_sample_rate(self, source_sample_rate: int | torch.Tensor | None) -> int:
        if source_sample_rate is None:
            return self.config.sample_rate
        if isinstance(source_sample_rate, torch.Tensor):
            if source_sample_rate.numel() == 0:
                return self.config.sample_rate
            flat = source_sample_rate.detach().cpu().reshape(-1)
            first = int(flat[0].item())
            if not torch.all(flat == flat[0]):
                raise ValueError("Batches with mixed source sample rates are not supported yet.")
            return first
        return int(source_sample_rate)

