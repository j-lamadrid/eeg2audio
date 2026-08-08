from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap

bootstrap()

from eeg_audio_reconstruction.data import MADEEGDataset
from eeg_audio_reconstruction.diffusion import DiffusionSchedule
from eeg_audio_reconstruction.features import LogMelExtractor
from eeg_audio_reconstruction.latent import denormalize_latent, normalize_latent
from eeg_audio_reconstruction.models import ConditionalLatentDenoiser, DirectMelRegressor, SpectrogramAutoencoder
from eeg_audio_reconstruction.train_utils import get_device, seed_everything


def main() -> None:
    seed_everything(123)
    device = get_device("auto")
    dataset = MADEEGDataset(
        split="all",
        max_subjects=1,
        max_stimuli_per_subject=1,
        max_samples=2,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    eeg = batch["eeg"].to(device)
    audio = batch["audio"].to(device)

    extractor = LogMelExtractor().to(device)
    mel = extractor(audio, batch["audio_sample_rate"])

    autoencoder = SpectrogramAutoencoder().to(device)
    recon = autoencoder(mel)
    latent = autoencoder.encode(mel)
    latent_stats = {
        "mean": latent.mean(dim=(0, 2, 3)).detach().cpu().tolist(),
        "std": latent.std(dim=(0, 2, 3)).clamp_min(1e-6).detach().cpu().tolist(),
        "eps": 1e-6,
    }
    normalized_latent = normalize_latent(latent, latent_stats)
    restored_latent = denormalize_latent(normalized_latent, latent_stats)

    schedule = DiffusionSchedule(timesteps=4, target_alpha_bar=1e-3).to(device)
    denoiser = ConditionalLatentDenoiser(latent_channels=latent.shape[1], model_channels=32, num_res_blocks=2).to(device)
    timesteps = torch.randint(0, schedule.timesteps, (normalized_latent.shape[0],), device=device)
    noise = torch.randn_like(normalized_latent)
    noisy = schedule.q_sample(normalized_latent, timesteps, noise)
    pred_noise = denoiser(noisy, timesteps, eeg)

    baseline = DirectMelRegressor().to(device)
    baseline_pred = baseline(eeg)

    print(f"device: {device}")
    print(f"eeg: {tuple(eeg.shape)}")
    print(f"audio: {tuple(audio.shape)}")
    print(f"mel: {tuple(mel.shape)}")
    print(f"ae_recon: {tuple(recon.shape)}")
    print(f"latent: {tuple(latent.shape)}")
    print(f"latent_roundtrip_ok: {torch.allclose(restored_latent, latent, atol=1e-6)}")
    print(f"terminal_alpha_bar: {float(schedule.alpha_bars[-1]):.6f}")
    print(f"pred_noise: {tuple(pred_noise.shape)}")
    print(f"baseline_pred: {tuple(baseline_pred.shape)}")
    print("smoke_test_ok")
    dataset.close()


if __name__ == "__main__":
    main()
