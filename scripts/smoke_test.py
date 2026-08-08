from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap

bootstrap()

from eeg_audio_reconstruction.data import MADEEGDataset
from eeg_audio_reconstruction.diffusion import DiffusionSchedule
from eeg_audio_reconstruction.features import LogMelExtractor
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

    schedule = DiffusionSchedule(timesteps=4).to(device)
    denoiser = ConditionalLatentDenoiser(latent_channels=latent.shape[1], model_channels=32, num_res_blocks=2).to(device)
    timesteps = torch.randint(0, schedule.timesteps, (latent.shape[0],), device=device)
    noise = torch.randn_like(latent)
    noisy = schedule.q_sample(latent, timesteps, noise)
    pred_noise = denoiser(noisy, timesteps, eeg)

    baseline = DirectMelRegressor().to(device)
    baseline_pred = baseline(eeg)

    print(f"device: {device}")
    print(f"eeg: {tuple(eeg.shape)}")
    print(f"audio: {tuple(audio.shape)}")
    print(f"mel: {tuple(mel.shape)}")
    print(f"ae_recon: {tuple(recon.shape)}")
    print(f"latent: {tuple(latent.shape)}")
    print(f"pred_noise: {tuple(pred_noise.shape)}")
    print(f"baseline_pred: {tuple(baseline_pred.shape)}")
    print("smoke_test_ok")
    dataset.close()


if __name__ == "__main__":
    main()

