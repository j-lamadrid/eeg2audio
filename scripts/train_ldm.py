from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
import torch.nn.functional as torch_f
from torch.utils.data import DataLoader

from _bootstrap import bootstrap

bootstrap()

from eeg_audio_reconstruction.config import DenoiserConfig, DiffusionConfig, FeatureConfig
from eeg_audio_reconstruction.data import MADEEGDataset
from eeg_audio_reconstruction.diffusion import DiffusionSchedule
from eeg_audio_reconstruction.features import LogMelExtractor
from eeg_audio_reconstruction.latent import estimate_latent_stats, normalize_latent
from eeg_audio_reconstruction.models import ConditionalLatentDenoiser, SpectrogramAutoencoder
from eeg_audio_reconstruction.train_utils import (
    count_parameters,
    get_device,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EEG-conditioned latent diffusion model.")
    parser.add_argument("--hdf5", default="data/madeeg_preprocessed.hdf5")
    parser.add_argument("--metadata", default="data/madeeg_preprocessed.yaml")
    parser.add_argument("--autoencoder", default="checkpoints/autoencoder.pt")
    parser.add_argument("--output", default="checkpoints/ldm.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--target-alpha-bar", type=float, default=1e-3)
    parser.add_argument("--latent-stats-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-stimuli-per-subject", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-channels", type=int, default=96)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--num-res-blocks", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)

    ae_checkpoint = load_checkpoint(args.autoencoder, map_location=device)
    feature_config = FeatureConfig(**ae_checkpoint["feature_config"])
    ae_config = ae_checkpoint["model_config"]
    extractor = LogMelExtractor(feature_config).to(device)
    autoencoder = SpectrogramAutoencoder(
        n_mels=feature_config.n_mels,
        frames=feature_config.frames,
        **ae_config,
    ).to(device)
    autoencoder.load_state_dict(ae_checkpoint["model_state"])
    autoencoder.eval().requires_grad_(False)

    denoiser_config = DenoiserConfig(
        model_channels=args.model_channels,
        cond_dim=args.cond_dim,
        num_res_blocks=args.num_res_blocks,
    )
    diffusion_config = DiffusionConfig(timesteps=args.timesteps, target_alpha_bar=args.target_alpha_bar)
    denoiser = ConditionalLatentDenoiser(
        latent_channels=autoencoder.latent_channels,
        model_channels=denoiser_config.model_channels,
        cond_dim=denoiser_config.cond_dim,
        num_res_blocks=denoiser_config.num_res_blocks,
    ).to(device)
    schedule = DiffusionSchedule(**asdict(diffusion_config)).to(device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=args.lr)

    dataset = MADEEGDataset(
        hdf5_path=args.hdf5,
        metadata_path=args.metadata,
        split=args.split,
        max_subjects=args.max_subjects,
        max_stimuli_per_subject=args.max_stimuli_per_subject,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"device: {device}")
    print(f"samples: {len(dataset)}")
    print(f"trainable parameters: {count_parameters(denoiser):,}")
    latent_stats = estimate_latent_stats(
        autoencoder,
        extractor,
        loader,
        device,
        max_batches=args.latent_stats_batches,
    )
    print(f"latent mean: {[round(value, 4) for value in latent_stats['mean']]}")
    print(f"latent std: {[round(value, 4) for value in latent_stats['std']]}")
    print(f"terminal alpha_bar: {float(schedule.alpha_bars[-1]):.6f}")

    for epoch in range(1, args.epochs + 1):
        denoiser.train()
        running = 0.0
        seen = 0
        for batch in loader:
            eeg = batch["eeg"].to(device)
            audio = batch["audio"].to(device)
            source_sr = batch["audio_sample_rate"]
            with torch.no_grad():
                mel = extractor(audio, source_sr).to(device)
                clean_latent = normalize_latent(autoencoder.encode(mel), latent_stats)

            noise = torch.randn_like(clean_latent)
            timesteps = torch.randint(0, schedule.timesteps, (clean_latent.shape[0],), device=device)
            noisy_latent = schedule.q_sample(clean_latent, timesteps, noise)
            pred_noise = denoiser(noisy_latent, timesteps, eeg)
            loss = torch_f.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()

            running += float(loss.item()) * eeg.shape[0]
            seen += eeg.shape[0]

        avg_loss = running / max(seen, 1)
        print(f"epoch {epoch:03d} | diffusion_loss={avg_loss:.6f}")
        save_checkpoint(
            args.output,
            epoch=epoch,
            loss=avg_loss,
            model_state=denoiser.state_dict(),
            model_config=asdict(denoiser_config),
            diffusion_config=asdict(diffusion_config),
            feature_config=asdict(feature_config),
            autoencoder_config=ae_config,
            latent_shape=autoencoder.latent_shape,
            latent_stats=latent_stats,
        )
    dataset.close()


if __name__ == "__main__":
    main()
