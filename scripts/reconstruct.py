from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio

from _bootstrap import bootstrap

bootstrap()

from eeg_audio_reconstruction.config import DiffusionConfig, FeatureConfig
from eeg_audio_reconstruction.data import MADEEGDataset
from eeg_audio_reconstruction.diffusion import DiffusionSchedule
from eeg_audio_reconstruction.features import LogMelExtractor
from eeg_audio_reconstruction.latent import denormalize_latent
from eeg_audio_reconstruction.models import ConditionalLatentDenoiser, DirectMelRegressor, SpectrogramAutoencoder
from eeg_audio_reconstruction.train_utils import get_device, load_checkpoint, seed_everything
from eeg_audio_reconstruction.visualization import save_mel_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct a one-second audio window from EEG.")
    parser.add_argument("--method", choices=["ldm", "baseline"], default="ldm")
    parser.add_argument("--hdf5", default="data/madeeg_preprocessed.hdf5")
    parser.add_argument("--metadata", default="data/madeeg_preprocessed.yaml")
    parser.add_argument("--autoencoder", default="checkpoints/autoencoder.pt")
    parser.add_argument("--ldm", default="checkpoints/ldm.pt")
    parser.add_argument("--baseline", default="checkpoints/baseline.pt")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/reconstruction")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic DDPM sampling instead of deterministic posterior means.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MADEEGDataset(args.hdf5, args.metadata, split="all", max_samples=args.sample_index + 1)
    sample = dataset[args.sample_index]
    eeg = sample["eeg"].unsqueeze(0).to(device)

    if args.method == "ldm":
        mel = reconstruct_ldm(args, eeg, device)
    else:
        mel = reconstruct_baseline(args, eeg, device)

    feature_config = load_feature_config(args)
    extractor = LogMelExtractor(feature_config).to(device)
    target_mel = extractor(sample["audio"].unsqueeze(0).to(device), sample["audio_sample_rate"])

    save_mel_image(mel, output_dir / "predicted_mel.png", title=f"{args.method.upper()} prediction")
    save_mel_image(target_mel, output_dir / "target_mel.png", title="Target mel")
    torch.save({"predicted_mel": mel.cpu(), "target_mel": target_mel.cpu(), "sample": sample}, output_dir / "reconstruction.pt")

    if not args.no_audio:
        waveform = extractor.mel_to_waveform(mel).cpu()
        torchaudio.save(str(output_dir / "predicted.wav"), waveform[0], feature_config.sample_rate)
        torchaudio.save(str(output_dir / "target.wav"), sample["audio"], int(sample["audio_sample_rate"]))

    print(f"wrote reconstruction artifacts to {output_dir.resolve()}")
    dataset.close()


def reconstruct_ldm(args: argparse.Namespace, eeg: torch.Tensor, device: torch.device) -> torch.Tensor:
    ae_checkpoint = load_checkpoint(args.autoencoder, map_location=device)
    ldm_checkpoint = load_checkpoint(args.ldm, map_location=device)
    feature_config = FeatureConfig(**ae_checkpoint["feature_config"])
    autoencoder = SpectrogramAutoencoder(
        n_mels=feature_config.n_mels,
        frames=feature_config.frames,
        **ae_checkpoint["model_config"],
    ).to(device)
    autoencoder.load_state_dict(ae_checkpoint["model_state"])
    autoencoder.eval().requires_grad_(False)

    denoiser = ConditionalLatentDenoiser(
        latent_channels=autoencoder.latent_channels,
        **ldm_checkpoint["model_config"],
    ).to(device)
    denoiser.load_state_dict(ldm_checkpoint["model_state"])
    denoiser.eval()

    diffusion_config = DiffusionConfig(**ldm_checkpoint["diffusion_config"])
    schedule = DiffusionSchedule(**diffusion_config.__dict__).to(device)
    latent_shape = tuple(ldm_checkpoint.get("latent_shape", autoencoder.latent_shape))
    with torch.no_grad():
        latent = schedule.sample(
            denoiser,
            eeg=eeg,
            shape=(1, *latent_shape),
            device=device,
            stochastic=args.stochastic,
        )
        latent = denormalize_latent(latent, ldm_checkpoint.get("latent_stats"))
        return autoencoder.decode(latent).clamp(-1.0, 1.0)


def reconstruct_baseline(args: argparse.Namespace, eeg: torch.Tensor, device: torch.device) -> torch.Tensor:
    checkpoint = load_checkpoint(args.baseline, map_location=device)
    feature_config = FeatureConfig(**checkpoint["feature_config"])
    model = DirectMelRegressor(n_mels=feature_config.n_mels, frames=feature_config.frames).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        return model(eeg).clamp(-1.0, 1.0)


def load_feature_config(args: argparse.Namespace) -> FeatureConfig:
    path = args.autoencoder if args.method == "ldm" else args.baseline
    checkpoint = load_checkpoint(path, map_location="cpu")
    return FeatureConfig(**checkpoint["feature_config"])


if __name__ == "__main__":
    main()
