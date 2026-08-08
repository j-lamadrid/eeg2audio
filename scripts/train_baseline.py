from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
import torch.nn.functional as torch_f
from torch.utils.data import DataLoader

from _bootstrap import bootstrap

bootstrap()

from eeg_audio_reconstruction.config import FeatureConfig
from eeg_audio_reconstruction.data import MADEEGDataset
from eeg_audio_reconstruction.features import LogMelExtractor
from eeg_audio_reconstruction.models import DirectMelRegressor
from eeg_audio_reconstruction.train_utils import count_parameters, get_device, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a direct EEG-to-mel baseline.")
    parser.add_argument("--hdf5", default="data/madeeg_preprocessed.hdf5")
    parser.add_argument("--metadata", default="data/madeeg_preprocessed.yaml")
    parser.add_argument("--output", default="checkpoints/baseline.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-stimuli-per-subject", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    feature_config = FeatureConfig()
    extractor = LogMelExtractor(feature_config).to(device)
    model = DirectMelRegressor(n_mels=feature_config.n_mels, frames=feature_config.frames).to(device)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device: {device}")
    print(f"samples: {len(dataset)}")
    print(f"trainable parameters: {count_parameters(model):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in loader:
            eeg = batch["eeg"].to(device)
            audio = batch["audio"].to(device)
            source_sr = batch["audio_sample_rate"]
            with torch.no_grad():
                target = extractor(audio, source_sr).to(device)
            pred = model(eeg)
            loss = torch_f.l1_loss(pred, target) + 0.5 * torch_f.mse_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * eeg.shape[0]
            seen += eeg.shape[0]

        avg_loss = running / max(seen, 1)
        print(f"epoch {epoch:03d} | baseline_loss={avg_loss:.6f}")
        save_checkpoint(
            args.output,
            epoch=epoch,
            loss=avg_loss,
            model_state=model.state_dict(),
            feature_config=asdict(feature_config),
        )
    dataset.close()


if __name__ == "__main__":
    main()

