from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_audio_reconstruction.data import MADEEGDataset
from eeg_audio_reconstruction.diffusion import DiffusionSchedule
from eeg_audio_reconstruction.features import LogMelExtractor
from eeg_audio_reconstruction.latent import denormalize_latent, normalize_latent
from eeg_audio_reconstruction.models import ConditionalLatentDenoiser, DirectMelRegressor, SpectrogramAutoencoder


class CoreSmokeTests(unittest.TestCase):
    def test_dataset_indexes_synthetic_hdf5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hdf5_path, yaml_path = self._make_synthetic_dataset(Path(tmp))
            dataset = MADEEGDataset(hdf5_path, yaml_path, split="all")
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(tuple(sample["eeg"].shape), (20, 256))
            self.assertEqual(tuple(sample["audio"].shape), (1, 44100))
            self.assertEqual(sample["audio_sample_rate"], 44100)
            dataset.close()

    def test_feature_and_model_shapes(self) -> None:
        audio = torch.randn(2, 1, 44100)
        eeg = torch.randn(2, 20, 256)
        extractor = LogMelExtractor()
        mel = extractor(audio, torch.tensor([44100, 44100]))
        self.assertEqual(tuple(mel.shape), (2, 1, 128, 64))

        autoencoder = SpectrogramAutoencoder()
        recon = autoencoder(mel)
        latent = autoencoder.encode(mel)
        self.assertEqual(tuple(recon.shape), tuple(mel.shape))
        self.assertEqual(tuple(latent.shape), (2, 8, 16, 8))

        schedule = DiffusionSchedule(timesteps=4)
        denoiser = ConditionalLatentDenoiser(latent_channels=8, model_channels=32, num_res_blocks=2)
        timesteps = torch.tensor([0, 3])
        noisy = schedule.q_sample(latent, timesteps)
        pred = denoiser(noisy, timesteps, eeg)
        self.assertEqual(tuple(pred.shape), tuple(latent.shape))

        baseline = DirectMelRegressor()
        baseline_pred = baseline(eeg)
        self.assertEqual(tuple(baseline_pred.shape), (2, 1, 128, 64))

    def test_scaled_diffusion_schedule_and_latent_stats(self) -> None:
        schedule = DiffusionSchedule(timesteps=8, target_alpha_bar=1e-3)
        self.assertLess(abs(float(schedule.alpha_bars[-1]) - 1e-3), 1e-5)

        latent = torch.randn(2, 3, 4, 5)
        stats = {
            "mean": [0.5, -0.25, 1.0],
            "std": [2.0, 0.5, 4.0],
            "eps": 1e-6,
        }
        normalized = normalize_latent(latent, stats)
        restored = denormalize_latent(normalized, stats)
        self.assertTrue(torch.allclose(restored, latent, atol=1e-6))

    def _make_synthetic_dataset(self, root: Path) -> tuple[Path, Path]:
        hdf5_path = root / "synthetic.hdf5"
        yaml_path = root / "synthetic.yaml"
        subject = "0001"
        stimulus = "synthetic_song"
        rng = np.random.default_rng(123)

        with h5py.File(hdf5_path, "w") as h5:
            group = h5.create_group(f"{subject}/{stimulus}")
            group.create_dataset("response", data=rng.normal(size=(20, 1024)).astype(np.float32))
            group.create_dataset("stimulus", data=rng.normal(size=(2, 44100)).astype(np.float32))
            group.create_dataset("soli", data=rng.normal(size=(3, 44100)).astype(np.float32))

        metadata = {
            subject: {
                stimulus: {
                    "instruments": ["A", "B", "C"],
                    "target": "A",
                    "wav_info": {"sfreq": 44100},
                }
            }
        }
        yaml_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
        return hdf5_path, yaml_path


if __name__ == "__main__":
    unittest.main()
