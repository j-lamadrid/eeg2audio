# EEG Audio Reconstruction

This repository is now organized around a runnable EEG-to-audio pipeline for the
MAD-EEG-style dataset in `data/`. The primary method is a compact latent
diffusion model (LDM):

1. Convert paired audio windows into normalized log-mel spectrograms.
2. Train a small spectrogram autoencoder and use its bottleneck as the latent
   audio space.
3. Train an EEG-conditioned DDPM denoiser in that latent space.
4. Decode sampled latents back to mel spectrograms and optionally reconstruct
   waveform audio with Griffin-Lim.

The consolidated project demo is `notebook.ipynb`.
Figures and the presentation are under `assets/`; broken prototype scripts are
under `legacy/`.

## Layout

- `src/eeg_audio_reconstruction/`: package code for data, features, models,
  diffusion, checkpointing, and visualization.
- `notebook.ipynb`: one end-to-end demo covering the CNN baseline and LDM.
- `scripts/`: command-line workflows.
- `configs/`: documented default settings.
- `tests/`: shape and dataset smoke tests.
- `data/`: local raw data. This directory is ignored by git because the files
  are large.

With the current HDF5/YAML files, the default one-second windowing indexes
1,395 paired EEG/audio samples across 8 subjects.

## Setup

Use the existing virtualenv if it is available:

```powershell
.\.env\Scripts\python.exe scripts\inspect_dataset.py --max-subjects 1 --max-stimuli-per-subject 1
```

For a fresh environment:

```powershell
python -m venv .env
.\.env\Scripts\python.exe -m pip install -r requirements.txt
```

## LDM Workflow

Train the spectrogram autoencoder first:

```powershell
.\.env\Scripts\python.exe scripts\train_autoencoder.py --epochs 20 --batch-size 16
```

Then train the EEG-conditioned latent diffusion denoiser:

```powershell
.\.env\Scripts\python.exe scripts\train_ldm.py --autoencoder checkpoints\autoencoder.pt --epochs 40 --batch-size 16
```

The LDM trainer standardizes autoencoder latents before diffusion and stores
the latent statistics in `checkpoints/ldm.pt`. The default diffusion schedule is
rescaled to end near `alpha_bar=0.001`, so sampling starts from a noise-like
latent distribution instead of from a mismatched scale.

Reconstruct a sample after both checkpoints exist:

```powershell
.\.env\Scripts\python.exe scripts\reconstruct.py --method ldm --sample-index 0
```

Outputs are written to `outputs/reconstruction/`.

## Baseline

A direct EEG-to-mel baseline is included for comparison:

```powershell
.\.env\Scripts\python.exe scripts\train_baseline.py --epochs 30 --batch-size 16
.\.env\Scripts\python.exe scripts\reconstruct.py --method baseline --sample-index 0
```

## Verification

Run the unit tests:

```powershell
.\.env\Scripts\python.exe -m unittest discover -s tests
```

Run a smoke test against the local dataset:

```powershell
.\.env\Scripts\python.exe scripts\smoke_test.py
```

## Notes on the Previous AudioLDM Attempt

The removed exploratory notebooks tried to fine-tune `cvssp/audioldm2-music`
directly, but the AudioLDM UNet, prompt embeddings, VAE inputs, and scheduler
objective were not wired with compatible shapes. The new implementation keeps
the LDM idea as the primary method while using a project-local latent space that
is small enough to train on this dataset. AudioLDM can still be revisited later
as a pretrained decoder or prior, but it should be added as an adapter after
this compact LDM has produced measurable baselines.
