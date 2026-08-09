# EEG Audio Reconstruction (2024 Cognitive Science Research Fellowship)

CNN and latent-diffusion methods for reconstructing short audio windows from
music-evoked EEG responses.

This repository implements a compact, runnable EEG-to-audio reconstruction
pipeline for a MAD-EEG-style HDF5 dataset. The goal is to map a one-second EEG
response window to an audio-domain representation, compare a direct CNN
baseline against an EEG-conditioned latent diffusion model (LDM), and export
predicted spectrograms and waveform reconstructions for inspection.

The consolidated demo is [notebook.ipynb](notebook.ipynb). It is intended to be
the single notebook entry point for the project.

## Overview

Reconstructing sound from EEG is a difficult inverse problem: EEG recordings
are low signal-to-noise, spatially coarse, and only indirectly related to the
audio waveform that evoked them. Direct waveform reconstruction is especially
hard because phase and fine temporal structure are weakly constrained by EEG.

This project therefore reconstructs audio in a mel-spectrogram space. A
normalized log-mel spectrogram provides a stable, perceptually meaningful
target that can later be converted back to a waveform using inverse mel scaling
and Griffin-Lim. The implemented methods are:

1. **CNN baseline**: an EEG encoder predicts the target mel spectrogram
   directly.
2. **Latent diffusion model**: a spectrogram autoencoder first learns a compact
   latent audio space; then an EEG-conditioned DDPM denoiser learns to sample
   latents that decode to mel spectrograms.

The current implementation is deliberately local and compact. Earlier
experiments attempted to fine-tune AudioLDM directly, but the pretrained UNet,
prompt embeddings, VAE inputs, and diffusion objective were not shape-compatible
with this dataset without a more careful adapter. This repository keeps the LDM
idea while using a project-local latent space that is small enough to train and
debug on the available EEG/audio pairs.

## Current Status

The codebase currently supports:

- Lazy MAD-EEG HDF5 indexing and one-second EEG/audio pairing.
- Per-channel EEG normalization and repeated-trial averaging.
- Log-mel feature extraction with fixed frame count.
- A direct CNN EEG-to-mel baseline.
- A spectrogram autoencoder used as the LDM latent space.
- An EEG-conditioned latent DDPM denoiser.
- Latent standardization before diffusion and inverse standardization before
  decoding.
- Deterministic LDM reconstruction by default, with stochastic DDPM sampling as
  an opt-in mode.
- A root-level notebook demo covering data, features, CNN, autoencoder, LDM,
  visual comparisons, and waveform export.

The notebook will load full checkpoints if they exist. If
`checkpoints/autoencoder.pt`, `checkpoints/ldm.pt`, and
`checkpoints/baseline.pt` are missing, it trains cached single-sample demo
checkpoints named `notebook_demo_v2_*.pt`. Those cached demo checkpoints are
only an overfit sanity check: they prove the pipeline is wired correctly and no
longer collapses to flat spectrograms, but they are not evidence of held-out
decoding performance.

## Data

The expected local data files are:

| File | Role |
| --- | --- |
| `data/madeeg_preprocessed.hdf5` | HDF5 store containing subject/stimulus EEG responses and audio arrays. |
| `data/madeeg_preprocessed.yaml` | Metadata, including audio sample rates and instrument labels. |

The default dataset reader indexes one-second windows. With the current local
HDF5/YAML files, this produces 1,395 paired EEG/audio samples across 8 subjects.

Each sample returned by `MADEEGDataset` contains:

| Field | Shape / Type | Description |
| --- | --- | --- |
| `eeg` | `(20, 256)` | One second of EEG at 256 Hz. |
| `audio` | `(1, T)` | Matching stimulus or source audio window. |
| `audio_sample_rate` | `int` | Original audio sample rate, commonly 44.1 kHz. |
| `subject` | `str` | Subject identifier. |
| `stimulus` | `str` | Stimulus identifier. |
| `segment_index` | `int` | One-second segment number within the stimulus. |

By default, the response matrix is treated as repeated EEG trials concatenated
in time. The dataset averages the same one-second interval across four
repetitions before pairing it with the corresponding audio interval. EEG is
standardized independently per channel:

```text
eeg_norm[c, t] = (eeg[c, t] - mean(eeg[c, :])) / (std(eeg[c, :]) + 1e-6)
```

Targets can be configured with `target="stimulus"` for the full stimulus mix or
with source-oriented targets such as `soli` / `soli:<index>` where the HDF5 file
contains separated instrument tracks.

## Audio Feature Representation

All learned models operate on normalized log-mel images rather than raw
waveforms. The default feature configuration is:

| Parameter | Value |
| --- | --- |
| Sample rate | 16,000 Hz |
| FFT size | 1,024 |
| Hop length | 256 |
| Mel bins | 128 |
| Frames | 64 |
| Frequency range | 20 Hz to Nyquist |
| Dynamic range | 80 dB |

For each waveform window:

1. Convert to mono if needed.
2. Resample to 16 kHz.
3. Peak-normalize the waveform.
4. Compute a power mel spectrogram.
5. Convert to decibels relative to the local peak.
6. Clamp to an 80 dB range.
7. Normalize to `[-1, 1]`.
8. Pad or crop to 64 frames.

In simplified form:

```text
M = MelSpectrogram(audio)
D = 10 * log10(max(M, 1e-10))
D_rel = clamp(D - max(D), -top_db, 0)
mel_norm = 2 * ((D_rel + top_db) / top_db) - 1
```

Waveform export is performed after prediction by denormalizing the mel power,
using `torchaudio.transforms.InverseMelScale`, and estimating phase with
Griffin-Lim. This is useful for listening checks, but it is not a high-fidelity
neural vocoder.

## Methods

### 1. CNN Baseline

The direct baseline is `DirectMelRegressor`. It is a convolutional
EEG-to-spectrogram model with no diffusion step.

Architecture:

1. `EEGEncoder`
   - Input: `(B, 20, 256)`.
   - Several 1D convolutional blocks with BatchNorm and SiLU activations.
   - Temporal downsampling through strided convolutions.
   - Adaptive average pooling to a fixed embedding.
   - Two linear layers produce a 256-dimensional EEG embedding.
2. Mel decoder
   - A fully connected projection expands the EEG embedding to a low-resolution
     image grid.
   - Three transposed-convolution stages upsample to `(B, 1, 128, 64)`.
   - Final `tanh` constrains predictions to the normalized mel range.

Training objective:

```text
loss_cnn = L1(pred_mel, target_mel) + 0.5 * MSE(pred_mel, target_mel)
```

The CNN baseline is important because it gives a deterministic, low-complexity
reference for the generative model. If the CNN cannot learn a useful mapping,
the EEG conditioning signal or data split may be too weak for the LDM to
improve on.

### 2. Spectrogram Autoencoder

The autoencoder learns the latent audio space used by the LDM.

Input and output:

```text
mel:    (B, 1, 128, 64)
latent: (B, latent_channels, 16, 8)
recon:  (B, 1, 128, 64)
```

Default latent configuration:

| Parameter | Value |
| --- | --- |
| `latent_channels` | 8 |
| `base_channels` | 32 |
| Latent spatial size | `16 x 8` |

Encoder:

- Three stride-2 2D convolutional stages reduce mel resolution by a factor of 8.
- A final convolution maps to `latent_channels`.

Decoder:

- Three transposed-convolution stages upsample the latent image.
- A final convolution plus `tanh` returns normalized mel values.

Training objective:

```text
loss_ae = L1(recon_mel, target_mel) + 0.5 * MSE(recon_mel, target_mel)
```

The autoencoder is the first quality gate for the LDM. If it reconstructs a
flat or averaged spectrogram, the LDM cannot produce useful audio because every
sample is decoded through this bottleneck.

### 3. Latent Diffusion Model

The LDM trains a conditional denoiser in the autoencoder latent space instead
of directly in mel space.

First, the trained autoencoder encodes target spectrograms:

```text
z_raw = Encoder(target_mel)
```

Because this is a deterministic autoencoder rather than a variational model,
the raw latent distribution can have arbitrary channel means and scales. The
trainer estimates per-channel latent statistics over the training loader:

```text
mu[c] = mean(z_raw[:, c, :, :])
sigma[c] = std(z_raw[:, c, :, :])
z_0 = (z_raw - mu) / sigma
```

These statistics are saved in the LDM checkpoint and applied in reverse before
decoding sampled latents.

The forward diffusion process is the standard DDPM corruption process:

```text
q(z_t | z_0) = sqrt(alpha_bar_t) * z_0
             + sqrt(1 - alpha_bar_t) * epsilon

epsilon ~ Normal(0, I)
```

The default schedule starts from a linear beta curve and rescales it so that
the final cumulative signal power is approximately:

```text
alpha_bar_T = 0.001
```

This matters in this project because the earlier unscaled short schedule ended
with too much clean signal remaining. Sampling then started from pure Gaussian
noise even though the model had never been trained near a pure-noise terminal
state.

The denoiser is `ConditionalLatentDenoiser`:

- A 1D CNN EEG encoder maps EEG to a conditioning vector.
- A sinusoidal timestep embedding maps the DDPM timestep to the same dimension.
- EEG and timestep embeddings are fused by an MLP.
- A stack of conditional residual 2D convolutional blocks predicts the added
  latent noise.

Training objective:

```text
t ~ Uniform({0, ..., T - 1})
epsilon ~ Normal(0, I)
z_t = q(z_t | z_0, epsilon)
epsilon_hat = Denoiser(z_t, t, eeg)

loss_ldm = MSE(epsilon_hat, epsilon)
```

Sampling:

1. Start from `z_T ~ Normal(0, I)` in normalized latent space.
2. Iteratively apply the learned reverse DDPM update conditioned on EEG.
3. By default, use deterministic posterior-mean sampling for reconstruction.
4. Optionally use stochastic sampling with `--stochastic`.
5. Convert the sampled normalized latent back to raw autoencoder scale.
6. Decode with the autoencoder.

```text
z_raw_sample = z_norm_sample * sigma + mu
pred_mel = Decoder(z_raw_sample)
```

## Demo Notebook

Run the root notebook:

```powershell
jupyter notebook notebook.ipynb
```

The notebook demonstrates:

- Dataset summary and sample inspection.
- EEG and waveform visualization.
- Log-mel feature extraction.
- Autoencoder reconstruction checks.
- CNN baseline prediction.
- LDM noise-prediction objective.
- EEG-conditioned LDM sampling.
- Side-by-side mel/error plots.
- WAV export for target, autoencoder, CNN, and LDM outputs.

If full checkpoints exist, the notebook loads them:

```text
checkpoints/autoencoder.pt
checkpoints/ldm.pt
checkpoints/baseline.pt
```

If they do not exist, it trains cached single-sample demo checkpoints:

```text
checkpoints/notebook_demo_v2_autoencoder.pt
checkpoints/notebook_demo_v2_ldm.pt
checkpoints/notebook_demo_v2_cnn.pt
```

Again, those notebook demo checkpoints are intentionally overfit to one sample.
They are a functional smoke test and visualization aid, not a final evaluation.

Notebook outputs are written to:

```text
outputs/notebook_demo/
```

## Command-Line Workflows

Use the existing virtual environment if available:

```powershell
.\.env\Scripts\python.exe scripts\inspect_dataset.py --max-subjects 1 --max-stimuli-per-subject 1
```

For a fresh environment:

```powershell
python -m venv .env
.\.env\Scripts\python.exe -m pip install -r requirements.txt
```

Inspect the dataset:

```powershell
.\.env\Scripts\python.exe scripts\inspect_dataset.py --split all
```

Train the spectrogram autoencoder:

```powershell
.\.env\Scripts\python.exe scripts\train_autoencoder.py --epochs 20 --batch-size 16
```

Train the EEG-conditioned LDM:

```powershell
.\.env\Scripts\python.exe scripts\train_ldm.py --autoencoder checkpoints\autoencoder.pt --epochs 40 --batch-size 16
```

Train the CNN baseline:

```powershell
.\.env\Scripts\python.exe scripts\train_baseline.py --epochs 30 --batch-size 16
```

Reconstruct a sample with the LDM:

```powershell
.\.env\Scripts\python.exe scripts\reconstruct.py --method ldm --sample-index 0
```

Reconstruct a sample with the CNN baseline:

```powershell
.\.env\Scripts\python.exe scripts\reconstruct.py --method baseline --sample-index 0
```

Use stochastic LDM sampling instead of deterministic posterior means:

```powershell
.\.env\Scripts\python.exe scripts\reconstruct.py --method ldm --sample-index 0 --stochastic
```

Reconstruction outputs are written to:

```text
outputs/reconstruction/
```

## Evaluation and Diagnostics

The repository currently emphasizes executable reconstruction diagnostics:

- Tensor shape checks for data, features, autoencoder, CNN, and LDM.
- Autoencoder reconstruction loss in normalized mel space.
- CNN prediction loss in normalized mel space.
- LDM noise-prediction MSE.
- Side-by-side mel spectrograms and absolute-error maps.
- Griffin-Lim waveform exports for listening.

Useful quantitative checks:

| Check | Meaning |
| --- | --- |
| Autoencoder L1/MSE | Whether the latent bottleneck preserves target mel structure. |
| CNN L1/MSE | Direct EEG-to-mel baseline fit. |
| LDM noise MSE | Whether the denoiser learns the DDPM objective. |
| LDM mel L1/MSE | Reconstruction quality after sampling and decoding. |
| Prediction standard deviation | Quick collapse check; flat outputs have very low variance. |

Future evaluation should add held-out splits, subject-wise generalization,
spectral similarity metrics, audio-domain metrics, and comparisons against
stimulus identity or retrieval baselines.

## Repository Structure

```text
.
|-- notebook.ipynb                  # Single project demo notebook
|-- README.md                       # This methods and usage document
|-- pyproject.toml                  # Package metadata
|-- requirements.txt                # Python dependencies
|-- configs/
|   |-- ldm.yaml                    # Default LDM settings
|   `-- baseline.yaml               # Default CNN baseline settings
|-- src/eeg_audio_reconstruction/
|   |-- data.py                     # Lazy HDF5 dataset and window indexing
|   |-- features.py                 # Log-mel and waveform reconstruction
|   |-- models.py                   # EEG encoder, CNN baseline, AE, denoiser
|   |-- diffusion.py                # DDPM schedule, q-sampling, reverse sampling
|   |-- latent.py                   # Latent mean/std estimation and transforms
|   |-- train_utils.py              # Checkpointing, seeding, device helpers
|   |-- visualization.py            # Mel image export
|   `-- config.py                   # Dataclass configs
|-- scripts/
|   |-- inspect_dataset.py          # Dataset summary
|   |-- train_autoencoder.py        # AE training
|   |-- train_ldm.py                # LDM training
|   |-- train_baseline.py           # CNN baseline training
|   |-- reconstruct.py              # Inference and artifact export
|   |-- smoke_test.py               # End-to-end shape/plumbing smoke test
|   `-- download_youtube_audio.py   # Utility script
|-- tests/
|   `-- test_core.py                # Unit tests for data/model/diffusion pieces
|-- assets/
|   |-- figures/                    # Project figures
|   `-- presentations/              # Presentation assets
|-- legacy/                         # Older prototype scripts
|-- data/                           # Local raw/preprocessed data, ignored by git
|-- checkpoints/                    # Model checkpoints, ignored by git
`-- outputs/                        # Reconstruction outputs, ignored by git
```

## Dependencies

| Package | Role |
| --- | --- |
| `torch` | Neural networks and training. |
| `torchaudio` | Mel transforms, resampling, inverse mel, Griffin-Lim, WAV export. |
| `h5py` | HDF5 dataset access. |
| `PyYAML` | Metadata and config loading. |
| `numpy`, `scipy`, `pandas` | Data utilities. |
| `matplotlib` | Spectrogram and diagnostic plots. |
| `scikit-learn` | General analysis dependency. |
| `tqdm` | Progress utilities. |
| `yt-dlp` | Optional audio-download utility. |

## Verification

Run unit tests:

```powershell
.\.env\Scripts\python.exe -m unittest discover -s tests
```

Run the dataset/model smoke test:

```powershell
.\.env\Scripts\python.exe scripts\smoke_test.py
```

Compile source and scripts:

```powershell
.\.env\Scripts\python.exe -m compileall src scripts
```

## Limitations

- The notebook fallback checkpoints are single-sample overfit demos.
- Griffin-Lim audio is useful for inspection but can sound phasey or synthetic.
- The current train/validation/test split is random over indexed windows; for
  stronger claims, subject-wise and stimulus-wise splits should be added.
- The compact LDM is not a pretrained AudioLDM model. It is a local latent
  diffusion model trained from the project data.
- EEG-to-audio decoding is underdetermined; spectrogram similarity does not
  necessarily imply perceptual or semantic correctness.
- The project currently emphasizes runnable methods over exhaustive evaluation.

## Research Context

This project sits at the intersection of neural decoding, auditory EEG
analysis, audio spectrogram modeling, and diffusion-based generative modeling.
The CNN baseline is related to direct EEG-to-mel reconstruction approaches such
as EEG2Mel. The LDM approach follows the broader idea of mapping brain signals
to a latent generative space, as seen in EEG-to-image and EEG/music latent
diffusion work. The use of a compact local autoencoder is a practical response
to the small-data and shape-compatibility constraints of this project.

## References

[1] Yunpeng Bai, Xintao Wang, Yan-pei Cao, Yixiao Ge, Chun Yuan, and Ying Shan.
"DreamDiffusion: Generating High-Quality Images from Brain EEG Signals." 2023.
arXiv:2306.16934.

[2] Adolfo G. Ramirez-Aristizabal and Chris Kello. "EEG2Mel: Reconstructing
Sound from Brain Responses to Music." 2022. arXiv:2207.13845.

[3] Emilian Postolache, Natalia Polouliakh, Hiroaki Kitano, Akima Connelly,
Emanuele Rodola, Luca Cosmo, and Taketo Akama. "Naturalistic Music Decoding
from EEG Data via Latent Diffusion Models." 2024. arXiv:2405.09062.

[4] Prajwal Singh, Pankaj Pandey, Krishna Miyapuram, and Shanmuganathan Raman.
"EEG2IMAGE: Image Reconstruction from EEG Brain Signals." 2023.
arXiv:2302.10121.

[5] Andreas Jansson, Eric Humphrey, Nicola Montecchio, Rachel M. Bittner,
Aparna Kumar, and Tillman Weyde. "Wave-U-Net: A Multi-Scale Neural Network for
End-to-End Audio Source Separation." 2018. arXiv:1806.03185.

[6] Steven Losorelli, Duc T. Nguyen, Jacek P. Dmochowski, and Blair Kaneshiro.
"NMED-T: A Tempo-Focused Dataset of Cortical and Behavioral Responses to
Naturalistic Music." To appear in Proceedings of the 18th International Society
for Music Information Retrieval Conference, Suzhou, China.

[7] Giorgia Cantisani, Gabriel Tregoat, Slim Essid, and Gael Richard.
"MAD-EEG: an EEG Dataset for Decoding Auditory Attention to a Target Instrument
in Polyphonic Music." Speech, Music and Mind (SMM), Satellite Workshop of
Interspeech 2019, Vienna, Austria. hal-02291882v1.
