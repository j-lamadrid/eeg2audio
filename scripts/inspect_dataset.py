from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from eeg_audio_reconstruction.data import MADEEGDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the MAD-EEG HDF5 dataset.")
    parser.add_argument("--hdf5", default="data/madeeg_preprocessed.hdf5")
    parser.add_argument("--metadata", default="data/madeeg_preprocessed.yaml")
    parser.add_argument("--split", default="all")
    parser.add_argument("--target", default="stimulus", choices=["stimulus", "mix", "target", "soli"])
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-stimuli-per-subject", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = MADEEGDataset(
        hdf5_path=args.hdf5,
        metadata_path=args.metadata,
        split=args.split,
        target=args.target,
        segment_seconds=args.segment_seconds,
        max_subjects=args.max_subjects,
        max_stimuli_per_subject=args.max_stimuli_per_subject,
        max_samples=args.max_samples,
    )
    print("Dataset summary")
    for key, value in dataset.describe().items():
        print(f"  {key}: {value}")
    if len(dataset):
        sample = dataset[0]
        print("First sample")
        print(f"  eeg: {tuple(sample['eeg'].shape)}")
        print(f"  audio: {tuple(sample['audio'].shape)} at {sample['audio_sample_rate']} Hz")
        print(f"  subject/stimulus/segment: {sample['subject']} / {sample['stimulus']} / {sample['segment_index']}")
    dataset.close()


if __name__ == "__main__":
    main()

