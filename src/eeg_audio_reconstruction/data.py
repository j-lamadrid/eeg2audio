"""Lazy MAD-EEG HDF5 dataset access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml


@dataclass(frozen=True)
class SampleRef:
    subject: str
    stimulus: str
    segment_index: int
    eeg_start: int
    eeg_stop: int
    eeg_interval_length: int
    audio_start: int
    audio_stop: int
    audio_sample_rate: int


class MADEEGDataset(Dataset):
    """Windowed EEG/audio pairs from ``madeeg_preprocessed.hdf5``.

    The response matrix appears to store repeated EEG trials concatenated in
    time. By default each sample averages the same one-second window across
    four repetitions before pairing it with the corresponding audio window.
    """

    def __init__(
        self,
        hdf5_path: str | Path = "data/madeeg_preprocessed.hdf5",
        metadata_path: str | Path = "data/madeeg_preprocessed.yaml",
        segment_seconds: float = 1.0,
        eeg_sample_rate: int = 256,
        average_repetitions: int = 4,
        target: str = "stimulus",
        normalize_eeg: bool = True,
        split: str | None = "train",
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        max_subjects: int | None = None,
        max_stimuli_per_subject: int | None = None,
        max_samples: int | None = None,
        subject_ids: Iterable[str] | None = None,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.metadata_path = Path(metadata_path)
        self.segment_seconds = float(segment_seconds)
        self.eeg_sample_rate = int(eeg_sample_rate)
        self.average_repetitions = max(1, int(average_repetitions))
        self.target = target
        self.normalize_eeg = normalize_eeg
        self.split = split
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.seed = int(seed)
        self.max_subjects = max_subjects
        self.max_stimuli_per_subject = max_stimuli_per_subject
        self.max_samples = max_samples
        self.subject_ids = list(subject_ids) if subject_ids is not None else None
        self._h5: h5py.File | None = None

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.hdf5_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata YAML not found: {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata: dict[str, Any] = yaml.safe_load(handle) or {}

        refs = self._build_index()
        refs = self._apply_split(refs)
        if max_samples is not None:
            refs = refs[: int(max_samples)]
        self.refs = refs

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index]
        group = self._file[ref.subject][ref.stimulus]

        eeg = self._read_eeg(group["response"], ref)
        audio = self._read_audio(group, ref)

        return {
            "eeg": torch.from_numpy(eeg),
            "audio": torch.from_numpy(audio).unsqueeze(0),
            "audio_sample_rate": ref.audio_sample_rate,
            "subject": ref.subject,
            "stimulus": ref.stimulus,
            "segment_index": ref.segment_index,
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    @property
    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def describe(self) -> dict[str, Any]:
        subjects = {ref.subject for ref in self.refs}
        stimuli = {ref.stimulus for ref in self.refs}
        seconds = len(self.refs) * self.segment_seconds
        return {
            "samples": len(self.refs),
            "subjects": len(subjects),
            "stimuli": len(stimuli),
            "hours": seconds / 3600.0,
            "target": self.target,
            "split": self.split or "all",
        }

    def _build_index(self) -> list[SampleRef]:
        eeg_window = int(round(self.segment_seconds * self.eeg_sample_rate))
        refs: list[SampleRef] = []

        with h5py.File(self.hdf5_path, "r") as h5:
            subjects = self.subject_ids or list(h5.keys())
            if self.max_subjects is not None:
                subjects = subjects[: int(self.max_subjects)]

            for subject in subjects:
                if subject not in h5:
                    continue
                stimuli = list(h5[subject].keys())
                if self.max_stimuli_per_subject is not None:
                    stimuli = stimuli[: int(self.max_stimuli_per_subject)]

                for stimulus in stimuli:
                    group = h5[subject][stimulus]
                    if "response" not in group:
                        continue
                    if not self._has_target(group):
                        continue

                    meta = self.metadata.get(subject, {}).get(stimulus, {})
                    source_sr = int(meta.get("wav_info", {}).get("sfreq", 44100))
                    audio_window = int(round(self.segment_seconds * source_sr))

                    response_len = int(group["response"].shape[1])
                    interval_length = response_len // self.average_repetitions
                    usable_eeg_len = interval_length if self.average_repetitions > 1 else response_len
                    audio_len = self._target_audio_length(group)
                    n_segments = min(usable_eeg_len // eeg_window, audio_len // audio_window)

                    for segment in range(n_segments):
                        refs.append(
                            SampleRef(
                                subject=subject,
                                stimulus=stimulus,
                                segment_index=segment,
                                eeg_start=segment * eeg_window,
                                eeg_stop=(segment + 1) * eeg_window,
                                eeg_interval_length=interval_length,
                                audio_start=segment * audio_window,
                                audio_stop=(segment + 1) * audio_window,
                                audio_sample_rate=source_sr,
                            )
                        )
        return refs

    def _apply_split(self, refs: list[SampleRef]) -> list[SampleRef]:
        if self.split is None or str(self.split).lower() in {"all", "none"}:
            return refs

        if self.val_fraction < 0 or self.test_fraction < 0:
            raise ValueError("Split fractions must be non-negative.")
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be less than 1.")

        split = str(self.split).lower()
        shuffled = list(refs)
        random.Random(self.seed).shuffle(shuffled)

        n_total = len(shuffled)
        n_test = int(round(n_total * self.test_fraction))
        n_val = int(round(n_total * self.val_fraction))

        test_refs = shuffled[:n_test]
        val_refs = shuffled[n_test : n_test + n_val]
        train_refs = shuffled[n_test + n_val :]

        if split == "train":
            return train_refs
        if split in {"val", "valid", "validation"}:
            return val_refs
        if split == "test":
            return test_refs
        raise ValueError(f"Unknown split: {self.split}")

    def _has_target(self, group: h5py.Group) -> bool:
        if self.target in {"stimulus", "mix"}:
            return "stimulus" in group
        return "soli" in group

    def _target_audio_length(self, group: h5py.Group) -> int:
        if self.target in {"stimulus", "mix"}:
            return int(group["stimulus"].shape[1])
        return int(group["soli"].shape[1])

    def _read_eeg(self, response: h5py.Dataset, ref: SampleRef) -> np.ndarray:
        if self.average_repetitions <= 1:
            eeg = response[:, ref.eeg_start : ref.eeg_stop]
        else:
            pieces = []
            for rep in range(self.average_repetitions):
                start = rep * ref.eeg_interval_length + ref.eeg_start
                stop = rep * ref.eeg_interval_length + ref.eeg_stop
                pieces.append(response[:, start:stop])
            eeg = np.stack(pieces, axis=0).mean(axis=0)

        eeg = np.asarray(eeg, dtype=np.float32)
        if self.normalize_eeg:
            mean = eeg.mean(axis=1, keepdims=True)
            std = eeg.std(axis=1, keepdims=True)
            eeg = (eeg - mean) / (std + 1e-6)
        return np.nan_to_num(eeg, copy=False).astype(np.float32)

    def _read_audio(self, group: h5py.Group, ref: SampleRef) -> np.ndarray:
        start, stop = ref.audio_start, ref.audio_stop
        if self.target in {"stimulus", "mix"}:
            audio = np.asarray(group["stimulus"][:, start:stop], dtype=np.float32).mean(axis=0)
        else:
            index = self._soli_index(ref)
            audio = np.asarray(group["soli"][index, start:stop], dtype=np.float32)

        expected = stop - start
        if audio.shape[-1] < expected:
            audio = np.pad(audio, (0, expected - audio.shape[-1]))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        return np.nan_to_num(audio, copy=False).astype(np.float32)

    def _soli_index(self, ref: SampleRef) -> int:
        if self.target.startswith("soli:"):
            return int(self.target.split(":", 1)[1])

        meta = self.metadata.get(ref.subject, {}).get(ref.stimulus, {})
        instruments = list(meta.get("instruments", []))
        target_name = meta.get("target")
        if target_name in instruments:
            return instruments.index(target_name)
        return 0

