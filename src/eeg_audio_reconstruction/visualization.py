"""Visualization helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_mel_image(mel: torch.Tensor, path: str | Path, title: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = mel.detach().cpu()
    if image.dim() == 4:
        image = image[0, 0]
    elif image.dim() == 3:
        image = image[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(image, origin="lower", aspect="auto", cmap="magma", vmin=-1, vmax=1)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bins")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

