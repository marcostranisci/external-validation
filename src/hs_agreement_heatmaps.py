"""Heatmaps of pairwise inter-model agreement (Cohen's kappa) in hate-speech
classification, one per treatment condition (zero-shot, generated belief,
verbalized belief).

Reads the ``pairwise_model_agreement_{zero_shot,steered}.csv`` files already
produced by ``HateSpeechSteeringAnalyzer`` (see ``src/hs_detection_analysis.py``)
and renders them as a single figure with a shared color scale, so the three
conditions are visually comparable.

Usage
-----
    from src.hs_agreement_heatmaps import plot_agreement_heatmaps

    plot_agreement_heatmaps(
        zero_shot_csv="data_analysis/hs_detection/pairwise_model_agreement_zero_shot.csv",
        generated_csv="data_analysis/hs_detection/pairwise_model_agreement_steered.csv",
        verbalized_csv="data_analysis/hs_detection_verbalized_shared_baseline/pairwise_model_agreement_steered.csv",
        out_path="data_analysis/figures/hs_agreement_heatmaps.png",
    )
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Sequential blue ramp, steps 100->700, from the house data-viz palette
# (references/palette.md): lightest = near-zero agreement, darkest = near-1.
_BLUE_SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CMAP = LinearSegmentedColormap.from_list("blue_sequential", _BLUE_SEQUENTIAL)

C_SURFACE = "#fcfcfb"
C_INK_PRIMARY = "#0b0b0b"
C_INK_SECONDARY = "#52514e"
C_INK_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_DIAGONAL = "#eceae4"

# Short display labels, in a fixed order shared with the rest of the paper
# (Section 2.2, Models).
MODEL_ORDER = [
    "Llama-3.1-8B-Instruct",
    "Qwen3-8B",
    "Ministral-3-8B-Instruct-2512",
    "Falcon3-7B-Instruct",
    "Olmo-3-7B-Instruct",
    "Apertus-8B-Instruct",
]
MODEL_LABELS = {
    "Llama-3.1-8B-Instruct": "Llama",
    "Qwen3-8B": "Qwen",
    "Ministral-3-8B-Instruct-2512": "Ministral",
    "Falcon3-7B-Instruct": "Falcon",
    "Olmo-3-7B-Instruct": "OLMo",
    "Apertus-8B-Instruct": "Apertus",
}


def _build_matrix(csv_path: str | Path, models: list[str] = MODEL_ORDER) -> np.ndarray:
    """Build a symmetric kappa matrix (diagonal = NaN) from a pairwise
    agreement CSV with ``model_a``, ``model_b``, ``cohens_kappa`` columns."""
    df = pd.read_csv(csv_path)
    n = len(models)
    mat = np.full((n, n), np.nan)
    idx = {m: i for i, m in enumerate(models)}
    for _, row in df.iterrows():
        if row["model_a"] not in idx or row["model_b"] not in idx:
            continue
        i, j = idx[row["model_a"]], idx[row["model_b"]]
        mat[i, j] = mat[j, i] = row["cohens_kappa"]
    return mat


def _draw_heatmap(ax, mat: np.ndarray, labels: list[str], title: str,
                   vmin: float, vmax: float) -> None:
    n = mat.shape[0]
    masked = np.ma.masked_invalid(mat)
    ax.set_facecolor(C_DIAGONAL)
    im = ax.imshow(masked, cmap=CMAP, vmin=vmin, vmax=vmax, aspect="equal")

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            value = mat[i, j]
            norm = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_color = "white" if norm > 0.55 else C_INK_PRIMARY
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                     fontsize=9.5, color=text_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9.5, color=C_INK_SECONDARY)
    ax.set_yticklabels(labels, fontsize=9.5, color=C_INK_SECONDARY)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=C_SURFACE, linewidth=2.5)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12, color=C_INK_PRIMARY, fontweight="bold", pad=10)
    return im


def plot_agreement_heatmaps(
    zero_shot_csv: str | Path,
    generated_csv: str | Path,
    verbalized_csv: str | Path,
    out_path: str | Path,
    models: list[str] = MODEL_ORDER,
    suptitle: str = "Pairwise inter-model agreement in hate-speech classification (Cohen's $\\kappa$)",
) -> Path:
    """Render zero-shot / generated-belief / verbalized-belief agreement
    matrices side by side on a shared color scale and save to ``out_path``
    (both the given extension and a .pdf alongside it)."""
    labels = [MODEL_LABELS.get(m, m) for m in models]
    mats = {
        "Zero-shot": _build_matrix(zero_shot_csv, models),
        "Generated belief": _build_matrix(generated_csv, models),
        "Verbalized belief": _build_matrix(verbalized_csv, models),
    }

    all_values = np.concatenate([m[~np.isnan(m)] for m in mats.values()])
    vmin, vmax = float(np.floor(all_values.min() * 20) / 20), float(np.ceil(all_values.max() * 20) / 20)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), facecolor=C_SURFACE)
    im = None
    for ax, (title, mat) in zip(axes, mats.items()):
        im = _draw_heatmap(ax, mat, labels, title, vmin, vmax)

    fig.suptitle(suptitle, fontsize=13.5, color=C_INK_PRIMARY, fontweight="bold", y=1.03)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Cohen's $\\kappa$", fontsize=10.5, color=C_INK_SECONDARY)
    cbar.ax.tick_params(labelsize=9, color=C_INK_MUTED, labelcolor=C_INK_SECONDARY)
    cbar.outline.set_visible(False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=C_SURFACE)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=C_SURFACE)
    plt.close(fig)
    return out_path
