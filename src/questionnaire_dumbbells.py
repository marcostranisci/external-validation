"""Dumbbell charts comparing each model's mean numeric (questionnaire) score
against its mean human-rated open-ended (interview) score, per model, for
MFT and PVQ side by side.

Reads the per-model processed survey files written by
``QuestionnaireAnalyzer.process_folder`` (``opinion`` = numeric Likert score,
``external_opinion`` = annotators' mean rating of the open-ended reply).

Usage
-----
    from src.questionnaire_dumbbells import plot_score_dumbbells

    plot_score_dumbbells(
        mft_dir="data_analysis/processed/mft",
        pvq_dir="data_analysis/processed/pvq",
        out_path="data_analysis/figures/questionnaire_vs_interview_dumbbells.png",
    )
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Categorical palette (house data-viz palette, slots 1-2): blue / orange.
C_QUESTIONNAIRE = "#2a78d6"
C_INTERVIEW = "#eb6834"
C_LINE = "#c3c2b7"
C_INK_PRIMARY = "#0b0b0b"
C_INK_SECONDARY = "#52514e"
C_GRID = "#e1e0d9"
C_SURFACE = "#fcfcfb"

# Fixed model order, shared with the other figures in the paper.
MODEL_ORDER = [
    "Llama-3.1-8B-Instruct",
    "Qwen3-8B",
    "Ministral-3-8B-Instruct-2512",
    "Falcon3-7B-Instruct",
    "Olmo-3-7B-Instruct",
    "Apertus-8B-Instruct",
]

# Display label per model (file stem -> label shown on the chart); only
# Ministral differs from its file stem, dropping the internal "-2512" tag.
MODEL_DISPLAY_LABELS = {m: m for m in MODEL_ORDER}
MODEL_DISPLAY_LABELS["Ministral-3-8B-Instruct-2512"] = "Ministral-3-8B-Instruct"


def _model_means(folder: str | Path, models: list[str] = MODEL_ORDER) -> pd.DataFrame:
    """Mean ``opinion`` (questionnaire) and ``external_opinion`` (interview)
    per model, for every ``<model>.csv`` found in ``folder``."""
    folder = Path(folder)
    rows = []
    for model in models:
        path = folder / f"{model}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        rows.append({
            "model": model,
            "questionnaire": df["opinion"].mean(),
            "interview": df["external_opinion"].mean(),
        })
    return pd.DataFrame(rows)


def _draw_dumbbell(ax, means: pd.DataFrame, title: str) -> None:
    # top-to-bottom in MODEL_ORDER order
    means = means.iloc[::-1].reset_index(drop=True)
    y = range(len(means))

    ax.hlines(y, means["questionnaire"], means["interview"], color=C_LINE, linewidth=2, zorder=1)
    ax.scatter(means["questionnaire"], y, s=170, color=C_QUESTIONNAIRE, zorder=3,
               label="Questionnaire", edgecolor=C_SURFACE, linewidth=0.8)
    ax.scatter(means["interview"], y, s=170, color=C_INTERVIEW, zorder=3,
               label="Interview", edgecolor=C_SURFACE, linewidth=0.8)

    ax.set_yticks(list(y))
    ax.set_yticklabels([MODEL_DISPLAY_LABELS.get(m, m) for m in means["model"]],
                        fontsize=10.5, color=C_INK_PRIMARY)
    ax.set_xlabel("Mean score", fontsize=10.5, color=C_INK_SECONDARY)
    ax.set_title(title, fontsize=13, color=C_INK_PRIMARY, fontweight="bold", pad=12)

    ax.grid(axis="x", color=C_GRID, linewidth=1, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10, colors=C_INK_SECONDARY, length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(C_INK_PRIMARY)
        spine.set_linewidth(1.1)
    ax.set_facecolor(C_SURFACE)


def plot_score_dumbbells(
    mft_dir: str | Path,
    pvq_dir: str | Path,
    out_path: str | Path,
    models: list[str] = MODEL_ORDER,
    suptitle: str = "Numeric score vs. open-ended reply, by model",
) -> Path:
    """Render MFT and PVQ questionnaire-vs-interview dumbbell charts side by
    side and save to ``out_path`` (both the given extension and a .pdf
    alongside it)."""
    mft_means = _model_means(mft_dir, models)
    pvq_means = _model_means(pvq_dir, models)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), facecolor=C_SURFACE)
    _draw_dumbbell(axes[0], mft_means, "MFT")
    _draw_dumbbell(axes[1], pvq_means, "PVQ")

    fig.suptitle(suptitle, fontsize=14.5, color=C_INK_PRIMARY, fontweight="bold", y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=2, frameon=True, fontsize=11, edgecolor=C_INK_PRIMARY)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=C_SURFACE)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=C_SURFACE)
    plt.close(fig)
    return out_path
