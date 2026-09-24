"""Grouped bar chart of per-model recall under the three hate-speech
classification conditions: zero-shot, generated belief, verbalized belief.

Zero-shot and generated-belief recall come from the free-text elicited-belief
JSON (``hs_detection``); verbalized-belief recall comes from the
confound-controlled shared-baseline JSON (``hs_detection_verbalized_shared_
baseline``), consistent with the rest of the analysis pipeline. Generated/
verbalized recall per model is the mean recall across all 76 belief-steered
items (MFT + PVQ).

Usage
-----
    from src.hs_recall_barplot import plot_recall_barplot

    plot_recall_barplot(
        generated_json="hs_detection/implicit_hate_all_models.json",
        verbalized_json="hs_detection/implicit_hate_verbalized_all_models.json",
        out_path="data_analysis/figures/hs_recall_barplot.png",
    )
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.hs_detection_analysis import HateSpeechSteeringAnalyzer

# Categorical palette (house data-viz palette, slots 1-3): blue / orange / aqua.
C_ZERO_SHOT = "#2a78d6"
C_GENERATED = "#eb6834"
C_VERBALIZED = "#1baf7a"
C_INK_PRIMARY = "#0b0b0b"
C_INK_SECONDARY = "#52514e"
C_GRID = "#e1e0d9"
C_SURFACE = "#fcfcfb"

MODEL_ORDER = [
    "Llama-3.1-8B-Instruct",
    "Qwen3-8B",
    "Ministral-3-8B-Instruct-2512",
    "Falcon3-7B-Instruct",
    "Olmo-3-7B-Instruct",
    "Apertus-8B-Instruct",
]
MODEL_DISPLAY_LABELS = {m: m for m in MODEL_ORDER}
MODEL_DISPLAY_LABELS["Ministral-3-8B-Instruct-2512"] = "Ministral-3-8B-Instruct"


def _per_model_recall(
    json_path: str | Path, baseline_json_path: str | Path | None = None,
    models: list[str] = MODEL_ORDER,
) -> dict[str, dict[str, float]]:
    """Per model: zero-shot recall (single value) and mean recall across
    all 76 belief-steered items (mft + pvq)."""
    analyzer = HateSpeechSteeringAnalyzer(json_path=json_path, baseline_json_path=baseline_json_path)
    table = analyzer.build_recall_table()
    out = {}
    for model in models:
        sub = table[table["model"] == model]
        zero_shot = sub.loc[sub["condition"] == "zero_shot", "recall"].iloc[0]
        steered = sub.loc[sub["condition"].isin(["mft", "pvq"]), "recall"].mean()
        out[model] = {"zero_shot": zero_shot, "steered": steered}
    return out


def plot_recall_barplot(
    generated_json: str | Path,
    verbalized_json: str | Path,
    out_path: str | Path,
    zero_shot_baseline_json: str | Path | None = None,
    models: list[str] = MODEL_ORDER,
    title: str = "Hate-speech classification recall by model and condition",
) -> Path:
    """Render the 3-bars-per-model grouped bar chart and save to
    ``out_path`` (both the given extension and a .pdf alongside it).

    ``zero_shot_baseline_json`` lets the verbalized file's own zero-shot be
    overridden by a shared baseline (matching
    ``HateSpeechSteeringAnalyzer.baseline_json_path``); defaults to using
    ``generated_json``'s zero-shot as that shared baseline, since the
    zero-shot bar is drawn once per model, not once per condition.
    """
    gen = _per_model_recall(generated_json, models=models)
    verb = _per_model_recall(
        verbalized_json,
        baseline_json_path=zero_shot_baseline_json or generated_json,
        models=models,
    )

    zero_shot = [gen[m]["zero_shot"] for m in models]
    generated = [gen[m]["steered"] for m in models]
    verbalized = [verb[m]["steered"] for m in models]
    labels = [MODEL_DISPLAY_LABELS.get(m, m) for m in models]

    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12.5, 6), facecolor=C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    bars = [
        (x - width, zero_shot, C_ZERO_SHOT, "Zero-shot"),
        (x, generated, C_GENERATED, "Generated belief"),
        (x + width, verbalized, C_VERBALIZED, "Verbalized belief"),
    ]
    for pos, values, color, label in bars:
        ax.bar(pos, values, width=width * 0.92, color=color, label=label, zorder=3,
               edgecolor=C_SURFACE, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5, color=C_INK_PRIMARY)
    ax.set_ylabel("Recall (positive / HS class)", fontsize=11, color=C_INK_SECONDARY)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=14, color=C_INK_PRIMARY, fontweight="bold", pad=14)

    ax.grid(axis="y", color=C_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10, colors=C_INK_SECONDARY, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(C_INK_PRIMARY)
    ax.spines["bottom"].set_linewidth(1.1)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
               frameon=False, fontsize=10.5)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=C_SURFACE)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=C_SURFACE)
    plt.close(fig)
    return out_path
