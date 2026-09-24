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
from src.questionnaire_analysis import _benjamini_hochberg

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


def _significance_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _per_model_recall(
    json_path: str | Path, baseline_json_path: str | Path | None = None,
    models: list[str] = MODEL_ORDER, output_subdir: str = "hs_detection",
    condition: str = "combined",
) -> dict[str, dict[str, float]]:
    """Per model: zero-shot recall (single value), mean recall across the
    belief-steered items for ``condition`` (``"combined"`` = all 76 mft+pvq
    items, or ``"mft"``/``"pvq"`` for just that questionnaire's items), and
    whether that steered-vs-zero-shot shift is significant (one-sample
    Wilcoxon signed-rank test on the per-item deltas,
    ``steering_effect_per_model``'s matching row, Benjamini-Hochberg
    corrected across the 6 models).

    ``output_subdir`` keeps this in line with the matching
    run_hs_detection*_analysis.py pipeline for the same JSON, so the
    ``steering_effect_per_model.csv`` this recomputes overwrites the same
    file that pipeline already writes, rather than a stray duplicate."""
    analyzer = HateSpeechSteeringAnalyzer(
        json_path=json_path, baseline_json_path=baseline_json_path, output_subdir=output_subdir,
    )
    table = analyzer.build_recall_table()
    effect = analyzer.steering_effect_per_model(table)
    rows = effect[effect["condition"] == condition].set_index("model")
    rows["wilcoxon_p_fdr_bh"] = _benjamini_hochberg(rows["wilcoxon_p"])

    steered_conditions = ["mft", "pvq"] if condition == "combined" else [condition]
    out = {}
    for model in models:
        sub = table[table["model"] == model]
        zero_shot = sub.loc[sub["condition"] == "zero_shot", "recall"].iloc[0]
        steered = sub.loc[sub["condition"].isin(steered_conditions), "recall"].mean()
        p_fdr = rows.loc[model, "wilcoxon_p_fdr_bh"] if model in rows.index else np.nan
        out[model] = {"zero_shot": zero_shot, "steered": steered, "p_fdr": p_fdr}
    return out


def _draw_recall_bars(ax, gen: dict, verb: dict, models: list[str], title: str) -> None:
    zero_shot = [gen[m]["zero_shot"] for m in models]
    generated = [gen[m]["steered"] for m in models]
    verbalized = [verb[m]["steered"] for m in models]
    generated_stars = [_significance_stars(gen[m]["p_fdr"]) for m in models]
    verbalized_stars = [_significance_stars(verb[m]["p_fdr"]) for m in models]
    labels = [MODEL_DISPLAY_LABELS.get(m, m) for m in models]

    x = np.arange(len(models))
    width = 0.26
    ax.set_facecolor(C_SURFACE)

    bars = [
        (x - width, zero_shot, C_ZERO_SHOT, "Zero-shot", None),
        (x, generated, C_GENERATED, "Generated belief", generated_stars),
        (x + width, verbalized, C_VERBALIZED, "Verbalized belief", verbalized_stars),
    ]
    for pos, values, color, label, stars in bars:
        ax.bar(pos, values, width=width * 0.92, color=color, label=label, zorder=3,
               edgecolor=C_SURFACE, linewidth=0.8)
        if stars is not None:
            for xi, yi, star in zip(pos, values, stars):
                if star:
                    ax.text(xi, yi + 0.015, star, ha="center", va="bottom",
                            fontsize=12, color=C_INK_PRIMARY, fontweight="bold", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10, color=C_INK_PRIMARY)
    ax.set_ylabel("Recall (positive / HS class)", fontsize=11, color=C_INK_SECONDARY)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=13.5, color=C_INK_PRIMARY, fontweight="bold", pad=12)

    ax.grid(axis="y", color=C_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10, colors=C_INK_SECONDARY, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(C_INK_PRIMARY)
    ax.spines["bottom"].set_linewidth(1.1)


def plot_recall_barplot(
    generated_json: str | Path,
    verbalized_json: str | Path,
    out_path: str | Path,
    zero_shot_baseline_json: str | Path | None = None,
    models: list[str] = MODEL_ORDER,
    title: str = "Hate-speech classification recall by model and condition",
) -> Path:
    """Render the 3-bars-per-model grouped bar chart (all 76 mft+pvq items
    pooled) and save to ``out_path`` (both the given extension and a .pdf
    alongside it).

    ``zero_shot_baseline_json`` lets the verbalized file's own zero-shot be
    overridden by a shared baseline (matching
    ``HateSpeechSteeringAnalyzer.baseline_json_path``); defaults to using
    ``generated_json``'s zero-shot as that shared baseline, since the
    zero-shot bar is drawn once per model, not once per condition.
    """
    gen = _per_model_recall(generated_json, models=models, output_subdir="hs_detection")
    verb = _per_model_recall(
        verbalized_json,
        baseline_json_path=zero_shot_baseline_json or generated_json,
        models=models,
        output_subdir="hs_detection_verbalized_shared_baseline",
    )

    fig, ax = plt.subplots(figsize=(12.5, 6), facecolor=C_SURFACE)
    _draw_recall_bars(ax, gen, verb, models, title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
               frameon=False, fontsize=10.5)
    ax.text(0.5, -0.25, "* $p_{\\mathrm{FDR}}$<0.05, ** $p_{\\mathrm{FDR}}$<0.01, *** $p_{\\mathrm{FDR}}$<0.001 "
                          "(Wilcoxon signed-rank vs. zero-shot, BH-corrected across models)",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color=C_INK_SECONDARY)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=C_SURFACE)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=C_SURFACE)
    plt.close(fig)
    return out_path


def plot_recall_barplot_by_questionnaire(
    generated_json: str | Path,
    verbalized_json: str | Path,
    out_path: str | Path,
    zero_shot_baseline_json: str | Path | None = None,
    models: list[str] = MODEL_ORDER,
    suptitle: str = "Hate-speech classification recall by model, condition, and questionnaire",
) -> Path:
    """Render MFT and PVQ recall bar charts side by side (each model's
    zero-shot / generated-belief / verbalized-belief recall computed only
    over that questionnaire's items, with its own significance test) and
    save to ``out_path`` (both the given extension and a .pdf alongside
    it)."""
    baseline = zero_shot_baseline_json or generated_json

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), facecolor=C_SURFACE)
    for ax, questionnaire, title in zip(axes, ("mft", "pvq"), ("MFT", "PVQ")):
        gen = _per_model_recall(generated_json, models=models, output_subdir="hs_detection",
                                 condition=questionnaire)
        verb = _per_model_recall(verbalized_json, baseline_json_path=baseline, models=models,
                                  output_subdir="hs_detection_verbalized_shared_baseline",
                                  condition=questionnaire)
        _draw_recall_bars(ax, gen, verb, models, title)

    fig.suptitle(suptitle, fontsize=14.5, color=C_INK_PRIMARY, fontweight="bold", y=1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1),
               ncol=3, frameon=False, fontsize=10.5)
    fig.text(0.5, -0.16, "* $p_{\\mathrm{FDR}}$<0.05, ** $p_{\\mathrm{FDR}}$<0.01, *** $p_{\\mathrm{FDR}}$<0.001 "
                          "(Wilcoxon signed-rank vs. zero-shot, BH-corrected across models, within questionnaire)",
             ha="center", fontsize=9, color=C_INK_SECONDARY)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=C_SURFACE)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=C_SURFACE)
    plt.close(fig)
    return out_path
