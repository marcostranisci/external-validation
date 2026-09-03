"""Analysis of belief-steered hate-speech detection recall.

For each model, ``hs_detection/implicit_hate_all_models.json`` holds hate
speech classifications (500 messages, balanced 250/250) under:

- ``zero_shot`` — one run, no questionnaire context.
- ``mft`` / ``pvq`` — one run per questionnaire item (36 MFT, 40 PVQ),
  each steered by that *same* model's own free-text reply to that item
  (``"prediction_setup": "paired_by_model"`` in the file's metadata — there
  is no cross-model steering condition in this file).

This module checks two things:

1. Does steering (belief context) shift recall relative to zero-shot,
   within each model?
2. Does the size/direction of that shift differ between models?

Usage
-----
    from src.hs_detection_analysis import HateSpeechSteeringAnalyzer

    analyzer = HateSpeechSteeringAnalyzer(json_path="hs_detection/implicit_hate_all_models.json")
    analyzer.run_all()
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.questionnaire_analysis import _correlations, _kruskal, _mannwhitney  # noqa: F401 (reuse)


def _recall(predictions: list[dict]) -> tuple[float, int, int]:
    """Recall on the positive (hate speech, label 1) class.

    Returns (recall, true_positives, n_positive).
    """
    positives = [p for p in predictions if int(p["dataset_label"]) == 1]
    n_pos = len(positives)
    if n_pos == 0:
        return float("nan"), 0, 0
    tp = sum(1 for p in positives if str(p["answer"]) == "1")
    return tp / n_pos, tp, n_pos


@dataclass
class HateSpeechSteeringAnalyzer:
    """Loads the belief-steered hate-speech JSON and tests whether
    questionnaire-belief steering shifts recall, and whether that shift
    differs across models."""

    json_path: str | Path
    output_dir: str | Path = "data_analysis"

    def __post_init__(self) -> None:
        self.json_path = Path(self.json_path)
        self.output_dir = Path(self.output_dir)
        with open(self.json_path) as fh:
            self._data = json.load(fh)

    # ------------------------------------------------------------------
    # Recall table
    # ------------------------------------------------------------------
    def build_recall_table(self) -> pd.DataFrame:
        """One row per (model, condition, item): recall on the 500-message
        set. ``condition`` is ``zero_shot``, ``mft``, or ``pvq``; zero_shot
        has a single row per model (``belief_id``/``test_statement`` null)."""
        rows = []
        for model, model_data in self._data["models"].items():
            recall, tp, n_pos = _recall(model_data["zero_shot"])
            rows.append({
                "model": model, "condition": "zero_shot", "belief_id": np.nan,
                "test_statement": None, "recall": recall, "tp": tp, "n_positive": n_pos,
            })
            for condition in ("mft", "pvq"):
                for item in model_data[condition]:
                    recall, tp, n_pos = _recall(item["predictions"])
                    rows.append({
                        "model": model, "condition": condition,
                        "belief_id": item["belief_id"],
                        "test_statement": item["test_statement"],
                        "recall": recall, "tp": tp, "n_positive": n_pos,
                    })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Does steering shift recall? (within each model)
    # ------------------------------------------------------------------
    def steering_effect_per_model(self, table: pd.DataFrame) -> pd.DataFrame:
        """Per model (and per questionnaire, plus combined): compare the
        distribution of belief-steered recalls against the model's own
        zero-shot recall with a one-sample Wilcoxon signed-rank test on
        (steered_recall - zero_shot_recall), plus descriptive stats."""
        rows = []
        for model, group in table.groupby("model"):
            zero_shot_recall = group.loc[group["condition"] == "zero_shot", "recall"].iloc[0]
            for condition, steered in [
                ("mft", group[group["condition"] == "mft"]),
                ("pvq", group[group["condition"] == "pvq"]),
                ("combined", group[group["condition"].isin(["mft", "pvq"])]),
            ]:
                deltas = (steered["recall"] - zero_shot_recall).dropna()
                row = {
                    "model": model, "condition": condition, "n_items": len(deltas),
                    "zero_shot_recall": zero_shot_recall,
                    "mean_steered_recall": steered["recall"].mean(),
                    "median_steered_recall": steered["recall"].median(),
                    "mean_delta": deltas.mean(), "median_delta": deltas.median(),
                    "wilcoxon_stat": np.nan, "wilcoxon_p": np.nan, "notes": "",
                }
                if len(deltas) < 5 or (deltas == 0).all():
                    row["notes"] = "fewer than 5 non-trivial deltas"
                else:
                    try:
                        stat, p = stats.wilcoxon(deltas)
                        row["wilcoxon_stat"], row["wilcoxon_p"] = float(stat), float(p)
                    except ValueError as exc:
                        row["notes"] = str(exc)
                rows.append(row)
        result = pd.DataFrame(rows)
        self._save(result, "steering_effect_per_model.csv")
        return result

    # ------------------------------------------------------------------
    # Does the steering effect differ between models?
    # ------------------------------------------------------------------
    def between_model_steering_effect(self, table: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Compare the per-item delta (steered_recall - that model's own
        zero_shot_recall) across models: a pooled Kruskal-Wallis test plus
        pairwise Mann-Whitney U tests, separately for mft, pvq, and both
        combined."""
        zero_shot = table.loc[table["condition"] == "zero_shot", ["model", "recall"]] \
            .rename(columns={"recall": "zero_shot_recall"})
        merged = table[table["condition"] != "zero_shot"].merge(zero_shot, on="model")
        merged["delta"] = merged["recall"] - merged["zero_shot_recall"]

        kruskal_rows = []
        pairwise_rows = []
        for condition, sub in [
            ("mft", merged[merged["condition"] == "mft"]),
            ("pvq", merged[merged["condition"] == "pvq"]),
            ("combined", merged),
        ]:
            groups = {model: g["delta"] for model, g in sub.groupby("model")}
            res = _kruskal(groups)
            kruskal_rows.append({"condition": condition, **res})

            for model_a, model_b in itertools.combinations(sorted(groups), 2):
                mw = _mannwhitney(groups[model_a], groups[model_b])
                pairwise_rows.append({
                    "condition": condition, "model_a": model_a, "model_b": model_b, **mw
                })

        kruskal_df = pd.DataFrame(kruskal_rows)
        pairwise_df = pd.DataFrame(pairwise_rows)
        self._save(kruskal_df, "between_model_steering_kruskal.csv")
        self._save(pairwise_df, "between_model_steering_pairwise_mannwhitney.csv")
        return {"kruskal": kruskal_df, "pairwise": pairwise_df}

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def _save(self, df: pd.DataFrame, filename: str) -> None:
        out_folder = self.output_dir / "hs_detection"
        out_folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_folder / filename, index=False)

    def run_all(self) -> dict[str, pd.DataFrame]:
        table = self.build_recall_table()
        self._save(table, "recall_by_model_condition_item.csv")
        per_model = self.steering_effect_per_model(table)
        between = self.between_model_steering_effect(table)
        return {"recall_table": table, "per_model": per_model, **between}
