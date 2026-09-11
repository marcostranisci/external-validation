"""Compares two runs of HateSpeechSteeringAnalyzer against each other —
built for comparing the free-text-opinion steering condition
(``hs_detection/implicit_hate_all_models.json``) against the verbalized
questionnaire-item steering condition
(``hs_detection/implicit_hate_verbalized_all_models.json``), but works on
any two of its output directories.

Usage
-----
    from src.hs_detection_comparison import SteeringConditionComparison

    cmp = SteeringConditionComparison(
        dir_a="data_analysis/hs_detection", label_a="free_text",
        dir_b="data_analysis/hs_detection_verbalized", label_b="verbalized",
        output_dir="data_analysis/hs_detection_comparison",
    )
    cmp.run_all()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SteeringConditionComparison:
    dir_a: str | Path
    dir_b: str | Path
    label_a: str = "condition_a"
    label_b: str = "condition_b"
    output_dir: str | Path = "data_analysis/hs_detection_comparison"

    def __post_init__(self) -> None:
        self.dir_a = Path(self.dir_a)
        self.dir_b = Path(self.dir_b)
        self.output_dir = Path(self.output_dir)

    def _load(self, filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        return pd.read_csv(self.dir_a / filename), pd.read_csv(self.dir_b / filename)

    def _save(self, df: pd.DataFrame, filename: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_dir / filename, index=False)

    # ------------------------------------------------------------------
    def compare_steering_effect(self) -> pd.DataFrame:
        """Per model: recall-shift magnitude/direction/significance under
        each condition, side by side, plus whether the two conditions
        agree on direction."""
        a, b = self._load("steering_effect_per_model.csv")
        a = a[a["condition"] == "combined"]
        b = b[b["condition"] == "combined"]
        merged = a.merge(b, on="model", suffixes=(f"_{self.label_a}", f"_{self.label_b}"))
        cols = ["model",
                f"zero_shot_recall_{self.label_a}",
                f"mean_delta_{self.label_a}", f"wilcoxon_p_{self.label_a}",
                f"mean_delta_{self.label_b}", f"wilcoxon_p_{self.label_b}"]
        result = merged[cols].copy()
        result["same_direction"] = np.sign(result[f"mean_delta_{self.label_a}"]) == \
            np.sign(result[f"mean_delta_{self.label_b}"])
        result["delta_of_deltas"] = (
            result[f"mean_delta_{self.label_b}"] - result[f"mean_delta_{self.label_a}"]
        )
        result = result.sort_values("delta_of_deltas", ascending=False)
        self._save(result, "steering_effect_comparison.csv")
        return result

    def compare_flip_counts(self) -> pd.DataFrame:
        """Per model: total flips and flip direction under each condition,
        side by side."""
        a, b = self._load("flip_counts_by_model.csv")
        a = a[a["condition"] == "combined"]
        b = b[b["condition"] == "combined"]
        merged = a.merge(b, on="model", suffixes=(f"_{self.label_a}", f"_{self.label_b}"))
        cols = ["model",
                f"total_flips_{self.label_a}", f"total_flips_to_hate_{self.label_a}",
                f"total_flips_to_not_hate_{self.label_a}",
                f"total_flips_{self.label_b}", f"total_flips_to_hate_{self.label_b}",
                f"total_flips_to_not_hate_{self.label_b}"]
        result = merged[cols].sort_values(f"total_flips_{self.label_b}", ascending=False)
        self._save(result, "flip_counts_comparison.csv")
        return result

    def compare_category_patterns(self, kind: str) -> pd.DataFrame:
        """Pooled (``model="ALL"``) mean delta-recall by content category
        (``kind="pvq"`` for Schwartz value, ``kind="mft"`` for Moral
        Foundation), side by side between the two conditions."""
        filename = "pvq_value_patterns.csv" if kind == "pvq" else "mft_foundation_patterns.csv"
        category_col = "pvq_value_label" if kind == "pvq" else "foundation"
        a, b = self._load(filename)
        a = a[a["model"] == "ALL"]
        b = b[b["model"] == "ALL"]
        merged = a.merge(b, on=category_col, suffixes=(f"_{self.label_a}", f"_{self.label_b}"))
        cols = [category_col,
                f"mean_delta_recall_{self.label_a}", f"n_significant_{self.label_a}",
                f"mean_delta_recall_{self.label_b}", f"n_significant_{self.label_b}"]
        result = merged[cols].sort_values(f"mean_delta_recall_{self.label_b}", ascending=False)
        self._save(result, f"{kind}_category_comparison.csv")
        return result

    def compare_item_level(self) -> pd.DataFrame:
        """Per model: Pearson/Spearman correlation between the two
        conditions' per-item ``delta_recall`` (joined on condition +
        belief_id) — do the same items drive the effect under both
        steering methods?"""
        a, b = self._load("item_significance_per_model.csv")
        rows = []
        for model in sorted(set(a["model"]) & set(b["model"])):
            ma = a[a["model"] == model][["condition", "belief_id", "delta_recall"]]
            mb = b[b["model"] == model][["condition", "belief_id", "delta_recall"]]
            merged = ma.merge(mb, on=["condition", "belief_id"], suffixes=("_a", "_b")).dropna()
            n = len(merged)
            row = {"model": model, "n_items": n, "pearson_r": np.nan, "pearson_p": np.nan,
                   "spearman_r": np.nan, "spearman_p": np.nan, "notes": ""}
            if n < 3 or merged["delta_recall_a"].std() == 0 or merged["delta_recall_b"].std() == 0:
                row["notes"] = "fewer than 3 valid items or zero variance"
            else:
                pear = stats.pearsonr(merged["delta_recall_a"], merged["delta_recall_b"])
                spear = stats.spearmanr(merged["delta_recall_a"], merged["delta_recall_b"])
                row.update(pearson_r=float(pear.statistic), pearson_p=float(pear.pvalue),
                           spearman_r=float(spear.statistic), spearman_p=float(spear.pvalue))
            rows.append(row)
        result = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
        self._save(result, "item_level_correlation_between_conditions.csv")
        return result

    def run_all(self) -> dict[str, pd.DataFrame]:
        return {
            "steering_effect": self.compare_steering_effect(),
            "flip_counts": self.compare_flip_counts(),
            "pvq_patterns": self.compare_category_patterns("pvq"),
            "mft_patterns": self.compare_category_patterns("mft"),
            "item_level_correlation": self.compare_item_level(),
        }
