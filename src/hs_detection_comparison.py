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

    @staticmethod
    def _correlate(x: pd.Series, y: pd.Series) -> dict:
        n = len(x)
        row = {"n": n, "pearson_r": np.nan, "pearson_p": np.nan,
               "spearman_r": np.nan, "spearman_p": np.nan, "notes": ""}
        if n < 3 or x.std() == 0 or y.std() == 0:
            row["notes"] = "fewer than 3 valid items or zero variance"
            return row
        pear = stats.pearsonr(x, y)
        spear = stats.spearmanr(x, y)
        row.update(pearson_r=float(pear.statistic), pearson_p=float(pear.pvalue),
                   spearman_r=float(spear.statistic), spearman_p=float(spear.pvalue))
        return row

    def compare_flip_counts_by_value(self, kind: str) -> pd.DataFrame:
        """Model x value-type raw flip-count comparison (``kind="pvq"`` for
        Schwartz value, ``kind="mft"`` for Moral Foundation) between the
        two conditions — the flip-magnitude counterpart of
        ``compare_category_patterns`` (which compares recall-significance
        patterns, not raw flip volume)."""
        condition = kind
        a, b = self._load("flip_counts_by_model_and_value.csv")
        a = a[a["condition"] == condition]
        b = b[b["condition"] == condition]
        merged = a.merge(b, on=["model", "value_type"], suffixes=(f"_{self.label_a}", f"_{self.label_b}"))
        cols = ["model", "value_type",
                f"total_flips_{self.label_a}", f"mean_flip_rate_{self.label_a}",
                f"total_flips_{self.label_b}", f"mean_flip_rate_{self.label_b}"]
        result = merged[cols].sort_values(
            ["model", f"total_flips_{self.label_b}"], ascending=[True, False]
        )
        self._save(result, f"flip_counts_by_value_comparison_{kind}.csv")
        return result

    def compare_instance_flip_counts(self) -> tuple[pd.DataFrame, dict]:
        """Per message: flip count/rate/direction (relative to each
        condition's own zero-shot, pooled across all 6 models) side by
        side between the two conditions, plus a correlation summary of
        ``flip_rate`` across all 500 messages — are the *same* messages
        unstable under both steering methods?"""
        a, b = self._load("flip_counts_by_instance.csv")
        merged = a.merge(b, on=["id", "dataset_label"], suffixes=(f"_{self.label_a}", f"_{self.label_b}"))
        cols = ["id", "dataset_label",
                f"n_flips_{self.label_a}", f"flip_rate_{self.label_a}",
                f"n_flips_to_hate_{self.label_a}", f"n_flips_to_not_hate_{self.label_a}",
                f"n_flips_{self.label_b}", f"flip_rate_{self.label_b}",
                f"n_flips_to_hate_{self.label_b}", f"n_flips_to_not_hate_{self.label_b}"]
        result = merged[cols].copy()
        result["abs_diff_flip_rate"] = (
            result[f"flip_rate_{self.label_b}"] - result[f"flip_rate_{self.label_a}"]
        ).abs()
        result = result.sort_values("abs_diff_flip_rate", ascending=False)
        self._save(result, "instance_flip_counts_comparison.csv")

        corr = self._correlate(merged[f"flip_rate_{self.label_a}"], merged[f"flip_rate_{self.label_b}"])
        self._save(pd.DataFrame([corr]), "instance_flip_rate_correlation.csv")
        return result, corr

    def compare_instance_prediction_profile(self) -> tuple[pd.DataFrame, dict]:
        """Per message (pooled ``model="ALL"``): predicted-label stability
        (``proportion_hate``, no zero-shot baseline) side by side between
        the two conditions, plus a correlation summary across all 500
        messages — does a message's overall predicted-label consistency
        carry over regardless of which steering method is used?"""
        a, b = self._load("instance_prediction_profile.csv")
        a = a[a["model"] == "ALL"]
        b = b[b["model"] == "ALL"]
        merged = a.merge(b, on=["id", "dataset_label"], suffixes=(f"_{self.label_a}", f"_{self.label_b}"))
        cols = ["id", "dataset_label",
                f"proportion_hate_{self.label_a}", f"proportion_hate_{self.label_b}"]
        result = merged[cols].copy()
        result["abs_diff"] = (
            result[f"proportion_hate_{self.label_b}"] - result[f"proportion_hate_{self.label_a}"]
        ).abs()
        result = result.sort_values("abs_diff", ascending=False)
        self._save(result, "instance_prediction_profile_comparison.csv")

        corr = self._correlate(
            merged[f"proportion_hate_{self.label_a}"], merged[f"proportion_hate_{self.label_b}"]
        )
        self._save(pd.DataFrame([corr]), "instance_proportion_hate_correlation.csv")
        return result, corr

    def compare_flip_magnitude(self) -> pd.DataFrame:
        """Per model: does flip *magnitude* (per-item flip_rate,
        Kruskal-Wallis-tested for between-model differences within each
        condition in ``flip_magnitude_kruskal.csv``) rank the same way
        under both steering conditions? Joins each condition's per-model
        mean flip rate (from ``flip_counts_by_model.csv``, ``combined``
        row) side by side, since the two conditions' Kruskal-Wallis tests
        aren't directly comparable to each other (different H0)."""
        a, b = self._load("flip_counts_by_model.csv")
        a = a[a["condition"] == "combined"][["model", "mean_flip_rate"]]
        b = b[b["condition"] == "combined"][["model", "mean_flip_rate"]]
        merged = a.merge(b, on="model", suffixes=(f"_{self.label_a}", f"_{self.label_b}"))
        merged["rank_a"] = merged[f"mean_flip_rate_{self.label_a}"].rank(ascending=False)
        merged["rank_b"] = merged[f"mean_flip_rate_{self.label_b}"].rank(ascending=False)
        merged["rank_shift"] = merged["rank_a"] - merged["rank_b"]
        merged = merged.sort_values(f"mean_flip_rate_{self.label_b}", ascending=False)
        self._save(merged, "flip_magnitude_comparison.csv")
        return merged

    def compare_flip_direction(self) -> pd.DataFrame:
        """Per model: does the *direction* of flips (proportion toward the
        hate label, from ``flip_direction_proportion_by_model.csv``) agree
        between the two conditions? A 2x2 chi-square test (condition x
        direction) per model flags whether the shift in proportion is
        itself statistically real, not just numerically different."""
        a, b = self._load("flip_direction_proportion_by_model.csv")
        a = a[a["condition"] == "combined"]
        b = b[b["condition"] == "combined"]
        rows = []
        for model in sorted(set(a["model"]) & set(b["model"])):
            ra = a[a["model"] == model].iloc[0]
            rb = b[b["model"] == model].iloc[0]
            table = np.array([
                [ra["n_flips_to_hate"], ra["n_flips_to_not_hate"]],
                [rb["n_flips_to_hate"], rb["n_flips_to_not_hate"]],
            ])
            row = {
                "model": model,
                f"proportion_to_hate_{self.label_a}": ra["proportion_to_hate"],
                f"proportion_to_hate_{self.label_b}": rb["proportion_to_hate"],
                "same_direction_bias": (ra["proportion_to_hate"] >= 0.5) == (rb["proportion_to_hate"] >= 0.5),
                "chi2_stat": np.nan, "p": np.nan,
            }
            if table.sum() > 0 and table.min(axis=0).sum() >= 0 and (table.sum(axis=0) > 0).all():
                chi2, p, _, _ = stats.chi2_contingency(table)
                row.update(chi2_stat=float(chi2), p=float(p))
            rows.append(row)
        result = pd.DataFrame(rows)
        result["p_fdr_bh"] = stats.false_discovery_control(result["p"].to_numpy(), method="bh")
        result = result.sort_values("p_fdr_bh")
        self._save(result, "flip_direction_comparison.csv")
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
            corr = self._correlate(merged["delta_recall_a"], merged["delta_recall_b"])
            rows.append({"model": model, "n_items": corr["n"],
                         "pearson_r": corr["pearson_r"], "pearson_p": corr["pearson_p"],
                         "spearman_r": corr["spearman_r"], "spearman_p": corr["spearman_p"],
                         "notes": corr["notes"]})
        result = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
        self._save(result, "item_level_correlation_between_conditions.csv")
        return result

    def run_all(self) -> dict:
        instance_flips, instance_flip_corr = self.compare_instance_flip_counts()
        instance_profile, instance_profile_corr = self.compare_instance_prediction_profile()
        return {
            "steering_effect": self.compare_steering_effect(),
            "flip_counts": self.compare_flip_counts(),
            "flip_magnitude": self.compare_flip_magnitude(),
            "flip_direction": self.compare_flip_direction(),
            "flip_counts_by_value_pvq": self.compare_flip_counts_by_value("pvq"),
            "flip_counts_by_value_mft": self.compare_flip_counts_by_value("mft"),
            "pvq_patterns": self.compare_category_patterns("pvq"),
            "mft_patterns": self.compare_category_patterns("mft"),
            "item_level_correlation": self.compare_item_level(),
            "instance_flips": instance_flips, "instance_flip_corr": instance_flip_corr,
            "instance_profile": instance_profile, "instance_profile_corr": instance_profile_corr,
        }
