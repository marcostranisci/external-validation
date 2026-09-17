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

    def compare_steering_effect_shared_baseline(self) -> pd.DataFrame:
        """Robustness check for the wording confound (`zero_shot_prompt`
        differs between the two files): recomputes condition B's steering
        delta against condition A's zero-shot recall instead of B's own —
        forcing both conditions to share one baseline. If a model's
        direction flip (test 1) is a genuine content effect rather than an
        artifact of the baseline itself having shifted from wording alone,
        it should survive this — condition B's steered recall minus
        condition A's zero-shot should still point the same direction as
        condition B's own-baseline delta. Uses only
        `recall_by_model_condition_item.csv` from both directories (no new
        model runs needed)."""
        a, b = self._load("recall_by_model_condition_item.csv")
        a_zs = a[a["condition"] == "zero_shot"].set_index("model")["recall"]
        a_items = a[a["condition"] != "zero_shot"]
        b_items = b[b["condition"] != "zero_shot"]
        b_zs = b[b["condition"] == "zero_shot"].set_index("model")["recall"]

        rows = []
        for model in sorted(set(a_zs.index) & set(b_zs.index)):
            a_delta = a_items[a_items["model"] == model]["recall"] - a_zs[model]
            b_delta_own = b_items[b_items["model"] == model]["recall"] - b_zs[model]
            b_delta_shared = b_items[b_items["model"] == model]["recall"] - a_zs[model]
            wa = stats.wilcoxon(a_delta)
            wb_own = stats.wilcoxon(b_delta_own)
            wb_shared = stats.wilcoxon(b_delta_shared)
            rows.append({
                "model": model,
                f"zero_shot_recall_{self.label_a}": float(a_zs[model]),
                f"zero_shot_recall_{self.label_b}": float(b_zs[model]),
                "zero_shot_shift_from_wording": float(b_zs[model] - a_zs[model]),
                f"mean_delta_{self.label_a}": float(a_delta.mean()), "wilcoxon_p_a": float(wa.pvalue),
                f"mean_delta_{self.label_b}_own_baseline": float(b_delta_own.mean()),
                "wilcoxon_p_b_own_baseline": float(wb_own.pvalue),
                f"mean_delta_{self.label_b}_shared_baseline": float(b_delta_shared.mean()),
                "wilcoxon_p_b_shared_baseline": float(wb_shared.pvalue),
            })
        result = pd.DataFrame(rows)
        # Does B's own-baseline direction survive being forced onto A's
        # baseline? False means the model's own-baseline result is at
        # least partly an artifact of the wording-shifted baseline itself.
        result["b_direction_survives_shared_baseline"] = np.sign(
            result[f"mean_delta_{self.label_b}_own_baseline"]
        ) == np.sign(result[f"mean_delta_{self.label_b}_shared_baseline"])
        # Does the original A-vs-B direction flip (test 1) still hold when
        # B is forced onto A's baseline instead of comparing own-baseline
        # deltas?
        result["ab_flip_survives_shared_baseline"] = np.sign(
            result[f"mean_delta_{self.label_a}"]
        ) != np.sign(result[f"mean_delta_{self.label_b}_shared_baseline"])
        self._save(result, "steering_effect_shared_baseline_check.csv")
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

    def compare_category_patterns_excluding_model(self, kind: str, exclude_model: str) -> pd.DataFrame:
        """The same category comparison as ``compare_category_patterns``,
        but averaging each category's ``mean_delta_recall`` across the
        5 models excluding ``exclude_model`` instead of reading the
        precomputed ``model="ALL"`` pooled row — for isolating whether a
        pattern in the pooled-6-model number is being driven by one
        outlier model (e.g. Apertus, whose depressed baseline dominates
        any pooled mean it's part of)."""
        filename = "pvq_value_patterns.csv" if kind == "pvq" else "mft_foundation_patterns.csv"
        category_col = "pvq_value_label" if kind == "pvq" else "foundation"
        a, b = self._load(filename)
        a_excl = a[(a["model"] != "ALL") & (a["model"] != exclude_model)]
        b_excl = b[(b["model"] != "ALL") & (b["model"] != exclude_model)]
        a_mean = a_excl.groupby(category_col)["mean_delta_recall"].mean().rename(f"mean_delta_recall_{self.label_a}")
        b_mean = b_excl.groupby(category_col)["mean_delta_recall"].mean().rename(f"mean_delta_recall_{self.label_b}")
        result = pd.concat([a_mean, b_mean], axis=1).reset_index()
        result = result.sort_values(f"mean_delta_recall_{self.label_b}", ascending=False)
        self._save(result, f"{kind}_category_comparison_excl_{exclude_model.split('-')[0].lower()}.csv")
        return result

    def compare_within_model_rank(self, kind: str) -> pd.DataFrame:
        """Do the two conditions agree on each category's *within-model*
        rank (from ``within_model_category_rank`` — rank 1 = that model's
        weakest lever), pooled across the 6 models via ``mean_rank``? This
        rank is invariant to any per-model additive baseline shift, so
        unlike ``compare_category_patterns`` it needs no shared-baseline
        correction to be a fair comparison. A high, significant Spearman
        correlation means a category that's relatively weak for a model
        under one elicitation method is also relatively weak under the
        other; a low one means the *relative* pattern itself depends on
        elicitation method, not just its overall level."""
        category_col = "pvq_value_label" if kind == "pvq" else "foundation"
        a, b = self._load(f"within_model_rank_summary_{category_col}.csv")
        merged = a.merge(b, on=category_col, suffixes=(f"_{self.label_a}", f"_{self.label_b}"))
        cols = [category_col, f"mean_rank_{self.label_a}", f"mean_rank_{self.label_b}"]
        result = merged[cols].sort_values(f"mean_rank_{self.label_a}")
        corr = self._correlate(
            merged[f"mean_rank_{self.label_a}"], merged[f"mean_rank_{self.label_b}"]
        )
        self._save(result, f"within_model_rank_comparison_{kind}.csv")
        self._save(pd.DataFrame([{kind: kind, **corr}]), f"within_model_rank_correlation_{kind}.csv")
        return result

    def compare_model_agreement(self, kind: str) -> pd.DataFrame:
        """Does the same pair of models agree the most (or least) with
        each other regardless of elicitation method? ``kind="zero_shot"``
        or ``kind="steered"`` selects which of the two pairwise-agreement
        tables to compare. Joins each pair's `cohens_kappa` between the
        two conditions and reports the Spearman/Pearson correlation across
        all 15 pairs — a high one means "which models see hate speech the
        same way" is a stable property of the model pair, not an artifact
        of how the belief was elicited."""
        filename = f"pairwise_model_agreement_{kind}.csv"
        a, b = self._load(filename)
        merged = a.merge(b, on=["model_a", "model_b"], suffixes=(f"_{self.label_a}", f"_{self.label_b}"))
        cols = ["model_a", "model_b", f"cohens_kappa_{self.label_a}", f"cohens_kappa_{self.label_b}"]
        result = merged[cols].sort_values(f"cohens_kappa_{self.label_a}", ascending=False)
        corr = self._correlate(
            merged[f"cohens_kappa_{self.label_a}"], merged[f"cohens_kappa_{self.label_b}"]
        )
        self._save(result, f"model_agreement_comparison_{kind}.csv")
        self._save(pd.DataFrame([{"kind": kind, **corr}]), f"model_agreement_correlation_{kind}.csv")
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

    def compare_flip_rate_tiers(self) -> dict[str, pd.DataFrame]:
        """Do the same messages fall in the same no/mild/strong-flip tier
        under both steering conditions? Builds the 3x3 contingency table
        (tier under A x tier under B) from ``flip_rate_tiers_by_instance.csv``,
        reports overall/per-tier overlap, a chi-square test of association,
        and Cohen's kappa (agreement beyond chance; 0 = chance level, 1 =
        perfect agreement, computed directly from the contingency table so
        no extra dependency is needed)."""
        a, b = self._load("flip_rate_tiers_by_instance.csv")
        merged = a.merge(b, on=["id", "dataset_label"], suffixes=(f"_{self.label_a}", f"_{self.label_b}"))
        tiers = ["no_flip", "mild_flip", "strong_flip"]
        contingency = pd.crosstab(
            merged[f"tier_{self.label_a}"], merged[f"tier_{self.label_b}"]
        ).reindex(index=tiers, columns=tiers, fill_value=0)
        self._save(contingency.reset_index(), "flip_rate_tier_contingency.csv")

        n = contingency.to_numpy().sum()
        observed_agree = np.trace(contingency.to_numpy())
        p_o = observed_agree / n
        row_marg = contingency.sum(axis=1).to_numpy() / n
        col_marg = contingency.sum(axis=0).to_numpy() / n
        p_e = float((row_marg * col_marg).sum())
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else np.nan
        chi2, p, dof, _ = stats.chi2_contingency(contingency.to_numpy())

        per_tier_rows = []
        for t in tiers:
            n_a = int(contingency.loc[t].sum())
            n_same = int(contingency.loc[t, t])
            per_tier_rows.append({
                "tier": t, f"n_{self.label_a}": n_a,
                "n_stayed_same_tier": n_same,
                "proportion_stayed": n_same / n_a if n_a else np.nan,
            })
        per_tier = pd.DataFrame(per_tier_rows)
        self._save(per_tier, "flip_rate_tier_stability.csv")

        summary = pd.DataFrame([{
            "n": int(n), "n_same_tier": int(observed_agree),
            "proportion_same_tier": p_o, "expected_by_chance": p_e,
            "cohens_kappa": kappa, "chi2_stat": float(chi2), "chi2_p": float(p), "dof": int(dof),
        }])
        self._save(summary, "flip_rate_tier_agreement.csv")
        return {"contingency": contingency, "per_tier": per_tier, "summary": summary}

    def compare_item_steerability_tiers(self) -> dict[str, pd.DataFrame]:
        """The item-level counterpart of ``compare_flip_rate_tiers``: do
        the same 76 items fall in the same low/mid/high-steerability tier
        (from ``item_steerability_tiers.csv``) under both conditions? Joins
        on ``condition`` + ``test_statement`` (items are tiered separately
        within MFT and PVQ, so cross-tabulates within each questionnaire
        too), reports the 3x3 contingency table, per-tier stability, and
        overall agreement (proportion in the same tier vs. the chance
        baseline implied by the marginals, Cohen's kappa, chi-square)."""
        a, b = self._load("item_steerability_tiers.csv")
        merged = a.merge(b, on=["condition", "test_statement"], suffixes=(f"_{self.label_a}", f"_{self.label_b}"))
        tiers = ["low_steerability", "mid_steerability", "high_steerability"]

        contingencies = {}
        summaries = []
        for condition, sub in [("mft", merged[merged["condition"] == "mft"]),
                                ("pvq", merged[merged["condition"] == "pvq"]),
                                ("combined", merged)]:
            contingency = pd.crosstab(
                sub[f"steerability_tier_{self.label_a}"], sub[f"steerability_tier_{self.label_b}"]
            ).reindex(index=tiers, columns=tiers, fill_value=0)
            contingencies[condition] = contingency

            n = contingency.to_numpy().sum()
            observed_agree = np.trace(contingency.to_numpy())
            p_o = observed_agree / n
            row_marg = contingency.sum(axis=1).to_numpy() / n
            col_marg = contingency.sum(axis=0).to_numpy() / n
            p_e = float((row_marg * col_marg).sum())
            kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else np.nan
            chi2, p, dof, _ = stats.chi2_contingency(contingency.to_numpy())
            summaries.append({
                "condition": condition, "n": int(n), "n_same_tier": int(observed_agree),
                "proportion_same_tier": p_o, "expected_by_chance": p_e,
                "cohens_kappa": kappa, "chi2_stat": float(chi2), "chi2_p": float(p), "dof": int(dof),
            })

        combined_contingency = contingencies["combined"].reset_index()
        self._save(combined_contingency, "item_steerability_tier_contingency.csv")
        summary_df = pd.DataFrame(summaries)
        self._save(summary_df, "item_steerability_tier_agreement.csv")
        return {"contingency": combined_contingency, "summary": summary_df}

    def robust_high_steerability_items(self) -> pd.DataFrame:
        """Which specific items land in the high-steerability tier under
        *both* conditions — a method-independent, item-level signal, as
        opposed to a category-level pattern that could hold on average
        while no individual item is robust. Saved to
        ``robust_high_steerability_items.csv``, sorted by condition and
        combined mean flip rate."""
        a, b = self._load("item_steerability_tiers.csv")
        merged = a.merge(b, on=["condition", "test_statement"], suffixes=(f"_{self.label_a}", f"_{self.label_b}"))
        both_high = merged[
            (merged[f"steerability_tier_{self.label_a}"] == "high_steerability") &
            (merged[f"steerability_tier_{self.label_b}"] == "high_steerability")
        ].copy()
        category_col_a = f"foundation_{self.label_a}"
        category_col_b = f"pvq_value_label_{self.label_a}"
        both_high["category"] = both_high[category_col_a].fillna(both_high[category_col_b])
        cols = ["condition", "test_statement", "category",
                f"mean_flip_rate_{self.label_a}", f"mean_flip_rate_{self.label_b}"]
        result = both_high[cols].sort_values(
            ["condition", f"mean_flip_rate_{self.label_b}"], ascending=[True, False]
        )
        self._save(result, "robust_high_steerability_items.csv")
        return result

    def robust_high_steerability_by_category(self, robust_items: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """For each questionnaire, is any single MFT foundation / Schwartz
        PVQ value over-represented among items that are robustly
        high-steerability in *both* conditions (from
        ``robust_high_steerability_items``)? One-vs-rest Fisher exact test
        per category, BH-corrected within each questionnaire — the item-
        overlap analogue of ``steerability_tier_patterns``'s single-
        condition version."""
        a, _ = self._load("item_steerability_tiers.csv")
        out: dict[str, pd.DataFrame] = {}
        for condition, category_col, filename_stub in [
            ("mft", "foundation", "foundation"), ("pvq", "pvq_value_label", "pvq_value"),
        ]:
            sub = a[a["condition"] == condition][["test_statement", category_col]].drop_duplicates()
            robust_ids = set(robust_items[robust_items["condition"] == condition]["test_statement"])
            sub = sub.copy()
            sub["both_high"] = sub["test_statement"].isin(robust_ids)

            n_total = len(sub)
            n_high_total = sub["both_high"].sum()
            rows = []
            for category, g in sub.groupby(category_col):
                n_cat = len(g)
                n_cat_high = int(g["both_high"].sum())
                table = [[n_cat_high, n_cat - n_cat_high],
                         [n_high_total - n_cat_high, n_total - n_cat - (n_high_total - n_cat_high)]]
                odds_ratio, p = stats.fisher_exact(table, alternative="two-sided")
                rows.append({
                    "condition": condition, category_col: category, "n_items": n_cat,
                    "n_robust_high": n_cat_high, "odds_ratio": float(odds_ratio), "p": float(p),
                })
            fisher_df = pd.DataFrame(rows)
            fisher_df["p_fdr_bh"] = stats.false_discovery_control(fisher_df["p"].to_numpy(), method="bh")
            fisher_df = fisher_df.sort_values("p_fdr_bh")
            out[filename_stub] = fisher_df
            self._save(fisher_df, f"robust_high_steerability_by_{filename_stub}.csv")
        return out

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
        flip_rate_tiers = self.compare_flip_rate_tiers()
        item_tiers = self.compare_item_steerability_tiers()
        robust_items = self.robust_high_steerability_items()
        robust_by_category = self.robust_high_steerability_by_category(robust_items)
        return {
            "steering_effect": self.compare_steering_effect(),
            "steering_effect_shared_baseline": self.compare_steering_effect_shared_baseline(),
            "flip_counts": self.compare_flip_counts(),
            "flip_magnitude": self.compare_flip_magnitude(),
            "flip_direction": self.compare_flip_direction(),
            "flip_rate_tiers": flip_rate_tiers,
            "item_steerability_tiers": item_tiers,
            "robust_high_steerability_items": robust_items,
            "robust_high_steerability_by_category": robust_by_category,
            "flip_counts_by_value_pvq": self.compare_flip_counts_by_value("pvq"),
            "flip_counts_by_value_mft": self.compare_flip_counts_by_value("mft"),
            "pvq_patterns": self.compare_category_patterns("pvq"),
            "mft_patterns": self.compare_category_patterns("mft"),
            "pvq_patterns_excl_apertus": self.compare_category_patterns_excluding_model(
                "pvq", "Apertus-8B-Instruct"),
            "mft_patterns_excl_apertus": self.compare_category_patterns_excluding_model(
                "mft", "Apertus-8B-Instruct"),
            "pvq_rank_comparison": self.compare_within_model_rank("pvq"),
            "mft_rank_comparison": self.compare_within_model_rank("mft"),
            "model_agreement_zero_shot": self.compare_model_agreement("zero_shot"),
            "model_agreement_steered": self.compare_model_agreement("steered"),
            "item_level_correlation": self.compare_item_level(),
            "instance_flips": instance_flips, "instance_flip_corr": instance_flip_corr,
            "instance_profile": instance_profile, "instance_profile_corr": instance_profile_corr,
        }
