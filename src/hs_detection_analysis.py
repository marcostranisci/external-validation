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


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR-adjusted p-values, leaving NaN entries as NaN
    and correcting only across the valid p-values (one hypothesis family)."""
    valid = pvalues.dropna()
    adjusted = pd.Series(np.nan, index=pvalues.index)
    if len(valid) == 0:
        return adjusted
    adjusted.loc[valid.index] = stats.false_discovery_control(valid.to_numpy(), method="bh")
    return adjusted


#: Schwartz's 10 basic values, as numbered in the PVQ item mapping (1-10).
SCHWARTZ_PVQ_VALUES = {
    1: "Universalism", 2: "Benevolence", 3: "Tradition", 4: "Conformity",
    5: "Security", 6: "Power", 7: "Achievement", 8: "Hedonism",
    9: "Stimulation", 10: "Self-Direction",
}


def _mcnemar_exact(b: int, c: int) -> float:
    """Exact (binomial) McNemar test p-value on the discordant pair counts
    ``b`` (correct under zero-shot, wrong under steering) and ``c`` (the
    reverse). Appropriate here since per-item discordant counts can be
    small (out of 250 positive-class messages)."""
    if b + c == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)


def _agreement(a: dict, b: dict) -> dict:
    """Pairwise agreement between two models' predicted labels on the same
    set of items (``a``/``b``: id -> predicted label, as strings "0"/"1").
    Reports raw agreement, Cohen's kappa (chance-corrected; 0 = chance
    level, 1 = perfect), and a chi-square test of independence between the
    two models' label distributions (with a phi coefficient, the 2x2
    effect-size analogue of Cramer's V) — tests whether the two models'
    predictions are associated at all, as opposed to kappa's "how much
    better than chance," which can be near 0 even when association is
    statistically real but weak."""
    ids = sorted(set(a) & set(b))
    n = len(ids)
    result = {"n": n, "raw_agreement": np.nan, "cohens_kappa": np.nan,
              "chi2_stat": np.nan, "chi2_p": np.nan, "phi": np.nan, "notes": ""}
    if n == 0:
        result["notes"] = "no shared ids"
        return result
    xa = np.array([str(a[i]) for i in ids])
    xb = np.array([str(b[i]) for i in ids])
    contingency = pd.crosstab(pd.Series(xa, name="a"), pd.Series(xb, name="b"))
    contingency = contingency.reindex(index=["0", "1"], columns=["0", "1"], fill_value=0)
    observed_agree = np.trace(contingency.to_numpy())
    p_o = observed_agree / n
    row_marg = contingency.sum(axis=1).to_numpy() / n
    col_marg = contingency.sum(axis=0).to_numpy() / n
    p_e = float((row_marg * col_marg).sum())
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else np.nan
    result.update(raw_agreement=p_o, cohens_kappa=kappa)
    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2 and contingency.to_numpy().sum() > 0:
        try:
            chi2, p, _, _ = stats.chi2_contingency(contingency.to_numpy())
            phi = float(np.sqrt(chi2 / n))
            result.update(chi2_stat=float(chi2), chi2_p=float(p), phi=phi)
        except ValueError as exc:
            result["notes"] = str(exc)
    return result


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
    output_subdir: str = "hs_detection"
    baseline_json_path: str | Path | None = None
    """If set, every model's ``zero_shot`` predictions are replaced with
    that model's ``zero_shot`` predictions from this other JSON file
    before any analysis runs — every method below reads
    ``model_data["zero_shot"]``, so this one substitution makes the whole
    pipeline (recall, flips, item significance, instance profiles, ...)
    use a single shared baseline instead of each file's own. Built to
    control for a wording difference in ``zero_shot_prompt`` between two
    files that would otherwise confound a same-model, own-baseline
    comparison between them."""

    def __post_init__(self) -> None:
        self.json_path = Path(self.json_path)
        self.output_dir = Path(self.output_dir)
        with open(self.json_path) as fh:
            self._data = json.load(fh)
        if self.baseline_json_path is not None:
            self.baseline_json_path = Path(self.baseline_json_path)
            with open(self.baseline_json_path) as fh:
                baseline_data = json.load(fh)
            for model, model_data in self._data["models"].items():
                model_data["zero_shot"] = baseline_data["models"][model]["zero_shot"]

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
                "test_statement": None, "foundation": None,
                "recall": recall, "tp": tp, "n_positive": n_pos,
            })
            for condition in ("mft", "pvq"):
                for item in model_data[condition]:
                    recall, tp, n_pos = _recall(item["predictions"])
                    rows.append({
                        "model": model, "condition": condition,
                        "belief_id": item["belief_id"],
                        "test_statement": item["test_statement"],
                        "foundation": item.get("foundation"),
                        "recall": recall, "tp": tp, "n_positive": n_pos,
                    })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Do models agree with each other on the predicted label?
    # ------------------------------------------------------------------
    def pairwise_model_agreement_zero_shot(self) -> pd.DataFrame:
        """Pairwise agreement between every pair of models' *zero-shot*
        predicted labels on the same 500 messages — the cleanest "do these
        two models fundamentally see hate speech the same way" comparison,
        unconfounded by which belief steered which model. See `_agreement`
        for the metrics (raw agreement, Cohen's kappa, chi-square, phi).
        Saved to `pairwise_model_agreement_zero_shot.csv`, sorted by
        `cohens_kappa` descending (most-agreeing pair first)."""
        preds = {
            model: {p["id"]: p["answer"] for p in model_data["zero_shot"]}
            for model, model_data in self._data["models"].items()
        }
        rows = []
        for model_a, model_b in itertools.combinations(sorted(preds), 2):
            row = _agreement(preds[model_a], preds[model_b])
            rows.append({"model_a": model_a, "model_b": model_b, **row})
        result = pd.DataFrame(rows).sort_values("cohens_kappa", ascending=False)
        self._save(result, "pairwise_model_agreement_zero_shot.csv")
        return result

    def pairwise_model_agreement_steered(self) -> pd.DataFrame:
        """Pairwise agreement between every pair of models' predicted
        labels under belief-steering, pooled across all 456 (id,
        belief_id, condition) combinations shared between the two models
        (36 MFT + 40 PVQ items x 500 messages) — much higher-powered than
        the zero-shot comparison. Note this compares model A steered by
        *its own* belief on item X against model B steered by *its own*
        belief on item X (`"prediction_setup": "paired_by_model"`): the
        content slot is the same, but the actual belief text differs per
        model, so this measures agreement under "the same kind of moral
        context," not literally the same steering text. Saved to
        `pairwise_model_agreement_steered.csv`, sorted by `cohens_kappa`
        descending."""
        preds: dict[str, dict[tuple, str]] = {}
        for model, model_data in self._data["models"].items():
            model_preds = {}
            for condition in ("mft", "pvq"):
                for item in model_data[condition]:
                    for p in item["predictions"]:
                        model_preds[(condition, item["belief_id"], p["id"])] = p["answer"]
            preds[model] = model_preds
        rows = []
        for model_a, model_b in itertools.combinations(sorted(preds), 2):
            row = _agreement(preds[model_a], preds[model_b])
            rows.append({"model_a": model_a, "model_b": model_b, **row})
        result = pd.DataFrame(rows).sort_values("cohens_kappa", ascending=False)
        self._save(result, "pairwise_model_agreement_steered.csv")
        return result

    def model_agreement_summary(
        self, zero_shot_agreement: pd.DataFrame, steered_agreement: pd.DataFrame,
    ) -> pd.DataFrame:
        """Per model: its mean pairwise Cohen's kappa with the other 5
        models, under zero-shot and under steering — "how much does this
        model agree with the panel on average," a single ranked score per
        model rather than a 15-pair matrix. Also reports the Spearman
        correlation between the two conditions' `cohens_kappa` across the
        15 pairs (does the "which pair agrees most" ranking hold up under
        steering, or is it specific to zero-shot?)."""
        rows = []
        for model in sorted(set(zero_shot_agreement["model_a"]) | set(zero_shot_agreement["model_b"])):
            zs = zero_shot_agreement[
                (zero_shot_agreement["model_a"] == model) | (zero_shot_agreement["model_b"] == model)
            ]["cohens_kappa"]
            st = steered_agreement[
                (steered_agreement["model_a"] == model) | (steered_agreement["model_b"] == model)
            ]["cohens_kappa"]
            rows.append({
                "model": model,
                "mean_kappa_with_panel_zero_shot": zs.mean(),
                "mean_kappa_with_panel_steered": st.mean(),
            })
        result = pd.DataFrame(rows).sort_values("mean_kappa_with_panel_zero_shot", ascending=False)
        self._save(result, "model_agreement_summary.csv")

        merged = zero_shot_agreement.merge(
            steered_agreement, on=["model_a", "model_b"], suffixes=("_zero_shot", "_steered")
        )
        corr = stats.spearmanr(merged["cohens_kappa_zero_shot"], merged["cohens_kappa_steered"])
        corr_row = pd.DataFrame([{
            "n_pairs": len(merged), "spearman_r": float(corr.statistic), "spearman_p": float(corr.pvalue),
        }])
        self._save(corr_row, "model_agreement_zero_shot_vs_steered_correlation.csv")
        return result

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
    # Which items steer recall the most?
    # ------------------------------------------------------------------
    def item_level_effect(self, table: pd.DataFrame) -> pd.DataFrame:
        """Per questionnaire item: across the 6 models, how much does being
        steered by *that model's own* reply to this item shift recall away
        from that model's zero-shot baseline?

        For each item this reports the delta's mean, mean absolute value
        (the item's overall "how much does it move recall, either way"
        score), std, min/max across the 6 models, and a one-sample
        Wilcoxon signed-rank test on those 6 per-model deltas (note: n=6 is
        a small sample, so treat the p-value as indicative, not
        confirmatory — the mean/mean-abs delta and how consistent the sign
        is across models are more informative for ranking items).
        ``wilcoxon_p_fdr_bh`` Benjamini-Hochberg-corrects across all items'
        p-values (one hypothesis family: 76 tests, one per item)."""
        zero_shot = table.loc[table["condition"] == "zero_shot", ["model", "recall"]] \
            .rename(columns={"recall": "zero_shot_recall"})
        merged = table[table["condition"].isin(["mft", "pvq"])].merge(zero_shot, on="model")
        merged["delta"] = merged["recall"] - merged["zero_shot_recall"]

        rows = []
        key_cols = ["condition", "belief_id", "test_statement", "foundation"]
        for keys, group in merged.groupby(key_cols, dropna=False):
            condition, belief_id, test_statement, foundation = keys
            deltas = group["delta"].dropna()
            row = {
                "condition": condition, "belief_id": belief_id,
                "test_statement": test_statement, "foundation": foundation,
                "n_models": len(deltas),
                "mean_delta": deltas.mean(), "mean_abs_delta": deltas.abs().mean(),
                "std_delta": deltas.std(ddof=0),
                "min_delta": deltas.min(), "max_delta": deltas.max(),
                "n_positive": int((deltas > 0).sum()), "n_negative": int((deltas < 0).sum()),
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
        result = pd.DataFrame(rows).sort_values("mean_abs_delta", ascending=False)
        result["wilcoxon_p_fdr_bh"] = _benjamini_hochberg(result["wilcoxon_p"])
        self._save(result, "item_level_steering_effect.csv")
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
    # Which items significantly change recall, for each model individually?
    # ------------------------------------------------------------------
    def item_significance_per_model(
        self, pvq_mapping_path: str | Path | None = None
    ) -> pd.DataFrame:
        """For each model and each questionnaire item, test via an exact
        (binomial) McNemar test whether steering by that model's own reply
        to this item significantly changes recall relative to zero-shot —
        using the 250 actual-positive messages, paired by message id
        between the zero-shot and steered runs (so this needs the raw
        per-message predictions, not the aggregate recall table).

        McNemar compares the discordant pairs: ``b`` = messages correct
        zero-shot but wrong after steering (a recall loss), ``c`` = wrong
        zero-shot but correct after steering (a recall gain). p-values are
        Benjamini-Hochberg corrected *within each model's own family* of
        76 items (36 MFT + 40 PVQ) — this is a per-model question, so each
        model gets its own correction rather than pooling all 456 tests.

        If ``pvq_mapping_path`` is given (a CSV with ``test_statement`` and
        ``pvq_value`` columns, Schwartz PVQ values numbered 1-10), PVQ
        items are labeled with their Schwartz value name.
        """
        pvq_map: dict[str, int] = {}
        if pvq_mapping_path is not None:
            map_df = pd.read_csv(pvq_mapping_path)
            pvq_map = dict(zip(map_df["test_statement"], map_df["pvq_value"]))

        rows = []
        for model, model_data in self._data["models"].items():
            zero_shot_by_id = {p["id"]: p["answer"] for p in model_data["zero_shot"]}
            for condition in ("mft", "pvq"):
                for item in model_data[condition]:
                    positives = [p for p in item["predictions"] if int(p["dataset_label"]) == 1]
                    a = b = c = d = 0
                    for p in positives:
                        zs_correct = str(zero_shot_by_id[p["id"]]) == "1"
                        st_correct = str(p["answer"]) == "1"
                        if zs_correct and st_correct:
                            a += 1
                        elif zs_correct and not st_correct:
                            b += 1
                        elif not zs_correct and st_correct:
                            c += 1
                        else:
                            d += 1
                    n_pos = a + b + c + d
                    delta_recall = (c - b) / n_pos if n_pos else np.nan
                    direction = "increase" if c > b else ("decrease" if b > c else "no_change")
                    pvq_value = pvq_map.get(item["test_statement"]) if condition == "pvq" else None
                    rows.append({
                        "model": model, "condition": condition, "belief_id": item["belief_id"],
                        "test_statement": item["test_statement"], "foundation": item.get("foundation"),
                        "pvq_value": pvq_value,
                        "pvq_value_label": SCHWARTZ_PVQ_VALUES.get(pvq_value),
                        "n_positive": n_pos, "n_lost": b, "n_gained": c,
                        "delta_recall": delta_recall, "direction": direction,
                        "mcnemar_p": _mcnemar_exact(b, c),
                    })
        result = pd.DataFrame(rows)
        result["mcnemar_p_fdr_bh"] = result.groupby("model")["mcnemar_p"].transform(_benjamini_hochberg)
        result["significant_fdr_05"] = result["mcnemar_p_fdr_bh"] < 0.05
        result = result.sort_values(["model", "mcnemar_p_fdr_bh"])
        self._save(result, "item_significance_per_model.csv")
        return result

    @staticmethod
    def _summarize_by_category(group: pd.DataFrame) -> pd.Series:
        sig = group[group["significant_fdr_05"]]
        return pd.Series({
            "n_items": len(group),
            "n_significant": len(sig),
            "n_significant_increase": int((sig["direction"] == "increase").sum()),
            "n_significant_decrease": int((sig["direction"] == "decrease").sum()),
            "mean_delta_recall": group["delta_recall"].mean(),
        })

    def _category_patterns(
        self, item_significance: pd.DataFrame, condition: str,
        category_column: str, filename: str,
    ) -> pd.DataFrame:
        """Shared implementation for ``pvq_value_patterns``/
        ``mft_foundation_patterns``: summarize the (per-model-significant)
        recall shift by ``category_column``, per model and pooled across
        all 6 (``model="ALL"``)."""
        rows = item_significance[item_significance["condition"] == condition].copy()
        per_model = (
            rows.groupby(["model", category_column], dropna=False)
            .apply(self._summarize_by_category, include_groups=False).reset_index()
        )
        pooled = (
            rows.groupby(category_column, dropna=False)
            .apply(self._summarize_by_category, include_groups=False).reset_index()
        )
        pooled.insert(0, "model", "ALL")
        result = pd.concat([per_model, pooled], ignore_index=True) \
            .sort_values(["model", "n_significant"], ascending=[True, False])
        self._save(result, filename)
        return result

    def pvq_value_patterns(self, item_significance: pd.DataFrame) -> pd.DataFrame:
        """Summarize how the recall shift breaks down by Schwartz PVQ
        value. Only meaningful for PVQ items — MFT items have no PVQ value
        and are excluded."""
        return self._category_patterns(
            item_significance, "pvq", "pvq_value_label", "pvq_value_patterns.csv"
        )

    def mft_foundation_patterns(self, item_significance: pd.DataFrame) -> pd.DataFrame:
        """Summarize how the recall shift breaks down by Moral Foundation.
        Only meaningful for MFT items — PVQ items have no foundation and
        are excluded."""
        return self._category_patterns(
            item_significance, "mft", "foundation", "mft_foundation_patterns.csv"
        )

    def within_model_category_rank(self, category_patterns: pd.DataFrame, category_col: str) -> dict[str, pd.DataFrame]:
        """Ranks each category's `mean_delta_recall` *within* each model
        (rank 1 = that model's weakest/most-negative category) instead of
        pooling across models. Pooling first (the `model="ALL"` row in
        `pvq_value_patterns`/`mft_foundation_patterns`) can hide a
        consistent *relative* pattern when models differ hugely in their
        overall effect level (e.g. one model's mean is strongly positive,
        another's strongly negative) — averaging their absolute deltas
        washes out which category is *that model's own* weakest lever.
        Ranking within each model first avoids that.

        Returns the per-(model, category) rank, plus a per-category
        summary (mean rank across models, and a one-sample Wilcoxon test
        of whether that mean rank differs from the chance expectation
        of (n_categories + 1) / 2)."""
        df = category_patterns[category_patterns["model"] != "ALL"]
        pivot = df.pivot(index="model", columns=category_col, values="mean_delta_recall")
        n_categories = pivot.shape[1]
        ranks = pivot.rank(axis=1)
        long = ranks.reset_index().melt(id_vars="model", var_name=category_col, value_name="rank_within_model")
        self._save(long, f"within_model_rank_{category_col}.csv")

        expected = (n_categories + 1) / 2
        rows = []
        for category, group in long.groupby(category_col):
            r = group["rank_within_model"].dropna()
            row = {category_col: category, "n_models": len(r), "mean_rank": r.mean(),
                   "expected_rank_by_chance": expected, "p": np.nan}
            if len(r) >= 3 and r.std() > 0:
                w = stats.wilcoxon(r - expected)
                row["p"] = float(w.pvalue)
            rows.append(row)
        summary = pd.DataFrame(rows).sort_values("mean_rank")
        summary["p_fdr_bh"] = _benjamini_hochberg(summary["p"])
        self._save(summary, f"within_model_rank_summary_{category_col}.csv")
        return {"ranks": long, "summary": summary}

    # ------------------------------------------------------------------
    # Model-wise: how often does steering flip the predicted label at all?
    # ------------------------------------------------------------------
    def flip_counts_by_item(self, pvq_mapping_path: str | Path | None = None) -> pd.DataFrame:
        """For each model and item, count how many of the 500 messages
        flip their predicted label between zero-shot and steered —
        regardless of ground truth or correctness. This is a raw measure
        of how much a belief shifts the model's stated opinion on hate
        speech, as distinct from ``item_significance_per_model`` (which
        only looks at whether *correctness on the positive class* changes).

        If ``pvq_mapping_path`` is given, PVQ items are labeled with their
        Schwartz value name (as in ``item_significance_per_model``).
        """
        pvq_map: dict[str, int] = {}
        if pvq_mapping_path is not None:
            map_df = pd.read_csv(pvq_mapping_path)
            pvq_map = dict(zip(map_df["test_statement"], map_df["pvq_value"]))

        rows = []
        for model, model_data in self._data["models"].items():
            zero_shot_by_id = {p["id"]: p["answer"] for p in model_data["zero_shot"]}
            for condition in ("mft", "pvq"):
                for item in model_data[condition]:
                    preds = item["predictions"]
                    n = len(preds)
                    to_hate = to_not_hate = 0
                    for p in preds:
                        zs, st = str(zero_shot_by_id[p["id"]]), str(p["answer"])
                        if zs != st:
                            if st == "1":
                                to_hate += 1
                            else:
                                to_not_hate += 1
                    n_flips = to_hate + to_not_hate
                    pvq_value = pvq_map.get(item["test_statement"]) if condition == "pvq" else None
                    rows.append({
                        "model": model, "condition": condition, "belief_id": item["belief_id"],
                        "test_statement": item["test_statement"], "foundation": item.get("foundation"),
                        "pvq_value": pvq_value,
                        "pvq_value_label": SCHWARTZ_PVQ_VALUES.get(pvq_value),
                        "n_messages": n, "n_flips": n_flips, "flip_rate": n_flips / n,
                        "n_flips_to_hate": to_hate, "n_flips_to_not_hate": to_not_hate,
                    })
        result = pd.DataFrame(rows).sort_values(["model", "n_flips"], ascending=[True, False])
        self._save(result, "flip_counts_by_item.csv")
        return result

    def item_steerability(self, flips_by_item: pd.DataFrame) -> pd.DataFrame:
        """Collapse ``flip_counts_by_item`` (one row per model x item) to
        one row per item, averaging ``flip_rate`` across the 6 models —
        the item-level analogue of ``mean_flip_rate`` in
        ``model_variance_by_instance``. Also reports the between-model
        standard deviation of flip_rate on that item, for context."""
        group_cols = ["condition", "belief_id", "test_statement", "foundation",
                      "pvq_value", "pvq_value_label"]
        agg = flips_by_item.groupby(group_cols, dropna=False)["flip_rate"].agg(
            mean_flip_rate="mean", std_flip_rate="std", n_models="count"
        ).reset_index()
        result = agg.sort_values("mean_flip_rate", ascending=False)
        self._save(result, "item_steerability.csv")
        return result

    def item_steerability_tiers(self, item_steerability: pd.DataFrame, n_tiers: int = 3) -> pd.DataFrame:
        """Bin items into ``n_tiers`` steerability tiers (default 3:
        `low_steerability`, `mid_steerability`, `high_steerability`) by
        ``mean_flip_rate`` — unlike the message-level tiers (which use a
        fixed 0 / 0.15 cutoff, since many messages never flip at all), no
        item has zero flip rate once averaged over 500 messages and 6
        models, so tiers are assigned by tertile (roughly equal-sized
        groups) *within each questionnaire* (MFT's 36 items and PVQ's 40
        items are tiered separately, since foundation/value patterns are
        only meaningful within their own questionnaire)."""
        labels = [f"tier_{i+1}_of_{n_tiers}" for i in range(n_tiers)] if n_tiers != 3 else \
            ["low_steerability", "mid_steerability", "high_steerability"]
        result = item_steerability.copy()
        result["steerability_tier"] = result.groupby("condition")["mean_flip_rate"].transform(
            lambda s: pd.qcut(s, n_tiers, labels=labels, duplicates="drop")
        )
        self._save(result, "item_steerability_tiers.csv")
        return result

    def steerability_tier_patterns(self, tiers: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Does steerability tier associate with MFT foundation / Schwartz
        PVQ value? For each questionnaire, builds a tier x category
        contingency table (counts and row/column proportions), a
        chi-square test of independence (caveat: with ~36-40 items split
        across 6-10 categories and 3 tiers, expected cell counts are small
        — read this as descriptive, not confirmatory), and a one-vs-rest
        Fisher exact test per category asking specifically "is this
        category over/under-represented in the *high*-steerability tier
        compared to every other category combined" (BH-corrected across
        categories, within each questionnaire — the more targeted, better
        powered version of the question)."""
        out: dict[str, pd.DataFrame] = {}
        for condition, category_col, filename_stub in [
            ("mft", "foundation", "foundation"), ("pvq", "pvq_value_label", "pvq_value"),
        ]:
            sub = tiers[tiers["condition"] == condition]
            contingency = pd.crosstab(sub[category_col], sub["steerability_tier"])
            contingency["total_items"] = contingency.sum(axis=1)
            for tier_col in [c for c in contingency.columns if c != "total_items"]:
                contingency[f"proportion_{tier_col}"] = contingency[tier_col] / contingency["total_items"]
            contingency = contingency.reset_index()
            self._save(contingency, f"steerability_tier_by_{filename_stub}.csv")
            out[f"{filename_stub}_contingency"] = contingency

            counts_only = pd.crosstab(sub[category_col], sub["steerability_tier"])
            chi2_row = {"condition": condition, "chi2_stat": np.nan, "p": np.nan, "dof": np.nan, "notes": ""}
            if counts_only.shape[0] >= 2 and counts_only.shape[1] >= 2:
                chi2, p, dof, expected = stats.chi2_contingency(counts_only.to_numpy())
                min_expected = expected.min()
                chi2_row.update(chi2_stat=float(chi2), p=float(p), dof=int(dof))
                if min_expected < 5:
                    chi2_row["notes"] = f"min expected cell count {min_expected:.2f} < 5; treat as descriptive"
            out[f"{filename_stub}_chi2"] = pd.DataFrame([chi2_row])
            self._save(out[f"{filename_stub}_chi2"], f"steerability_tier_association_{filename_stub}.csv")

            high_tier = "high_steerability" if "high_steerability" in counts_only.columns else counts_only.columns[-1]
            fisher_rows = []
            n_total = counts_only.sum().sum()
            n_high_total = counts_only[high_tier].sum()
            for category in counts_only.index:
                n_cat = counts_only.loc[category].sum()
                n_cat_high = counts_only.loc[category, high_tier]
                table = [[n_cat_high, n_cat - n_cat_high],
                         [n_high_total - n_cat_high, n_total - n_cat - (n_high_total - n_cat_high)]]
                odds_ratio, p = stats.fisher_exact(table, alternative="two-sided")
                fisher_rows.append({
                    "condition": condition, category_col: category,
                    "n_items": int(n_cat), "n_high_steerability": int(n_cat_high),
                    "proportion_high_steerability": n_cat_high / n_cat if n_cat else np.nan,
                    "overall_proportion_high_steerability": n_high_total / n_total,
                    "odds_ratio": float(odds_ratio), "p": float(p),
                })
            fisher_df = pd.DataFrame(fisher_rows)
            fisher_df["p_fdr_bh"] = _benjamini_hochberg(fisher_df["p"])
            fisher_df = fisher_df.sort_values("p_fdr_bh")
            out[f"{filename_stub}_fisher"] = fisher_df
            self._save(fisher_df, f"steerability_tier_onevsrest_fisher_{filename_stub}.csv")
        return out

    def flip_counts_by_model(self, flips: pd.DataFrame) -> pd.DataFrame:
        """Aggregate item-level flip counts to one row per model (and per
        model x MFT/PVQ/combined)."""
        rows = []
        for model, group in flips.groupby("model"):
            for condition, sub in [
                ("mft", group[group["condition"] == "mft"]),
                ("pvq", group[group["condition"] == "pvq"]),
                ("combined", group),
            ]:
                rows.append({
                    "model": model, "condition": condition, "n_items": len(sub),
                    "total_flips": int(sub["n_flips"].sum()),
                    "mean_flips_per_item": sub["n_flips"].mean(),
                    "mean_flip_rate": sub["flip_rate"].mean(),
                    "total_flips_to_hate": int(sub["n_flips_to_hate"].sum()),
                    "total_flips_to_not_hate": int(sub["n_flips_to_not_hate"].sum()),
                })
        result = pd.DataFrame(rows).sort_values(["condition", "total_flips"], ascending=[True, False])
        self._save(result, "flip_counts_by_model.csv")
        return result

    def flip_magnitude_by_model(self, flips: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Do models differ in how *often* their prediction flips when
        steered? Kruskal-Wallis on per-item ``flip_rate`` across the 6
        models (n=76 items, or 36/40 within mft/pvq alone), plus pairwise
        Mann-Whitney U tests — the same magnitude-comparison approach as
        ``between_model_steering_effect``, but on raw flip rate rather than
        on the recall delta (a flip doesn't have to change the item's
        correctness to count here)."""
        kruskal_rows = []
        pairwise_rows = []
        for condition, sub in [
            ("mft", flips[flips["condition"] == "mft"]),
            ("pvq", flips[flips["condition"] == "pvq"]),
            ("combined", flips),
        ]:
            groups = {model: g["flip_rate"] for model, g in sub.groupby("model")}
            res = _kruskal(groups)
            kruskal_rows.append({"condition": condition, **res})
            for model_a, model_b in itertools.combinations(sorted(groups), 2):
                mw = _mannwhitney(groups[model_a], groups[model_b])
                pairwise_rows.append({
                    "condition": condition, "model_a": model_a, "model_b": model_b, **mw
                })
        kruskal_df = pd.DataFrame(kruskal_rows)
        pairwise_df = pd.DataFrame(pairwise_rows)
        pairwise_df["p_fdr_bh"] = pairwise_df.groupby("condition")["p"].transform(_benjamini_hochberg)
        self._save(kruskal_df, "flip_magnitude_kruskal.csv")
        self._save(pairwise_df, "flip_magnitude_pairwise_mannwhitney.csv")
        return {"kruskal": kruskal_df, "pairwise": pairwise_df}

    def flip_direction_by_model(self, flips: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Do models differ in *which direction* they flip (toward vs. away
        from the hate-speech label) when steered? A chi-square test of
        independence between model identity and flip direction, on the
        pooled to-hate/to-not-hate counts (summed across items), separately
        for mft, pvq, and both combined. Cramer's V is reported as effect
        size (0 = no association, 1 = perfect association)."""
        rows = []
        for condition, sub in [
            ("mft", flips[flips["condition"] == "mft"]),
            ("pvq", flips[flips["condition"] == "pvq"]),
            ("combined", flips),
        ]:
            counts = sub.groupby("model")[["n_flips_to_hate", "n_flips_to_not_hate"]].sum()
            counts = counts[(counts.sum(axis=1)) > 0]
            row = {"condition": condition, "n_models": len(counts),
                   "chi2_stat": np.nan, "p": np.nan, "dof": np.nan, "cramers_v": np.nan,
                   "notes": ""}
            if len(counts) < 2:
                row["notes"] = "fewer than 2 models with at least 1 flip"
                rows.append(row)
                continue
            chi2, p, dof, _ = stats.chi2_contingency(counts.to_numpy())
            n = counts.to_numpy().sum()
            k = min(counts.shape) - 1
            cramers_v = float(np.sqrt((chi2 / n) / k)) if k > 0 else np.nan
            row.update(chi2_stat=float(chi2), p=float(p), dof=int(dof), cramers_v=cramers_v)
            rows.append(row)
        result = pd.DataFrame(rows)
        self._save(result, "flip_direction_chi2_by_model.csv")

        # Per-model proportion of flips that go toward "hate", for readability
        # alongside the pooled test above.
        prop_rows = []
        for condition, sub in [
            ("mft", flips[flips["condition"] == "mft"]),
            ("pvq", flips[flips["condition"] == "pvq"]),
            ("combined", flips),
        ]:
            for model, g in sub.groupby("model"):
                to_hate = int(g["n_flips_to_hate"].sum())
                to_not_hate = int(g["n_flips_to_not_hate"].sum())
                total = to_hate + to_not_hate
                prop_rows.append({
                    "condition": condition, "model": model,
                    "total_flips": total, "n_flips_to_hate": to_hate,
                    "n_flips_to_not_hate": to_not_hate,
                    "proportion_to_hate": to_hate / total if total else np.nan,
                })
        prop_df = pd.DataFrame(prop_rows).sort_values(["condition", "proportion_to_hate"], ascending=[True, False])
        self._save(prop_df, "flip_direction_proportion_by_model.csv")
        return {"chi2": result, "proportions": prop_df}

    def flip_counts_by_model_and_value(self, flips: pd.DataFrame) -> pd.DataFrame:
        """Model x value-type breakdown of flip counts: MFT foundation for
        MFT items, Schwartz PVQ value for PVQ items (requires
        ``flip_counts_by_item`` to have been run with ``pvq_mapping_path``
        set, otherwise PVQ rows fall back to a single ``None`` value)."""
        flips = flips.copy()
        flips["value_type"] = flips["foundation"].where(
            flips["condition"] == "mft", flips["pvq_value_label"]
        )
        rows = []
        for (model, condition, value_type), group in flips.groupby(
            ["model", "condition", "value_type"], dropna=False
        ):
            rows.append({
                "model": model, "condition": condition, "value_type": value_type,
                "n_items": len(group), "total_flips": int(group["n_flips"].sum()),
                "mean_flips_per_item": group["n_flips"].mean(),
                "mean_flip_rate": group["flip_rate"].mean(),
            })
        result = pd.DataFrame(rows).sort_values(
            ["model", "condition", "total_flips"], ascending=[True, True, False]
        )
        self._save(result, "flip_counts_by_model_and_value.csv")
        return result

    # ------------------------------------------------------------------
    # Instance-wise: how often is each message's prediction flipped?
    # ------------------------------------------------------------------
    def flip_counts_by_instance(self) -> pd.DataFrame:
        """For each of the 500 messages, count how many (model, item)
        steering conditions (up to 6 models x 76 items = 456) flip its
        predicted label relative to that same model's zero-shot prediction
        on it — i.e. how often, across every model and every moral-belief
        context, this specific message's classification is unstable."""
        counts: dict[int, dict] = {}
        for model, model_data in self._data["models"].items():
            zero_shot_by_id = {p["id"]: p["answer"] for p in model_data["zero_shot"]}
            for condition in ("mft", "pvq"):
                for item in model_data[condition]:
                    for p in item["predictions"]:
                        mid = p["id"]
                        rec = counts.setdefault(mid, {
                            "dataset_label": int(p["dataset_label"]),
                            "n_conditions": 0, "n_flips": 0,
                            "n_flips_to_hate": 0, "n_flips_to_not_hate": 0,
                        })
                        rec["n_conditions"] += 1
                        zs, st = str(zero_shot_by_id[mid]), str(p["answer"])
                        if zs != st:
                            rec["n_flips"] += 1
                            if st == "1":
                                rec["n_flips_to_hate"] += 1
                            else:
                                rec["n_flips_to_not_hate"] += 1
        rows = [{"id": mid, **rec, "flip_rate": rec["n_flips"] / rec["n_conditions"]}
                for mid, rec in counts.items()]
        result = pd.DataFrame(rows).sort_values("n_flips", ascending=False)
        self._save(result, "flip_counts_by_instance.csv")
        return result

    def flip_rate_tiers_by_instance(
        self, flips_by_instance: pd.DataFrame, mild_cutoff: float = 0.15
    ) -> pd.DataFrame:
        """Bin each message's ``flip_rate`` (from ``flip_counts_by_instance``,
        pooled across all 6 models x 76 items) into three tiers: ``no_flip``
        (flip_rate == 0, stable under every model/belief), ``mild_flip``
        (0 < flip_rate <= ``mild_cutoff``), and ``strong_flip`` (>
        ``mild_cutoff``). Built for comparing which messages fall in which
        tier between the free-text and verbalized steering conditions."""
        df = flips_by_instance.copy()

        def tier(rate: float) -> str:
            if rate == 0:
                return "no_flip"
            if rate <= mild_cutoff:
                return "mild_flip"
            return "strong_flip"

        df["tier"] = df["flip_rate"].apply(tier)
        result = df[["id", "dataset_label", "flip_rate", "tier"]].copy()
        self._save(result, "flip_rate_tiers_by_instance.csv")
        return result

    def model_variance_by_instance(self) -> pd.DataFrame:
        """For each message, compute *each model's own* flip rate (over its
        76 items) rather than pooling across models, then the variance of
        those 6 per-model rates. Tests whether more-steerable messages
        (high overall flip rate) are also where models diverge from each
        other the most, or whether steerability and between-model
        disagreement are independent.

        ``dispersion_ratio`` divides the observed variance by the
        theoretical maximum variance 6 values bounded in [0, 1] can have
        given that mean (``mean * (1 - mean) * 6/5``, i.e. the
        maximum-spread case where some models sit at 0 and others at 1) —
        this matters because that ceiling itself shrinks toward 0 as the
        mean approaches 0 or 1, which would otherwise make "more
        steerable => more between-model variance" a near-tautology for
        low-flip-rate messages. A dispersion ratio that still correlates
        with the mean, after this normalization, is not just an artifact
        of that ceiling."""
        counts: dict[str, dict[int, list[int]]] = {}
        dataset_label: dict[int, int] = {}
        for model, model_data in self._data["models"].items():
            zero_shot_by_id = {p["id"]: p["answer"] for p in model_data["zero_shot"]}
            model_counts: dict[int, list[int]] = {}
            for condition in ("mft", "pvq"):
                for item in model_data[condition]:
                    for p in item["predictions"]:
                        mid = p["id"]
                        dataset_label[mid] = int(p["dataset_label"])
                        rec = model_counts.setdefault(mid, [0, 0])
                        rec[1] += 1
                        if str(zero_shot_by_id[mid]) != str(p["answer"]):
                            rec[0] += 1
            counts[model] = model_counts

        rows = []
        all_ids = sorted({mid for model_counts in counts.values() for mid in model_counts})
        for mid in all_ids:
            rates = np.array([
                counts[model][mid][0] / counts[model][mid][1]
                for model in counts if mid in counts[model]
            ])
            mean_rate = float(rates.mean())
            var = float(rates.var(ddof=1)) if len(rates) > 1 else np.nan
            max_var = mean_rate * (1 - mean_rate) * (len(rates) / (len(rates) - 1)) \
                if 0 < mean_rate < 1 and len(rates) > 1 else 0.0
            rows.append({
                "id": mid, "dataset_label": dataset_label[mid], "n_models": len(rates),
                "mean_flip_rate": mean_rate, "between_model_var": var,
                "max_possible_var": max_var,
                "dispersion_ratio": var / max_var if max_var > 0 else np.nan,
            })
        result = pd.DataFrame(rows).sort_values("mean_flip_rate", ascending=False)
        self._save(result, "model_variance_by_instance.csv")
        return result

    def steerability_vs_model_divergence(self, variance_by_instance: pd.DataFrame) -> pd.DataFrame:
        """Correlate a message's overall steerability (``mean_flip_rate``)
        against how much models diverge from each other on it
        (``between_model_var`` and the ceiling-normalized
        ``dispersion_ratio``), across all 500 messages. High, significant
        correlations mean between-model differences aren't spread evenly
        across all messages — they concentrate on the messages that are
        steerable at all."""
        rows = []
        for col in ("between_model_var", "dispersion_ratio"):
            sub = variance_by_instance.dropna(subset=["mean_flip_rate", col])
            pear = stats.pearsonr(sub["mean_flip_rate"], sub[col])
            spear = stats.spearmanr(sub["mean_flip_rate"], sub[col])
            rows.append({
                "against": col, "n": len(sub),
                "pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue),
                "spearman_r": float(spear.statistic), "spearman_p": float(spear.pvalue),
            })
        result = pd.DataFrame(rows)
        self._save(result, "steerability_vs_model_divergence.csv")
        return result

    def instance_prediction_profile(self) -> pd.DataFrame:
        """For each message, across *every* prediction ever made on it —
        zero-shot plus all 76 belief-steered runs, for each model (77
        predictions per model, 462 pooled across all 6) — count how many
        times it was predicted class 1 ("hate") vs. class 0 ("not hate"),
        and the resulting proportion predicted hate. Unlike the flip-count
        analyses, this doesn't reference a zero-shot baseline at all: it's
        the raw stability of the predicted label across every context the
        message was ever classified under.

        One row per (model, id), plus a pooled ``model="ALL"`` row per id
        summing across all 6 models."""
        per_model_counts: dict[str, dict[int, dict]] = {}
        for model, model_data in self._data["models"].items():
            id_counts: dict[int, dict] = {}
            all_preds = list(model_data["zero_shot"])
            for condition in ("mft", "pvq"):
                for item in model_data[condition]:
                    all_preds.extend(item["predictions"])
            for p in all_preds:
                mid = p["id"]
                rec = id_counts.setdefault(mid, {"dataset_label": int(p["dataset_label"]), "n_0": 0, "n_1": 0})
                if str(p["answer"]) == "1":
                    rec["n_1"] += 1
                else:
                    rec["n_0"] += 1
            per_model_counts[model] = id_counts

        rows = []
        pooled: dict[int, dict] = {}
        for model, id_counts in per_model_counts.items():
            for mid, rec in id_counts.items():
                n_total = rec["n_0"] + rec["n_1"]
                rows.append({
                    "model": model, "id": mid, "dataset_label": rec["dataset_label"],
                    "n_0": rec["n_0"], "n_1": rec["n_1"], "n_total": n_total,
                    "proportion_hate": rec["n_1"] / n_total,
                })
                p = pooled.setdefault(mid, {"dataset_label": rec["dataset_label"], "n_0": 0, "n_1": 0})
                p["n_0"] += rec["n_0"]
                p["n_1"] += rec["n_1"]
        for mid, rec in pooled.items():
            n_total = rec["n_0"] + rec["n_1"]
            rows.append({
                "model": "ALL", "id": mid, "dataset_label": rec["dataset_label"],
                "n_0": rec["n_0"], "n_1": rec["n_1"], "n_total": n_total,
                "proportion_hate": rec["n_1"] / n_total,
            })
        result = pd.DataFrame(rows).sort_values(["model", "proportion_hate"], ascending=[True, False])
        self._save(result, "instance_prediction_profile.csv")
        return result

    def instance_prediction_bins(self, profile: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
        """Bin ``proportion_hate`` from ``instance_prediction_profile`` into
        ``n_bins`` equal-width bins over [0, 1], counted per model (plus
        pooled ``model="ALL"``), each bin split by ground-truth
        ``dataset_label`` so a bin's count can be read against how many of
        its messages are actually hate speech."""
        df = profile.copy()
        edges = np.linspace(0, 1, n_bins + 1)
        df["bin"] = pd.cut(df["proportion_hate"], bins=edges, include_lowest=True)

        rows = []
        for (model, bin_), group in df.groupby(["model", "bin"], observed=True):
            rows.append({
                "model": model, "bin": str(bin_),
                "bin_left": bin_.left, "bin_right": bin_.right,
                "n_instances": len(group),
                "n_actual_hate": int((group["dataset_label"] == 1).sum()),
                "n_actual_not_hate": int((group["dataset_label"] == 0).sum()),
            })
        result = pd.DataFrame(rows).sort_values(["model", "bin_left"])
        self._save(result, "instance_prediction_bins.csv")
        return result

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def _save(self, df: pd.DataFrame, filename: str) -> None:
        out_folder = self.output_dir / self.output_subdir
        out_folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_folder / filename, index=False)

    def run_all(self, pvq_mapping_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
        table = self.build_recall_table()
        self._save(table, "recall_by_model_condition_item.csv")
        zero_shot_agreement = self.pairwise_model_agreement_zero_shot()
        steered_agreement = self.pairwise_model_agreement_steered()
        agreement_summary = self.model_agreement_summary(zero_shot_agreement, steered_agreement)
        per_model = self.steering_effect_per_model(table)
        by_item = self.item_level_effect(table)
        between = self.between_model_steering_effect(table)
        item_significance = self.item_significance_per_model(pvq_mapping_path)
        pvq_patterns = self.pvq_value_patterns(item_significance)
        foundation_patterns = self.mft_foundation_patterns(item_significance)
        pvq_rank = self.within_model_category_rank(pvq_patterns, "pvq_value_label")
        foundation_rank = self.within_model_category_rank(foundation_patterns, "foundation")

        flips_by_item = self.flip_counts_by_item(pvq_mapping_path)
        item_steerability = self.item_steerability(flips_by_item)
        item_tiers = self.item_steerability_tiers(item_steerability)
        tier_patterns = self.steerability_tier_patterns(item_tiers)
        flips_by_model = self.flip_counts_by_model(flips_by_item)
        flips_by_model_value = self.flip_counts_by_model_and_value(flips_by_item)
        flips_by_instance = self.flip_counts_by_instance()
        flip_rate_tiers = self.flip_rate_tiers_by_instance(flips_by_instance)
        model_variance = self.model_variance_by_instance()
        steerability_divergence = self.steerability_vs_model_divergence(model_variance)
        flip_magnitude = self.flip_magnitude_by_model(flips_by_item)
        flip_direction = self.flip_direction_by_model(flips_by_item)

        prediction_profile = self.instance_prediction_profile()
        prediction_bins = self.instance_prediction_bins(prediction_profile)

        return {
            "recall_table": table,
            "zero_shot_agreement": zero_shot_agreement, "steered_agreement": steered_agreement,
            "agreement_summary": agreement_summary,
            "per_model": per_model, "by_item": by_item,
            "item_significance": item_significance, "pvq_patterns": pvq_patterns,
            "foundation_patterns": foundation_patterns,
            "pvq_rank": pvq_rank, "foundation_rank": foundation_rank,
            "flips_by_item": flips_by_item, "flips_by_model": flips_by_model,
            "item_steerability": item_steerability, "item_tiers": item_tiers,
            "tier_patterns": tier_patterns,
            "flips_by_model_value": flips_by_model_value, "flips_by_instance": flips_by_instance,
            "flip_rate_tiers": flip_rate_tiers,
            "model_variance_by_instance": model_variance,
            "steerability_vs_model_divergence": steerability_divergence,
            "flip_magnitude_kruskal": flip_magnitude["kruskal"],
            "flip_magnitude_pairwise": flip_magnitude["pairwise"],
            "flip_direction_chi2": flip_direction["chi2"],
            "flip_direction_proportions": flip_direction["proportions"],
            "prediction_profile": prediction_profile, "prediction_bins": prediction_bins,
            **between,
        }
