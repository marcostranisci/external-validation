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

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def _save(self, df: pd.DataFrame, filename: str) -> None:
        out_folder = self.output_dir / "hs_detection"
        out_folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_folder / filename, index=False)

    def run_all(self, pvq_mapping_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
        table = self.build_recall_table()
        self._save(table, "recall_by_model_condition_item.csv")
        per_model = self.steering_effect_per_model(table)
        by_item = self.item_level_effect(table)
        between = self.between_model_steering_effect(table)
        item_significance = self.item_significance_per_model(pvq_mapping_path)
        pvq_patterns = self.pvq_value_patterns(item_significance)
        foundation_patterns = self.mft_foundation_patterns(item_significance)

        flips_by_item = self.flip_counts_by_item(pvq_mapping_path)
        flips_by_model = self.flip_counts_by_model(flips_by_item)
        flips_by_model_value = self.flip_counts_by_model_and_value(flips_by_item)
        flips_by_instance = self.flip_counts_by_instance()

        return {
            "recall_table": table, "per_model": per_model, "by_item": by_item,
            "item_significance": item_significance, "pvq_patterns": pvq_patterns,
            "foundation_patterns": foundation_patterns,
            "flips_by_item": flips_by_item, "flips_by_model": flips_by_model,
            "flips_by_model_value": flips_by_model_value, "flips_by_instance": flips_by_instance,
            **between,
        }
