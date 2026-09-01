"""Reusable analysis pipeline for the moral-questionnaire survey results (MFT, PVQ, ...).

For every model's response file in a survey folder (e.g. ``surveys/mft`` or
``surveys/pvq``) this module:

1. parses the per-annotator ratings stored in the ``*_disaggregated`` column,
2. derives ``annotator_disagreement`` (mean absolute deviation of the
   annotators around their mean) and ``external_opinion`` (the annotators'
   mean rating),
3. z-score normalizes the ``opinion`` and ``external_opinion`` columns,
4. runs correlation / independence tests between ``opinion`` and
   ``external_opinion`` within each model, between models, and between
   annotator disagreement and the opinion/external-opinion delta.

Usage
-----
    from src.questionnaire_analysis import QuestionnaireAnalyzer

    analyzer = QuestionnaireAnalyzer(surveys_dir="surveys", output_dir="data_analysis")
    analyzer.run_all(["mft", "pvq"])
"""

from __future__ import annotations

import ast
import itertools
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _find_disaggregated_column(columns: Iterable[str]) -> str | None:
    """Return the name of the column holding per-annotator ratings, if any."""
    for col in columns:
        if "disaggregated" in col:
            return col
    return None


def _parse_rating_list(value) -> list[float] | None:
    """Parse a stringified list of annotator ratings (e.g. ``"[5.0, 4.0]"``)."""
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if pd.isna(value):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) == 0:
        return None
    return [float(v) for v in parsed]


def mean_absolute_deviation(values: list[float]) -> float:
    """Mean absolute deviation of ``values`` around their own mean.

    Used as the annotator-disagreement metric: the average distance between
    each annotator's rating and the group mean rating for that item.
    """
    arr = np.asarray(values, dtype=float)
    return float(np.mean(np.abs(arr - arr.mean())))


def zscore(series: pd.Series) -> pd.Series:
    """Z-score normalize a numeric series, leaving NaNs as NaN.

    A zero-variance column normalizes to all zeros instead of NaN/Inf.
    """
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)
    if not std or np.isnan(std):
        return pd.Series(np.where(values.isna(), np.nan, 0.0), index=series.index)
    return (values - mean) / std


def _paired_valid(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    mask = a.notna() & b.notna()
    return a[mask].to_numpy(dtype=float), b[mask].to_numpy(dtype=float)


def _correlations(a: pd.Series, b: pd.Series) -> dict:
    """Pearson + Spearman correlation between two paired series, with guards
    for too few observations or zero variance."""
    x, y = _paired_valid(a, b)
    n = len(x)
    result = {"n": n, "pearson_r": np.nan, "pearson_p": np.nan,
              "spearman_r": np.nan, "spearman_p": np.nan, "notes": ""}
    if n < 3:
        result["notes"] = "fewer than 3 valid paired observations"
        return result
    if np.std(x) == 0 or np.std(y) == 0:
        result["notes"] = "zero variance in at least one variable"
        return result
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    result.update(
        pearson_r=float(pear.statistic), pearson_p=float(pear.pvalue),
        spearman_r=float(spear.statistic), spearman_p=float(spear.pvalue),
    )
    return result


@dataclass
class QuestionnaireAnalyzer:
    """Loads, augments, and statistically analyzes questionnaire survey files."""

    surveys_dir: str | Path = "surveys"
    output_dir: str | Path = "data_analysis"
    warnings: list[str] = field(default_factory=list)
    #: number of equal-frequency quantile buckets used to discretize
    #: continuous z-scored (``*_normalized``) columns for chi-square tests
    normalized_bins: int = 4

    def __post_init__(self) -> None:
        self.surveys_dir = Path(self.surveys_dir)
        self.output_dir = Path(self.output_dir)

    # ------------------------------------------------------------------
    # Loading & per-file feature engineering
    # ------------------------------------------------------------------
    def _warn(self, message: str) -> None:
        logger.warning(message)
        self.warnings.append(message)

    def load_and_process_file(self, filepath: Path) -> pd.DataFrame:
        """Load one survey CSV and add annotator_disagreement / external_opinion /
        opinion_normalized / external_opinion_normalized columns."""
        df = pd.read_csv(filepath)
        model = filepath.stem

        if "opinion" not in df.columns:
            raise ValueError(f"{filepath} has no 'opinion' column")

        opinion_numeric = pd.to_numeric(df["opinion"], errors="coerce")
        n_bad = int(opinion_numeric.isna().sum() - df["opinion"].isna().sum())
        if n_bad > 0:
            self._warn(
                f"[{model}] 'opinion' had {n_bad} non-numeric value(s); coerced to NaN."
            )
        df["opinion"] = opinion_numeric

        disagg_col = _find_disaggregated_column(df.columns)
        if disagg_col is None:
            self._warn(
                f"[{model}] no '*_disaggregated' column found; "
                "'annotator_disagreement' and 'external_opinion' set to NaN."
            )
            df["annotator_disagreement"] = np.nan
            df["external_opinion"] = np.nan
        else:
            parsed = df[disagg_col].apply(_parse_rating_list)
            n_unparsed = int(parsed.isna().sum())
            if n_unparsed > 0:
                self._warn(
                    f"[{model}] {n_unparsed} row(s) in '{disagg_col}' could not be parsed."
                )
            df["annotator_disagreement"] = parsed.apply(
                lambda v: mean_absolute_deviation(v) if v is not None else np.nan
            )
            df["external_opinion"] = parsed.apply(
                lambda v: float(np.mean(v)) if v is not None else np.nan
            )

        df["opinion_normalized"] = zscore(df["opinion"])
        df["external_opinion_normalized"] = zscore(df["external_opinion"])
        df["model"] = model
        return df

    def process_folder(self, folder_name: str) -> dict[str, pd.DataFrame]:
        """Process every CSV in ``surveys_dir/folder_name``.

        Returns a dict mapping model name (file stem) -> processed DataFrame,
        and writes each processed DataFrame to
        ``output_dir/processed/<folder_name>/<model>.csv``.
        """
        folder = self.surveys_dir / folder_name
        out_folder = self.output_dir / "processed" / folder_name
        out_folder.mkdir(parents=True, exist_ok=True)

        results: dict[str, pd.DataFrame] = {}
        for filepath in sorted(folder.glob("*.csv")):
            df = self.load_and_process_file(filepath)
            results[filepath.stem] = df
            df.to_csv(out_folder / filepath.name, index=False)
        return results

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _paired_valid(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        return _paired_valid(a, b)

    @classmethod
    def _correlations(cls, a: pd.Series, b: pd.Series) -> dict:
        return _correlations(a, b)

    @staticmethod
    def _discretize(series: pd.Series, bins: int | None) -> pd.Series | None:
        """Bucket a numeric series into categories for a chi-square test.

        ``bins=None`` rounds to the nearest integer (fits raw Likert-scale
        ratings). ``bins=k`` cuts the series into ``k`` equal-frequency
        quantile buckets instead (fits continuous z-scored/normalized data,
        where rounding would mostly yield unique values).
        """
        if bins is None:
            return series.round()
        try:
            return pd.qcut(series, q=bins, duplicates="drop")
        except (ValueError, IndexError):
            return None

    @classmethod
    def _chi2_independence(cls, a: pd.Series, b: pd.Series, bins: int | None = None) -> dict:
        mask = a.notna() & b.notna()
        a, b = a[mask], b[mask]
        result = {"n": int(mask.sum()), "chi2_stat": np.nan, "chi2_p": np.nan,
                  "chi2_dof": np.nan, "notes": ""}
        if mask.sum() < 3:
            result["notes"] = "fewer than 3 valid paired observations"
            return result
        cat_a, cat_b = cls._discretize(a, bins), cls._discretize(b, bins)
        if cat_a is None or cat_b is None:
            result["notes"] = "could not bin values into categories"
            return result
        table = pd.crosstab(cat_a, cat_b)
        if table.shape[0] < 2 or table.shape[1] < 2:
            result["notes"] = "fewer than 2 distinct categories on one axis"
            return result
        chi2, p, dof, _ = stats.chi2_contingency(table)
        result.update(chi2_stat=float(chi2), chi2_p=float(p), chi2_dof=int(dof))
        return result

    # ------------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------------
    def opinion_vs_external_per_model(
        self,
        dfs: dict[str, pd.DataFrame],
        folder_name: str,
        column_a: str = "opinion",
        column_b: str = "external_opinion",
        bins: int | None = None,
        filename: str | None = None,
    ) -> pd.DataFrame:
        """Independence + correlation tests between ``column_a`` and ``column_b``,
        run separately for each model.

        ``bins`` is passed to the chi-square test's discretization step: leave
        it ``None`` for raw Likert-scale columns (rounded to the nearest
        integer) or set it to a bucket count for continuous/normalized
        columns (quantile-binned instead). Note that Pearson/Spearman
        correlations are invariant to z-score normalization, so running this
        on ``*_normalized`` columns changes only the chi-square result.
        """
        rows = []
        for model, df in dfs.items():
            corr = self._correlations(df[column_a], df[column_b])
            chi2 = self._chi2_independence(df[column_a], df[column_b], bins=bins)
            rows.append({
                "model": model,
                "n": corr["n"],
                "pearson_r": corr["pearson_r"], "pearson_p": corr["pearson_p"],
                "spearman_r": corr["spearman_r"], "spearman_p": corr["spearman_p"],
                "chi2_stat": chi2["chi2_stat"], "chi2_p": chi2["chi2_p"],
                "chi2_dof": chi2["chi2_dof"],
                "notes": "; ".join(n for n in (corr["notes"], chi2["notes"]) if n),
            })
        result = pd.DataFrame(rows)
        self._save(result, folder_name, filename or f"{column_a}_vs_{column_b}_per_model.csv")
        return result

    def between_model_correlations(
        self, dfs: dict[str, pd.DataFrame], folder_name: str, column: str
    ) -> pd.DataFrame:
        """Pairwise correlation tests of ``column`` between every pair of models,
        aligning rows on the shared ``test_statement``."""
        rows = []
        for model_a, model_b in itertools.combinations(sorted(dfs), 2):
            merged = pd.merge(
                dfs[model_a][["test_statement", column]],
                dfs[model_b][["test_statement", column]],
                on="test_statement", suffixes=("_a", "_b"),
            )
            corr = self._correlations(merged[f"{column}_a"], merged[f"{column}_b"])
            rows.append({
                "model_a": model_a, "model_b": model_b, "column": column,
                "n": corr["n"],
                "pearson_r": corr["pearson_r"], "pearson_p": corr["pearson_p"],
                "spearman_r": corr["spearman_r"], "spearman_p": corr["spearman_p"],
                "notes": corr["notes"],
            })
        result = pd.DataFrame(rows)
        self._save(result, folder_name, f"between_model_correlations_{column}.csv")
        return result

    def pairwise_model_independence(
        self, dfs: dict[str, pd.DataFrame], folder_name: str, column: str,
        bins: int | None = None,
    ) -> pd.DataFrame:
        """Chi-square test of independence between model identity and the
        rating distribution of ``column``, for every pair of models.

        ``bins=None`` rounds ``column`` to the nearest integer (raw
        Likert-scale columns); pass a bucket count to quantile-bin it
        instead (for continuous/normalized columns)."""
        rows = []
        for model_a, model_b in itertools.combinations(sorted(dfs), 2):
            long = pd.concat([
                dfs[model_a][[column]].assign(model=model_a),
                dfs[model_b][[column]].assign(model=model_b),
            ], ignore_index=True).dropna()
            row = {"model_a": model_a, "model_b": model_b, "column": column,
                   "n": len(long), "chi2_stat": np.nan, "chi2_p": np.nan,
                   "chi2_dof": np.nan, "notes": ""}
            if len(long) < 3:
                row["notes"] = "fewer than 3 valid observations"
            else:
                categories = self._discretize(long[column], bins)
                if categories is None:
                    row["notes"] = "could not bin values into categories"
                else:
                    table = pd.crosstab(long["model"], categories)
                    if table.shape[0] < 2 or table.shape[1] < 2:
                        row["notes"] = "fewer than 2 distinct categories"
                    else:
                        chi2, p, dof, _ = stats.chi2_contingency(table)
                        row.update(chi2_stat=float(chi2), chi2_p=float(p), chi2_dof=int(dof))
            rows.append(row)
        result = pd.DataFrame(rows)
        self._save(result, folder_name, f"pairwise_model_independence_{column}.csv")
        return result

    def between_model_independence(
        self, dfs: dict[str, pd.DataFrame], folder_name: str, column: str,
        bins: int | None = None,
    ) -> pd.DataFrame:
        """Chi-square test of independence between model identity and the
        rating distribution of ``column``, across all models.

        ``bins=None`` rounds ``column`` to the nearest integer (raw
        Likert-scale columns); pass a bucket count to quantile-bin it
        instead (for continuous/normalized columns)."""
        long = pd.concat(
            [df[["model", column]] for df in dfs.values()], ignore_index=True
        ).dropna()
        result_row = {"column": column, "n": len(long), "chi2_stat": np.nan,
                       "chi2_p": np.nan, "chi2_dof": np.nan, "notes": ""}
        if long["model"].nunique() < 2:
            result_row["notes"] = "fewer than 2 models with data"
        else:
            categories = self._discretize(long[column], bins)
            if categories is None:
                result_row["notes"] = "could not bin values into categories"
            else:
                table = pd.crosstab(long["model"], categories)
                if table.shape[0] < 2 or table.shape[1] < 2:
                    result_row["notes"] = "fewer than 2 distinct categories"
                else:
                    chi2, p, dof, _ = stats.chi2_contingency(table)
                    result_row.update(chi2_stat=float(chi2), chi2_p=float(p), chi2_dof=int(dof))
        result = pd.DataFrame([result_row])
        self._save(result, folder_name, f"between_model_independence_{column}.csv")
        return result

    def disagreement_vs_delta(
        self, dfs: dict[str, pd.DataFrame], folder_name: str
    ) -> pd.DataFrame:
        """Correlation between annotator_disagreement and the opinion vs.
        external_opinion delta (signed and absolute), per model."""
        rows = []
        for model, df in dfs.items():
            delta = df["opinion"] - df["external_opinion"]
            abs_delta = delta.abs()
            corr_signed = self._correlations(df["annotator_disagreement"], delta)
            corr_abs = self._correlations(df["annotator_disagreement"], abs_delta)
            rows.append({
                "model": model,
                "n": corr_signed["n"],
                "pearson_r_delta": corr_signed["pearson_r"],
                "pearson_p_delta": corr_signed["pearson_p"],
                "spearman_r_delta": corr_signed["spearman_r"],
                "spearman_p_delta": corr_signed["spearman_p"],
                "pearson_r_abs_delta": corr_abs["pearson_r"],
                "pearson_p_abs_delta": corr_abs["pearson_p"],
                "spearman_r_abs_delta": corr_abs["spearman_r"],
                "spearman_p_abs_delta": corr_abs["spearman_p"],
                "notes": "; ".join(n for n in (corr_signed["notes"], corr_abs["notes"]) if n),
            })
        result = pd.DataFrame(rows)
        self._save(result, folder_name, "annotator_disagreement_vs_delta.csv")
        return result

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def _save(self, df: pd.DataFrame, folder_name: str, filename: str) -> None:
        out_folder = self.output_dir / folder_name
        out_folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_folder / filename, index=False)

    def run_folder(self, folder_name: str) -> dict[str, pd.DataFrame]:
        """Run the full pipeline (feature engineering + all tests) for one folder."""
        dfs = self.process_folder(folder_name)

        # raw (Likert-scale) columns: chi-square categories from rounding
        self.opinion_vs_external_per_model(dfs, folder_name)
        self.between_model_correlations(dfs, folder_name, "opinion")
        self.between_model_correlations(dfs, folder_name, "external_opinion")
        self.pairwise_model_independence(dfs, folder_name, "opinion")
        self.pairwise_model_independence(dfs, folder_name, "external_opinion")
        self.between_model_independence(dfs, folder_name, "opinion")
        self.between_model_independence(dfs, folder_name, "external_opinion")

        # normalized (z-scored) columns: chi-square categories from quantile bins.
        # Pearson/Spearman are invariant to z-scoring, so only chi-square
        # results are new here; the per-model file's correlation columns are
        # identical to the raw version's and kept only for a self-contained report.
        bins = self.normalized_bins
        self.opinion_vs_external_per_model(
            dfs, folder_name,
            column_a="opinion_normalized", column_b="external_opinion_normalized",
            bins=bins,
            filename="opinion_vs_external_opinion_per_model_normalized.csv",
        )
        self.pairwise_model_independence(dfs, folder_name, "opinion_normalized", bins=bins)
        self.pairwise_model_independence(dfs, folder_name, "external_opinion_normalized", bins=bins)
        self.between_model_independence(dfs, folder_name, "opinion_normalized", bins=bins)
        self.between_model_independence(dfs, folder_name, "external_opinion_normalized", bins=bins)

        self.disagreement_vs_delta(dfs, folder_name)
        return dfs

    def run_all(self, folder_names: Iterable[str] = ("mft", "pvq")) -> None:
        """Run the pipeline for every folder and write a consolidated warnings log."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for folder_name in folder_names:
            self.run_folder(folder_name)

        with open(self.output_dir / "data_quality_warnings.log", "w") as fh:
            if self.warnings:
                fh.write("\n".join(self.warnings) + "\n")
            else:
                fh.write("No data quality issues detected.\n")


@dataclass
class AnnotatorSurveyAnalyzer:
    """Analyzes per-annotator merged survey exports (one row per annotator).

    Each file is expected to hold, for the same set of items, two parallel
    blocks of columns:

    - ``<prefix>_<NN>`` — the annotator's own evaluation of a model's reply
      to item NN (e.g. ``MF_01`` .. ``MF_36``).
    - ``<prefix>02_<NN>`` — the annotator's own reply to that same
      questionnaire item (e.g. ``MF02_01`` .. ``MF02_36``).

    This class checks whether an annotator's personal stance on an item
    relates to how they evaluate a model's reply to it: per annotator
    (row-wise, across items), per item (column-wise, across annotators),
    and overall (every annotator-item pair pooled together).
    """

    output_dir: str | Path = "data_analysis"
    id_column: str = "Participant id"
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)

    def _warn(self, message: str) -> None:
        logger.warning(message)
        self.warnings.append(message)

    @staticmethod
    def _item_columns(
        df: pd.DataFrame, prefix: str
    ) -> tuple[list[int], dict[int, str], dict[int, str], list[int]]:
        """Match ``<prefix>_<NN>`` (evaluation) and ``<prefix>02_<NN>`` (own
        reply) columns and return the item numbers present in both."""
        own_pattern = re.compile(rf"^{re.escape(prefix)}02_(\d+)$")
        eval_pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
        own_cols: dict[int, str] = {}
        eval_cols: dict[int, str] = {}
        for col in df.columns:
            m = own_pattern.match(col)
            if m:
                own_cols[int(m.group(1))] = col
                continue
            m = eval_pattern.match(col)
            if m:
                eval_cols[int(m.group(1))] = col
        items = sorted(set(own_cols) & set(eval_cols))
        missing = sorted(set(own_cols) ^ set(eval_cols))
        return items, own_cols, eval_cols, missing

    def _load(
        self, filepath: str | Path, prefix: str
    ) -> tuple[pd.DataFrame, list[int], dict[int, str], dict[int, str]]:
        filepath = Path(filepath)
        df = pd.read_csv(filepath)
        items, own_cols, eval_cols, missing = self._item_columns(df, prefix)
        if missing:
            self._warn(
                f"[{filepath.name}] item number(s) {missing} have only an "
                f"'{prefix}_NN' or only an '{prefix}02_NN' column; skipped."
            )
        if not items:
            raise ValueError(
                f"{filepath}: no '{prefix}_NN' / '{prefix}02_NN' column pairs found"
            )
        return df, items, own_cols, eval_cols

    def per_annotator_correlation(self, filepath: str | Path, prefix: str) -> pd.DataFrame:
        """For each annotator (row), correlate their own item-by-item
        questionnaire replies against their item-by-item evaluations of
        model replies, across the shared items."""
        df, items, own_cols, eval_cols = self._load(filepath, prefix)
        id_col = self.id_column if self.id_column in df.columns else None
        rows = []
        for idx, row in df.iterrows():
            own = pd.Series([row[own_cols[i]] for i in items])
            evaluation = pd.Series([row[eval_cols[i]] for i in items])
            corr = _correlations(own, evaluation)
            rows.append({
                "annotator": row[id_col] if id_col else idx,
                "n_items": corr["n"],
                "pearson_r": corr["pearson_r"], "pearson_p": corr["pearson_p"],
                "spearman_r": corr["spearman_r"], "spearman_p": corr["spearman_p"],
                "notes": corr["notes"],
            })
        result = pd.DataFrame(rows)
        self._save(result, f"{prefix.lower()}_annotator_correlations.csv")
        return result

    def per_item_correlation(self, filepath: str | Path, prefix: str) -> pd.DataFrame:
        """For each item, correlate annotators' own replies against their
        evaluations of the model's reply to that item, across annotators."""
        df, items, own_cols, eval_cols = self._load(filepath, prefix)
        rows = []
        for i in items:
            corr = _correlations(df[own_cols[i]], df[eval_cols[i]])
            rows.append({
                "item": f"{prefix}_{i:02d}",
                "n_annotators": corr["n"],
                "pearson_r": corr["pearson_r"], "pearson_p": corr["pearson_p"],
                "spearman_r": corr["spearman_r"], "spearman_p": corr["spearman_p"],
                "notes": corr["notes"],
            })
        result = pd.DataFrame(rows)
        self._save(result, f"{prefix.lower()}_item_correlations.csv")
        return result

    def overall_correlation(self, filepath: str | Path, prefix: str) -> pd.DataFrame:
        """Pool every (annotator, item) pair together and correlate own
        reply vs. model-reply evaluation overall, ignoring annotator/item
        identity."""
        df, items, own_cols, eval_cols = self._load(filepath, prefix)
        own_long = pd.concat(
            [df[own_cols[i]].rename("own") for i in items], ignore_index=True
        )
        eval_long = pd.concat(
            [df[eval_cols[i]].rename("eval") for i in items], ignore_index=True
        )
        corr = _correlations(own_long, eval_long)
        result = pd.DataFrame([{
            "n": corr["n"],
            "pearson_r": corr["pearson_r"], "pearson_p": corr["pearson_p"],
            "spearman_r": corr["spearman_r"], "spearman_p": corr["spearman_p"],
            "notes": corr["notes"],
        }])
        self._save(result, f"{prefix.lower()}_overall_correlation.csv")
        return result

    def _save(self, df: pd.DataFrame, filename: str) -> None:
        out_folder = self.output_dir / "annotators"
        out_folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_folder / filename, index=False)

    def run(self, filepath: str | Path, prefix: str) -> dict[str, pd.DataFrame]:
        """Run all three correlation analyses for one merged annotator file."""
        return {
            "per_annotator": self.per_annotator_correlation(filepath, prefix),
            "per_item": self.per_item_correlation(filepath, prefix),
            "overall": self.overall_correlation(filepath, prefix),
        }

    def run_all(self, files: dict[str, str | Path]) -> None:
        """Run for multiple ``{prefix: filepath}`` pairs, e.g.
        ``{"MF": "surveys/mf_merged.csv", "PV": "surveys/pv_merged.csv"}``,
        and write a consolidated warnings log."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for prefix, filepath in files.items():
            self.run(filepath, prefix)

        out_folder = self.output_dir / "annotators"
        out_folder.mkdir(parents=True, exist_ok=True)
        with open(out_folder / "data_quality_warnings.log", "w") as fh:
            if self.warnings:
                fh.write("\n".join(self.warnings) + "\n")
            else:
                fh.write("No data quality issues detected.\n")
