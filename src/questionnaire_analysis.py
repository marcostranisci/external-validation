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


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR-adjusted p-values, leaving NaN entries as NaN
    and correcting only across the valid p-values (one hypothesis family)."""
    valid = pvalues.dropna()
    adjusted = pd.Series(np.nan, index=pvalues.index)
    if len(valid) == 0:
        return adjusted
    adjusted.loc[valid.index] = stats.false_discovery_control(valid.to_numpy(), method="bh")
    return adjusted


def _mannwhitney(a: pd.Series, b: pd.Series) -> dict:
    """Two-sample Mann-Whitney U test between two independent (unpaired)
    groups, with a rank-biserial correlation as effect size.

    Unlike a chi-square test on binned data, this works directly on
    continuous values (no binning, no arbitrary bin-count choice) and
    respects the ordering of ordinal/Likert data rather than treating
    ratings as unordered categories.
    """
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    result = {"n_a": len(x), "n_b": len(y), "u_stat": np.nan, "p": np.nan,
              "rank_biserial_r": np.nan, "notes": ""}
    if len(x) < 3 or len(y) < 3:
        result["notes"] = "fewer than 3 valid observations in at least one group"
        return result
    try:
        res = stats.mannwhitneyu(x, y, alternative="two-sided")
    except ValueError as exc:
        result["notes"] = str(exc)
        return result
    u = float(res.statistic)
    rank_biserial = 1 - (2 * u) / (len(x) * len(y))
    result.update(u_stat=u, p=float(res.pvalue), rank_biserial_r=float(rank_biserial))
    return result


def _kruskal(groups: dict[str, pd.Series]) -> dict:
    """Kruskal-Wallis H test across k independent groups, with an
    eta-squared effect size. Like Mann-Whitney, works on continuous or
    ordinal values directly without binning."""
    clean = {k: pd.to_numeric(v, errors="coerce").dropna().to_numpy() for k, v in groups.items()}
    clean = {k: v for k, v in clean.items() if len(v) >= 3}
    n = sum(len(v) for v in clean.values())
    result = {"n_groups": len(clean), "n": n, "h_stat": np.nan, "p": np.nan,
              "dof": np.nan, "eta_squared": np.nan, "notes": ""}
    if len(clean) < 2:
        result["notes"] = "fewer than 2 groups with at least 3 valid observations"
        return result
    try:
        h, p = stats.kruskal(*clean.values())
    except ValueError as exc:
        result["notes"] = str(exc)
        return result
    k = len(clean)
    eta_sq = (h - k + 1) / (n - k) if n > k else np.nan
    result.update(h_stat=float(h), p=float(p), dof=k - 1, eta_squared=float(eta_sq))
    return result


@dataclass
class QuestionnaireAnalyzer:
    """Loads, augments, and statistically analyzes questionnaire survey files."""

    surveys_dir: str | Path = "surveys"
    output_dir: str | Path = "data_analysis"
    warnings: list[str] = field(default_factory=list)

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
    def _chi2_independence(a: pd.Series, b: pd.Series) -> dict:
        """Chi-square test on integer-rounded values. Intended for raw
        Likert-scale ratings only; use ``pairwise_model_mannwhitney``/
        ``kruskal_wallis`` for continuous values instead."""
        mask = a.notna() & b.notna()
        a, b = a[mask], b[mask]
        result = {"n": int(mask.sum()), "chi2_stat": np.nan, "chi2_p": np.nan,
                  "chi2_dof": np.nan, "notes": ""}
        if mask.sum() < 3:
            result["notes"] = "fewer than 3 valid paired observations"
            return result
        table = pd.crosstab(a.round(), b.round())
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
        filename: str | None = None,
    ) -> pd.DataFrame:
        """Independence + correlation tests between ``column_a`` and ``column_b``,
        run separately for each model. Intended for raw Likert-scale columns
        (the chi-square step rounds values to the nearest integer).

        ``pearson_p_fdr_bh`` / ``spearman_p_fdr_bh`` Benjamini-Hochberg
        correct across the models in this one call (one hypothesis family
        per folder)."""
        rows = []
        for model, df in dfs.items():
            corr = self._correlations(df[column_a], df[column_b])
            chi2 = self._chi2_independence(df[column_a], df[column_b])
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
        result["pearson_p_fdr_bh"] = _benjamini_hochberg(result["pearson_p"])
        result["spearman_p_fdr_bh"] = _benjamini_hochberg(result["spearman_p"])
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
    ) -> pd.DataFrame:
        """Chi-square test of independence between model identity and the
        (integer-rounded) rating distribution of ``column``, for every pair
        of models. Intended for raw Likert-scale columns."""
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
                table = pd.crosstab(long["model"], long[column].round())
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
    ) -> pd.DataFrame:
        """Chi-square test of independence between model identity and the
        (integer-rounded) rating distribution of ``column``, across all
        models. Intended for raw Likert-scale columns."""
        long = pd.concat(
            [df[["model", column]] for df in dfs.values()], ignore_index=True
        ).dropna()
        result_row = {"column": column, "n": len(long), "chi2_stat": np.nan,
                       "chi2_p": np.nan, "chi2_dof": np.nan, "notes": ""}
        if long["model"].nunique() < 2:
            result_row["notes"] = "fewer than 2 models with data"
        else:
            table = pd.crosstab(long["model"], long[column].round())
            if table.shape[0] < 2 or table.shape[1] < 2:
                result_row["notes"] = "fewer than 2 distinct categories"
            else:
                chi2, p, dof, _ = stats.chi2_contingency(table)
                result_row.update(chi2_stat=float(chi2), chi2_p=float(p), chi2_dof=int(dof))
        result = pd.DataFrame([result_row])
        self._save(result, folder_name, f"between_model_independence_{column}.csv")
        return result

    def pairwise_model_mannwhitney(
        self, dfs: dict[str, pd.DataFrame], folder_name: str, column: str
    ) -> pd.DataFrame:
        """Two-sample Mann-Whitney U test of ``column`` between every pair of
        models. Distribution-free alternative to the binned chi-square test:
        works directly on continuous/ordinal values, no bin-count choice."""
        rows = []
        for model_a, model_b in itertools.combinations(sorted(dfs), 2):
            res = _mannwhitney(dfs[model_a][column], dfs[model_b][column])
            rows.append({"model_a": model_a, "model_b": model_b, "column": column, **res})
        result = pd.DataFrame(rows)
        self._save(result, folder_name, f"pairwise_model_mannwhitney_{column}.csv")
        return result

    def kruskal_wallis(
        self, dfs: dict[str, pd.DataFrame], folder_name: str, column: str
    ) -> pd.DataFrame:
        """Kruskal-Wallis H test of ``column`` across all models at once.
        Distribution-free alternative to the pooled chi-square test."""
        res = _kruskal({model: df[column] for model, df in dfs.items()})
        result = pd.DataFrame([{"column": column, **res}])
        self._save(result, folder_name, f"kruskal_wallis_{column}.csv")
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
        """Run the full pipeline (feature engineering + all tests) for one
        folder, on the raw (Likert-scale) ``opinion``/``external_opinion``
        columns. ``opinion_normalized``/``external_opinion_normalized`` are
        still computed and saved in the processed files (see
        ``load_and_process_file``), but are not used for statistical
        testing: the raw scale is common and externally fixed across models
        (unlike a subjective per-rater scale), so a model's absolute rating
        level is itself a meaningful result rather than noise to normalize
        away."""
        dfs = self.process_folder(folder_name)

        self.opinion_vs_external_per_model(dfs, folder_name)
        self.between_model_correlations(dfs, folder_name, "opinion")
        self.between_model_correlations(dfs, folder_name, "external_opinion")

        # chi-square treats ratings as unordered categories; kept for
        # continuity, but see the rank-based tests below for the more
        # appropriate treatment of ordinal Likert data.
        self.pairwise_model_independence(dfs, folder_name, "opinion")
        self.pairwise_model_independence(dfs, folder_name, "external_opinion")
        self.between_model_independence(dfs, folder_name, "opinion")
        self.between_model_independence(dfs, folder_name, "external_opinion")

        # rank-based tests: respect the ordering of ordinal Likert ratings
        # (unlike chi-square) and need no discretization.
        self.pairwise_model_mannwhitney(dfs, folder_name, "opinion")
        self.pairwise_model_mannwhitney(dfs, folder_name, "external_opinion")
        self.kruskal_wallis(dfs, folder_name, "opinion")
        self.kruskal_wallis(dfs, folder_name, "external_opinion")

        self.disagreement_vs_delta(dfs, folder_name)
        return dfs

    def compare_between_model_correlations(
        self, folder_a: str, folder_b: str, columns: Iterable[str] = ("opinion", "external_opinion"),
    ) -> pd.DataFrame:
        """Compare how strongly models agree with each other on ``columns``
        between two questionnaires (e.g. is between-model correlation on
        `mft` systematically weaker than on `pvq`?), reading each folder's
        already-saved ``between_model_correlations_<column>.csv``.

        Since Pearson r is not itself normally distributed, the two
        folders' per-pair r values are compared via a Fisher z-transform
        (``arctanh``) with a two-sample Mann-Whitney U test (distribution-
        free) and a Welch t-test (parametric, assumes the transformed
        values are approximately normal) — reported side by side."""
        rows = []
        for column in columns:
            a = pd.read_csv(self.output_dir / folder_a / f"between_model_correlations_{column}.csv")
            b = pd.read_csv(self.output_dir / folder_b / f"between_model_correlations_{column}.csv")
            a = a.dropna(subset=["pearson_r"])
            b = b.dropna(subset=["pearson_r"])
            row = {
                "column": column,
                f"n_{folder_a}": len(a), f"n_{folder_b}": len(b),
                f"median_r_{folder_a}": a["pearson_r"].median(), f"median_r_{folder_b}": b["pearson_r"].median(),
                f"n_significant_{folder_a}": int((a["pearson_p"] < 0.05).sum()),
                f"n_significant_{folder_b}": int((b["pearson_p"] < 0.05).sum()),
                "mannwhitney_u": np.nan, "mannwhitney_p": np.nan,
                "welch_t": np.nan, "welch_p": np.nan, "notes": "",
            }
            if len(a) < 3 or len(b) < 3:
                row["notes"] = "fewer than 3 valid pairs in at least one folder"
            else:
                za = np.arctanh(a["pearson_r"].clip(-0.999, 0.999))
                zb = np.arctanh(b["pearson_r"].clip(-0.999, 0.999))
                u, p_u = stats.mannwhitneyu(za, zb, alternative="two-sided")
                t, p_t = stats.ttest_ind(za, zb, equal_var=False)
                row.update(mannwhitney_u=float(u), mannwhitney_p=float(p_u),
                           welch_t=float(t), welch_p=float(p_t))
            rows.append(row)
        result = pd.DataFrame(rows)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(self.output_dir / f"between_model_correlations_{folder_a}_vs_{folder_b}.csv", index=False)
        return result

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
        self, source: str | Path | pd.DataFrame, prefix: str
    ) -> tuple[pd.DataFrame, list[int], dict[int, str], dict[int, str]]:
        if isinstance(source, pd.DataFrame):
            df, label = source, "<dataframe>"
        else:
            df, label = pd.read_csv(source), Path(source).name
        items, own_cols, eval_cols, missing = self._item_columns(df, prefix)
        if missing:
            self._warn(
                f"[{label}] item number(s) {missing} have only an "
                f"'{prefix}_NN' or only an '{prefix}02_NN' column; skipped."
            )
        if not items:
            raise ValueError(
                f"{label}: no '{prefix}_NN' / '{prefix}02_NN' column pairs found"
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

    def infer_model_mapping(
        self, source: str | Path | pd.DataFrame, prefix: str,
        model_glob: str, group_column: str = "QUESTNNR",
    ) -> dict[str, str]:
        """Identify which model each ``group_column`` value (e.g. a
        ``QUESTNNR`` block) of annotators evaluated, by matching the sorted
        multiset of ratings they gave per item against the disaggregated
        ratings in each model survey file (order-independent, since
        annotator order need not match between the two file formats).

        Returns ``{group_value: model_file_stem}`` for groups with exactly
        one matching file; a group with zero or multiple matches is left
        out and a warning is logged.
        """
        df, items, own_cols, eval_cols = self._load(source, prefix)
        if group_column not in df.columns:
            raise ValueError(f"no '{group_column}' column found")
        ordered_eval_cols = [eval_cols[i] for i in items]

        def signature(rows: pd.DataFrame, cols: list[str]) -> tuple:
            return tuple(tuple(sorted(rows[c].tolist())) for c in cols)

        group_sigs = {g: signature(rows, ordered_eval_cols) for g, rows in df.groupby(group_column)}

        model_sigs: dict[str, tuple] = {}
        for model_path in sorted(Path().glob(model_glob)):
            model_df = pd.read_csv(model_path)
            disagg_col = _find_disaggregated_column(model_df.columns)
            if disagg_col is None:
                continue
            parsed = model_df[disagg_col].apply(_parse_rating_list)
            if parsed.isna().any():
                continue
            model_sigs[model_path.stem] = tuple(tuple(sorted(v)) for v in parsed)

        mapping = {}
        for group_value, sig in group_sigs.items():
            matches = [name for name, model_sig in model_sigs.items() if model_sig == sig]
            if len(matches) == 1:
                mapping[group_value] = matches[0]
            else:
                self._warn(
                    f"'{group_column}' value '{group_value}' matched "
                    f"{len(matches)} model file(s) under '{model_glob}'; left unmapped."
                )
        return mapping

    def demographic_agreement(
        self, source: str | Path | pd.DataFrame, prefix: str,
        demographic_columns: list[str], group_column: str = "QUESTNNR",
        model_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """For each ``group_column`` value (the annotators who evaluated one
        model), test whether annotators sharing a demographic attribute
        agree with each other more than annotators who don't, on their
        evaluations of that model's replies (the ``<prefix>_NN`` columns).

        Agreement between two annotators is measured two ways across their
        shared evaluation items: Pearson correlation (higher = more similar
        *pattern* across items) and mean absolute difference, negated so
        higher also means more agreement here (``neg_mad``; more similar
        absolute *level*). Pairs are split into "same" vs "different" on
        each demographic column and compared with a two-sample Mann-Whitney
        U test. Per model this is underpowered (a 9-annotator group has 36
        pairs total, split further into same/different), so an additional
        ``model="ALL (pooled)"`` row pools every model's pairs together for
        a properly-powered version of the same test."""
        df, items, own_cols, eval_cols = self._load(source, prefix)
        if group_column not in df.columns:
            raise ValueError(f"no '{group_column}' column found")
        ordered_eval_cols = [eval_cols[i] for i in items]
        model_mapping = model_mapping or {}

        def test_rows(pairs: pd.DataFrame, same: list[bool], label: dict) -> list[dict]:
            out = []
            for metric in ("pearson_r", "neg_mad"):
                same_vals = pairs.loc[same, metric].dropna()
                diff_vals = pairs.loc[[not s for s in same], metric].dropna()
                res = _mannwhitney(
                    pd.Series(same_vals.to_numpy()), pd.Series(diff_vals.to_numpy())
                )
                out.append({
                    **label, "metric": metric,
                    "n_same_pairs": len(same_vals), "n_diff_pairs": len(diff_vals),
                    "median_same": float(same_vals.median()) if len(same_vals) else np.nan,
                    "median_diff": float(diff_vals.median()) if len(diff_vals) else np.nan,
                    "u_stat": res["u_stat"], "p": res["p"],
                    "rank_biserial_r": res["rank_biserial_r"],
                    "notes": res["notes"],
                })
            return out

        rows = []
        pooled_pairs = {col: [] for col in demographic_columns}
        pooled_same = {col: [] for col in demographic_columns}
        for group_value, group in df.groupby(group_column):
            model_name = model_mapping.get(group_value, group_value)
            idx = group.index.tolist()
            matrix = group[ordered_eval_cols].to_numpy(dtype=float)
            pair_records = []
            for a, b in itertools.combinations(range(len(idx)), 2):
                x, y = matrix[a], matrix[b]
                r = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan
                pair_records.append({
                    "row_a": idx[a], "row_b": idx[b],
                    "pearson_r": r, "neg_mad": -float(np.mean(np.abs(x - y))),
                })
            pairs = pd.DataFrame(pair_records)

            for demo_col in demographic_columns:
                if demo_col not in group.columns:
                    self._warn(f"demographic column '{demo_col}' not found; skipped.")
                    continue
                values = group[demo_col]
                same = [values.loc[r["row_a"]] == values.loc[r["row_b"]] for r in pair_records]
                label = {group_column: group_value, "model": model_name, "demographic": demo_col}
                rows.extend(test_rows(pairs, same, label))
                pooled_pairs[demo_col].append(pairs)
                pooled_same[demo_col].extend(same)

        for demo_col in demographic_columns:
            if not pooled_pairs[demo_col]:
                continue
            all_pairs = pd.concat(pooled_pairs[demo_col], ignore_index=True)
            label = {group_column: "ALL", "model": "ALL (pooled)", "demographic": demo_col}
            rows.extend(test_rows(all_pairs, pooled_same[demo_col], label))

        result = pd.DataFrame(rows)
        self._save(result, f"{prefix.lower()}_demographic_agreement.csv")
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
