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
        mask = a.notna() & b.notna()
        return a[mask].to_numpy(dtype=float), b[mask].to_numpy(dtype=float)

    @classmethod
    def _correlations(cls, a: pd.Series, b: pd.Series) -> dict:
        x, y = cls._paired_valid(a, b)
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

    @staticmethod
    def _chi2_independence(a: pd.Series, b: pd.Series) -> dict:
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
        self, dfs: dict[str, pd.DataFrame], folder_name: str
    ) -> pd.DataFrame:
        """Independence + correlation tests between opinion and external_opinion,
        run separately for each model."""
        rows = []
        for model, df in dfs.items():
            corr = self._correlations(df["opinion"], df["external_opinion"])
            chi2 = self._chi2_independence(df["opinion"], df["external_opinion"])
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
        self._save(result, folder_name, "opinion_vs_external_opinion_per_model.csv")
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
        self, dfs: dict[str, pd.DataFrame], folder_name: str, column: str
    ) -> pd.DataFrame:
        """Chi-square test of independence between model identity and the
        (rounded) rating distribution of ``column``, for every pair of models."""
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
        self, dfs: dict[str, pd.DataFrame], folder_name: str, column: str
    ) -> pd.DataFrame:
        """Chi-square test of independence between model identity and the
        (rounded) rating distribution of ``column``, across all models."""
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
        self.opinion_vs_external_per_model(dfs, folder_name)
        self.between_model_correlations(dfs, folder_name, "opinion")
        self.between_model_correlations(dfs, folder_name, "external_opinion")
        self.pairwise_model_independence(dfs, folder_name, "opinion")
        self.pairwise_model_independence(dfs, folder_name, "external_opinion")
        self.between_model_independence(dfs, folder_name, "opinion")
        self.between_model_independence(dfs, folder_name, "external_opinion")
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
