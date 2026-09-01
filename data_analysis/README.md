# Questionnaire Analysis

This folder contains the output of the reusable analysis pipeline for the
moral-questionnaire survey results in `surveys/mft` (Moral Foundations
Theory) and `surveys/pvq` (Portrait Values Questionnaire).

## Code

- `src/questionnaire_analysis.py` — `QuestionnaireAnalyzer` class with all
  the reusable logic (feature engineering + statistical tests).
- `run_questionnaire_analysis.py` — entry point that runs the full pipeline
  on both `surveys/mft` and `surveys/pvq` and writes everything here.

### Re-running the analysis

```bash
pip install -r requirements.txt   # pandas, numpy, scipy already included
python run_questionnaire_analysis.py
```

### Reusing the class on a new folder/dataset

```python
from src.questionnaire_analysis import QuestionnaireAnalyzer

analyzer = QuestionnaireAnalyzer(surveys_dir="surveys", output_dir="data_analysis")
dfs = analyzer.run_folder("mft")     # process one folder + run all tests
analyzer.run_all(["mft", "pvq"])     # or process every folder at once
```

Each survey CSV is expected to have (at least):

- `model_name`, `test_statement`, `opinion` — the model's own Likert answer.
- a column whose name contains `disaggregated` (e.g. `pvq_disaggregated`) —
  a stringified list of the individual human annotators' ratings for that
  item, e.g. `"[5.0, 4.0, 4.0, 5.0, 5.0, 3.0, 4.0, 5.0, 4.0]"`.

## What was computed

For every CSV file in `surveys/mft/` and `surveys/pvq/` (one file per
model), the pipeline adds four columns:

| column | meaning |
|---|---|
| `annotator_disagreement` | Mean absolute deviation of the human annotators' ratings around their own mean for that item — a measure of how much the annotators disagreed with each other. |
| `external_opinion` | Mean of the human annotators' ratings for that item — the "crowd" opinion, to compare against the model's own `opinion`. |
| `opinion_normalized` | Z-score of `opinion`, computed per model file (mean 0, std 1 across that model's items). |
| `external_opinion_normalized` | Z-score of `external_opinion`, computed the same way. |

A zero-variance column (e.g. a model that answered the same value on every
item) is normalized to all zeros instead of producing `NaN`/`inf`.

The augmented per-model files are saved to:

```
data_analysis/processed/mft/<model>.csv
data_analysis/processed/pvq/<model>.csv
```

## Statistical tests

All tests are run separately for the `mft` and `pvq` folders, and results
are saved under `data_analysis/mft/` and `data_analysis/pvq/` respectively.

1. **`opinion_vs_external_opinion_per_model.csv`**
   Per model: Pearson correlation, Spearman correlation, and a chi-square
   test of independence (on rounded ratings) between the model's own
   `opinion` and the human `external_opinion`.

2. **`between_model_correlations_opinion.csv`** /
   **`between_model_correlations_external_opinion.csv`**
   Pairwise Pearson and Spearman correlations between every pair of models,
   computed on `opinion` (resp. `external_opinion`) after aligning rows by
   `test_statement` (i.e. comparing models on the same questionnaire item).

3. **`between_model_independence_opinion.csv`** /
   **`between_model_independence_external_opinion.csv`**
   A chi-square test of independence between model identity and the
   (rounded) rating distribution, pooling all models — tests whether rating
   distributions depend on which model produced them.

4. **`annotator_disagreement_vs_delta.csv`**
   Per model: Pearson/Spearman correlation between `annotator_disagreement`
   and the delta between the model's `opinion` and the human
   `external_opinion` (both the signed delta `opinion - external_opinion`
   and its absolute value).

Every test row reports `n` (valid paired observations used) and a `notes`
column explaining why a test could not be run (e.g. `"zero variance in at
least one variable"`, `"fewer than 3 valid paired observations"`).

## Data quality notes

See `data_analysis/data_quality_warnings.log` for issues detected while
processing the raw files, notably:

- `surveys/mft/Ministral-3-8B-Instruct-2512.csv` and
  `surveys/pvq/Ministral-3-8B-Instruct-2512.csv` have no `*_disaggregated`
  column, so `annotator_disagreement`/`external_opinion` are `NaN` and all
  tests involving them are skipped for that model.
- `surveys/pvq/Ministral-3-8B-Instruct.csv` has free-text values in its
  `opinion` column instead of numeric ratings; these are coerced to `NaN`,
  so tests involving `opinion` are skipped for that model in the `pvq`
  folder (its `annotator_disagreement`/`external_opinion` are still
  computed from the disaggregated ratings it does have).
- `surveys/mft/Ministral-3-8B-Instruct.csv` and
  `surveys/mft/Ministral-3-8B-Instruct-2512.csv` both contain the
  `model_name` value `Ministral-3-8B-Instruct-2512` and (for the `opinion`
  column) identical data — they are kept as two separate models in the
  output, keyed by file name, since that is what the two files represent.
