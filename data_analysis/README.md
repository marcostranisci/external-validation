# Questionnaire Analysis

This folder contains the output of the reusable analysis pipeline for the
moral-questionnaire survey results in `surveys/mft` (Moral Foundations
Theory) and `surveys/pvq` (Portrait Values Questionnaire), plus a second
analysis of the raw human-annotator exports in `surveys/mf_merged.csv` and
`surveys/pv_merged.csv` (see "Annotator-level analysis" below).

## Code

- `src/questionnaire_analysis.py` — `QuestionnaireAnalyzer` class (model
  survey files, one file per model) and `AnnotatorSurveyAnalyzer` class
  (merged annotator files, one row per annotator).
- `run_questionnaire_analysis.py` — entry point that runs `QuestionnaireAnalyzer`
  on both `surveys/mft` and `surveys/pvq` and writes everything under
  `data_analysis/{mft,pvq}/`.
- `run_annotator_analysis.py` — entry point that runs `AnnotatorSurveyAnalyzer`
  on `surveys/mf_merged.csv` and `surveys/pv_merged.csv` and writes
  everything under `data_analysis/annotators/`.

### Re-running the analysis

```bash
pip install -r requirements.txt   # pandas, numpy, scipy already included
python run_questionnaire_analysis.py
python run_annotator_analysis.py
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

3. **`pairwise_model_independence_opinion.csv`** /
   **`pairwise_model_independence_external_opinion.csv`**
   Chi-square test of independence between model identity and the (rounded)
   rating distribution, for every pair of models — tests whether the two
   models' rating distributions differ from each other.

4. **`between_model_independence_opinion.csv`** /
   **`between_model_independence_external_opinion.csv`**
   The same chi-square test of independence, pooling all models at once
   instead of pairwise — tests whether rating distributions depend on which
   model produced them, overall.

5. **`annotator_disagreement_vs_delta.csv`**
   Per model: Pearson/Spearman correlation between `annotator_disagreement`
   and the delta between the model's `opinion` and the human
   `external_opinion` (both the signed delta `opinion - external_opinion`
   and its absolute value).

Every test row reports `n` (valid paired observations used) and a `notes`
column explaining why a test could not be run (e.g. `"zero variance in at
least one variable"`, `"fewer than 3 valid paired observations"`).

### Independence tests on the normalized scores

Tests 1 and 3 above are repeated on the z-scored `opinion_normalized` /
`external_opinion_normalized` columns, producing:

- `opinion_vs_external_opinion_per_model_normalized.csv`
- `pairwise_model_independence_opinion_normalized.csv` /
  `pairwise_model_independence_external_opinion_normalized.csv`
- `between_model_independence_opinion_normalized.csv` /
  `between_model_independence_external_opinion_normalized.csv`

Two things differ from the raw-column versions:

- **Discretization.** Raw Likert ratings are integer-valued, so the
  chi-square contingency tables are built by rounding to the nearest
  integer. Normalized scores are continuous z-scores, so rounding would
  mostly produce unique values; instead they are split into
  `QuestionnaireAnalyzer.normalized_bins` (default 4) equal-frequency
  quantile buckets before building the table.
- **What changes vs. the raw version.** Pearson/Spearman correlation is
  invariant to per-column z-scoring, so the `pearson_r`/`spearman_r`
  columns in `opinion_vs_external_opinion_per_model_normalized.csv` are
  identical to the raw file's — only the chi-square result differs, and
  it can differ meaningfully: normalizing removes each model's own
  mean/scale usage, so the independence tests on `opinion_normalized`/
  `external_opinion_normalized` isolate differences in the *shape* of a
  model's rating distribution (e.g. skew, how tightly it clusters around
  its own average) rather than differences caused simply by one model
  favoring different raw numbers than another.

### Mann-Whitney / Kruskal-Wallis: the more appropriate test for the normalized scores

Chi-square on binned quantiles is a workable approximation, but it's not
the right tool for genuinely continuous z-scored data: it throws away
within-bin information and its result depends on an arbitrary bin-count
choice (`normalized_bins`, default 4). The standard, bin-free tests for
comparing continuous/ordinal distributions between groups are:

- **`pairwise_model_mannwhitney_opinion_normalized.csv`** /
  **`pairwise_model_mannwhitney_external_opinion_normalized.csv`** — a
  two-sample Mann-Whitney U test for every model pair, with a
  rank-biserial correlation (`rank_biserial_r`, range -1 to 1) as effect
  size.
- **`kruskal_wallis_opinion_normalized.csv`** /
  **`kruskal_wallis_external_opinion_normalized.csv`** — a Kruskal-Wallis H
  test across all models at once, with an eta-squared effect size
  (`eta_squared`; Cohen-style benchmarks: ~0.01 small, ~0.06 medium, ~0.14
  large).

**These substantially revise the chi-square-based conclusion.** The pooled
chi-square test on `opinion_normalized` looked like near-total separation
(MFT: χ²=363, p≈5e-66; median pairwise Cramér's V=0.83). Kruskal-Wallis,
which doesn't depend on a bin-count choice, still finds a real difference
between models but a *small* one: MFT H=20.7, p=0.002, η²=0.06; PVQ
H=12.9, p=0.025, η²=0.03. At the pairwise level, Mann-Whitney finds 12/20
valid MFT pairs and 6/15 valid PVQ pairs significant (p<0.05) — real, but
far from the near-universal separation chi-square implied. For
`external_opinion_normalized`, Kruskal-Wallis found **no** significant
difference at all (MFT p=0.998, PVQ p=0.9998; 0/15 pairwise Mann-Whitney
pairs significant in either questionnaire) — this actually sharpens the
chi-square-based finding (which was already non-significant, p≈0.79-0.80)
into a much more decisive null result.

**Takeaway:** treat the chi-square/Cramér's V numbers on the normalized
columns as an upper-bound sanity check at most, not as the effect-size
estimate to report — use the Mann-Whitney/Kruskal-Wallis results instead.
Models genuinely differ in the shape of their own normalized `opinion`
(the effect survives the more conservative test), but the difference is
small-to-medium, not the large separation the binned chi-square suggested.
The `external_opinion_normalized` null result — annotator rating patterns
don't detectably differ by which model produced the reply being rated —
holds up under both the chi-square and the rank-based test alike.

## Annotator-level analysis: own replies vs. model evaluations

`surveys/mf_merged.csv` and `surveys/pv_merged.csv` are a different shape of
data: one row per human annotator (54 annotators in each file), rather than
one row per model. For a shared set of questionnaire items, each annotator
has two parallel blocks of columns:

- `MF_01..MF_36` / `PV_01..PV_40` — the annotator's own evaluation of a
  model's reply to that item.
- `MF02_01..MF02_36` / `PV02_01..PV02_40` — the annotator's own reply to
  that same questionnaire item, as themselves.

`AnnotatorSurveyAnalyzer` correlates these two blocks — does an annotator's
personal stance on an item relate to how they evaluate a model's reply to
it? — at three levels, for each of `MF` and `PV`:

1. **`{mf,pv}_annotator_correlations.csv`** — per annotator (row-wise):
   Pearson/Spearman correlation between that one annotator's own-reply
   vector and their model-evaluation vector, across all shared items
   (`n_items` = 36 for MF, 40 for PV).
2. **`{mf,pv}_item_correlations.csv`** — per item (column-wise): Pearson/
   Spearman correlation between annotators' own replies and their
   evaluations for that one item, across all 54 annotators.
3. **`{mf,pv}_overall_correlation.csv`** — every (annotator, item) pair
   pooled into one long vector and correlated overall, ignoring annotator/
   item identity.

Results (excluding the `notes` column, empty when a test ran cleanly):

| | annotators: median r (significant / 54) | items: median r (significant / N items) | overall r (n pairs, p) |
|---|---|---|---|
| MF | 0.14 (17 positive, 1 negative) | 0.24 (16 / 36) | 0.31 (n=1944, p≈2e-45) |
| PV | 0.30 (26 positive, 0 negative) | 0.35 (30 / 40) | 0.40 (n=2160, p≈4e-82) |

Own replies and model evaluations are positively correlated overall in both
questionnaires — annotators who personally lean toward agreement on an item
also tend to rate a model's reply to it more favorably — but the
association is modest (r≈0.3-0.4 pooled) and far from universal at the
individual level: about a third of MF annotators and half of PV annotators
show a statistically significant positive correlation, essentially none
show a significant negative one, and the rest show no detectable
relationship at their own individual n=36/40.

Rerun with:

```bash
python run_annotator_analysis.py
```

Reuse on a new merged file:

```python
from src.questionnaire_analysis import AnnotatorSurveyAnalyzer

analyzer = AnnotatorSurveyAnalyzer(output_dir="data_analysis")
analyzer.run("surveys/mf_merged.csv", prefix="MF")   # one file
analyzer.run_all({"MF": "surveys/mf_merged.csv", "PV": "surveys/pv_merged.csv"})
```

`prefix` must match the column naming convention `<prefix>_NN` /
`<prefix>02_NN` (case-sensitive); item numbers present in only one of the
two blocks are skipped and reported in
`data_analysis/annotators/data_quality_warnings.log`. No such gaps were
found in `mf_merged.csv`/`pv_merged.csv` — both files had complete,
non-missing data for every annotator and item.

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
