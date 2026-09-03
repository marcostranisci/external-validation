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
- `run_annotator_demographics.py` — entry point that checks whether
  annotators sharing a demographic (gender, continent of birth) agree more
  with each other when evaluating a model's replies (see "Demographic
  agreement" below).

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

**The statistical tests below all use the raw `opinion`/`external_opinion`
columns, not the normalized ones.** The rating scale here (1-5 for MFT,
1-6 for PVQ) is the same fixed scale for every model and every annotator —
unlike a subjective per-rater anchor, there's no reason to treat one
model's average level as a nuisance to normalize away. A model that
systematically rates higher or lower than another is itself a real,
reportable result, and z-scoring per model would erase exactly that by
construction. `opinion_normalized`/`external_opinion_normalized` are kept
in the processed files for anyone who wants to ask the narrower question
"controlling for level and spread, does the *shape* of a model's answers
still differ?", but that's not what's tested here.

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

5. **`pairwise_model_mannwhitney_opinion.csv`** /
   **`pairwise_model_mannwhitney_external_opinion.csv`**
   A two-sample Mann-Whitney U test between every pair of models, with a
   rank-biserial correlation (`rank_biserial_r`, range -1 to 1) as effect
   size. Unlike chi-square, this works directly on the ordinal ratings
   without rounding/binning, respecting their ordering rather than
   treating them as unordered categories — the more appropriate test for
   ordinal Likert data, and the one to prefer over tests 3-4 above.

6. **`kruskal_wallis_opinion.csv`** / **`kruskal_wallis_external_opinion.csv`**
   The pooled, all-models-at-once counterpart of test 5 (a Kruskal-Wallis H
   test), with an eta-squared effect size (`eta_squared`; Cohen-style
   benchmarks: ~0.01 small, ~0.06 medium, ~0.14 large) — the rank-based
   alternative to test 4.

7. **`annotator_disagreement_vs_delta.csv`**
   Per model: Pearson/Spearman correlation between `annotator_disagreement`
   and the delta between the model's `opinion` and the human
   `external_opinion` (both the signed delta `opinion - external_opinion`
   and its absolute value).

Every test row reports `n` (valid paired observations used) and a `notes`
column explaining why a test could not be run (e.g. `"zero variance in at
least one variable"`, `"fewer than 3 valid paired observations"`).

**Read tests 5-6, not 3-4, for the headline "do models differ" result.**
Chi-square (tests 3-4) treats ratings as unordered nominal categories,
discarding the fact that a 4 is closer to a 5 than to a 1; Mann-Whitney/
Kruskal-Wallis use ranks and don't have that problem. On the raw scale
both approaches agree there's a real, large difference between models:
Kruskal-Wallis on `opinion` gives η²≈0.45-0.46 in both MFT and PVQ (a large
effect by Cohen's benchmarks — models genuinely differ a lot in their
absolute rating level), and on `external_opinion` gives η²≈0.11 (MFT) /
0.14 (PVQ) (medium-to-large — annotators' absolute ratings of different
models' replies differ too, not just the shape of the pattern). Pairwise
Mann-Whitney: 18/20 valid MFT pairs and 12/15 valid PVQ pairs differ
significantly on `opinion`; 5/15 (MFT) and 9/15 (PVQ) differ on
`external_opinion`.

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

## Demographic agreement: does gender/origin predict inter-annotator agreement?

Each `QUESTNNR` value in the merged files identifies the group of 9
annotators who evaluated one specific model's replies (the `<prefix>_NN`
columns) — not the model by name, so `AnnotatorSurveyAnalyzer.infer_model_mapping`
recovers that mapping by matching each group's sorted per-item ratings
against the disaggregated ratings in `surveys/mft/*.csv` / `surveys/pvq/*.csv`
(order-independent, since annotator order differs between the two file
formats). Every `QUESTNNR` group matched exactly one model file. The
annotator pool is stratified 3-3-3 by gender (woman/man/non-binary) and by
continent of birth (Africa/Asia/Europe, derived from `Country of birth`)
within every group of 9.

`AnnotatorSurveyAnalyzer.demographic_agreement` asks: for a given model, do
annotators sharing a demographic attribute agree with each other *more*
than annotators who don't, when evaluating that model's replies? For every
pair of annotators within a `QUESTNNR` group it computes two agreement
metrics across their shared evaluation items — Pearson correlation
(agreement in *pattern* across items) and negated mean absolute difference,
`neg_mad` (agreement in absolute *level*, higher = closer) — then splits
the pairs into "same demographic" vs "different demographic" and compares
with a two-sample Mann-Whitney U test. Per model this is underpowered (9
annotators → 36 pairs, split further into 9 same-gender vs 27
different-gender, or similarly for continent), so an additional
`model="ALL (pooled)"` row pools all 6 models' pairs (54 same vs. 162
different) for a properly-powered version of the same test. Results:
`data_analysis/annotators/{mf,pv}_demographic_agreement.csv`.

Rerun with:

```bash
python run_annotator_demographics.py
```

**Result: no reliable evidence of gender-based in-group agreement, and only
weak, non-replicating evidence for continent.** Pooled (the numbers to
trust):

| | Gender: pearson_r p | Gender: neg_mad p | Continent: pearson_r p | Continent: neg_mad p |
|---|---|---|---|---|
| MF | 0.93 | 0.74 | 0.26 | **0.031** |
| PV | 0.91 | 0.88 | 0.79 | 0.50 |

Gender is a clean null in both questionnaires, on both metrics (all
p≥0.74). Continent reaches p=0.031 in MF on the level-agreement metric
(`neg_mad`; same-continent pairs agree slightly more closely in absolute
level: median -1.03 vs. -1.11) but not on the pattern metric in MF
(p=0.26), and doesn't replicate in PV at all (p=0.50 on the same metric).
Given 8 pooled tests were run in total (2 questionnaires × 2 demographics ×
2 metrics) plus 24 unpowered per-model tests underneath them, a single
p≈0.03 hit that fails to replicate across questionnaires is exactly what
you'd expect from chance alone (≈0.4 false positives expected at α=0.05
across the 8 pooled tests) — treat the continent result as suggestive at
most, not as an established effect, and don't lead with the per-model
numbers (`data_analysis/annotators/{mf,pv}_demographic_agreement.csv`
rows other than `model="ALL (pooled)"`) as evidence on their own; they're
kept for transparency/inspection, not for drawing conclusions from
individually.

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
