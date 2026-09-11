# Questionnaire Analysis

This folder contains the output of the reusable analysis pipeline for the
moral-questionnaire survey results in `surveys/mft` (Moral Foundations
Theory) and `surveys/pvq` (Portrait Values Questionnaire), a second
analysis of the raw human-annotator exports in `surveys/mf_merged.csv` and
`surveys/pv_merged.csv` (see "Annotator-level analysis" below), and a
third analysis of belief-steered hate-speech detection (see "Belief-steered
hate-speech detection" below). **See `SUMMARY.md` for the cross-experiment
findings write-up** — this file documents methodology, output files, and
per-test results in detail.

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
- `src/hs_detection_analysis.py` — `HateSpeechSteeringAnalyzer` class (does
  questionnaire-belief steering shift hate-speech recall?).
- `run_hs_detection_analysis.py` — entry point that runs it on
  `hs_detection/implicit_hate_all_models.json` and writes everything under
  `data_analysis/hs_detection/` (see "Belief-steered hate-speech detection"
  below).
- `run_hs_detection_verbalized_analysis.py` — the same analyzer, on
  `hs_detection/implicit_hate_verbalized_all_models.json` (steering by the
  verbalized questionnaire item/Likert score instead of free-text opinion),
  writing to `data_analysis/hs_detection_verbalized/`.
- `src/hs_detection_comparison.py` — `SteeringConditionComparison` class
  and `run_hs_detection_comparison.py` — compares the two runs above (see
  "Free-text opinion vs. verbalized questionnaire item" below).

### Re-running the analysis

```bash
pip install -r requirements.txt   # pandas, numpy, scipy already included
python run_questionnaire_analysis.py
python run_annotator_analysis.py
python run_annotator_demographics.py
python run_hs_detection_analysis.py
python run_hs_detection_verbalized_analysis.py
python run_hs_detection_comparison.py
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
both approaches agree there's a real, large difference between the 6
models: Kruskal-Wallis on `opinion` gives η²=0.49 (MFT) / 0.46 (PVQ) (a
large effect by Cohen's benchmarks — models genuinely differ a lot in
their absolute rating level), and on `external_opinion` gives η²=0.11
(MFT) / 0.14 (PVQ) (medium-to-large — annotators' absolute ratings of
different models' replies differ too, not just the shape of the pattern).
Pairwise Mann-Whitney (15 pairs = C(6,2), all valid): 13/15 MFT and 12/15
PVQ pairs differ significantly on `opinion`; 5/15 (MFT) and 9/15 (PVQ)
differ on `external_opinion`.

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

## Belief-steered hate-speech detection

`hs_detection/implicit_hate_all_models.json` holds hate-speech
classifications of the same 500 messages (balanced 250 hate / 250 not) for
each of the 6 models, under three conditions: `zero_shot` (one run, no
questionnaire context), and `mft`/`pvq` (one run per questionnaire item —
36 MFT, 40 PVQ — each steered by *that same model's own* free-text reply to
that item; the file's metadata records this as `"prediction_setup":
"paired_by_model"` — there is no cross-model steering condition in this
file, i.e. Model A is never steered by Model B's replies).

`HateSpeechSteeringAnalyzer` (`src/hs_detection_analysis.py`) asks two
questions:

1. **Does steering shift recall, within each model?** For each model,
   recall (TP / 250 actual-positive messages) is computed for `zero_shot`
   and for every one of the 76 belief-steered runs (36 MFT + 40 PVQ). A
   one-sample Wilcoxon signed-rank test on
   `steered_recall - zero_shot_recall` (per model, separately for MFT,
   PVQ, and combined) tests whether steering shifts recall in a
   consistent direction. Saved to `steering_effect_per_model.csv`.
2. **Does the size/direction of that shift differ between models?** ("own
   vs. other models", reinterpreted as comparing models' own-steering
   effects to each other, since there's no cross-steering condition to
   compare directly — see above.) The per-item delta
   (`steered_recall - that model's own zero_shot_recall`) is compared
   across all 6 models with a pooled Kruskal-Wallis test and pairwise
   Mann-Whitney U tests, separately for MFT, PVQ, and combined. Saved to
   `between_model_steering_kruskal.csv` /
   `between_model_steering_pairwise_mannwhitney.csv`.
3. **Which items steer recall the most?** For each of the 76 items,
   `item_level_effect` takes the 6 models' per-item deltas and reports
   their mean, mean absolute value (the item's overall "how much does it
   move recall, either way" score), how many models it pushed up vs. down,
   and a one-sample Wilcoxon signed-rank test on those 6 deltas
   (`wilcoxon_p`), plus a Benjamini-Hochberg FDR correction across all 76
   of them (`wilcoxon_p_fdr_bh`, one hypothesis family). n=6 caps the
   smallest achievable `wilcoxon_p` at 1/32≈0.031 (reached whenever all 6
   models move the same direction) — **after BH correction, 0 of 76 items
   reach `wilcoxon_p_fdr_bh` < 0.05** (the smallest corrected value is
   0.198). So no single item's effect is individually confirmed at a
   corrected significance level; the per-item numbers below are
   descriptive ranking/pattern-spotting, not confirmed findings. Saved to
   `item_level_steering_effect.csv`, sorted by mean absolute delta.

4. **Which items significantly change recall, for each model
   individually?** Test 3's n=6 (one data point per model) is too
   underpowered to ever survive correction. This test instead uses the
   250 positive-class messages themselves as the paired sample: for a
   given model and item, compare each message's zero-shot vs. steered
   correctness with an exact (binomial) **McNemar test** — the standard
   test for paired binary outcomes — on the discordant pairs (`n_lost` =
   correct zero-shot, wrong after steering; `n_gained` = the reverse).
   p-values are Benjamini-Hochberg corrected *within each model's own*
   76-item family (a per-model question, so each model gets its own
   correction rather than pooling all 456 tests). If
   `pvq_mapping_path` is given (a CSV with `test_statement`/`pvq_value`
   columns; `hs_detection/pvq_items_mapping.csv` here, values 1-10 per
   Schwartz's Portrait Values Questionnaire), PVQ items are also labeled
   with their Schwartz value name. Saved to
   `item_significance_per_model.csv`.
5. **Pattern check: does the effect cluster by Schwartz PVQ value, or by
   Moral Foundation?** `pvq_value_patterns` / `mft_foundation_patterns`
   (sharing one `_category_patterns` implementation) aggregate test 4's
   PVQ/MFT results by Schwartz value or Moral Foundation respectively, per
   model and pooled (`model="ALL"`): item count, how many are significant,
   in which direction, and the mean delta. Saved to `pvq_value_patterns.csv`
   / `mft_foundation_patterns.csv`.
6. **Model-wise: how often does steering flip the predicted label at
   all?** Tests 1-5 only look at the positive class and at whether
   *correctness* changes. `flip_counts_by_item` instead counts, for every
   model and item, how many of the 500 messages get a different predicted
   label under steering than under zero-shot — regardless of ground truth
   or correctness, split into flips toward "hate" vs. toward "not hate".
   `flip_counts_by_model` aggregates this to one row per model (and per
   model × MFT/PVQ/combined); `flip_counts_by_model_and_value` breaks it
   down further by MFT foundation / Schwartz PVQ value (needs
   `pvq_mapping_path`, as in test 4). Saved to `flip_counts_by_item.csv`,
   `flip_counts_by_model.csv`, `flip_counts_by_model_and_value.csv`.
7. **Instance-wise: how often is each message's prediction flipped?**
   `flip_counts_by_instance` inverts the view: for each of the 500
   messages, across all 456 (6 models × 76 items) steering conditions, how
   many times does the predicted label differ from that same model's own
   zero-shot prediction on it — i.e. which specific messages are most
   unstable under moral steering, pooling across every model and belief.
   Saved to `flip_counts_by_instance.csv`.
8. **Instance-wise: raw predicted-label stability (no baseline).**
   `instance_prediction_profile` counts, for each message, how many of
   *every* prediction ever made on it — zero-shot plus all 76
   belief-steered runs, per model (77 per model, 462 pooled across all 6,
   `model="ALL"`) — landed on class 1 ("hate") vs. class 0, and the
   resulting `proportion_hate`. Unlike test 7, this doesn't reference a
   zero-shot baseline at all: it's the raw consistency of the predicted
   label across every context the message was ever classified under.
   `instance_prediction_bins` bins `proportion_hate` into 10 equal-width
   bins (0-0.1, 0.1-0.2, ..., 0.9-1.0), per model and pooled, each bin
   split by the message's actual `dataset_label` so a bin's count can be
   read against how many of its messages are truly hate speech. Saved to
   `instance_prediction_profile.csv` / `instance_prediction_bins.csv`.

The full per-item recall table (one row per model × condition × item) is
in `recall_by_model_condition_item.csv`.

**Item significance results.** Unlike test 3, this test has real power —
most items turn out significant per model: Ministral 74/76, Falcon 56/76,
Llama 51/76, Olmo 46/76, Qwen 45/76, Apertus 23/76 (lowest, consistent
with its near-zero average effect from test 1). The *direction* sharpens
the earlier picture a lot: for Falcon, Llama, Ministral, and Qwen,
essentially every significant item is an **increase** (53/56, 51/51,
74/74, 45/45) — steering isn't just helpful on average for these models,
it's almost never harmful item-by-item. Olmo is the mirror image: most of
its significant items (35/46) are **decreases**. Apertus is genuinely
mixed (13 decreases, 10 increases) rather than simply "unaffected" —
test 1's near-zero average was masking real, opposite-signed, item-level
effects that roughly cancel out.

**PVQ value pattern.** Pooling all 6 models, every Schwartz value has a
net *positive* mean delta (steering helps) **except Universalism
(-0.048) and Benevolence (-0.044)** — the two values in Schwartz's
"self-transcendence" quadrant (concern for others' welfare broadly).
Security, Achievement, and Stimulation show the largest positive average
shifts (+0.05 to +0.06). This isn't purely one outlier model: Olmo's
Universalism (-0.39) and Benevolence (-0.32) deltas are far larger in
magnitude than anything else in the table and dominate the pooled
average, but Apertus *also* shows Universalism/Benevolence as its most
negative values (-0.032 / -0.061, its only significant items in either
category are decreases), and Falcon's Universalism delta (-0.027) is its
only negative value across all 10 Schwartz categories. Llama, Ministral,
and Qwen, by contrast, show Universalism/Benevolence as clearly positive,
in line with their general "steering helps" pattern. So: 2 of 6 models
(Apertus, Olmo) are consistently hurt specifically by
universalism/benevolence content, a 3rd (Falcon) leans that way for
Universalism only, and the other 3 show no such exception — a real,
if partial, pattern worth investigating further rather than a
description of all 6 models.

**MFT foundation pattern — converges with the PVQ result.** Pooled: every
foundation has a net positive mean delta, but **care (+0.017) and purity
(+0.007) are far behind the other four** (proportionality +0.052, loyalty
+0.050, equality +0.048, authority +0.034), and have the fewest
significant items too (care 11/36, purity 15/36, vs. 25-31/36 for the
other four). This is the same split as the PVQ result, in the same two
models: **care is negative for Apertus (-0.041) and Olmo (-0.022)**, and
**purity is negative for Apertus (-0.031) and Olmo (-0.069)** — the only
negative cells in either model's foundation row. `care` is MFT's closest
analogue to PVQ's Universalism/Benevolence (concern for others' welfare),
so this is an independent replication, from a completely different
questionnaire, of the same finding: **Apertus and Olmo are specifically
resistant to (or actively hurt by) "care for others" moral content as
steering context, while the other four models are helped by it just like
everything else.** For the other four models, care/purity aren't
negative, but they're still consistently their two *weakest* foundations
(smallest positive delta of the six) — so the pattern holds directionally
even where it doesn't flip to a net negative.

**Flip counts (model-wise).** Ranking models by total label flips across
all 76 items (`flip_counts_by_model.csv`, `condition="combined"`) gives
the same ordering as the recall-shift results, but adds a mechanism:
Olmo flips the most overall (6030 flips, mean flip rate 15.9% of the 500
messages per item) and flips overwhelmingly *toward "not hate"* (4730 vs.
1300 toward "hate") — it isn't just losing recall on average, it's
actively relabeling messages away from hate speech under steering.
Ministral flips almost as often (5245) but in the opposite direction —
overwhelmingly *toward "hate"* (5117 vs. 128) — the mechanism behind its
recall gain. Qwen and Falcon show the same toward-hate skew, more mildly;
Llama is toward-hate too but flips least overall (3025); Apertus is the
only model with a roughly even split (1572 vs. 1602), consistent with its
mixed, cancelling-out item-level effects. Breaking flips down by value
type (`flip_counts_by_model_and_value.csv`) reproduces the
Universalism/Benevolence/care/purity pattern at the mechanism level too:
Olmo's Universalism items alone cause 992 flips — more than any other
single (model, value) cell in the whole table by a wide margin (next
highest is Olmo's own Benevolence at 565) — while every other model's
Universalism/Benevolence flip counts are in the same range as their other
values.

**Flip counts (instance-wise).** `flip_counts_by_instance.csv` pools all
456 (6 models × 76 items) steering conditions per message and counts how
often each message's label is unstable. Instability is not concentrated
in a handful of messages: flip rates range continuously from 0 up to 53%
(message id 465, an actual non-hate message that gets flipped both
directions roughly evenly). Non-hate messages flip slightly more often on
average than actual hate messages (mean flip rate 12.2% vs. 9.7%) — so
steering is somewhat more likely to churn a model's opinion on borderline
non-hate content than on clear-cut hate speech, across models and beliefs
alike.

**Predicted-label stability (instance-wise, no baseline).** Pooled across
all 462 predictions per message, splitting the 10 bins into three tiers
gives a cleaner picture than "confident vs. ambiguous" alone:

| tier | `proportion_hate` | n | % of corpus | actually hate speech |
|---|---|---|---|---|
| Confident | ≤0.1 or ≥0.9 | 240 | 48.0% | 137 (131 in the "hate" extreme, 6 in the "not hate" extreme) |
| Leaning | (0.1, 0.3] or [0.7, 0.9) | 138 | 27.6% | 64 |
| Balanced | (0.3, 0.7) | 122 | 24.4% | 49 |

Just under half the corpus (240/500) is classified with high,
context-independent consistency in one direction, regardless of which
model or which moral belief is steering it — and checked against ground
truth these confident bins are mostly right but not purely so: of the
175 confidently-"hate" messages, 131 are actually hate speech and 44 are
not (44 persistent false positives, confidently mislabeled across nearly
every model and belief); of the 65 confidently-"not hate" messages, only
6 are actually hate speech (59 correct, 6 persistent false negatives —
implicit hate that essentially no context makes any model flag). Another
138 (27.6%) lean one way without reaching that consistency, and only the
remaining 122 (24.4%) are genuinely balanced (`proportion_hate` between
0.3 and 0.7) — real context-dependence, where the label actually depends
on which model and which belief is asked, rather than the "everything
outside the confident bins is ambiguous" reading.

The per-model bins (`model` column in `instance_prediction_bins.csv`)
show each model's base rate for calling something "hate speech" at all,
consistent with the flip-count and recall results above: Llama and
Ministral put the most messages in the confidently-"hate" bin (379/500
and 365/500) and the fewest in confidently-"not hate" (63, 59); Olmo is
the mirror image (94 confidently-"hate", 214 confidently-"not hate");
Falcon, Qwen, and Apertus fall in between. A model's overall willingness
to call something hate speech, largely independent of the specific
steering belief, is itself a stable per-model property.

Rerun with:

```bash
python run_hs_detection_analysis.py
```

**Results.** Zero-shot recall varies a lot by model to start with: Llama
0.896, Apertus 0.856, Ministral 0.856, Qwen 0.716, Olmo 0.628, Falcon 0.62.
Steering (combined MFT+PVQ) significantly shifts recall for 5 of 6 models:

| model | zero-shot recall | mean steered recall | delta | p (Wilcoxon) |
|---|---|---|---|---|
| Ministral-3-8B-Instruct-2512 | 0.856 | 0.946 | **+0.090** | 3.5e-14 |
| Falcon3-7B-Instruct | 0.620 | 0.691 | **+0.071** | 4.0e-11 |
| Qwen3-8B | 0.716 | 0.767 | **+0.051** | 2.6e-12 |
| Llama-3.1-8B-Instruct | 0.896 | 0.936 | **+0.040** | 1.3e-13 |
| Apertus-8B-Instruct | 0.856 | 0.853 | -0.003 | 0.63 (n.s.) |
| Olmo-3-7B-Instruct | 0.628 | 0.532 | **-0.096** | 1.3e-5 |

Four models get a real recall *boost* from being shown their own
questionnaire beliefs before classifying (Ministral most, then Falcon,
Qwen, Llama); one (Olmo) gets significantly *worse*, by about the same
magnitude as Ministral's gain; Apertus is unaffected. The direction isn't
simply "steering always helps" or tied to how good the model already was
zero-shot (the two worst zero-shot models, Olmo and Falcon, go opposite
ways).

**Which items steer the most.** None of this survives Benjamini-Hochberg
correction (0/76 items at `wilcoxon_p_fdr_bh` < 0.05) — read it as a
descriptive pattern to investigate further with more models/items, not a
confirmed result. Two different questions here give two different
answers:

- *Largest effect regardless of direction* (top of
  `item_level_steering_effect.csv` by `mean_abs_delta`, ≈0.10-0.15): these
  are almost all PVQ items about care/fairness/equality/universalism
  content ("protect the weak", "equal opportunities for everyone", "help
  the people around him", "forgive people who hurt him"). But they have a
  *mixed* sign across models (typically 3 of 6 models up, 3 down) — a big
  effect whose direction is model-specific, not a property of the item
  alone.
- *Most reliably one-directional* (`n_positive == 6` or `n_negative == 6`
  in that file, i.e. every model moved the same way): 12 items, **all 12
  positive** — no item pushed recall down for every model. These cluster
  thematically around proportionality/authority/tradition/achievement
  content ("people who are more hard-working should end up with more
  money", "we all need to learn from our elders", "traditions serve a
  valuable function", "being successful/impressing others", "having a
  stable government") — moderate magnitude (+0.05 to +0.09) but consistent
  across every model.

Consistent with that split, averaging `mean_delta` by MFT foundation shows
every foundation nudges recall up on average, but by very different
amounts: proportionality (+0.052), loyalty (+0.050), and equality (+0.048)
move it the most; authority (+0.034) is in between; care (+0.017) and
purity (+0.007) are close to no effect on average. Given each foundation
only has 6 items and each item only has 6 model-level data points, treat
this foundation-level pattern as suggestive rather than conclusive.

The between-model comparison confirms this isn't noise: pooled
Kruskal-Wallis on the per-item delta gives η²≈0.47-0.52 (MFT/PVQ/combined
alike) — a large effect, models clearly differ in how steering affects
them — and all 15 pairwise Mann-Whitney comparisons between models are
significant (p≤0.021 in every case), meaning every model's steering
response is statistically distinguishable from every other model's, not
just the two extremes (Ministral vs. Olmo, p=6e-23) from each other.

## Free-text opinion vs. verbalized questionnaire item: does steering method matter?

`hs_detection/implicit_hate_verbalized_all_models.json` is a second
steering run: same models, same 500 messages, same 76 MFT/PVQ items
(confirmed identical `test_statement` sets), same `"prediction_setup":
"paired_by_model"`, but the `belief` is a one-line verbalization of the
model's own *Likert rating* itself (e.g. `"Caring for people who have
suffered is an important virtue" describes you extremely well.`, plus
scale text) rather than its free-text explanation — confirmed the `score`
field exactly equals that model's own `opinion` in `surveys/mft`/`surveys/pvq`.
Because `HateSpeechSteeringAnalyzer` only assumes the belief-steering JSON
shape, no code changes were needed — `run_hs_detection_verbalized_analysis.py`
runs the identical pipeline (tests 1-8 above) against it, writing to
`data_analysis/hs_detection_verbalized/`.

**Important confound: the two files do not hold the instruction prompt
constant.** `zero_shot_prompt` and `belief_prompt` differ in wording
between the two files (the verbalized file's adds an explicit "If it is,
label it as 1, otherwise label it as 0. Answer with only 1 or 0." and
minor rephrasing elsewhere) — this is *not* an isolated
belief-content-only manipulation. Because `delta_recall` is always
computed against *that file's own* zero-shot baseline, the within-file
"does steering shift recall" conclusions are still valid — but the
prompt wording alone (with **no belief/steering involved at all**) shifts
zero-shot recall substantially for some models:

| model | zero-shot recall (free-text file's prompt) | zero-shot recall (verbalized file's prompt) | shift from wording alone |
|---|---|---|---|
| Apertus | 0.856 | 0.264 | **−0.592** |
| Llama | 0.896 | 0.816 | −0.080 |
| Qwen | 0.716 | 0.660 | −0.056 |
| Falcon | 0.620 | 0.572 | −0.048 |
| Olmo | 0.628 | 0.684 | +0.056 |
| Ministral | 0.856 | 0.868 | +0.012 |

Apertus's zero-shot recall collapses by 59 points from prompt wording
alone — larger than any steering effect measured anywhere in this
analysis. **Apertus's comparison result below should be read as
unreliable**: its "flip from no effect to a large positive effect" is
at least partly, possibly mostly, an artifact of an anomalously low,
easy-to-improve-on baseline rather than a genuine property of verbalized
steering. The other five models' zero-shot shift from wording alone
(−0.08 to +0.06) is far smaller than their measured steering effects, so
their comparisons below are less contaminated but not perfectly clean —
treat the cross-condition comparison as suggestive, not as a controlled
isolation of "belief content type" as the only variable.

`src/hs_detection_comparison.py` (`SteeringConditionComparison`,
`run_hs_detection_comparison.py`) then compares the two conditions'
outputs directly, writing to `data_analysis/hs_detection_comparison/`:

- **`steering_effect_comparison.csv`** — per model: recall-shift
  magnitude, significance, and direction agreement between the two
  conditions.
- **`flip_counts_comparison.csv`** — per model: total flips and flip
  direction (toward "hate" vs. "not hate") under each condition.
- **`pvq_category_comparison.csv`** / **`mft_category_comparison.csv`** —
  pooled mean delta-recall by Schwartz value / Moral Foundation, side by
  side.
- **`item_level_correlation_between_conditions.csv`** — per model,
  Pearson/Spearman correlation between the two conditions' per-item
  `delta_recall` (same items, same model, different steering text) — do
  the same items drive the effect under both steering methods?

Rerun with:

```bash
python run_hs_detection_verbalized_analysis.py
python run_hs_detection_comparison.py
```

**Result: steering method changes more than magnitude — it flips
direction for half the models, and the two conditions barely agree on
*which items* matter.**

| model | Δrecall (free-text) | Δrecall (verbalized) | same direction? |
|---|---|---|---|
| Apertus | −0.003 (n.s.) | **+0.173** (p=4e-14) | **No** |
| Olmo | **−0.096** (p=1e-5) | **+0.053** (p=3e-13) | **No** |
| Qwen | **+0.051** (p=3e-12) | **−0.057** (p=3e-12) | **No** |
| Falcon | +0.071 | +0.200 | Yes (bigger) |
| Llama | +0.040 | +0.108 | Yes (bigger) |
| Ministral | +0.090 | +0.042 | Yes (smaller) |

Half the models (Apertus, Olmo, Qwen) don't just change magnitude, they
flip sign. Apertus's flip (no effect → one of the largest positive
effects) is the one to discount, given its 59-point zero-shot swing from
prompt wording alone (see confound note above) — its baseline was so
depressed under the verbalized file's prompt that almost any steering
would look like a large improvement. **Olmo and Qwen are the clean
cases**: their zero-shot recall barely moves from wording alone (+0.056,
−0.056) — far smaller than their steering deltas — so their flips are not
explained by the same artifact. Olmo goes from significantly *hurt*
(−0.096) to significantly *helped* (+0.053); Qwen goes from significantly
*helped* (+0.051) to significantly *hurt* (−0.057). Verbalized steering
is also a stronger signal on average for the models with a clean
comparison (Falcon, Llama: 2-3x larger; Olmo: reversed and comparable
magnitude), consistent with a short, unhedged declarative statement ("X
describes you extremely well") being a more direct steering signal than a
long, often-hedged free-text paragraph — though Ministral is the
counterexample (smaller under verbalized), so this isn't universal
either.

**Item-level agreement between the two conditions is weak to absent**
(`item_level_correlation_between_conditions.csv`): Pearson r ranges from
−0.03 to 0.25 across all 6 models, and only Falcon's is even nominally
significant (r=0.25, p=0.03, and that wouldn't survive correction for 6
tests). So which *specific* item drives a model's recall shift is
essentially uncorrelated between free-text and verbalized steering, even
within the same model — the two steering mechanisms appear to operate
through different pathways, not just different-strength versions of the
same one.

**The Universalism/Benevolence/care/purity exception does not
replicate.** This is the most important caveat to the Experiment 2
findings above: under free-text steering, Universalism and Benevolence
were the only PVQ values with a net *negative* pooled effect, and MFT's
care/purity were far behind the other four foundations. Under verbalized
steering, **every single PVQ value and every MFT foundation has a net
positive effect** — Universalism actually swings from −0.048 to **+0.090**
and Benevolence from −0.044 to **+0.074**, no longer standing out at all;
MFT's care (+0.076) and purity (+0.066) land in the same range as the
other foundations (+0.063 to +0.123) rather than trailing them. Read
together with the weak item-level correlation, this indicates the
"self-transcendence content is the weak/harmful lever, and Apertus/Olmo
are specifically vulnerable to it" finding is a property of *free-text
elicitation specifically* (plausibly its length, hedging, or rhetorical
style for that content), not a property of self-transcendence moral
content in general. That finding should be scoped explicitly to the
free-text steering condition in any write-up, not stated as a general
claim about moral content.

This reversal is not an artifact of the Apertus confound: recomputing
both pooled means with Apertus excluded entirely gives the same story —
free-text Universalism/Benevolence mean delta is −0.051/−0.041 (still
negative), care/purity is +0.029/+0.015 (still trailing); verbalized
Universalism/Benevolence is +0.074/+0.057, care/purity +0.061/+0.044
(still uniformly positive, still no longer trailing). The other five
models alone reproduce both the original free-text exception and its
disappearance under verbalized steering.

## Data quality notes

`data_analysis/data_quality_warnings.log` currently reports no issues —
`run_questionnaire_analysis.py` processes exactly 6 clean model files per
folder, each with a numeric `opinion` column and a matching
`*_disaggregated` column.

This was not always true. `surveys/{mft,pvq}/` used to contain two files
for the same model — `Ministral-3-8B-Instruct.csv` and
`Ministral-3-8B-Instruct-2512.csv` — both with the internal `model_name`
`Ministral-3-8B-Instruct-2512`, i.e. two incomplete halves of one export
rather than two different models (confirmed against
`hs_detection/implicit_hate_all_models.json`, which only ever had one
Ministral entry). Since the pipeline keys models by filename, this meant
every between-model output (pairwise correlations/independence/
Mann-Whitney, and the pooled chi-square/Kruskal-Wallis tests) silently
treated `mft`/`pvq` as 7-model datasets instead of 6, with one model's
signal double-counted and a spurious "Ministral vs. Ministral-2512"
self-comparison row in every pairwise table (recognizable in hindsight by
suspiciously perfect agreement: r≈1.0, χ²=0/p=1, U=648/p=1). The fix
consolidated both files into one `Ministral-3-8B-Instruct-2512.csv` per
questionnaire, taking numeric `opinion`/`pvq_value` from whichever file
had them and `*_disaggregated` from whichever file had that; every CSV in
`data_analysis/mft/` and `data_analysis/pvq/` has been regenerated against
the corrected 6-model data, and every reported figure in this file and in
`SUMMARY.md` reflects that regeneration.
