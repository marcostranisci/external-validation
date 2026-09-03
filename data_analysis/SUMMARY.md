# Findings Summary

Six models: Apertus-8B-Instruct, Falcon3-7B-Instruct, Llama-3.1-8B-Instruct,
Ministral-3-8B-Instruct(-2512), Olmo-3-7B-Instruct, Qwen3-8B. Two
questionnaires: MFT (Moral Foundations, 36 items) and PVQ (Portrait
Values, 40 items, mapped to Schwartz's 10 basic values). Full methodology,
per-test caveats, and CSVs are in `data_analysis/README.md`; this is the
cross-experiment synthesis.

## 1. Do models' Likert-scale answers match how humans read their free-text opinions?

Each model gave a numeric Likert answer (`opinion`) *and* a free-text
explanation to every questionnaire item; 9 human annotators rated their
agreement with each model's free-text explanation per item, aggregated
into `external_opinion` (mean) and `annotator_disagreement`.

- **Weak, inconsistent alignment.** Correlating a model's own `opinion`
  against how humans rated its free-text explanation (`external_opinion`)
  is only significant for 1-2 of 6 models per questionnaire (MFT: Apertus
  r=0.60, Ministral r=0.39; PVQ: Qwen r=0.58 only). For most models, the
  Likert self-report and the human-perceived content of the explanation
  don't reliably track each other at all.
- **Models differ enormously from each other in absolute answer level,
  much more than in relative pattern.** Kruskal-Wallis on raw `opinion`:
  η²≈0.45-0.46 (large) in both questionnaires — almost half the variance
  in ratings is "which model is this." One model (Llama, PVQ) gave the
  identical answer to all 40 items — no discrimination at all. Humans'
  ratings of the free text (`external_opinion`) differ by model too, but
  less (η²≈0.11-0.14, medium-large).
- **What humans see in the free text is driven by item content, not by
  which model wrote it.** Despite the level differences above, the
  *pattern* of which items draw more/less human agreement is highly
  consistent across models' explanations in PVQ (pairwise correlation
  median r=0.76, all 15 model pairs significant) and directionally
  consistent (always positive, weaker) in MFT (median r=0.29). A
  chi-square/Mann-Whitney check of whether this pattern's *shape* (not
  just level) depends on the model came back null for `external_opinion`
  either way.
- **Annotators' own moral stance is a minor factor in their evaluation of
  a model's reply.** Correlating each annotator's own reply to an item
  against their evaluation of a model's reply to that same item: pooled
  r=0.31 (MFT) / r=0.40 (PVQ) — R²≈10-16%. Real, replicated in both
  questionnaires, but most of what drives an evaluation isn't the
  annotator's personal opinion.
- **No reliable demographic effect on inter-annotator agreement.**
  Annotators sharing gender or continent-of-birth do not agree with each
  other more than annotators who don't (pooled Mann-Whitney, properly
  powered at 54 same-pairs vs. 162 different-pairs): gender is a clean
  null in both questionnaires (p≥0.74 throughout); continent hits p=0.031
  in MFT on one of four metrics, but doesn't replicate in PVQ (p=0.50) —
  consistent with chance given 8 pooled tests run.

**Bottom line:** models' explicit self-reported opinion and the
human-legible content of their free-text explanation are only loosely
coupled, and that coupling varies a lot by model. The human side of the
process (rating the free text) is internally coherent and driven by the
item's content rather than annotator identity or bias — so the weak
model↔human-rating correlation is a property of the models' outputs, not
noise in the annotation process.

## 2. Does steering with a model's own free-text opinions affect hate-speech recall?

Each model classified the same 500 balanced hate-speech messages
zero-shot, and again once per questionnaire item (76 runs) steered by
*that model's own* free-text opinion on that item (no cross-model
steering in this data — `"prediction_setup": "paired_by_model"`).

- **Steering is far from neutral, and the direction is model-specific.**
  5 of 6 models show a significant shift in recall (Wilcoxon on the
  76-item delta): **boosted** for Ministral (+0.090), Falcon (+0.071),
  Qwen (+0.051), Llama (+0.040); **hurt** for Olmo (-0.096); no net
  average effect for Apertus. The between-model difference itself is
  large (Kruskal-Wallis η²≈0.5; every pairwise model comparison
  significant).
- **Properly powered (McNemar, per model, FDR-corrected within model),
  most individual items do significantly move recall** — 23 to 74 of 76
  items depending on the model. For Falcon/Llama/Ministral/Qwen, almost
  every significant item is an *increase* — steering rarely hurts them
  item-by-item. For Olmo, almost every significant item is a *decrease*.
  **Apertus is genuinely mixed** (13 decreases, 10 increases) — its
  near-zero average was masking real, opposite-signed effects, not an
  absence of effect.
- **What kind of content drives it — and this replicates across two
  independent questionnaires.** No single item survives correction when
  tested across models (0/76, n=6 is too small), but aggregating by
  content category (much more power) shows a clear, convergent pattern:
  - *PVQ (Schwartz values):* every value has a net positive pooled effect
    **except Universalism (-0.048) and Benevolence (-0.044)** —
    Schwartz's "self-transcendence" (concern for others' welfare)
    quadrant.
  - *MFT (Moral Foundations):* **care (+0.017) and purity (+0.007)** are
    far behind the other four foundations (proportionality +0.052,
    loyalty +0.050, equality +0.048, authority +0.034) — `care` is MFT's
    closest analogue to PVQ's Universalism/Benevolence.
  - In both questionnaires, independently, the same two models
    (**Apertus and Olmo**) are the ones actually *hurt* by this specific
    content (negative deltas for care/purity and for
    Universalism/Benevolence), while the other four are helped by it
    just like everything else, only less so.

**Bottom line:** steering with a model's own moral/values opinions
changes hate-speech recall substantially, and the effect is not uniform —
it usually helps, sometimes hurts, and even where it helps, "care for
others" content is consistently the weakest lever (and for two models,
Apertus and Olmo, an actively harmful one) while
achievement/security/tradition/proportionality-type content is the most
reliable booster. That two independently-designed questionnaires converge
on the same two models and the same substantive content category is the
strongest signal in this analysis.
