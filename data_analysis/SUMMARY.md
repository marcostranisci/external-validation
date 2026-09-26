# Findings Summary

Six models: Apertus-8B-Instruct, Falcon3-7B-Instruct, Llama-3.1-8B-Instruct,
Ministral-3-8B-Instruct-2512, Olmo-3-7B-Instruct, Qwen3-8B. Two
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
  is nominally significant for 1-2 of 6 models per questionnaire (MFT:
  Apertus r=0.60 p=9.3e-5, Ministral r=0.39 p=0.019; PVQ: Qwen r=0.58
  p=7.5e-5 only), but after Benjamini-Hochberg correction (across the 6
  models per questionnaire) only **one model per questionnaire survives**:
  Apertus for MFT (p_fdr=0.00056) and Qwen for PVQ (p_fdr=0.00037) —
  Ministral's MFT case does not (p_fdr=0.058). For every other model, the
  Likert self-report and the human-perceived content of the explanation
  don't reliably track each other at all.
- **Models differ enormously from each other in absolute answer level,
  much more than in relative pattern.** Kruskal-Wallis on raw `opinion`:
  η²=0.49 (MFT) / 0.46 (PVQ), both large — almost half the variance in
  ratings is "which model is this." One model (Llama, PVQ) gave the
  identical answer to all 40 items — no discrimination at all. Humans'
  ratings of the free text (`external_opinion`) differ by model too, but
  less (η²=0.11 MFT / 0.14 PVQ, medium-large).
- **What humans see in the free text is driven by item content, not by
  which model wrote it — and much more so on PVQ than on MFT.** Despite
  the level differences above, the *pattern* of which items draw more/less
  human agreement is highly consistent across models' explanations in PVQ
  (pairwise correlation median r=0.76, all 15 model pairs significant) and
  directionally consistent but much weaker in MFT (median r=0.29, only
  6/15 significant). This PVQ-vs-MFT gap is itself statistically real
  (Fisher-z Mann-Whitney/Welch test, p≈4e-6) — but the same test on models'
  own `opinion` (not the human ratings) shows only a nominal, *not*
  significant, gap (median r=0.36 PVQ vs. 0.25 MFT, p≈0.14–0.19). So the
  finding is specifically about how humans read the models' free-text
  explanations, not (yet demonstrably) about the models' own self-report —
  plausibly because PVQ's third-person trait-similarity items ("how much
  is this person like you") are a more mechanical judgment than MFT's
  first-person moral endorsements ("I admire...", "I believe society
  should..."), which are more inherently divisive. A chi-square/Mann-Whitney
  check of whether the PVQ/MFT pattern's *shape* (not just level) depends
  on the model came back null for `external_opinion` either way.
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

- **Before steering enters the picture at all: models agree with each
  other substantially but incompletely, with no dramatic outlier and one
  consistently closest pair.** Pairwise Cohen's kappa on the 500 zero-shot
  predictions ranges 0.40–0.70 across the 15 model pairs (every pair's
  association is statistically overwhelming, p≈0, but even the closest
  pair still disagrees on ~13-15% of messages). **Llama–Ministral** and
  **Falcon–Qwen** are consistently the most-agreeing pair, both zero-shot
  and pooled across all steered predictions (κ up to ~0.70); **Llama–Olmo**
  and **Ministral–Olmo** are consistently the least (κ drops to ~0.25 under
  steering, "fair" agreement only). Per-model average agreement with the
  panel is compressed (0.54–0.59 zero-shot) — no model is a dramatic
  outlier in how much it agrees with the rest, at least in this free-text
  file. (§3 below revisits this: whether *which pair* agrees most under
  steering itself depends on how the belief is elicited.)
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
- **Binning the 76 items themselves by steerability (not just by
  recall-delta significance) gives the cleanest, best-powered confirmation
  of the Universalism finding in the whole analysis.** Splitting each
  questionnaire's items into low/mid/high-steerability tertiles by mean
  flip rate: **all 6 of PVQ's Universalism items land in the
  high-steerability tier** (vs. a 30% base rate) — a one-vs-rest Fisher
  exact test gives p=0.00024, and it's the only PVQ value to survive BH
  correction across all 10 (p_fdr=0.0024). This is the one item-level
  result in the entire hate-speech analysis that survives correction
  (test 3's per-item Wilcoxon test found 0/76 survive, because n=6 models
  per item is too small a sample; counting whole items into a category
  instead has real power). MFT shows the same directional pattern without
  reaching significance: equality is the most concentrated in the
  high-steerability tier (5/6 items, p_fdr≈0.06) and care/purity the least
  (0-1/6) — consistent with, but not as strong as, the PVQ result.
  **Caveat: this is a narrow-band ranking, not a qualitative split** — all
  76 items' flip rates fall within roughly 0.08-0.17, so
  "high-steerability" means an item ranks in the top third (only 2.6-3.8
  points of absolute flip-rate gap between the low and high tiers'
  means), not that it moves substantially more messages in absolute
  terms. Read the finding as "which items rank where," not "how much more
  disruptive these items are."
- **The mechanism behind the recall shift is raw label-flipping, and both
  how much a model flips and which way it flips are strongly
  model-dependent — not just how its recall moves on average.** Counting
  every predicted-label change vs. zero-shot (not just changes to
  positive-class correctness): total flips range from 3025 (Llama) to
  6030 (Olmo) per model across the 76 items, and a Kruskal-Wallis test on
  per-item flip rate confirms this is a real, large effect (η²=0.40,
  p=1e-37), not noise. Direction is just as strongly model-specific
  (chi-square on model × flip-direction, Cramer's V=0.62, p≈0): Olmo
  flips overwhelmingly *away* from "hate" (4730 vs. 1300), Ministral
  overwhelmingly *toward* it (5117 vs. 128), Qwen/Falcon/Llama skew
  toward "hate" more mildly, and Apertus is the only model with a
  roughly even split (1572 vs. 1602) — mirroring its mixed item-level
  effects. Breaking flips down by content category reproduces the
  Universalism/Benevolence finding at the mechanism level: Olmo's
  Universalism items alone cause 992 flips, 3x its next-highest category
  and the single largest (model, value) cell in the whole study.
- **Model differences aren't spread evenly across messages — they
  concentrate on the messages that are steerable at all; stable messages
  are stable precisely because models agree on them.** Computing each
  model's own flip rate on a message (over that model's 76 items) and the
  variance between those 6 rates: a message's overall steerability
  correlates with how much models diverge from each other on it at r=0.93
  (free-text) / r=0.91 (verbalized, shared baseline). Since flip rates are
  bounded in [0, 1], some of that is mechanical (a message nobody flips
  can't show disagreement by definition) — but normalizing by the
  theoretical variance ceiling given the mean, the correlation survives at
  r=0.82 / r=0.69 (both p<1e-58): models don't just have more *room* to disagree on
  steerable messages, they actually use more of that room. Combined with
  the tier-overlap finding below (§3), the two-part picture is: messages
  that don't flip are stable both *within* a steering method (low
  variance across models) and *across* methods (free-text vs. verbalized,
  77.8% stay `no_flip`), while messages that do flip are exactly where
  both model-to-model and method-to-method differences show up.

**Bottom line (free-text steering specifically — see §3 below):**
steering with a model's own *free-text* moral/values opinion changes
hate-speech recall substantially, and the effect is not uniform — it
usually helps, sometimes hurts, and even where it helps, "care for
others" content is consistently the weakest lever (and for two models,
Apertus and Olmo, an actively harmful one) while
achievement/security/tradition/proportionality-type content is the most
reliable booster. That two independently-designed questionnaires converge on the same two
models and an analogous content category was, at the time, the strongest
signal in this analysis — §3's more careful, baseline-shift-invariant
re-test shows this convergence is only partly real: MFT's "care/purity
are relatively weak levers" *does* survive a change in elicitation
method, but PVQ's "Universalism/Benevolence are the exception" does not.
So the general, questionnaire-independent version of this finding does
not hold — it should be read as a genuine (if not fully understood)
property of MFT's care/purity foundations specifically, plus a separate,
elicitation-method-specific artifact for PVQ's Universalism/Benevolence.

## 3. Does the steering *mechanism* matter — free-text opinion vs. the verbalized questionnaire item itself?

A second steering run (`hs_detection/implicit_hate_verbalized_all_models.json`)
uses the same models, 500 messages, 76 items, and `paired_by_model`
setup as §2, replacing the free-text belief with a one-line
verbalization of the model's own Likert rating (e.g. `"Caring for people
who have suffered is an important virtue" describes you extremely
well.`). **It is not a clean isolated manipulation of belief content,
though**: `zero_shot_prompt` and `belief_prompt` also differ in wording
between the two files, and prompt wording alone (no belief/steering
involved) shifts zero-shot recall for some models, most dramatically for
Apertus. **Because of this, every result below compares the verbalized
condition's steered recall against the free-text condition's zero-shot
baseline, not the verbalized file's own** — there is no own-baseline
verbalized analysis anywhere in this codebase to confound belief content
with prompt wording. This is the most conservative comparison available:
it removes any advantage a model could get from an artificially depressed
or inflated baseline within its own file.

- **Direction flips for 2 of 6 models, cleanly; Apertus is a different
  kind of case.** Olmo goes from significantly *hurt* (−0.096) to
  significantly *helped* (+0.109); Qwen goes from significantly *helped*
  (+0.051) to significantly *hurt* (−0.113) — genuine sign flips. Apertus
  has no real free-text effect to flip away from (−0.003, n.s.), but a
  large, significant *negative* effect under verbalized steering (−0.419)
  — the largest magnitude of any model-condition pair in this analysis.
  Falcon, Llama, and Ministral keep the same direction under both
  conditions (bigger under verbalized for Falcon, smaller for Llama and
  Ministral) — there's no universal rule that verbalized steering is
  stronger or weaker.
- **The two conditions barely agree on *which items* matter.** Per-model
  correlation between the two conditions' per-item recall deltas ranges
  from r=−0.03 to r=0.25 — essentially uncorrelated, and the one nominally
  significant case (Falcon, r=0.25, p=0.03) wouldn't survive correcting
  for 6 models. Same model, same items, same labels — different belief
  phrasing produces a near-unrelated pattern of which items move the
  needle.
- **The "which model is most volatile" ranking is a property of the
  elicitation method, not a stable model trait.** A 2×2 chi-square test
  (condition × flip-direction) on each model's proportion of flips toward
  "hate" is significant for every model even after BH correction (weakest
  case still p_fdr≈2e-39, for Llama) — including Falcon, Ministral, and
  Llama, which keep the *same* majority direction under both conditions
  but still shift its strength by a statistically real amount. Apertus's
  proportion of flips toward "hate" collapses from 49.5% (free-text) to
  essentially 0% (0.04%) under verbalized steering, consistent with its
  large negative recall shift above. The relative flip-magnitude ranking
  is unstable too: Apertus jumps from 5th to 1st most flip-prone model
  (rank shift +4), Falcon rises 2 ranks, Ministral and Olmo fall (−3, −2),
  Llama is unchanged, and Qwen shifts by only 1 rank. So "model X is
  unusually steerable" is not a fact about model X in isolation — it
  depends on how the belief was elicited.
- **Which pair of models agrees most under steering is itself
  elicitation-method-dependent.** Within the free-text file, zero-shot
  and steered agreement rankings across the 15 model pairs correlate
  significantly (Spearman ρ=0.77, p=0.0008) — a pair that agrees
  zero-shot tends to also agree once steered. But the *steered* ranking
  does not carry over *between* free-text and verbalized steering
  (ρ=0.175, p=0.53, n.s.) — so "which two models see hate speech the same
  way under moral-belief steering" is not a fixed trait of the model
  pair, it depends on how the belief was elicited, same as the recall and
  flip-magnitude findings above.
- **At the message level, the same picture holds: "fair" agreement, not
  a shared or an unrelated pattern.** Binning each message's flip rate
  into `no_flip`/`mild_flip`/`strong_flip` tiers under each condition and
  cross-tabulating: 58.2% of the 500 messages land in the same tier under
  both, against a 35.8% chance baseline (Cohen's κ=0.35, "fair" —
  significant, χ²=205.5, p=2e-43, but far short of strong agreement).
  Stability is uneven: messages that never flip under free-text mostly
  stay that way (77.8%), and strongly-flipping messages mostly stay
  volatile (86.1%) — but mildly-flipping messages are the least stable
  (40.4% stay mild). This mirrors and adds mechanism to the raw-correlation
  finding: absolute prediction tendency is fairly robust across steering
  methods (r=0.96), but how much a given message's *label gets perturbed*
  by steering (`flip_rate`, r=0.63) is only moderately consistent, and
  that inconsistency doesn't average out — it lands disproportionately on
  the mid-volatility messages.
- **Correction: the §2 exception disappears for PVQ, but not for MFT —
  pooling models' absolute mean deltas was the wrong lens.** Pooling all
  6 models' *absolute* deltas under verbalized steering makes every MFT
  foundation and every PVQ value look net-negative, dominated by Apertus's
  large effect. Excluding Apertus flips this picture for MFT entirely:
  every foundation stays positive under both conditions, and 5/6 are
  actually *larger* under verbalized (only proportionality shrinks) — so
  Apertus alone drove the pooled-negative appearance for MFT. PVQ is
  genuinely mixed even excluding Apertus: Universalism and Benevolence
  flip from negative under free-text to positive under verbalized, while
  the other 8 values stay positive under both, split roughly evenly
  between larger and smaller under verbalized. Averaging absolute deltas
  is the wrong test when models differ hugely in overall level; ranking
  each category *within* each model instead (rank 1 = that model's weakest
  lever) sidesteps this, since it's mathematically invariant to any
  per-model baseline shift. That ranking's correlation between free-text
  and verbalized steering is **ρ=0.93 (p=0.008) for MFT** — purity and
  care remain relatively weak levers for most models under *both*
  conditions, a pattern stable across elicitation method. **PVQ is
  genuinely different: ρ=0.22 (p=0.53, n.s.)** — which category is a
  model's weakest PVQ lever reshuffles between elicitation methods. So the
  "exception disappears under a different elicitation method" story is
  correct for PVQ, but for MFT the more accurate statement is: care/purity
  are consistently weak levers regardless of elicitation method.
- **The item-steerability-tier confirmation of the Universalism finding
  (§2) does not replicate under verbalized steering, and item-level tier
  agreement between conditions is weak to slightly below chance overall.**
  Universalism's striking free-text result — all 6 items in the
  high-steerability tier (p_fdr=0.0024), the one BH-corrected-significant
  item-level finding in the whole free-text analysis — does not appear at
  all under verbalized steering (0/6 Universalism items land in the
  high-steerability tier there; p_fdr=0.51). More broadly, only 28.9% of
  the 76 items land in the same steerability tier under both conditions
  (vs. a 33.4% chance baseline — κ=−0.07, not significantly different
  from chance, p=0.17). Only 4 items are `high_steerability` under *both*
  conditions: 2 MFT (`authority`, `proportionality`) and 2 PVQ
  (`Stimulation`, `Tradition`) — no foundation or value is
  over-represented among them (all p_fdr≥0.93). So under the
  confound-controlled comparison, there is no item, and no category, that
  is reliably high-steerability regardless of elicitation method — a
  stronger null than the earlier (own-baseline) version of this check,
  which had found a weak `equality` concentration.

**Bottom line:** the mechanism of eliciting the moral opinion is not a
neutral implementation detail — it changes which models are helped vs.
hurt and which items matter, and it determines whether the PVQ
Universalism/Benevolence exception exists at all (it's specific to
free-text elicitation — ρ=0.22, n.s., between conditions). It does
*not*, however, determine the MFT care/purity finding, which replicates
across elicitation methods at the level that matters (each model's
*relative* ranking of its weakest foundation, ρ=0.93, p=0.008) even
though each model's *overall* effect level still shifts with elicitation
method just like everything else. So the properly general claim is
two-layered: moral-belief steering reliably shifts hate-speech recall,
substantially and in a model-specific direction that depends on how the
belief is presented (robust across both questionnaires) — but *which
specific content is the weak lever* is itself sometimes a stable property
of the content (MFT's care/purity) and sometimes an artifact of the
elicitation method (PVQ's Universalism/Benevolence), and the two
questionnaires disagree on which case they're in.
