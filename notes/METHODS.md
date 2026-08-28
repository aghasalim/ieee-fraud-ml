# Methods

The long-form version of the work. These sections were moved out of the README
unchanged so the README could stay short. Nothing here has been rewritten or
retrimmed: same headings, same tables, same numbers. The figures they refer to
are embedded in the [README](../README.md).

The running decision trail, what I tried and what broke, is in
[NOTES.md](../NOTES.md).

---

## Abstract

IEEE-CIS Fraud Detection is usually reported as a single AUC. This work treats
the evaluation protocol as the object of study instead, holding the model,
features and rows fixed and varying only how the folds are cut. Across 590k real
transactions the resulting AUC spans 0.956 to 0.851, a 10.4-point range produced
entirely by protocol, larger than any modelling gain in the repository. The
mechanism is measured rather than assumed: card overlap between training and
validation falls from 86% under a shuffled K-fold to 31% under a chronological
split with a 30-day gap, and the AUC follows it down.

A feature ablation shows the same pattern from the other side. Target encoding
drives training AUC to exactly 1.000 and costs 3.1 points of validation AUC; a
leaderboard reporting training performance would have ranked it first. The
remaining train, validation gap of roughly 0.11 is characterised rather than
"fixed", because most of it is not closable at this sample size.

Model quality is then reported at the operating points a review team actually
has. At a 1% alert budget the model recovers 47% of fraud; per-segment AUC ranges
from 0.70 to 0.89, so the global figure averages over slices where the model is
substantially weaker.

**Contributions.** (i) A protocol experiment isolating evaluation design from
modelling, with the leakage mechanism measured. (ii) A feature ablation reporting
train, validation gap alongside score. (iii) A predicted leak that was measured and
then withdrawn when the data did not support it. (iv) Error analysis at fixed
review budgets rather than at a threshold nobody operates.

---

## 1. The headline: my own conclusion was wrong, and I caught it


Nothing about the model changes across those six bars. The right panel is the
mechanism: card overlap between train and validation falls from 86% to 31%, and
the AUC falls with it.


I built the validation splitter before touching a model, then ran a 2×2 to find
out how much a bad split flatters you: two split strategies × two ways of
computing a per-card target encoding. First on synthetic data, then, the part
that mattered, again on all 590,540 real transactions.

**Real data, 432 features, `card1` as the entity** (`make leakage-real`):

| split | target encoding | AUC |
|---|---|---|
| shuffled K-fold | global | **0.9557** |
| shuffled K-fold | fold-local | 0.9495 |
| chronological | global | 0.9318 |
| chronological | fold-local | 0.8866 |

On synthetic data I had concluded, confidently, that the leaky *feature* was the
dominant problem by a factor of four. Re-running on real data reversed it:

| leak source | synthetic | real |
|---|---|---|
| global target encoding | +0.2723 | +0.0452 |
| shuffled split | +0.0613 | +0.0629 |
| **ratio feature:split** | **4.4×** | **0.72×** |

Best guess at why: in the synthetic setup the leaky encoding was 1 of 6 features
and carried most of the signal. In the real data it is 1 of 433, competing with
C1, C14, D1, D15 and 339 V-columns that already encode much of what it knows
while the real drift across 182 days is far stronger than the drift term I wrote
into the generator, so shuffling time away helps the model more than I simulated.

The simulation was still worth building: it correctly predicted that both leaks
are real and that the honest number sits far below the flattering one. It got the
*relative magnitudes* wrong, and relative magnitudes were exactly what I'd used
it to conclude.

### Then the EDA invalidated my "honest" number too

The real test set begins **30 days after** the training period ends. My
chronological folds were contiguous, training right up to the day before
validation, which is a materially easier task than the real one. I'd been
calling that "honest" from the start.

| configuration | AUC |
|---|---|
| shuffled + global TE (most flattering) | 0.9557 |
| chronological + fold-local, contiguous | 0.8866 |
| **chronological + fold-local, 30-day embargo** | **0.8513** |

**0.1044 AUC** separates the most flattering configuration from the most
conservative one, roughly the distance between the top of this competition's
leaderboard and its middle.

### Then the leaderboard said the "honest" number was wrong too

All of the above is cross-validation on my own training data, so I submitted to
the competition to check it against a scorer I can't influence. I wrote the
prediction into `submit.py` before submitting: it should land near 0.8513.

It didn't.

| configuration | AUC | vs leaderboard |
|---|---|---|
| shuffled + global TE (most flattering) | 0.9557 | **+0.0471** |
| chronological + global TE | 0.9318 | +0.0232 |
| chronological + fold-local, contiguous | 0.8866 | −0.0220 |
| chronological + fold-local, 30-day embargo | 0.8513 | **−0.0573** |
| **private leaderboard (the actual answer)** | **0.9086** |, |

**The leakage finding survives**: a shuffled split with global target encoding
really is 0.047 optimistic, confirmed externally. **But the number I called most
defensible is off by nearly as much in the other direction**, and the *less*
careful chronological estimate was closer to the truth than the embargoed one I
replaced it with.

The explanation was already in my own results and I'd read it as a caveat rather
than a prediction: validation AUC climbs monotonically with training history
(0.8672 → 0.8855 → 0.9003 across folds), and the submitted model trains on all
182 days, more than any fold ever saw. Extrapolate that line and you get ~0.91.

**Expanding-window CV estimates a model trained on a fraction of your data, not
the one you ship.** The embargo makes it worse, because it removes another 30
days from each fold's training window, a cost the final model never pays. I had
treated "more conservative" as a synonym for "more correct"; a pessimistic
estimate is still a biased one, and on this number I'd have rejected a model that
was materially better than I believed.

Entry 5 of NOTES.md is left unedited rather than quietly rewritten, it was my
reasoning at the time, and it was wrong in an instructive way.

Full reasoning in **[NOTES.md](NOTES.md)**.

---

## 2. What the data actually looks like

| | |
|---|---|
| transactions | 590,540 × 394, joined to 144,233 identity rows |
| fraud rate | **3.499%** |
| identity coverage | **24.4%**: hence a left join, and `has_identity` as a feature |
| columns 50 to 90% missing | **172** (worst single column: 93.6%) |
| fraud rate spread | 5.7× across product codes (C: 11.7%, W: 2.0%) |
| train span | 182 days, test starts 30 days later |

---

## 3. The feature that backfired

Incremental ablation under the honest split, chronological folds, 30-day
embargo, every aggregate fold-local (`make train`):

| features | train AUC | val AUC | delta |
|---|---|---|---|
| raw columns only | 0.9945 | 0.8733 |, |
| + engineered base | 0.9962 | 0.8761 | +0.0028 |
| + frequency encoding | 0.9971 | **0.8839** | **+0.0078** |
| + uid aggregates | 0.9975 | 0.8843 | +0.0004 |
| + target encoding | **1.0000** | **0.8531** | **−0.0312** |

Per-entity target encoding made the model **worse**, and this is the *correct*
version, fold-local, no validation labels. Note the train column hitting
1.0000: with 13,553 cards it hands the model a near-unique key per customer, so
it memorises which customers defrauded during training rather than learning what
fraud looks like. Across a 30-day gap those customers are gone.

Which makes the feature bad in two separate ways, and it took both experiments to
see it: computed globally it *inflates* your score (+0.045), computed correctly
it *lowers* your real one (−0.031). The version that looks best and the version
that works are different features, and neither is the one you want.

`uid` aggregates were the other miss (+0.0004, i.e. noise), most likely
redundant with C1, C14 and D1, D15, which are already per-entity counters built by
people who had the raw data. Kept in the repo; a negative result is still a
result.


## 4. The overfitting gap that mostly isn't fixable

Train AUC 0.9934 to 1.0000 against validation 0.8672 to 0.9003, a gap of 0.09 to 0.13
everywhere. The instinct is to regularise, but the gap barely moves while
validation *improves* (0.1213 → 0.1132), which points at temporal shift rather
than model capacity.

| fold | val AUC | best_iter chosen on val |
|---|---|---|
| 1 (least history) | 0.8672 | **47** |
| 2 | 0.8855 | 118 |
| 3 (most history) | 0.9003 | **395** |

The optimal tree count varies **8×** across folds, so any single `n_estimators`
is wrong for most of them, worth knowing before quoting one tuned number as
"the" model score.

## 5. A leak I predicted, measured, and withdrew

I expected early stopping on the scored fold to be a meaningful hidden leak, and
wrote that into the code before testing it. Measured bonus: **−0.0014, +0.0031,
+0.0001**, a mean of +0.0006, and negative on one fold. The mechanism is real
but the AUC curve is flat near its optimum here, so picking the peak with
hindsight buys nothing. I kept the fixed iteration count as the more defensible
default but dropped the claim that it was protecting anything.

A decision trail containing only confirmed hypotheses is a highlight reel.

## 6. Error analysis

`make error-analysis`. The two weakest segments are also the two largest, and
they overlap, W-product transactions rarely carry an identity record:

| segment | n | AUC | recall@1% |
|---|---|---|---|
| **ProductCD = W** | **355,414** | **0.7030** | 0.141 |
| **no identity record** | **359,603** | **0.7066** | 0.145 |

So the aggregate is propped up by the minority of rows that have identity data,
while ~80% of volume scores near 0.70. The missed-fraud profile agrees:

| | hardest 25% of fraud | easiest 25% |
|---|---|---|
| has identity record | **30.4%** | **97.5%** |
| ProductCD = W | **68.6%** | **0.8%** |
| median C1 (card activity) | 1 | 11 |

Easy fraud is on established cards with identity records. Hard fraud is a quiet
W-product transaction on a card with no history and no identity data, rows that
genuinely carry less information, which no hyperparameter recovers.

**As a review queue**, which is how this would actually be used:

| review budget | recall | precision |
|---|---|---|
| 0.1% (442 cases) | 2.6% | **100.0%** |
| 1% (4,429) | 23.6% | 89.5% |
| 5% (22,145) | **49.5%** | 37.5% |

**Calibration is fine above 25% and badly off below 1%**, where it under-predicts
by nearly 7× (0.0022 predicted vs 0.0149 actual across 347,463 rows, about
5,200 frauds hiding in scores the model calls negligible). Irrelevant for AUC,
decisive for any "auto-approve under 1%" rule, and invisible in every other
metric here.

Two things I refused to conclude: pooled OOF AUC (0.7954) disagrees with mean
per-fold AUC (0.8839) because fold models are differently calibrated and ranking
across them inserts errors deployment never sees, I report per-fold and say why.
And AUC rising across the validation window is confounded with later folds having
more training history, so I can't separate the two.

---


A global AUC averages over segments where the model is materially weaker, and it
summarises thresholds nobody operates at. Both figures above are the same model
seen from a review team's side of the desk.

## 7. The one thing I'd want to be asked about

"Leakage" bundles two different problems, and separating them changed my design:

- **Training on future rows** is always wrong here. The competition's test set is
  the period right after train, so a shuffled split answers an easier question
  than the one being scored.
- **The same card in train and validation is *not* automatically wrong.** My
  first instinct was to force cards disjoint across folds. That instinct was
  wrong: at inference time you genuinely do know a card's history, so banning it
  would validate a system nobody would deploy. It only becomes a bug when the
  *aggregate* is computed globally.

So the splitter enforces time ordering structurally, computes entity aggregates
fold-locally, and **measures** entity overlap instead of banning it.

`split.py` also keeps a `random_kfold` that is deliberately wrong for this data,
because the gap is the evidence. A test asserts it still leaks, otherwise a
refactor could quietly make the headline comparison meaningless while CI stayed
green.

---

## 9. Scope

| | |
|---|---|
| ✅ | Chronological holdout + expanding-window CV, tested |
| ✅ | Entity-overlap and temporal-leak diagnostics |
| ✅ | The 2×2 leakage experiment, on synthetic **and** real data |
| ✅ | Memory-aware loading (590k × 434 doesn't fit at float64) |
| ✅ | EDA on the real data, missingness, imbalance, temporal structure |
| ✅ | LightGBM model + overfitting diagnosis |
| ✅ | Feature engineering log, including what failed |
| ✅ | Streamlit predictor with SHAP explanations + Docker |
| ✅ | Error analysis: segments, review budget, calibration, missed-fraud profile |
| ✅ | Kaggle submission, private LB **0.9086**, which refuted my own prediction |
| ✅ | [Hosted demo](https://ieee-fraud-ml.streamlit.app/): Streamlit Community Cloud |

The unchecked rows are genuinely not done yet. I'm not going to fill them in
with synthetic stand-ins.

---

## 10. Deploy

```bash
make docker && docker run -p 8501:8501 ieee-fraud-ml
```

The image runs as UID 1000 (verified), so it drops straight into any host that
runs containers unprivileged.

**Hugging Face Spaces no longer fits this app on the free tier.** As of this
writing only *static* Spaces are free on `cpu-basic`; Docker and Streamlit
Spaces both return `402 Payment Required` without a PRO subscription, and a
static Space has no Python backend to run LightGBM or SHAP. The `Dockerfile`
here is Spaces-ready (`app_port: 8501`) if you do have PRO.

**Streamlit Community Cloud** runs it unchanged, and is where the
[live demo](https://ieee-fraud-ml.streamlit.app/) is hosted: point it at this
repo with `app/streamlit_app.py` as the entry point. `packages.txt` supplies
`libgomp1`, which LightGBM needs and the default image lacks, without it the
app dies on import with an opaque OSError. Cloud runs Python 3.14 rather than
the 3.12 used locally; the pickled model and SHAP both load fine there, which
is worth checking rather than assuming, since a version mismatch breaks the
unpickle after deploy rather than before.

## Additional detail

Paragraphs kept from the earlier, longer README.

That reproduces the table above. No Kaggle account, no credentials, no cost
the experiment falls back to the synthetic generator when the competition data is
absent.

14 tests. They run the real code path at smaller size rather than mocking it, so
they'd catch the headline claim silently breaking.

To run it on the actual competition data, you need a Kaggle token
(Settings → API → Create New API Token) and to accept the
[rules](https://www.kaggle.com/c/ieee-fraud-detection/rules):

```bash
mkdir -p ~/.kaggle && echo 'KGAT_your_token_here' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```

```bash
make data && make eda && make leakage-real && make train
```

`make eda` regenerates every figure in the "What the data actually looks like"
table into `reports/eda_*.csv`.

```bash
make train-final && make app
```

```bash
docker build -t ieee-fraud-ml . && docker run -p 8501:8501 ieee-fraud-ml
```

`data.py` distinguishes "no token" from "rules not accepted", Kaggle reports both
as a bare 403, which is unhelpful when you're stuck.

MIT, see [LICENSE](LICENSE). The competition data is not redistributed here;
`make data` fetches it from Kaggle under their terms.
