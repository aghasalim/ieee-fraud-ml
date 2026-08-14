# Real-World Tabular ML — a decision trail, not a leaderboard score

[![ci](https://github.com/aghasalim/ieee-fraud-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/ieee-fraud-ml/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Working the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
competition end to end, by a third-year Applied Computer Science (AI) student.
The thing I'm actually trying to produce here is **[NOTES.md](NOTES.md)** — a
written record of what I tried, what broke, and what I caught. A model that's
slightly worse with an honest trail behind it is worth more than a good score
with no story.

## The headline: my own conclusion was wrong, and I caught it

I built the validation splitter before touching a model, then ran a 2×2 to find
out how much a bad split flatters you: two split strategies × two ways of
computing a per-card target encoding. First on synthetic data, then — the part
that mattered — again on all 590,540 real transactions.

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
C1–C14, D1–D15 and 339 V-columns that already encode much of what it knows —
while the real drift across 182 days is far stronger than the drift term I wrote
into the generator, so shuffling time away helps the model more than I simulated.

The simulation was still worth building: it correctly predicted that both leaks
are real and that the honest number sits far below the flattering one. It got the
*relative magnitudes* wrong, and relative magnitudes were exactly what I'd used
it to conclude.

### Then the EDA invalidated my "honest" number too

The real test set begins **30 days after** the training period ends. My
chronological folds were contiguous — training right up to the day before
validation — which is a materially easier task than the real one. I'd been
calling that "honest" from the start.

| configuration | AUC |
|---|---|
| shuffled + global TE (most flattering) | 0.9557 |
| chronological + fold-local, contiguous | 0.8866 |
| **chronological + fold-local, 30-day embargo** | **0.8513** |

**0.1044 AUC** separates the most flattering configuration from the most
defensible one — roughly the distance between the top of this competition's
leaderboard and its middle.

Full reasoning in **[NOTES.md](NOTES.md)**.

---

## What the data actually looks like

| | |
|---|---|
| transactions | 590,540 × 394, joined to 144,233 identity rows |
| fraud rate | **3.499%** |
| identity coverage | **24.4%** — hence a left join, and `has_identity` as a feature |
| columns 50–90% missing | **172** (worst single column: 93.6%) |
| fraud rate spread | 5.7× across product codes (C: 11.7%, W: 2.0%) |
| train span | 182 days, test starts 30 days later |

---

## The feature that backfired

Incremental ablation under the honest split — chronological folds, 30-day
embargo, every aggregate fold-local (`make train`):

| features | train AUC | val AUC | delta |
|---|---|---|---|
| raw columns only | 0.9945 | 0.8733 | — |
| + engineered base | 0.9962 | 0.8761 | +0.0028 |
| + frequency encoding | 0.9971 | **0.8839** | **+0.0078** |
| + uid aggregates | 0.9975 | 0.8843 | +0.0004 |
| + target encoding | **1.0000** | **0.8531** | **−0.0312** |

Per-entity target encoding made the model **worse**, and this is the *correct*
version — fold-local, no validation labels. Note the train column hitting
1.0000: with 13,553 cards it hands the model a near-unique key per customer, so
it memorises which customers defrauded during training rather than learning what
fraud looks like. Across a 30-day gap those customers are gone.

Which makes the feature bad in two separate ways, and it took both experiments to
see it: computed globally it *inflates* your score (+0.045), computed correctly
it *lowers* your real one (−0.031). The version that looks best and the version
that works are different features, and neither is the one you want.

`uid` aggregates were the other miss (+0.0004, i.e. noise) — most likely
redundant with C1–C14 and D1–D15, which are already per-entity counters built by
people who had the raw data. Kept in the repo; a negative result is still a
result.

## The overfitting gap that mostly isn't fixable

Train AUC 0.9934–1.0000 against validation 0.8672–0.9003 — a gap of 0.09–0.13
everywhere. The instinct is to regularise, but the gap barely moves while
validation *improves* (0.1213 → 0.1132), which points at temporal shift rather
than model capacity.

| fold | val AUC | best_iter chosen on val |
|---|---|---|
| 1 (least history) | 0.8672 | **47** |
| 2 | 0.8855 | 118 |
| 3 (most history) | 0.9003 | **395** |

The optimal tree count varies **8×** across folds, so any single `n_estimators`
is wrong for most of them — worth knowing before quoting one tuned number as
"the" model score.

## A leak I predicted, measured, and withdrew

I expected early stopping on the scored fold to be a meaningful hidden leak, and
wrote that into the code before testing it. Measured bonus: **−0.0014, +0.0031,
+0.0001** — a mean of +0.0006, and negative on one fold. The mechanism is real
but the AUC curve is flat near its optimum here, so picking the peak with
hindsight buys nothing. I kept the fixed iteration count as the more defensible
default but dropped the claim that it was protecting anything.

A decision trail containing only confirmed hypotheses is a highlight reel.

## Error analysis: strong on a slice, weak on the bulk

`make error-analysis`. The two weakest segments are also the two largest, and
they overlap — W-product transactions rarely carry an identity record:

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
W-product transaction on a card with no history and no identity data — rows that
genuinely carry less information, which no hyperparameter recovers.

**As a review queue**, which is how this would actually be used:

| review budget | recall | precision |
|---|---|---|
| 0.1% (442 cases) | 2.6% | **100.0%** |
| 1% (4,429) | 23.6% | 89.5% |
| 5% (22,145) | **49.5%** | 37.5% |

**Calibration is fine above 25% and badly off below 1%**, where it under-predicts
by nearly 7× (0.0022 predicted vs 0.0149 actual across 347,463 rows — about
5,200 frauds hiding in scores the model calls negligible). Irrelevant for AUC,
decisive for any "auto-approve under 1%" rule, and invisible in every other
metric here.

Two things I refused to conclude: pooled OOF AUC (0.7954) disagrees with mean
per-fold AUC (0.8839) because fold models are differently calibrated and ranking
across them inserts errors deployment never sees — I report per-fold and say why.
And AUC rising across the validation window is confounded with later folds having
more training history, so I can't separate the two.

---

## The one thing I'd want to be asked about

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
because the gap is the evidence. A test asserts it still leaks — otherwise a
refactor could quietly make the headline comparison meaningless while CI stayed
green.

---

## Running it

```bash
make setup && make validate
```

That reproduces the table above. No Kaggle account, no credentials, no cost —
the experiment falls back to the synthetic generator when the competition data is
absent.

```bash
make test
```

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

Then the predictor:

```bash
make train-final && make app
```

```bash
docker build -t ieee-fraud-ml . && docker run -p 8501:8501 ieee-fraud-ml
```

`data.py` distinguishes "no token" from "rules not accepted" — Kaggle reports both
as a bare 403, which is unhelpful when you're stuck.

---

## What's here, and what isn't

| | |
|---|---|
| ✅ | Chronological holdout + expanding-window CV, tested |
| ✅ | Entity-overlap and temporal-leak diagnostics |
| ✅ | The 2×2 leakage experiment, on synthetic **and** real data |
| ✅ | Memory-aware loading (590k × 434 doesn't fit at float64) |
| ✅ | EDA on the real data — missingness, imbalance, temporal structure |
| ✅ | LightGBM model + overfitting diagnosis |
| ✅ | Feature engineering log, including what failed |
| ✅ | Streamlit predictor with SHAP explanations + Docker |
| ✅ | Error analysis: segments, review budget, calibration, missed-fraud profile |
| ⬜ | Hosted demo — see Deploy below |

The unchecked rows are genuinely not done yet. I'm not going to fill them in
with synthetic stand-ins.

---

## Deploy

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

**Streamlit Community Cloud** is the free option that runs this unchanged:
point it at this repo with `app/streamlit_app.py` as the entry point.
`packages.txt` supplies `libgomp1`, which LightGBM needs and which the default
image lacks — without it the app dies on import with an opaque OSError.

## Layout

```
src/fraud/
  config.py                      experiment knobs in one place
  data.py                        download, join identity, downcast dtypes
  split.py                       the load-bearing file — chronological CV,
                                 entity overlap, embargo gap
  experiments/validation_gap.py  the 2×2 on synthetic data
  experiments/leakage_real.py    the 2×2 on 590k real transactions
tests/                           14 tests, synthetic data only
NOTES.md                         the decision trail
```

## License

MIT — see [LICENSE](LICENSE). The competition data is not redistributed here;
`make data` fetches it from Kaggle under their terms.
