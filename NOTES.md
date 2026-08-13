# Decision trail

A running log of what I tried, what broke, and what I caught. Newest entries at
the bottom. This file is the point of the project — the model is downstream of it.

Ground rule I set for myself: **no number in this file that I did not personally
run.** Where a result comes from synthetic data rather than the competition data,
it says so in the heading, not in a footnote.

---

## 1. I built the validation splitter before the model, on purpose

Every score a project like this produces is downstream of one decision: how the
data was partitioned. Get it wrong and every number afterwards is fiction — but
it's *confident* fiction, because a leaky split doesn't crash, it just quietly
hands you a better number than you earned.

So `src/fraud/split.py` came first, with tests, before any modelling.

While writing it I had to separate two things that get bundled together as
"leakage", and the distinction turned out to matter more than I expected:

**Temporal leakage** — training on rows that happen *after* the rows you're
validating on. Unambiguously wrong here. The competition's test set is the period
immediately after train, so a shuffled split is answering an easier question than
the one you get scored on.

**Entity overlap** — the same card appearing in both train and validation. My
first instinct was that this is also a bug and I should force cards to be
disjoint across folds. **That instinct was wrong**, and I'm glad I thought about
it before writing the code. At real inference time you genuinely *do* know a
card's history — a fraud system scoring a transaction has that card's past
behaviour available. Forcing disjoint cards would be validating a system nobody
would ever deploy. Overlap only becomes a bug when the *feature* is computed over
the whole dataset, because then training rows have absorbed statistics from the
validation period. That's future information wearing a per-entity disguise.

So: temporal ordering enforced structurally, entity aggregates computed
fold-locally, entity overlap measured and reported rather than banned.

I kept a deliberately-wrong `random_kfold` in the codebase to measure the gap,
and there's a test asserting it *still leaks* — otherwise a future refactor could
silently make the comparison vacuous while the suite stayed green.

---

## 2. The leak I expected was real, but it was the small one — synthetic data

**Status: measured on synthetic data.** The Kaggle competition data isn't
downloaded yet (needs my API token and rule acceptance). Everything below is a
real measurement from a real run, on data I generated to have the two properties
that make IEEE-CIS awkward — recurring cards with persistent latent risk, and a
base rate that drifts over time. It demonstrates the *mechanism*. It is not a
result about fraud, and I'm not going to present it as one.

I ran a 2×2: two split strategies × two ways of building a per-card target
encoding. 60,000 rows, 4,000 cards (~15 transactions each), 4.0% positive rate.

| split | target encoding | AUC | trains on future? |
|---|---|---|---|
| shuffled K-fold | global | **0.8975** | yes |
| shuffled K-fold | fold-local | 0.6779 | yes |
| chronological | global | 0.8889 | no |
| **chronological** | **fold-local** | **0.6166** | no |

Only the last row is defensible. The gap between best-looking and honest is
**+0.2809 AUC**.

**What I expected:** that the shuffled split would be the main villain. That's the
mistake everyone warns about, and it's the one I was primed to look for.

**What actually happened:** it was the smaller of the two effects, by a lot.

| leak source | isolated effect |
|---|---|
| global target encoding (split held correct) | **+0.2723** |
| shuffled split (encoding held correct) | +0.0613 |

The feature leak is about **4.4× larger** than the split leak.

The line that actually changed how I think: **chronological split + global
encoding still reads 0.8889.** You can do the famous thing right — respect time
order, feel good about it — and still be off by 0.27 AUC because a feature was
computed with `groupby` over the entire dataframe before splitting. Fixing your
split does not save you. It's the more seductive failure, too, because you've
already done the bit that gets talked about, so you stop looking.

This is why the 2×2 exists rather than a single before/after number. One
comparison would have shown "leaky 0.8975 vs honest 0.6166" and I'd have
attributed it to the split and been wrong about the mechanism.

**Caveat I want on the record:** the effect sizes are a property of my generator.
~15 transactions per card and a 4% base rate were chosen to resemble IEEE-CIS,
but the *magnitudes* would move if I changed either. What I'd defend is the
ordering and the mechanism, not the specific 0.27.

---

## Pending — needs the competition data

These are the entries this project actually promises, and they can't be written
from synthetic data. Placeholders, not claims:

- **EDA**: missingness structure (many V-columns are >90% NaN), the 4% imbalance,
  what `TransactionDT` actually is, mixed-type and high-cardinality columns.
- **A leakage catch on the real data** — the above re-run on IEEE-CIS, where I
  expect the numbers to differ and the ordering to be the interesting question.
- **An overfitting diagnosis** — train/validation divergence in LightGBM and what
  I changed.
- **A feature that backfired** — with my best guess as to why.
- **Error analysis** on the worst-scored cases.
