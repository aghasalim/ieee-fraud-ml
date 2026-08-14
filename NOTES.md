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

## 3. What the real data actually looks like

590,540 transactions × 394 columns, joined left onto 144,233 identity records.
**3.499% fraud.** Identity is present for only **24.4%** of transactions, which
is why the join is left and why `has_identity` is a feature rather than a
filter — an inner join would have silently discarded three quarters of the data.

Missingness is the defining feature of this dataset:

| share of column missing | number of columns |
|---|---|
| under 1% | 111 |
| 1–50% | 109 |
| **50–90%** | **172** |
| over 90% | 2 |

Worst single column is 93.6% missing. Fraud rate varies 5.7× across product
codes (C: 11.7%, W: 2.0%) on wildly different volumes (W is 439,670 of the
590,540 rows).

`TransactionDT` spans 182 days. The important detail is what comes next.

---

## 4. The finding I got wrong on synthetic data

I re-ran the exact 2×2 from entry 2 on the real data — same design, real card
entity (`card1`, 13,553 unique), all 432 features. `make leakage-real`.

| split | target encoding | AUC |
|---|---|---|
| shuffled K-fold | global | **0.9557** |
| shuffled K-fold | fold-local | 0.9495 |
| chronological | global | 0.9318 |
| chronological | fold-local | 0.8866 |

**The ordering reversed.**

| leak source | synthetic | real |
|---|---|---|
| global target encoding | +0.2723 | +0.0452 |
| shuffled split | +0.0613 | +0.0629 |
| **ratio feature:split** | **4.4×** | **0.72×** |

On synthetic data I concluded the leaky feature was the dominant problem by a
factor of four. On real data the leaky *split* is the larger of the two. The
confident sentence I wrote in entry 2 — "fixing your split does not save you" —
is still true in the sense that a global encoding costs a real 0.045, but the
emphasis was wrong, and I'd have carried that wrong emphasis into an interview
if I'd stopped at the simulation.

My best explanation is proportion. In the synthetic setup `card_te` was one of
six features and carried most of the available signal, so contaminating it moved
everything. In the real data it is one of 433 columns, competing with C1–C14,
D1–D15 and 339 V-columns that already encode a lot of what it knows. Meanwhile
the real temporal drift over 182 days is far stronger than the linear drift term
I wrote into the generator, so shuffling time away helps the model much more
than I simulated.

The lesson I'm taking is not "simulations are useless" — the simulation
correctly predicted that *both* leaks are real and that the honest number is far
below the flattering one. It got the *relative magnitudes* wrong, and relative
magnitudes were exactly what I used it to conclude.

---

## 5. Even my "honest" number was optimistic

The EDA turned up something that invalidated my own validation design: the
competition's test set starts **30 days after** the training period ends.

My chronological folds were contiguous — training right up to the day before
validation begins. That is a materially easier task than the real one, where the
model must survive a month of drift before it sees a single scored row. I had
been calling that setup "honest" since entry 1.

Adding an embargo gap matching the real one:

| configuration | AUC |
|---|---|
| shuffled + global TE (most flattering) | 0.9557 |
| chronological + fold-local, contiguous | 0.8866 |
| **chronological + fold-local, 30-day embargo** | **0.8513** |

The embargo costs a further **0.0353**, and the distance from the most
flattering configuration to the most defensible one is **0.1044 AUC**. For
scale, that is roughly the distance between the top of this competition's
leaderboard and the middle of it.

`expanding_window_folds` now takes a `gap` argument. It defaults to 0, but the
docstring says plainly that 0 is not the honest setting for this dataset.

---

## 6. A bug that only appears on pandas 3

`reduce_mem` decided which columns to downcast by matching dtype *names*
(`== "object"`). pandas 3.0 stores strings as a `str` dtype rather than
`object`, so `ProductCD` sailed past the guard and hit `astype("float32")`:

```
ValueError: could not convert string to float: 'W'
```

Fixed by testing `pd.api.types.is_numeric_dtype` instead of comparing dtype
names — asking the question I actually meant rather than one that happened to
be equivalent under pandas 2. The same class of bug is presumably sitting in a
lot of `reduce_mem_usage` copies floating around Kaggle notebooks.

---

## 7. The feature that backfired — and it wasn't even the leaky one

Incremental ablation under the honest split (chronological, 30-day embargo,
everything fold-local). `make train`.

| features | n | train AUC | val AUC | delta |
|---|---|---|---|---|
| raw columns only | 432 | 0.9945 | 0.8733 | — |
| + engineered base (time, amount transforms) | 438 | 0.9962 | 0.8761 | +0.0028 |
| + frequency encoding | 443 | 0.9971 | **0.8839** | **+0.0078** |
| + uid aggregates | 447 | 0.9975 | 0.8843 | +0.0004 |
| + target encoding | 449 | **1.0000** | **0.8531** | **−0.0312** |

**Per-entity target encoding made the model materially worse**, and this is the
*correct* version — fitted fold-locally, no labels from the validation period,
exactly the fix entry 2 said to apply. It still cost 0.0312 AUC.

Look at the train column: 1.0000. Perfect separation on the training rows. With
13,553 card values and a `_uid` of far higher cardinality, the encoding hands
the model a near-unique key per customer, and it memorises the training period's
fraud outcomes per customer instead of learning what fraud looks like. Across a
30-day embargo those memorised customers are largely gone or behaving
differently, so the memorised mapping is worse than useless — it displaces
signal the model would otherwise have used.

So this feature is bad in two independent ways, and I needed both experiments to
see it. Computed globally it *inflates* your score (entry 2, +0.045). Computed
correctly it *lowers* your real score (here, −0.031). The version that looks
best on a dashboard and the version that works are different features, and
neither is the one you want.

**uid aggregates were the other disappointment**: +0.0004, which is noise. The
hypothesis was that deviation from a customer's own average spend would flag
anomalies. My best guess at why it failed is redundancy — C1–C14 and D1–D15 are
already per-entity counters and time-deltas built by people who had the raw
data, so a mean and standard deviation of transaction amount adds little they
do not already carry. I kept the code, since it costs nothing and the negative
result is part of the record.

**Frequency encoding was the only real win** (+0.0078). It says how often a
card, address or email domain appears at all, which is a property of the entity
rather than of its outcomes — nothing to memorise.

---

## 8. Overfitting: an enormous gap that mostly is not fixable

Train AUC sits between 0.9934 and 1.0000 while validation sits between 0.8672
and 0.9003 — a gap of roughly **0.09 to 0.13** in every configuration, before
target encoding makes it worse.

The instinct is to regularise it away. I do not think that is the right reading
here. The gap barely moves across feature sets (0.1213 → 0.1132 as validation
*improves*), which suggests it is dominated by the temporal shift rather than by
model capacity: the model is not memorising noise so much as learning a fraud
distribution that has genuinely changed by the time validation starts.

The per-fold numbers support that:

| fold | val AUC | best_iter chosen on val |
|---|---|---|
| 1 (least history) | 0.8672 | **47** |
| 2 | 0.8855 | 118 |
| 3 (most history) | 0.9003 | **395** |

Validation improves monotonically with more training history, and the optimal
tree count varies **8×** across folds. An early fold with little data saturates
after 47 trees; a late fold is still improving at 395. Any single global
`n_estimators` is therefore wrong for most folds — which is worth knowing before
reporting one tuned number as *the* model's performance.

---

## 9. A leak I predicted, measured, and had to drop

I expected early stopping on the fold being scored to be a meaningful hidden
leak: it picks the iteration count using validation labels, so the reported
number should be a best case rather than an estimate. I wrote that claim into
the code comments before measuring it.

| fold | AUC w/ early stopping | AUC w/ fixed 400 | bonus |
|---|---|---|---|
| 1 | 0.8658 | 0.8672 | **−0.0014** |
| 2 | 0.8886 | 0.8855 | +0.0031 |
| 3 | 0.9004 | 0.9003 | +0.0001 |

Mean bonus ≈ **+0.0006**. It is nothing, and on one fold early stopping was
actively *worse* than the fixed count.

The mechanism I described is real, but its size depends on how sharply AUC peaks
in the iteration count, and here the curve is flat enough near the optimum that
choosing the peak with hindsight buys almost nothing. I have left the fixed
`n_estimators` in place because it is still the more defensible default, but I
am no longer claiming it protects a score that was measurably at risk.

Recording this because a decision trail that only contains confirmed hypotheses
is not a decision trail, it is a highlight reel.

---

## 10. Error analysis, starting with a number that contradicted itself

`make error-analysis`, on out-of-fold predictions from the honest split
(442,905 scored rows — the seed-history block is never validated).

**First surprise: pooled OOF AUC is 0.7954, but the mean per-fold AUC is
0.8839.** Same predictions, same rows, 0.088 apart.

Neither is a bug. Averaging per-fold AUC asks "how well does a model rank within
its own scoring period". Pooling asks the model to rank a week-3 transaction
against a week-20 one — but those scores came from *different* fold models with
different calibration, so the comparison inserts errors that would never happen
in deployment, where one model scores everything. The pooled number is the
pessimistic one here, and it is the wrong question.

I am reporting per-fold AUC as the headline and using pooled predictions only
for the *segment* analysis below, where every comparison is within-segment and
mostly within-fold. Worth writing down because picking whichever of two
defensible numbers looks better is precisely the habit this project exists to
avoid.

### The model is good at a small slice and weak on the bulk

| segment | n | fraud rate | AUC | recall@1% |
|---|---|---|---|---|
| **ProductCD = W** | **355,414** | 2.08% | **0.7030** | 0.141 |
| **no identity record** | **359,603** | 2.13% | **0.7066** | 0.145 |
| ProductCD = H | 11,639 | 9.86% | 0.7272 | 0.100 |
| ProductCD = S | 8,464 | 6.94% | 0.7320 | 0.138 |

The two weakest segments are also the two biggest, and they overlap: W
transactions rarely carry an identity record. So the aggregate score is being
propped up by the minority of transactions that have identity data, while ~80%
of the volume is scored at an AUC near 0.70.

The profile of missed fraud says the same thing from the other end:

| | hardest 25% of fraud | easiest 25% |
|---|---|---|
| share with identity record | **30.4%** | **97.5%** |
| share ProductCD = W | **68.6%** | **0.8%** |
| median C1 (card activity) | 1 | 11 |
| median amount | $82.32 | $49.48 |

Fraud the model catches easily is on established cards (C1 = 11) with identity
records, on non-W products. Fraud it misses is a quiet W-product transaction on
a card with almost no history and no identity data. That is not a tuning
problem — those rows genuinely carry less information, and no hyperparameter
recovers a signal that was never collected.

### What that means for a review queue

A 0.5 threshold is not how this gets used; a fraud team works a queue of fixed
size.

| review budget | fraud caught | recall | precision |
|---|---|---|---|
| 0.1% (442 cases) | 442 | 2.6% | **100.0%** |
| 1% (4,429) | 3,962 | 23.6% | 89.5% |
| 5% (22,145) | 8,306 | **49.5%** | 37.5% |

The top 0.1% is perfect — 442 alerts, 442 frauds. Reviewing 1% of transactions
catches a quarter of fraud at ~90% precision, which is a genuinely useful
operating point. Catching half requires reviewing 5% and accepting that two
thirds of the queue is clean.

### Calibration is badly off at the low end

| predicted | n | mean predicted | actual |
|---|---|---|---|
| 0–1% | 347,463 | 0.0022 | **0.0149** |
| 1–5% | 68,752 | 0.0210 | **0.0416** |
| 25–50% | 3,053 | 0.3512 | 0.3741 |
| 50–75% | 1,584 | 0.6232 | 0.6250 |
| 75–100% | 4,263 | 0.9288 | 0.9017 |

Above 25% it is well calibrated. Below 1% it under-predicts by nearly **7×** —
and that bucket holds 347,463 rows at a real 1.49% fraud rate, roughly 5,200
frauds sitting in scores the model calls negligible. For ranking (AUC) this is
irrelevant. For any decision phrased as "auto-approve anything under 1%" it
matters enormously, and it is invisible in every metric reported so far.

### One thing I could not conclude

AUC by week rises across the validation window (≈0.85 early, ≈0.90 late). The
tempting read is that the model handles later periods better. It is confounded:
later weeks belong to later folds, which trained on more history. I cannot
separate "later period is easier" from "more training data helps" with this
design, so I am not claiming either.

---

## Still outstanding

- Hyperparameter tuning — deliberately untouched, since every number above is
  about validation design, features and calibration rather than model capacity.
  The segment analysis suggests tuning is not where the remaining gains are.
- A hosted demo. The Docker image runs locally and is verified.
