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

## Still outstanding

- **An overfitting diagnosis** — train/validation divergence in LightGBM and
  what I changed in response.
- **A feature that backfired**, with my best guess as to why.
- **Error analysis** on the worst-scored cases.
- **The deployed predictor** with SHAP explanations.
