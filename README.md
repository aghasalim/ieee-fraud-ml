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

> ### Status, up front
>
> **The competition data is not downloaded yet** — it needs a Kaggle API token
> and rule acceptance. What's built and tested so far is the validation
> methodology, and one real experiment run on **synthetic** data.
>
> Every number below was produced by running the code in this repo. None of them
> are about fraud yet. I'd rather publish that sentence than a placeholder score.

---

## The finding so far

I built the validation splitter before touching a model, then ran a 2×2 to find
out how much a bad split actually flatters you: two split strategies × two ways
of computing a per-card target encoding.

*60,000 synthetic rows, 4,000 recurring cards, 4.0% positive rate — a generator
built to have the two properties that make this dataset awkward. Reproduce with
`make validate`.*

| split | target encoding | AUC | trains on future? |
|---|---|---|---|
| shuffled K-fold | global | **0.8975** | yes |
| shuffled K-fold | fold-local | 0.6779 | yes |
| chronological | global | 0.8889 | no |
| **chronological** | **fold-local** | **0.6166** | no |

Only the bottom row is defensible. Best-looking to honest is **+0.2809 AUC**.

I expected the shuffled split to be the main problem — it's the mistake everyone
warns about. It wasn't:

| leak source | isolated effect |
|---|---|
| global target encoding (split held correct) | **+0.2723** |
| shuffled split (encoding held correct) | +0.0613 |

**The feature leak is ~4.4× the split leak.** The row that matters most is the
third one: a *correct* chronological split still reads 0.8889 when the target
encoding was computed with a `groupby` over the whole dataframe. You can do the
famous thing right and still be off by 0.27, and it's the more dangerous failure
because you've already done the bit that gets talked about, so you stop looking.

That's also why this is a 2×2 and not a single before/after. One comparison would
have read "leaky 0.8975 vs honest 0.6166", and I'd have blamed the split and been
wrong about the mechanism.

Full reasoning, and the caveats about what the synthetic numbers can't tell me,
in **[NOTES.md](NOTES.md)**.

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
make data && make validate
```

`data.py` distinguishes "no token" from "rules not accepted" — Kaggle reports both
as a bare 403, which is unhelpful when you're stuck.

---

## What's here, and what isn't

| | |
|---|---|
| ✅ | Chronological holdout + expanding-window CV, tested |
| ✅ | Entity-overlap and temporal-leak diagnostics |
| ✅ | The 2×2 leakage experiment, runnable by anyone |
| ✅ | Memory-aware loading (590k × 434 doesn't fit at float64) |
| ⬜ | EDA on the real data — missingness, imbalance, temporal structure |
| ⬜ | LightGBM model + overfitting diagnosis |
| ⬜ | Feature engineering log, including what failed |
| ⬜ | Streamlit predictor with SHAP explanations, Docker, hosted demo |

The unchecked rows are the ones that need the competition data. I'm not going to
fill them in with synthetic stand-ins.

---

## Layout

```
src/fraud/
  config.py                     experiment knobs in one place
  data.py                       download, join identity, downcast dtypes
  split.py                      the load-bearing file — chronological CV
  experiments/validation_gap.py the 2×2
tests/                          14 tests, synthetic data only
NOTES.md                        the decision trail
```

## License

MIT — see [LICENSE](LICENSE). The competition data is not redistributed here;
`make data` fetches it from Kaggle under their terms.
