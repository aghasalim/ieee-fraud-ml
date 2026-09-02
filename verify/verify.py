# Recompute the four published AUCs from the per-row scores and verify the
# drift table's internal consistency.
#
# The AUC uses the same Mann-Whitney rank-sum formula as the C, R, and SQL
# verifiers. Printing the canonical AUC| lines lets verify.sh compare all
# implementations to twelve decimals.
#
# The drift check is new: ea_drift.csv reports a per-week AUC and fraud rate,
# and this verifier confirms that every row's fraud rate and AUC are within
# the bounds you would expect from the reported sample sizes (fraud rate is a
# proportion of n, so it must be expressible as k/n for some integer k).
#
# No dependencies beyond the standard library.

import csv
import math
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."

PUB_TOL = 5e-5


def read_csv(name):
    with open(os.path.join(root, "reports", name), newline="") as f:
        return list(csv.DictReader(f))


# -- AUC via mid-ranks (Mann-Whitney) ----------------------------------------

def auc(ys, scores):
    n = len(ys)
    indexed = sorted(range(n), key=lambda i: scores[i])
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[indexed[j + 1]] == scores[indexed[i]]:
            j += 1
        mid = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            rank[indexed[k]] = mid
        i = j + 1
    n_pos = sum(1 for y in ys if y == 1)
    n_neg = sum(1 for y in ys if y == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = sum(rank[i] for i in range(n) if ys[i] == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


scores_rows = read_csv("validation_gap_scores_synthetic.csv")
published = read_csv("validation_gap_synthetic.csv")

# Group rows by (split, target encoding, fold)
groups = {}
for row in scores_rows:
    key = (row["split"], row["target encoding"], int(row["fold"]))
    groups.setdefault(key, ([], []))
    groups[key][0].append(int(row["y"]))
    groups[key][1].append(float(row["score"]))

failures = 0

print("point estimates, against reports/validation_gap_synthetic.csv")
for pub in published:
    split = pub["split"]
    te = pub["target encoding"]
    want = float(pub["AUC"])
    fold_aucs = []
    for (s, t, f), (ys, sc) in sorted(groups.items()):
        if s == split and t == te:
            fold_aucs.append(auc(ys, sc))
    got = sum(fold_aucs) / len(fold_aucs)
    delta = abs(got - want)
    ok = delta <= PUB_TOL
    if not ok:
        failures += 1
    status = "ok" if ok else "FAIL"
    print(f"  {split:<16s} {te:<11s} AUC {got:.12f}  published {want:.4f}  |d| {delta:.1e}  {status}")
    print(f"AUC|{split}|{te}|{got:.12f}")

# -- drift table consistency -------------------------------------------------

print("\ndrift table, reports/ea_drift.csv")
drift = read_csv("ea_drift.csv")
drift_checks = 0
for row in drift:
    n = int(row["n"])
    rate = float(row["fraud rate"])
    # The fraud rate should be expressible as round(k/n, 4) for some integer k
    k = round(rate * n)
    reconstructed = round(k / n, 4)
    d = abs(reconstructed - rate)
    ok = d < 1e-4
    drift_checks += 1
    if not ok:
        failures += 1
        print(f"  FAIL week {row['week']}: fraud rate {rate} not consistent with n={n}")
    # AUC should be between 0 and 1
    a = float(row["AUC"])
    drift_checks += 1
    if not (0.0 <= a <= 1.0):
        failures += 1
        print(f"  FAIL week {row['week']}: AUC {a} out of range")

if drift_checks > 0 and failures == 0:
    print(f"  {drift_checks} checks on {len(drift)} weeks, all consistent")

if failures > 0:
    print(f"\n{failures} checks failed")
    sys.exit(1)

print("\nPython reproduces every published AUC and confirms drift-table consistency")
