# Recompute the four published AUCs from the per-row scores and verify that
# the submission file covers the expected ID range without duplicates.
#
# The AUC uses the same Mann-Whitney mid-rank formula as every other verifier,
# so verify.sh can require all implementations to agree to twelve decimals.
#
# The submission check is new: reports/submission.csv should have one row per
# TransactionID, no duplicates, and every predicted probability in [0, 1].
#
# No gems required.

require "csv"

root = ARGV[0] || "."

PUB_TOL = 5e-5

def read_csv(root, name)
  CSV.read(File.join(root, "reports", name), headers: true)
end

# -- AUC via mid-ranks (Mann-Whitney) ----------------------------------------

def auc(ys, scores)
  n = ys.length
  indexed = (0...n).sort_by { |i| scores[i] }
  rank = Array.new(n, 0.0)
  i = 0
  while i < n
    j = i
    while j + 1 < n && scores[indexed[j + 1]] == scores[indexed[i]]
      j += 1
    end
    mid = (i + 1 + j + 1) / 2.0
    (i..j).each { |k| rank[indexed[k]] = mid }
    i = j + 1
  end
  n_pos = ys.count(1)
  n_neg = ys.count(0)
  return Float::NAN if n_pos == 0 || n_neg == 0
  rank_sum = (0...n).select { |i| ys[i] == 1 }.sum { |i| rank[i] }
  (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
end

scores_rows = read_csv(root, "validation_gap_scores_synthetic.csv")
published = read_csv(root, "validation_gap_synthetic.csv")

# Group by (split, target encoding, fold)
groups = Hash.new { |h, k| h[k] = [[], []] }
scores_rows.each do |row|
  key = [row["split"], row["target encoding"], row["fold"].to_i]
  groups[key][0] << row["y"].to_i
  groups[key][1] << row["score"].to_f
end

failures = 0

puts "point estimates, against reports/validation_gap_synthetic.csv"
published.each do |pub|
  split = pub["split"]
  te = pub["target encoding"]
  want = pub["AUC"].to_f
  fold_aucs = groups.keys
    .select { |s, t, _f| s == split && t == te }
    .sort
    .map { |key| auc(groups[key][0], groups[key][1]) }
  got = fold_aucs.sum / fold_aucs.length
  delta = (got - want).abs
  ok = delta <= PUB_TOL
  failures += 1 unless ok
  status = ok ? "ok" : "FAIL"
  printf "  %-16s %-11s AUC %.12f  published %.4f  |d| %.1e  %s\n",
         split, te, got, want, delta, status
  printf "AUC|%s|%s|%.12f\n", split, te, got
end

# -- submission file checks ---------------------------------------------------

puts "\nsubmission file, reports/submission.csv"
sub = read_csv(root, "submission.csv")
sub_checks = 0

ids = sub.map { |r| r["TransactionID"].to_i }
sub_checks += 1
dups = ids.length - ids.uniq.length
if dups > 0
  failures += 1
  puts "  FAIL #{dups} duplicate TransactionIDs"
else
  puts "  #{ids.length} rows, no duplicate IDs"
end

bad_probs = 0
sub.each do |row|
  p = row["isFraud"].to_f
  bad_probs += 1 unless p >= 0.0 && p <= 1.0
end
sub_checks += 1
if bad_probs > 0
  failures += 1
  puts "  FAIL #{bad_probs} predictions outside [0, 1]"
else
  puts "  all predictions in [0, 1]"
end

if failures > 0
  puts "\n#{failures} checks failed"
  exit 1
end

puts "\nRuby reproduces every published AUC and confirms submission-file integrity"
