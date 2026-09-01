# Statistical inference on two claims the README makes and never tested.
#
# 1. "an inflation of 0.28 AUC". That is a point estimate with no interval on
#    it. This bootstraps the scored rows and reports a 95% interval, so the
#    headline stops being a single number with nothing under it.
#
# 2. "Calibration is fine above 25% and badly off below 1%, where it
#    under-predicts by nearly 7x". That is a quantitative claim about
#    reports/ea_calibration.csv, so it can be checked against that file rather
#    than taken on trust.
#
# The AUCs themselves are recomputed here as well, with R's own rank(), which
# makes this the fourth independent implementation after the Python, the SQL and
# the C.
#
# Base R only, so CI needs nothing beyond r-base-core.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
set.seed(20260901)

DRAWS <- 1000
PUB_TOL <- 5e-5          # the published table is rounded to four decimals

scores <- read.csv(file.path(root, "reports", "validation_gap_scores_synthetic.csv"),
                   check.names = FALSE)
published <- read.csv(file.path(root, "reports", "validation_gap_synthetic.csv"),
                      check.names = FALSE)
calib <- read.csv(file.path(root, "reports", "ea_calibration.csv"), check.names = FALSE)

for (needed in c("split", "target encoding", "fold", "y", "score")) {
    if (!needed %in% names(scores)) stop("score file has no column ", needed)
}

# Mann-Whitney over mid-ranks. rank() averages ties by default, which is the
# convention roc_auc_score uses; any other tie rule shifts the answer.
auc <- function(y, s) {
    r <- rank(s)
    np <- sum(y == 1L)
    nn <- sum(y == 0L)
    if (np == 0L || nn == 0L) return(NA_real_)
    (sum(r[y == 1L]) - np * (np + 1) / 2) / (np * nn)
}

config_key <- paste(scores$split, scores[["target encoding"]], sep = "|")
fold_key <- paste(config_key, scores$fold, sep = "|")
groups <- split(seq_len(nrow(scores)), fold_key)

config_auc <- function(split_name, te_name) {
    prefix <- paste(split_name, te_name, sep = "|")
    mine <- groups[startsWith(names(groups), paste0(prefix, "|"))]
    mine <- mine[order(names(mine))]
    mean(vapply(mine, function(i) auc(scores$y[i], scores$score[i]), numeric(1)))
}

cat("point estimates, against reports/validation_gap_synthetic.csv\n")
failures <- 0
for (i in seq_len(nrow(published))) {
    s <- published$split[i]
    t <- published[["target encoding"]][i]
    got <- config_auc(s, t)
    want <- published$AUC[i]
    delta <- abs(got - want)
    ok <- delta <= PUB_TOL
    failures <- failures + !ok
    cat(sprintf("  %-16s %-11s AUC %.12f  published %.4f  |d| %.1e  %s\n",
                s, t, got, want, delta, if (ok) "ok" else "FAIL"))
    cat(sprintf("AUC|%s|%s|%.12f\n", s, t, got))
}

# The two ends of the headline: the most flattering configuration and the one
# the repository argues is the defensible one.
best_i <- which.max(published$AUC)
honest_i <- which(published$split == "chronological" &
                  published[["target encoding"]] == "fold-local")[1]
if (is.na(honest_i)) stop("no chronological fold-local row to compare against")
best <- list(s = published$split[best_i], t = published[["target encoding"]][best_i])
honest <- list(s = published$split[honest_i], t = published[["target encoding"]][honest_i])
point <- config_auc(best$s, best$t) - config_auc(honest$s, honest$t)

# Resampling is within fold, because the folds are the unit the experiment
# averages over. The two configurations are resampled independently, which
# ignores that they score overlapping rows and so gives an interval a little
# wider than a paired one would. That direction is the safe one: if this
# interval still clears zero, a paired interval would too.
boot_config <- function(split_name, te_name, draws) {
    prefix <- paste(split_name, te_name, sep = "|")
    mine <- groups[startsWith(names(groups), paste0(prefix, "|"))]
    ys <- lapply(mine, function(i) scores$y[i])
    ss <- lapply(mine, function(i) scores$score[i])
    vapply(seq_len(draws), function(b) {
        mean(vapply(seq_along(ys), function(k) {
            n <- length(ys[[k]])
            pick <- sample.int(n, n, replace = TRUE)
            auc(ys[[k]][pick], ss[[k]][pick])
        }, numeric(1)))
    }, numeric(1))
}

cat(sprintf("\nbootstrap on the headline gap, %d draws\n", DRAWS))
gap <- boot_config(best$s, best$t, DRAWS) - boot_config(honest$s, honest$t, DRAWS)
ci <- quantile(gap, c(0.025, 0.975), names = FALSE)
cat(sprintf("  %s %s minus %s %s\n", best$s, best$t, honest$s, honest$t))
cat(sprintf("  point %.4f   95%% interval %.4f to %.4f   width %.4f   sd %.4f\n",
            point, ci[1], ci[2], ci[2] - ci[1], sd(gap)))
cat(sprintf("BOOT|gap|%.6f|%.6f|%.6f|%.6f\n", point, ci[1], ci[2], sd(gap)))

if (ci[1] <= 0) {
    cat("FAIL: the interval on the headline gap includes zero\n")
    failures <- failures + 1
}
if (point < ci[1] || point > ci[2]) {
    cat("FAIL: the point estimate falls outside its own bootstrap interval\n")
    failures <- failures + 1
}

# The calibration claim. Wilson intervals on the observed rate in each bucket,
# then the two halves of the sentence the README actually writes.
cat("\ncalibration, reports/ea_calibration.csv\n")
wilson <- function(k, n, z = 1.959964) {
    p <- k / n
    d <- 1 + z^2 / n
    centre <- (p + z^2 / (2 * n)) / d
    half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / d
    c(centre - half, centre + half)
}
ratio <- calib$actual / calib$mean_pred
for (i in seq_len(nrow(calib))) {
    k <- round(calib$n[i] * calib$actual[i])
    w <- wilson(k, calib$n[i])
    covered <- calib$mean_pred[i] >= w[1] && calib$mean_pred[i] <= w[2]
    cat(sprintf("  %-16s n %7d  predicted %.4f  observed %.4f [%.4f, %.4f]  %.2fx  %s\n",
                calib$bucket[i], calib$n[i], calib$mean_pred[i], calib$actual[i],
                w[1], w[2], ratio[i],
                if (covered) "prediction inside" else "prediction outside"))
}

low <- which.min(calib$mean_pred)
high <- calib$mean_pred > 0.25
cat(sprintf("\n  lowest bucket under-predicts by %.2fx\n", ratio[low]))
cat(sprintf("  worst departure above 0.25 is %.2fx over %d buckets\n",
            max(pmax(ratio[high], 1 / ratio[high])), sum(high)))
cat(sprintf("CALIB|%.4f|%.4f\n", ratio[low], max(pmax(ratio[high], 1 / ratio[high]))))

if (ratio[low] < 5) {
    cat("FAIL: the lowest bucket does not under-predict the way the README says\n")
    failures <- failures + 1
}
if (max(pmax(ratio[high], 1 / ratio[high])) > 1.2) {
    cat("FAIL: calibration above 0.25 is not as good as the README says\n")
    failures <- failures + 1
}

if (failures > 0) {
    cat(sprintf("\n%d checks failed\n", failures))
    quit(status = 1)
}
cat("\nR reproduces every published AUC, puts an interval on the headline gap,\n")
cat("and confirms both halves of the calibration sentence\n")
