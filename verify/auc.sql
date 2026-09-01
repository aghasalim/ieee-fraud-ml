-- Two jobs, both of them aggregation, which is what SQL is for.
--
-- 1. Rebuild the four published AUCs from the per-row scores, using window
--    functions for the mid-ranks. Same arithmetic as verify/auc.c and
--    verify/gocheck, written by different hands in a different language.
--
-- 2. Check that the five tables under reports/ that describe the error analysis
--    are describing the same population. ea_segments.csv partitions the scored
--    rows three separate ways, ea_calibration.csv bins them a fourth, and
--    ea_drift.csv splits them by week. Each was written by a different function
--    in src/fraud/error_analysis.py and nothing ever compared them. If any one
--    of them were computed on a different subset, or dropped rows, the totals
--    would not line up, and until now nothing would have said so.
--
-- Run: sqlite3 -init verify/auc.sql :memory: ""
-- Every result line is CHECK|name|ok or FAIL|..., and verify/verify.sh treats
-- any FAIL as a failure of the whole run.

.bail on
.mode list
.separator |
.headers off

.import --csv reports/validation_gap_scores_synthetic.csv raw_scores
.import --csv reports/validation_gap_synthetic.csv raw_published
.import --csv reports/ea_segments.csv raw_segments
.import --csv reports/ea_calibration.csv raw_calibration
.import --csv reports/ea_budget.csv raw_budget
.import --csv reports/ea_drift.csv raw_drift

-- .import gives every column TEXT affinity, so the casts are not decoration:
-- ordering scores as text would rank "9e-05" above "0.5". Naming the columns
-- here also means a renamed column upstream is an error rather than a silent
-- shift of position.
CREATE TEMP VIEW scores AS
SELECT split,
       "target encoding" AS te,
       CAST(fold AS INTEGER)  AS fold,
       CAST(y AS INTEGER)     AS y,
       CAST(score AS REAL)    AS score
FROM raw_scores;

CREATE TEMP VIEW published AS
SELECT split, "target encoding" AS te, CAST(AUC AS REAL) AS auc FROM raw_published;

-- Mid-rank of each score inside its fold. RANK ascending gives the first rank a
-- tied block spans and RANK descending gives the last, so their midpoint is the
-- average rank that Mann-Whitney needs.
CREATE TEMP TABLE ranked AS
SELECT split, te, fold, y,
       (RANK() OVER w_asc
        + (COUNT(*) OVER w_grp + 1 - RANK() OVER w_desc)) / 2.0 AS r
FROM scores
WINDOW w_asc  AS (PARTITION BY split, te, fold ORDER BY score ASC),
       w_desc AS (PARTITION BY split, te, fold ORDER BY score DESC),
       w_grp  AS (PARTITION BY split, te, fold);

CREATE TEMP VIEW fold_auc AS
SELECT split, te, fold,
       (SUM(CASE WHEN y = 1 THEN r ELSE 0.0 END)
        - SUM(y) * (SUM(y) + 1) / 2.0) / (SUM(y) * SUM(1 - y)) AS auc
FROM ranked
GROUP BY split, te, fold;

CREATE TEMP VIEW config_auc AS
SELECT split, te, AVG(auc) AS auc, COUNT(*) AS folds FROM fold_auc GROUP BY split, te;

SELECT 'AUC', c.split, c.te, printf('%.12f', c.auc)
FROM config_auc c ORDER BY c.split, c.te;

-- The published table is rounded to four decimals, so half of the last digit is
-- all the agreement it can carry.
SELECT 'CHECK', 'auc ' || p.split || ' ' || p.te,
       CASE WHEN c.auc IS NULL THEN 'FAIL no scored rows'
            WHEN abs(c.auc - p.auc) <= 5e-5 THEN 'ok'
            ELSE 'FAIL' END,
       printf('recomputed %.6f published %.4f over %d folds',
              c.auc, p.auc, c.folds)
FROM published p LEFT JOIN config_auc c ON c.split = p.split AND c.te = p.te
ORDER BY p.split, p.te;

-- Every way the error analysis partitions the scored rows.
CREATE TEMP VIEW partitions AS
    SELECT 'segments/' || segment AS part,
           SUM(CAST(n AS INTEGER)) AS n,
           SUM(CAST(n AS REAL) * CAST("fraud rate" AS REAL)) AS fraud
    FROM raw_segments GROUP BY segment
    UNION ALL
    SELECT 'calibration bins',
           SUM(CAST(n AS INTEGER)),
           SUM(CAST(n AS REAL) * CAST(actual AS REAL))
    FROM raw_calibration
    UNION ALL
    SELECT 'drift weeks',
           SUM(CAST(n AS INTEGER)),
           SUM(CAST(n AS REAL) * CAST("fraud rate" AS REAL))
    FROM raw_drift;

SELECT 'ROWS', part, n, printf('%.1f', fraud) FROM partitions ORDER BY part;

SELECT 'CHECK', 'every partition of the scored rows has the same size',
       CASE WHEN COUNT(DISTINCT n) = 1 THEN 'ok' ELSE 'FAIL' END,
       printf('%d partitions, sizes %d to %d', COUNT(*), MIN(n), MAX(n))
FROM partitions;

-- The rates are rounded to four decimals, so over 4.4e5 rows each implied fraud
-- count carries about +/- 22 of rounding. 0.2% of the count is a little over
-- twice that and still far tighter than any real disagreement would be.
SELECT 'CHECK', 'every partition implies the same fraud count',
       CASE WHEN (MAX(fraud) - MIN(fraud)) / AVG(fraud) <= 0.002 THEN 'ok' ELSE 'FAIL' END,
       printf('%.1f to %.1f, spread %.3f%%', MIN(fraud), MAX(fraud),
              100.0 * (MAX(fraud) - MIN(fraud)) / AVG(fraud))
FROM partitions;

-- reports/ea_budget.csv was written by yet another function. Its review sizes
-- have to be that same population truncated at each budget, its precision has
-- to be the ratio of its own two count columns, and the fraud total its recall
-- implies has to be the fraud total every other table implies.
CREATE TEMP VIEW budget AS
SELECT CAST(RTRIM("alert budget", '%') AS REAL) / 100.0 AS frac,
       CAST("n reviewed" AS INTEGER)   AS reviewed,
       CAST("fraud caught" AS INTEGER) AS caught,
       CAST(recall AS REAL)            AS recall,
       CAST(precision AS REAL)         AS prec
FROM raw_budget;

SELECT 'CHECK', 'review sizes are the same population truncated at each budget',
       CASE WHEN SUM(reviewed <> CAST(frac * (SELECT MAX(n) FROM partitions) AS INTEGER)) = 0
            THEN 'ok' ELSE 'FAIL' END,
       printf('%d budgets against a population of %d',
              COUNT(*), (SELECT MAX(n) FROM partitions))
FROM budget;

SELECT 'CHECK', 'precision is caught over reviewed',
       CASE WHEN MAX(abs(prec - CAST(caught AS REAL) / reviewed)) <= 5e-5
            THEN 'ok' ELSE 'FAIL' END,
       printf('largest residual %.2e', MAX(abs(prec - CAST(caught AS REAL) / reviewed)))
FROM budget;

SELECT 'CHECK', 'recall implies the same fraud total as the other tables',
       CASE WHEN MAX(abs(caught / recall - f) / f) <= 0.0025 THEN 'ok' ELSE 'FAIL' END,
       printf('largest departure %.3f%% from %.0f',
              100.0 * MAX(abs(caught / recall - f) / f), MAX(f))
FROM budget, (SELECT AVG(fraud) AS f FROM partitions);

