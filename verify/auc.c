/* Recompute the four published AUCs from the per-row scores, in C.
 *
 * reports/validation_gap_synthetic.csv is what the README quotes. It was
 * produced by sklearn's roc_auc_score inside
 * src/fraud/experiments/validation_gap.py, and nothing else ever looked at it.
 * This reads the labels and scores that experiment scored,
 * reports/validation_gap_scores_synthetic.csv, and rebuilds each number from
 * scratch: Mann-Whitney rank sums with mid-ranks for ties, averaged over folds
 * exactly the way the experiment averages them.
 *
 * Columns are resolved by name, so a column added upstream cannot silently
 * shift what this reads.
 *
 * Prints one canonical "AUC|split|encoding|value" line per configuration.
 * verify/verify.sh requires every implementation to print the same lines, which
 * is a far tighter test than each of them matching a table rounded to four
 * decimals.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE 4096
#define MAX_CFG 8
#define MAX_FOLD 16
/* The published table is rounded to four decimals, so half of the last digit
 * is the most agreement that table can express. */
#define PUB_TOL 5e-5

typedef struct {
    double *score;
    char *y;
    size_t n, cap;
} Group;

typedef struct {
    char split[64], te[64];
    double published;
    Group fold[MAX_FOLD + 1];
} Config;

static Config cfg[MAX_CFG];
static int n_cfg = 0;

static void push(Group *g, double s, char y)
{
    if (g->n == g->cap) {
        g->cap = g->cap ? g->cap * 2 : 1024;
        g->score = realloc(g->score, g->cap * sizeof *g->score);
        g->y = realloc(g->y, g->cap * sizeof *g->y);
        if (!g->score || !g->y) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    g->score[g->n] = s;
    g->y[g->n] = y;
    g->n++;
}

/* Index of a named column, or -1. Trailing newline and CR are stripped so the
 * last column name matches. */
static int column_of(const char *header, const char *name)
{
    char buf[LINE];
    snprintf(buf, sizeof buf, "%s", header);
    int i = 0;
    for (char *tok = strtok(buf, ",\r\n"); tok; tok = strtok(NULL, ",\r\n"), i++)
        if (strcmp(tok, name) == 0)
            return i;
    return -1;
}

/* Field `index` of a comma separated line, into `out`. No quoted fields here:
 * neither file this reads has any, and pretending otherwise would hide a
 * malformed row rather than trip on it. */
static int field(const char *line, int index, char *out, size_t cap)
{
    int col = 0;
    const char *p = line;
    while (col < index) {
        p = strchr(p, ',');
        if (!p) return 0;
        p++; col++;
    }
    const char *end = strchr(p, ',');
    size_t n = end ? (size_t)(end - p) : strlen(p);
    if (n >= cap) n = cap - 1;
    memcpy(out, p, n);
    out[n] = '\0';
    char *nl = strpbrk(out, "\r\n");
    if (nl) *nl = '\0';
    return 1;
}

typedef struct { double s; char y; } Pair;

static int by_score(const void *a, const void *b)
{
    const double x = ((const Pair *)a)->s, y = ((const Pair *)b)->s;
    return x < y ? -1 : x > y ? 1 : 0;
}

/* Mann-Whitney U over the mid-ranks, which is the definition roc_auc_score
 * implements. Ties get the average of the ranks they span; getting that wrong
 * is the classic way to be slightly and invisibly off. */
static double auc_of(Group *g)
{
    const size_t n = g->n;
    /* Sorting the pair keeps every score with its own label. */
    Pair *p = malloc(n * sizeof *p);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(2); }
    for (size_t i = 0; i < n; i++) { p[i].s = g->score[i]; p[i].y = g->y[i]; }
    qsort(p, n, sizeof *p, by_score);

    double rank_sum_pos = 0.0;
    size_t n_pos = 0, n_neg = 0;
    for (size_t i = 0; i < n; ) {
        size_t j = i;
        while (j + 1 < n && p[j + 1].s == p[i].s) j++;
        /* Ranks are one based, so the block spans i+1 .. j+1. */
        const double mid = ((double)(i + 1) + (double)(j + 1)) / 2.0;
        for (size_t k = i; k <= j; k++) {
            if (p[k].y) { rank_sum_pos += mid; n_pos++; }
            else n_neg++;
        }
        i = j + 1;
    }
    free(p);
    if (n_pos == 0 || n_neg == 0) return NAN;
    return (rank_sum_pos - (double)n_pos * (n_pos + 1) / 2.0)
           / ((double)n_pos * (double)n_neg);
}

static int find_cfg(const char *split, const char *te)
{
    for (int i = 0; i < n_cfg; i++)
        if (strcmp(cfg[i].split, split) == 0 && strcmp(cfg[i].te, te) == 0)
            return i;
    return -1;
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    char path[1024], line[LINE], header[LINE], a[64], b[64], c[64];

    /* The published table first: it defines which configurations exist. */
    snprintf(path, sizeof path, "%s/reports/validation_gap_synthetic.csv", root);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!fgets(header, sizeof header, f)) { fclose(f); return 2; }
    const int p_split = column_of(header, "split");
    const int p_te = column_of(header, "target encoding");
    const int p_auc = column_of(header, "AUC");
    if (p_split < 0 || p_te < 0 || p_auc < 0) {
        fprintf(stderr, "validation_gap_synthetic.csv is missing a column\n");
        fclose(f); return 2;
    }
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        if (n_cfg == MAX_CFG) { fprintf(stderr, "too many rows\n"); fclose(f); return 2; }
        field(line, p_split, a, sizeof a);
        field(line, p_te, b, sizeof b);
        field(line, p_auc, c, sizeof c);
        snprintf(cfg[n_cfg].split, sizeof cfg[n_cfg].split, "%s", a);
        snprintf(cfg[n_cfg].te, sizeof cfg[n_cfg].te, "%s", b);
        cfg[n_cfg].published = atof(c);
        n_cfg++;
    }
    fclose(f);

    snprintf(path, sizeof path, "%s/reports/validation_gap_scores_synthetic.csv", root);
    f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!fgets(header, sizeof header, f)) { fclose(f); return 2; }
    const int s_split = column_of(header, "split");
    const int s_te = column_of(header, "target encoding");
    const int s_fold = column_of(header, "fold");
    const int s_y = column_of(header, "y");
    const int s_score = column_of(header, "score");
    if (s_split < 0 || s_te < 0 || s_fold < 0 || s_y < 0 || s_score < 0) {
        fprintf(stderr, "score file is missing a column\n"); fclose(f); return 2;
    }

    long rows = 0, unknown = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        field(line, s_split, a, sizeof a);
        field(line, s_te, b, sizeof b);
        const int k = find_cfg(a, b);
        if (k < 0) { unknown++; continue; }
        field(line, s_fold, c, sizeof c);
        const int fold = atoi(c);
        if (fold < 1 || fold > MAX_FOLD) {
            fprintf(stderr, "fold %d out of range\n", fold); fclose(f); return 2;
        }
        field(line, s_y, c, sizeof c);
        const char y = (char)(atoi(c) != 0);
        field(line, s_score, c, sizeof c);
        push(&cfg[k].fold[fold], atof(c), y);
        rows++;
    }
    fclose(f);
    if (unknown) {
        fprintf(stderr, "%ld scored rows belong to no published configuration\n", unknown);
        return 1;
    }
    printf("read %ld scored rows for %d configurations\n", rows, n_cfg);

    int failures = 0;
    for (int i = 0; i < n_cfg; i++) {
        double sum = 0.0;
        int folds = 0;
        for (int k = 1; k <= MAX_FOLD; k++) {
            if (cfg[i].fold[k].n == 0) continue;
            const double a_ = auc_of(&cfg[i].fold[k]);
            if (isnan(a_)) {
                fprintf(stderr, "fold %d of %s/%s has one class only\n",
                        k, cfg[i].split, cfg[i].te);
                return 1;
            }
            sum += a_;
            folds++;
        }
        if (folds == 0) {
            fprintf(stderr, "no scored rows for %s/%s\n", cfg[i].split, cfg[i].te);
            return 1;
        }
        const double got = sum / folds;
        const double delta = fabs(got - cfg[i].published);
        const int bad = delta > PUB_TOL;
        failures += bad;
        printf("  %-16s %-11s %d folds  AUC %.12f  published %.4f  |d| %.1e  %s\n",
               cfg[i].split, cfg[i].te, folds, got, cfg[i].published, delta,
               bad ? "FAIL" : "ok");
        printf("AUC|%s|%s|%.12f\n", cfg[i].split, cfg[i].te, got);
    }

    if (failures) {
        printf("\n%d of %d configurations disagree with the published table\n",
               failures, n_cfg);
        return 1;
    }
    printf("\nC reproduces all %d published AUCs from the per-row scores "
           "(tolerance %.0e)\n", n_cfg, PUB_TOL);
    return 0;
}
