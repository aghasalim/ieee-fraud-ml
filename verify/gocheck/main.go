// Three jobs: check that the files under reports/ are well formed, recompute
// the published AUCs from the per-row scores, and check that every number the
// README quotes is still the number in the file it came from.
//
// The third is the one that was missing. Every table and every figure in this
// repository is generated, but the README was written by hand around them, so
// a number could drift out of a regenerated CSV and the prose would keep
// asserting the old value indefinitely. Nothing compared them. This does, and
// it is deliberately blunt about it: a claim is checked by rebuilding the exact
// string from the CSV and requiring the README to contain it.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// The published tables round to four decimals, so half of the last digit is the
// most agreement they can express.
const pubTol = 5e-5

type table struct {
	header []string
	rows   [][]string
}

func (t table) col(name string) int {
	for i, h := range t.header {
		if h == name {
			return i
		}
	}
	return -1
}

// get returns one cell by column name. Resolving by name means a column added
// or reordered upstream is an error here rather than a silent shift.
func (t table) get(row []string, name string) string {
	i := t.col(name)
	if i < 0 || i >= len(row) {
		return ""
	}
	return row[i]
}

func (t table) num(row []string, name string) float64 {
	v, err := strconv.ParseFloat(strings.TrimSpace(t.get(row, name)), 64)
	if err != nil {
		return math.NaN()
	}
	return v
}

func readCSV(path string) (table, error) {
	f, err := os.Open(path)
	if err != nil {
		return table{}, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		return table{}, err
	}
	if len(rows) < 2 {
		return table{}, fmt.Errorf("only %d rows", len(rows))
	}
	return table{header: rows[0], rows: rows[1:]}, nil
}

// validate reports every structural problem in a file rather than the first, so
// one pass diagnoses a broken run.
func validate(path string) []string {
	var problems []string
	t, err := readCSV(path)
	if err != nil {
		return []string{fmt.Sprintf("unreadable: %v", err)}
	}
	seen := map[string]bool{}
	for _, h := range t.header {
		if strings.TrimSpace(h) == "" {
			problems = append(problems, "a column has an empty name")
		}
		if seen[h] {
			problems = append(problems, fmt.Sprintf("duplicate column %q", h))
		}
		seen[h] = true
	}
	for i, row := range t.rows {
		for j, cell := range row {
			switch strings.ToLower(strings.TrimSpace(cell)) {
			case "nan", "inf", "-inf", "infinity", "-infinity":
				problems = append(problems,
					fmt.Sprintf("row %d column %s is %s", i+2, t.header[j], cell))
			}
		}
	}
	return problems
}

// auc is Mann-Whitney over mid-ranks, the definition roc_auc_score implements.
func auc(y []int, score []float64) float64 {
	idx := make([]int, len(score))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return score[idx[a]] < score[idx[b]] })

	rankSumPos, nPos, nNeg := 0.0, 0, 0
	for i := 0; i < len(idx); {
		j := i
		for j+1 < len(idx) && score[idx[j+1]] == score[idx[i]] {
			j++
		}
		mid := (float64(i+1) + float64(j+1)) / 2.0
		for k := i; k <= j; k++ {
			if y[idx[k]] == 1 {
				rankSumPos += mid
				nPos++
			} else {
				nNeg++
			}
		}
		i = j + 1
	}
	if nPos == 0 || nNeg == 0 {
		return math.NaN()
	}
	return (rankSumPos - float64(nPos)*float64(nPos+1)/2.0) /
		(float64(nPos) * float64(nNeg))
}

type key struct{ split, te string }

func recomputeAUC(root string) (map[key]float64, []string) {
	var problems []string
	scores, err := readCSV(filepath.Join(root, "reports", "validation_gap_scores_synthetic.csv"))
	if err != nil {
		return nil, []string{fmt.Sprintf("score file: %v", err)}
	}
	published, err := readCSV(filepath.Join(root, "reports", "validation_gap_synthetic.csv"))
	if err != nil {
		return nil, []string{fmt.Sprintf("published table: %v", err)}
	}

	type group struct {
		y     []int
		score []float64
	}
	folds := map[key]map[int]*group{}
	for _, row := range scores.rows {
		k := key{scores.get(row, "split"), scores.get(row, "target encoding")}
		fold, err := strconv.Atoi(strings.TrimSpace(scores.get(row, "fold")))
		if err != nil {
			return nil, []string{fmt.Sprintf("bad fold %q", scores.get(row, "fold"))}
		}
		if folds[k] == nil {
			folds[k] = map[int]*group{}
		}
		if folds[k][fold] == nil {
			folds[k][fold] = &group{}
		}
		g := folds[k][fold]
		yv, err := strconv.Atoi(strings.TrimSpace(scores.get(row, "y")))
		if err != nil {
			return nil, []string{fmt.Sprintf("bad label %q", scores.get(row, "y"))}
		}
		g.y = append(g.y, yv)
		g.score = append(g.score, scores.num(row, "score"))
	}

	got := map[key]float64{}
	for _, row := range published.rows {
		k := key{published.get(row, "split"), published.get(row, "target encoding")}
		byFold, ok := folds[k]
		if !ok {
			problems = append(problems,
				fmt.Sprintf("no scored rows for %s / %s", k.split, k.te))
			continue
		}
		ids := make([]int, 0, len(byFold))
		for f := range byFold {
			ids = append(ids, f)
		}
		sort.Ints(ids)

		sum := 0.0
		for _, f := range ids {
			a := auc(byFold[f].y, byFold[f].score)
			if math.IsNaN(a) {
				problems = append(problems,
					fmt.Sprintf("fold %d of %s / %s has one class only", f, k.split, k.te))
				continue
			}
			sum += a
		}
		mean := sum / float64(len(ids))
		got[k] = mean
		want := published.num(row, "AUC")
		delta := math.Abs(mean - want)
		status := "ok"
		if delta > pubTol {
			status = "FAIL"
			problems = append(problems,
				fmt.Sprintf("%s / %s: recomputed %.6f against published %.4f",
					k.split, k.te, mean, want))
		}
		fmt.Printf("  %-16s %-11s %d folds  AUC %.12f  published %.4f  |d| %.1e  %s\n",
			k.split, k.te, len(ids), mean, want, delta, status)
		fmt.Printf("AUC|%s|%s|%.12f\n", k.split, k.te, mean)
	}
	return got, problems
}

// ---- README traceability -------------------------------------------------

type readme struct {
	text  string
	lines []string
	bad   []string
	n     int
}

// wants requires a literal string, rebuilt from a CSV cell, to appear anywhere
// in the README.
func (r *readme) wants(what, s string) {
	r.n++
	if !strings.Contains(r.text, s) {
		r.bad = append(r.bad, fmt.Sprintf("%s: README does not contain %q", what, s))
	}
}

// onSameLine requires one line of the README to carry all of these at once,
// which is what stops a right number being attached to the wrong row.
func (r *readme) onSameLine(what string, parts ...string) bool {
	r.n++
	for _, line := range r.lines {
		ok := true
		for _, p := range parts {
			if !strings.Contains(line, p) {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	r.bad = append(r.bad, fmt.Sprintf("%s: no README line holds all of %v", what, parts))
	return false
}

// comma formats an integer the way the README writes one, 590540 -> "590,540".
func comma(n int) string {
	s := strconv.Itoa(n)
	var out []byte
	for i, c := range []byte(s) {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, c)
	}
	return string(out)
}

// tableAfter returns the body rows of the markdown table whose header line
// contains `marker`.
func tableAfter(lines []string, marker string) []string {
	for i, line := range lines {
		if !strings.Contains(line, marker) || !strings.HasPrefix(strings.TrimSpace(line), "|") {
			continue
		}
		var out []string
		for j := i + 1; j < len(lines) && strings.HasPrefix(strings.TrimSpace(lines[j]), "|"); j++ {
			out = append(out, lines[j])
		}
		return out
	}
	return nil
}

var fourDecimalRE = regexp.MustCompile(`[0-9]\.[0-9]{4}(?:[^0-9]|$)`)

// fourDecimals pulls every number written to exactly four decimals out of a
// line, which is how every AUC in this README is written.
func fourDecimals(line string) []float64 {
	var out []float64
	for _, m := range fourDecimalRE.FindAllString(line, -1) {
		v, err := strconv.ParseFloat(m[:6], 64)
		if err == nil {
			out = append(out, v)
		}
	}
	return out
}

func checkREADME(root string) (int, []string) {
	raw, err := os.ReadFile(filepath.Join(root, "README.md"))
	if err != nil {
		return 0, []string{fmt.Sprintf("README.md: %v", err)}
	}
	r := &readme{text: string(raw), lines: strings.Split(string(raw), "\n")}

	load := func(name string) table {
		t, err := readCSV(filepath.Join(root, "reports", name))
		if err != nil {
			r.bad = append(r.bad, fmt.Sprintf("%s: %v", name, err))
		}
		return t
	}

	// The headline. Both ends of the spread and the spread itself.
	leak := load("leakage_real.csv")
	if len(leak.rows) > 0 {
		best, worst := math.Inf(-1), math.Inf(1)
		hiOv, loOv := math.Inf(-1), math.Inf(1)
		quoted := 0
		for _, row := range leak.rows {
			a := leak.num(row, "AUC")
			best, worst = math.Max(best, a), math.Min(worst, a)
			o := leak.num(row, "card overlap (fold 1)")
			hiOv, loOv = math.Max(hiOv, o), math.Min(loOv, o)
			// A quoted AUC has to sit on a line that names its own encoding,
			// otherwise the right number is attached to the wrong row.
			s := fmt.Sprintf("%.4f", a)
			if strings.Contains(r.text, s) {
				quoted++
				r.onSameLine("leakage_real.csv row", s, leak.get(row, "target encoding"))
			}
		}
		r.wants("headline spread", fmt.Sprintf("%.1f AUC points", 100*(best-worst)))
		r.wants("best AUC", fmt.Sprintf("%.4f", best))
		r.wants("worst AUC", fmt.Sprintf("%.4f", worst))
		r.wants("card overlap high", fmt.Sprintf("%.0f%%", 100*hiOv))
		r.wants("card overlap low", fmt.Sprintf("%.0f%%", 100*loOv))
		if quoted < 4 {
			r.bad = append(r.bad,
				fmt.Sprintf("README quotes only %d of the %d rows of leakage_real.csv",
					quoted, len(leak.rows)))
		}

		// The leaderboard table. Its right hand column is the only arithmetic
		// the README does that no script ever did, and the leaderboard score
		// itself is the one number here that is not in any file, so it is read
		// back out of the table and the stated gaps are required to follow from
		// it. Every row is scanned for four decimal numbers: the row with one
		// is the leaderboard, a row with two is a configuration and its gap.
		block := tableAfter(r.lines, "vs leaderboard")
		if len(block) == 0 {
			r.bad = append(r.bad, "no leaderboard comparison table in the README")
		}
		lb, pairs := math.NaN(), 0
		for _, line := range block {
			if n := fourDecimals(line); len(n) == 1 {
				lb = n[0]
			}
		}
		for _, line := range block {
			n := fourDecimals(line)
			if len(n) != 2 || math.IsNaN(lb) {
				continue
			}
			pairs++
			r.n++
			known := false
			for _, row := range leak.rows {
				known = known || math.Abs(leak.num(row, "AUC")-n[0]) < 1e-9
			}
			if !known {
				r.bad = append(r.bad,
					fmt.Sprintf("leaderboard table quotes %.4f, which is not in leakage_real.csv", n[0]))
			} else if math.Abs(math.Abs(n[0]-lb)-n[1]) > 5e-5 {
				r.bad = append(r.bad,
					fmt.Sprintf("leaderboard table says %.4f is %.4f from %.4f, but it is %.4f",
						n[0], n[1], lb, math.Abs(n[0]-lb)))
			}
		}
		if pairs < 4 {
			r.bad = append(r.bad,
				fmt.Sprintf("leaderboard table compares only %d configurations", pairs))
		}
	}

	// The dataset description.
	if sum := load("eda_summary.csv"); len(sum.rows) > 0 {
		v := map[string]string{}
		for _, row := range sum.rows {
			v[sum.get(row, "metric")] = sum.get(row, "value")
		}
		if tx := strings.Fields(strings.ReplaceAll(v["transactions"], "x", " ")); len(tx) > 0 {
			if n, err := strconv.Atoi(tx[0]); err == nil {
				r.wants("transaction count", comma(n))
			}
		}
		if n, err := strconv.Atoi(v["identity rows"]); err == nil {
			r.wants("identity rows", comma(n))
		}
		for _, c := range []struct {
			what, metric, format string
		}{
			{"fraud rate", "fraud rate", "%.3f%%"},
			{"identity coverage", "identity coverage", "%.1f%%"},
			{"worst column missingness", "worst column missingness", "%.1f%%"},
		} {
			if f, err := strconv.ParseFloat(v[c.metric], 64); err == nil {
				r.wants(c.what, fmt.Sprintf(c.format, 100*f))
			}
		}
	}

	// Fraud is not spread evenly across product codes.
	if prod := load("eda_product.csv"); len(prod.rows) > 0 {
		hi, lo, biggest, biggestN := math.Inf(-1), math.Inf(1), "", 0
		for _, row := range prod.rows {
			f := prod.num(row, "fraud_rate")
			hi, lo = math.Max(hi, f), math.Min(lo, f)
			if n, err := strconv.Atoi(prod.get(row, "n")); err == nil && n > biggestN {
				biggestN, biggest = n, prod.get(row, "ProductCD")
			}
		}
		r.wants("highest product fraud rate", fmt.Sprintf("%.1f%%", 100*hi))
		r.wants("lowest product fraud rate", fmt.Sprintf("%.1f%%", 100*lo))
		r.wants("fraud rate spread", fmt.Sprintf("%.1fx", hi/lo))
		r.onSameLine("largest product code", biggest, comma(biggestN))
	}

	// The feature ablation.
	if ab := load("ablation.csv"); len(ab.rows) > 0 {
		for _, row := range ab.rows {
			r.onSameLine("ablation row", ab.get(row, "features"),
				fmt.Sprintf("%.4f", ab.num(row, "train AUC")),
				fmt.Sprintf("%.4f", ab.num(row, "val AUC")))
		}
	}

	// The review budget table.
	if bud := load("ea_budget.csv"); len(bud.rows) > 0 {
		quoted := 0
		for _, row := range bud.rows {
			n, err := strconv.Atoi(bud.get(row, "n reviewed"))
			if err != nil || !strings.Contains(r.text, comma(n)) {
				continue
			}
			quoted++
			r.onSameLine("review budget row", comma(n),
				fmt.Sprintf("%.1f%%", 100*bud.num(row, "recall")),
				fmt.Sprintf("%.1f%%", 100*bud.num(row, "precision")))
		}
		if quoted < 3 {
			r.bad = append(r.bad,
				fmt.Sprintf("README quotes only %d review budgets", quoted))
		}
	}

	// The two weakest segments, which are the two the README singles out.
	if seg := load("ea_segments.csv"); len(seg.rows) > 0 {
		quoted := 0
		for _, row := range seg.rows {
			n, err := strconv.Atoi(seg.get(row, "n"))
			if err != nil || !strings.Contains(r.text, comma(n)) {
				continue
			}
			quoted++
			r.onSameLine("segment row", comma(n),
				fmt.Sprintf("%.4f", seg.num(row, "AUC")),
				strings.TrimSpace(seg.get(row, "recall@1%")))
		}
		if quoted < 2 {
			r.bad = append(r.bad,
				fmt.Sprintf("README quotes only %d segment rows", quoted))
		}
	}

	// The synthetic run, which is the part anyone can reproduce.
	syn := load("validation_gap_synthetic.csv")
	if len(syn.rows) > 0 {
		best, honest := math.Inf(-1), math.NaN()
		for _, row := range syn.rows {
			a := syn.num(row, "AUC")
			r.wants("synthetic AUC", fmt.Sprintf("%.4f", a))
			best = math.Max(best, a)
			if syn.get(row, "split") == "chronological" &&
				syn.get(row, "target encoding") == "fold-local" {
				honest = a
			}
		}
		if !math.IsNaN(honest) {
			r.wants("synthetic inflation", fmt.Sprintf("%.2f", best-honest))
		}
	}

	return r.n, r.bad
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	files, err := filepath.Glob(filepath.Join(*root, "reports", "*.csv"))
	if err != nil || len(files) == 0 {
		fmt.Fprintf(os.Stderr, "no CSVs under %s/reports\n", *root)
		os.Exit(2)
	}
	sort.Strings(files)

	bad := 0
	fmt.Printf("validating %d files under reports/\n", len(files))
	for _, path := range files {
		for _, p := range validate(path) {
			fmt.Printf("  %s: %s\n", filepath.Base(path), p)
			bad++
		}
	}
	if bad == 0 {
		fmt.Println("  no ragged rows, duplicate columns, empty column names, NaN or Inf")
	}

	fmt.Println("\nrecomputing the published AUCs from the per-row scores")
	_, problems := recomputeAUC(*root)
	for _, p := range problems {
		fmt.Printf("  %s\n", p)
		bad++
	}

	fmt.Println("\ntracing every number the README quotes back to its file")
	claims, readmeProblems := checkREADME(*root)
	for _, p := range readmeProblems {
		fmt.Printf("  %s\n", p)
		bad++
	}
	if len(readmeProblems) == 0 {
		fmt.Printf("  all %d checked claims still match the file they came from\n", claims)
	}

	if bad > 0 {
		fmt.Printf("\n%d problems\n", bad)
		os.Exit(1)
	}
	fmt.Println("\nGo agrees with the published AUCs, reports/ is well formed, " +
		"and the README still matches it")
}
