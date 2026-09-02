#!/usr/bin/env bash
# Recompute the published numbers in every language here and require agreement.
#
# Everything this repository publishes came out of one Python process. The AUCs
# in the README came from sklearn, the tables came from pandas, and every figure
# was drawn from those same tables. The test suite checks that the code runs, not
# that its arithmetic is right, so a mistake in the metric or in an aggregation
# would be invisible: everything downstream reads the output of the thing that
# made the mistake.
#
# So the four published AUCs are rebuilt from the per-row scores by five
# independent implementations, and the summary tables are checked against each
# other and against the README. A mistake would have to be made identically in
# all of them to survive.
#
# Each implementation is skipped with a message if its toolchain is missing, so
# a partial install still runs the rest. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export PATH="$HOME/.cargo/bin:$PATH"

logs="$(mktemp -d)"
trap 'rm -rf "$logs"' EXIT

pass=0 fail=0 skip=0

run () {
    local slug="$1" name="$2" tool="$3"; shift 3
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@" > "$logs/$slug.log" 2>&1; then
        cat "$logs/$slug.log"; pass=$((pass + 1))
    else
        cat "$logs/$slug.log"; fail=$((fail + 1))
        printf '%s FAILED\n' "$name"
    fi
}

# sqlite3 exits 0 whatever its queries print, so the assertions live in the
# output and are read back out here.
check_sql () {
    local out n
    out=$(sqlite3 -init verify/auc.sql :memory: "" 2>&1) || { printf '%s\n' "$out"; return 1; }
    printf '%s\n' "$out"
    if printf '%s\n' "$out" | grep -q 'FAIL'; then return 1; fi
    n=$(printf '%s\n' "$out" | grep -c '^CHECK|')
    if [ "$n" -lt 8 ]; then printf 'only %s checks ran\n' "$n"; return 1; fi
    printf '\nSQL: %s checks, no failures\n' "$n"
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "$logs/auc" verify/auc.c -lm || return 1
    "$logs/auc" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/pairs && cargo run --release --quiet -- "$root" ); }

# The sharpest test in here. Five implementations print the same canonical
# AUC|split|encoding|value lines, and those lines have to be identical. Matching
# a table rounded to four decimals leaves room for two implementations to be
# differently wrong; matching each other to twelve does not.
compare_auc () {
    printf '\n=== cross-implementation agreement ===\n'
    local f base n=0 differ=0
    base=""
    for f in "$logs"/*.log; do
        grep '^AUC|' "$f" | sort > "$f.auc"
        [ -s "$f.auc" ] || continue
        n=$((n + 1))
        if [ -z "$base" ]; then base="$f.auc"; continue; fi
        if ! diff -q "$base" "$f.auc" >/dev/null; then
            printf 'disagreement between %s and %s:\n' \
                   "$(basename "$base" .log.auc)" "$(basename "$f" .log)"
            diff "$base" "$f.auc"
            differ=$((differ + 1))
        fi
    done
    if [ "$n" -lt 2 ]; then
        printf 'skipped: only %d implementation produced AUC lines\n' "$n"
        skip=$((skip + 1)); return
    fi
    if [ "$differ" -gt 0 ]; then
        printf '%d of %d implementations disagree\n' "$differ" "$n"
        fail=$((fail + 1)); return
    fi
    printf '%d implementations, %d values each, identical to twelve decimals:\n\n' \
           "$n" "$(wc -l < "$base" | tr -d ' ')"
    sed 's/^AUC|/  /' "$base"
    pass=$((pass + 1))
}

run sql   "SQL, aggregation and cross-table totals" sqlite3 check_sql
run c     "C, the metric kernel"                    cc      check_c
run go    "Go, file structure and README claims"    go      check_go
run js    "JavaScript, derived columns"             node    node verify/tables.js "$root"
run r     "R, bootstrap interval and calibration"   Rscript Rscript verify/verify.R "$root"
run rust  "Rust, AUC from its definition"           cargo   check_rust
run py    "Python, AUC and drift consistency"       python3 python3 verify/verify.py "$root"
run rb    "Ruby, AUC and submission integrity"      ruby    ruby verify/verify.rb "$root"
compare_auc

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
