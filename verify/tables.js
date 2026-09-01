// Recompute every column of a published table that is arithmetic on the other
// columns of that same table, and every number one table quotes from another.
//
// reports/ablation.csv and reports/overfit.csv each carry columns that are pure
// subtraction of two neighbours, written out by pandas so they could be read
// off the page. reports/eda_summary.csv is nine headline numbers, four of which
// are summaries of reports/eda_product.csv and reports/eda_missing.csv. All of
// it was produced once, by one script, and read by nobody since. A transposed
// column or a stale regeneration of one file and not another would sit there
// looking perfectly plausible.
//
// Run: node verify/tables.js [root]

const fs = require('fs');
const path = require('path');

const root = process.argv[2] || '.';
let failures = 0;
let checks = 0;

function report(name, ok, detail) {
    checks++;
    if (!ok) failures++;
    console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`);
}

// Minimal quote aware split. reports/ea_calibration.csv has bucket labels with
// commas inside them, so naive splitting is not safe across this directory.
function splitRow(line) {
    const out = [];
    let field = '';
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (c === '"') {
            if (quoted && line[i + 1] === '"') { field += '"'; i++; }
            else quoted = !quoted;
        } else if (c === ',' && !quoted) {
            out.push(field); field = '';
        } else {
            field += c;
        }
    }
    out.push(field);
    return out;
}

function readCSV(name) {
    const text = fs.readFileSync(path.join(root, 'reports', name), 'utf8');
    const lines = text.split('\n').filter((l) => l.trim() !== '');
    const header = splitRow(lines[0]);
    return lines.slice(1).map((line) => {
        const cells = splitRow(line);
        if (cells.length !== header.length) {
            throw new Error(`${name}: ragged row, ${cells.length} cells against ${header.length}`);
        }
        return Object.fromEntries(header.map((h, i) => [h, cells[i]]));
    });
}

// Columns are read by name, so a rename upstream is a loud error here.
function num(row, column, file) {
    if (!(column in row)) throw new Error(`${file}: no column named ${column}`);
    const v = parseFloat(row[column]);
    if (!Number.isFinite(v)) throw new Error(`${file}: ${column} is ${row[column]}`);
    return v;
}

// A column written as the difference of two others, recomputed. The published
// tables are rounded, so `tol` says how much of the residual is rounding: a
// difference of two four decimal numbers is exact, but a difference taken
// before rounding and then rounded can be out by one in the last place.
function difference(file, rows, target, left, right, tol) {
    let worst = 0;
    let where = '';
    for (const row of rows) {
        const got = num(row, left, file) - num(row, right, file);
        const d = Math.abs(got - num(row, target, file));
        if (d > worst) { worst = d; where = row[Object.keys(row)[0]]; }
    }
    report(`${file}: ${target} is ${left} minus ${right}`, worst <= tol,
           `largest residual ${worst.toExponential(1)}${where ? ` at "${where}"` : ''}`);
}

console.log('columns that are arithmetic on their own neighbours');

const ablation = readCSV('ablation.csv');
difference('ablation.csv', ablation, 'train-val gap', 'train AUC', 'val AUC', 1.5e-4);

// `delta` is the step in validation AUC from the row above, and the first row
// has nothing above it, so it is empty rather than zero.
{
    let worst = 0;
    for (let i = 1; i < ablation.length; i++) {
        const step = num(ablation[i], 'val AUC', 'ablation.csv') -
                     num(ablation[i - 1], 'val AUC', 'ablation.csv');
        worst = Math.max(worst, Math.abs(step - num(ablation[i], 'delta', 'ablation.csv')));
    }
    const firstBlank = ablation[0].delta.trim() === '';
    report('ablation.csv: delta is the step in val AUC from the row above',
           worst <= 1e-9 && firstBlank,
           `largest residual ${worst.toExponential(1)}, first row ${firstBlank ? 'blank' : 'not blank'}`);
}

const overfit = readCSV('overfit.csv');
difference('overfit.csv', overfit, 'early-stopping bonus',
           'AUC w/ early stopping', 'AUC w/ fixed 400', 1e-9);
difference('overfit.csv', overfit, 'train-val gap',
           'train AUC (fixed)', 'AUC w/ fixed 400', 1e-9);

console.log('\nnumbers one table quotes from another');

const summary = Object.fromEntries(
    readCSV('eda_summary.csv').map((r) => [r.metric, r.value]));
const product = readCSV('eda_product.csv');
const missing = readCSV('eda_missing.csv');

const rows = product.reduce((a, r) => a + num(r, 'n', 'eda_product.csv'), 0);
const stated = parseInt(summary.transactions.split(/[^0-9]+/)[0], 10);
report('eda_summary transactions is the total in eda_product',
       rows === stated, `${rows} against ${stated}`);

const fraud = product.reduce(
    (a, r) => a + num(r, 'n', 'eda_product.csv') * num(r, 'fraud_rate', 'eda_product.csv'), 0);
const rate = fraud / rows;
report('eda_summary fraud rate is the weighted rate in eda_product',
       Math.abs(rate - parseFloat(summary['fraud rate'])) <= 5e-6,
       `${rate.toFixed(6)} against ${summary['fraud rate']}`);

const rates = product.map((r) => num(r, 'fraud_rate', 'eda_product.csv'));
const spread = Math.max(...rates) / Math.min(...rates);
report('eda_summary product spread is the ratio in eda_product',
       Math.abs(spread - parseFloat(summary['fraud-rate spread across ProductCD'])) <= 5e-3,
       `${spread.toFixed(4)}x against ${summary['fraud-rate spread across ProductCD']}`);

const worst = Math.max(...missing.map((r) => num(r, 'missing_frac', 'eda_missing.csv')));
report('eda_summary worst missingness is the worst in eda_missing',
       Math.abs(worst - parseFloat(summary['worst column missingness'])) <= 5e-5,
       `${worst.toFixed(6)} against ${summary['worst column missingness']}`);

// eda_missing.csv is presented as the worst columns first, and the README reads
// the first row off it as "the worst single column".
{
    const fracs = missing.map((r) => num(r, 'missing_frac', 'eda_missing.csv'));
    const sorted = fracs.every((v, i) => i === 0 || v <= fracs[i - 1]);
    report('eda_missing is ordered worst first', sorted,
           `${fracs.length} columns, ${fracs[0].toFixed(4)} down to ${fracs[fracs.length - 1].toFixed(4)}`);
}

console.log(`\n${checks - failures} of ${checks} checks passed`);
if (failures > 0) process.exit(1);
console.log('every derived column and every cross-quoted number still holds');
