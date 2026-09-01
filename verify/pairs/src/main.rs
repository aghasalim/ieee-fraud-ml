//! Two things the Python could not afford to do.
//!
//! 1. **AUC from its definition.** Every other implementation here, sklearn's
//!    included, computes AUC from rank sums. That is a shortcut, correct only
//!    if ties are handled exactly right, and all five of them take the same
//!    shortcut. This one counts pairs: for every fraudulent row against every
//!    legitimate row, does the model score the fraud higher. That is what AUC
//!    means. It is quadratic, which is why nothing else does it, and on these
//!    folds it is about a hundred million comparisons.
//!
//! 2. **The price of the bootstrap.** verify/verify.R puts a 95% interval on
//!    the headline gap from 1000 draws. An interval estimated from 1000 draws
//!    carries its own error, and nothing measured it. This runs a much larger
//!    reference bootstrap, then twenty independent 1000 draw runs whose spread
//!    is the error bar on the interval R reports, and asks whether 1000 draws
//!    was enough for the claim being made.

use std::env;
use std::fs;
use std::process::exit;

const REFERENCE_DRAWS: usize = 10_000;
const R_DRAWS: usize = 1_000;
const REPLICATES: usize = 20;
/// The published table is rounded to four decimals.
const PUB_TOL: f64 = 5e-5;
/// How far above zero the lower end of the interval has to sit, measured in its
/// own Monte Carlo standard deviations, for the headline to be safe from noise.
const MIN_SIGMA: f64 = 10.0;

/// xorshift64*. Not cryptographic and not meant to be: it has to be uniform,
/// fast, and seeded reproducibly so a failure can be re-run.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

struct Fold {
    /// Labels and scores sorted by score, so a bootstrap draw is a pass over
    /// multiplicities rather than another sort.
    y: Vec<u8>,
    score: Vec<f64>,
}

struct Config {
    split: String,
    te: String,
    published: f64,
    folds: Vec<Fold>,
}

/// AUC by the definition: the share of fraud/legitimate pairs the model orders
/// correctly, counting a tie as half. Quadratic and deliberately so.
fn auc_pairs(f: &Fold) -> f64 {
    let pos: Vec<f64> = f
        .score
        .iter()
        .zip(&f.y)
        .filter(|(_, y)| **y == 1)
        .map(|(s, _)| *s)
        .collect();
    let neg: Vec<f64> = f
        .score
        .iter()
        .zip(&f.y)
        .filter(|(_, y)| **y == 0)
        .map(|(s, _)| *s)
        .collect();
    if pos.is_empty() || neg.is_empty() {
        eprintln!("a fold has only one class");
        exit(1);
    }
    let mut greater: u64 = 0;
    let mut equal: u64 = 0;
    for &p in &pos {
        for &n in &neg {
            if p > n {
                greater += 1;
            } else if p == n {
                equal += 1;
            }
        }
    }
    (greater as f64 + 0.5 * equal as f64) / (pos.len() as f64 * neg.len() as f64)
}

/// AUC of one bootstrap resample, given how many times each row was drawn.
/// The fold is already sorted by score, so ties are consecutive and the
/// mid-ranks come out of a single pass.
fn auc_weighted(f: &Fold, mult: &[u32]) -> f64 {
    let n = f.score.len();
    let (mut rank_sum_pos, mut cum, mut np, mut nn) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && f.score[j + 1] == f.score[i] {
            j += 1;
        }
        let (mut block, mut block_pos) = (0.0f64, 0.0f64);
        for k in i..=j {
            let m = mult[k] as f64;
            block += m;
            if f.y[k] == 1 {
                block_pos += m;
            }
        }
        if block > 0.0 {
            let mid = cum + (block + 1.0) / 2.0;
            rank_sum_pos += mid * block_pos;
            np += block_pos;
            nn += block - block_pos;
            cum += block;
        }
        i = j + 1;
    }
    if np == 0.0 || nn == 0.0 {
        return f64::NAN;
    }
    (rank_sum_pos - np * (np + 1.0) / 2.0) / (np * nn)
}

/// One bootstrap draw of a configuration's mean AUC: every fold resampled to
/// its own size, then averaged the way the experiment averages.
fn draw_mean_auc(c: &Config, rng: &mut Rng, scratch: &mut Vec<u32>) -> f64 {
    let mut total = 0.0;
    for f in &c.folds {
        let n = f.score.len();
        scratch.clear();
        scratch.resize(n, 0);
        for _ in 0..n {
            let k = rng.below(n);
            scratch[k] += 1;
        }
        total += auc_weighted(f, scratch);
    }
    total / c.folds.len() as f64
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

fn column(header: &[&str], name: &str) -> usize {
    match header.iter().position(|h| *h == name) {
        Some(i) => i,
        None => {
            eprintln!("no column named {}", name);
            exit(2);
        }
    }
}

fn read(path: &str) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {}", path, e);
        exit(2)
    })
}

fn load(root: &str) -> Vec<Config> {
    let pub_text = read(&format!("{}/reports/validation_gap_synthetic.csv", root));
    let mut lines = pub_text.lines();
    let header: Vec<&str> = lines.next().expect("empty table").split(',').collect();
    let (cs, ct, ca) = (
        column(&header, "split"),
        column(&header, "target encoding"),
        column(&header, "AUC"),
    );
    let mut configs: Vec<Config> = Vec::new();
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        configs.push(Config {
            split: f[cs].to_string(),
            te: f[ct].to_string(),
            published: f[ca].trim().parse().unwrap_or(f64::NAN),
            folds: Vec::new(),
        });
    }

    let text = read(&format!("{}/reports/validation_gap_scores_synthetic.csv", root));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("empty score file").split(',').collect();
    let (cs, ct, cf, cy, csc) = (
        column(&header, "split"),
        column(&header, "target encoding"),
        column(&header, "fold"),
        column(&header, "y"),
        column(&header, "score"),
    );

    // (config, fold) -> rows, keeping folds in ascending order so the mean over
    // them is taken in the same order as everywhere else.
    let mut buckets: Vec<Vec<(usize, Vec<u8>, Vec<f64>)>> =
        configs.iter().map(|_| Vec::new()).collect();
    let mut unknown = 0usize;
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        let ci = match configs.iter().position(|c| c.split == f[cs] && c.te == f[ct]) {
            Some(i) => i,
            None => {
                unknown += 1;
                continue;
            }
        };
        let fold: usize = f[cf].trim().parse().unwrap_or_else(|_| {
            eprintln!("bad fold {}", f[cf]);
            exit(2)
        });
        let y: u8 = if f[cy].trim() == "1" { 1 } else { 0 };
        let score: f64 = f[csc].trim().parse().unwrap_or_else(|_| {
            eprintln!("bad score {}", f[csc]);
            exit(2)
        });
        let b = &mut buckets[ci];
        match b.iter().position(|(k, _, _)| *k == fold) {
            Some(i) => {
                b[i].1.push(y);
                b[i].2.push(score);
            }
            None => b.push((fold, vec![y], vec![score])),
        }
    }
    if unknown > 0 {
        eprintln!("{} scored rows belong to no published configuration", unknown);
        exit(1);
    }

    for (c, mut b) in configs.iter_mut().zip(buckets) {
        b.sort_by_key(|(k, _, _)| *k);
        for (_, y, score) in b {
            let mut ord: Vec<usize> = (0..score.len()).collect();
            ord.sort_by(|a, b| score[*a].partial_cmp(&score[*b]).unwrap());
            c.folds.push(Fold {
                y: ord.iter().map(|&i| y[i]).collect(),
                score: ord.iter().map(|&i| score[i]).collect(),
            });
        }
        if c.folds.is_empty() {
            eprintln!("no scored rows for {} / {}", c.split, c.te);
            exit(1);
        }
    }
    configs
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let root = args.get(1).map(String::as_str).unwrap_or(".");
    let configs = load(root);

    let rows: usize = configs
        .iter()
        .flat_map(|c| c.folds.iter().map(|f| f.score.len()))
        .sum();
    println!("{} scored rows across {} configurations", rows, configs.len());
    println!("AUC from the pairwise definition, not from rank sums\n");

    let mut failures = 0;
    let mut means: Vec<f64> = Vec::new();
    for c in &configs {
        let mut total = 0.0;
        let mut pairs: u64 = 0;
        for f in &c.folds {
            total += auc_pairs(f);
            let p = f.y.iter().filter(|y| **y == 1).count() as u64;
            pairs += p * (f.y.len() as u64 - p);
        }
        let got = total / c.folds.len() as f64;
        means.push(got);
        let delta = (got - c.published).abs();
        let bad = delta > PUB_TOL;
        failures += bad as i32;
        println!(
            "  {:<16} {:<11} {:>11} pairs  AUC {:.12}  published {:.4}  |d| {:.1e}  {}",
            c.split,
            c.te,
            pairs,
            got,
            c.published,
            delta,
            if bad { "FAIL" } else { "ok" }
        );
        println!("AUC|{}|{}|{:.12}", c.split, c.te, got);
    }

    // The headline gap, and how much of the interval R reports is noise.
    let best = means
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |acc, (i, v)| {
            if *v > acc.1 {
                (i, *v)
            } else {
                acc
            }
        })
        .0;
    let honest = match configs
        .iter()
        .position(|c| c.split == "chronological" && c.te == "fold-local")
    {
        Some(i) => i,
        None => {
            eprintln!("no chronological fold-local configuration");
            exit(1);
        }
    };
    let point = means[best] - means[honest];

    let interval = |draws: usize, seed: u64| -> (f64, f64) {
        let mut rng = Rng::new(seed);
        let mut scratch: Vec<u32> = Vec::new();
        let mut gaps: Vec<f64> = Vec::with_capacity(draws);
        for _ in 0..draws {
            let a = draw_mean_auc(&configs[best], &mut rng, &mut scratch);
            let b = draw_mean_auc(&configs[honest], &mut rng, &mut scratch);
            gaps.push(a - b);
        }
        gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (quantile(&gaps, 0.025), quantile(&gaps, 0.975))
    };

    println!(
        "\nheadline gap: {} {} minus {} {}",
        configs[best].split, configs[best].te, configs[honest].split, configs[honest].te
    );
    let (rlo, rhi) = interval(REFERENCE_DRAWS, 0x5EED_1234);
    println!(
        "  point {:.4}   {} draw reference interval {:.4} to {:.4}",
        point, REFERENCE_DRAWS, rlo, rhi
    );

    let mut lows = Vec::with_capacity(REPLICATES);
    let mut highs = Vec::with_capacity(REPLICATES);
    for r in 0..REPLICATES {
        let (lo, hi) = interval(R_DRAWS, 0xC0FFEE + (r as u64) * 104_729);
        lows.push(lo);
        highs.push(hi);
    }
    let sd = |v: &Vec<f64>| {
        let m: f64 = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
    };
    let (sd_lo, sd_hi) = (sd(&lows), sd(&highs));
    println!(
        "  {} runs of {} draws: lower end sd {:.5}, upper end sd {:.5}",
        REPLICATES, R_DRAWS, sd_lo, sd_hi
    );
    let sigma = rlo / sd_lo.max(1e-12);
    println!(
        "  the lower end sits {:.0} of its own standard deviations above zero",
        sigma
    );
    println!("BOOT|{:.6}|{:.6}|{:.6}|{:.6}|{:.6}", point, rlo, rhi, sd_lo, sd_hi);

    if sigma < MIN_SIGMA {
        println!(
            "\nFAIL: at {:.1} sd the headline gap is not clear of bootstrap noise",
            sigma
        );
        exit(1);
    }
    if failures > 0 {
        println!("\n{} configurations disagree with the published table", failures);
        exit(1);
    }
    println!(
        "\nthe pairwise definition gives the same AUCs as the rank formula every other\n\
         implementation uses, and {} draws was more than enough for the headline",
        R_DRAWS
    );
}
