# Chapter 16: Experimentation and Bandits

Companion code for Chapter 16, which covers how to measure what a recommender change actually does: trustworthy A/B tests, variance reduction, sequential testing, novelty effects, interleaving, multi-armed and contextual bandits, and off-policy evaluation.

## Overview

The code is three simulations plus a figure script. Every result is scored against a known truth:

1. **A/B testing on real behaviour, with injected effects** (`ab_test_simulation.py`). Per-user listening variance comes from real Yambda users, arms are assigned by hashing, and treatment effects are injected with a known size. This is a *semi-synthetic* setup: it shows how the statistics behave on realistic heavy-tailed data and estimates no real effect of any real change.
2. **Interleaving versus A/B testing** (`interleaving_simulation.py`). A fully synthetic ranking world in which one ranker is truly better, compared by team-draft interleaving and by a user-split A/B test.
3. **Bandits and off-policy evaluation** (`bandit_ope_simulation.py`). A fully synthetic shelf-selection problem with known reward probabilities, used to measure regret and to score off-policy estimators against the true policy value.

Every script runs without the dataset: the A/B script takes `--synthetic`, and the other two are synthetic by design.

## Module Structure

```
chapter16_experimentation/
├── ab_test_simulation.py       # A/A, SRM, power, CUPED, ratio metrics, peeking, novelty
├── interleaving_simulation.py  # Team-draft interleaving vs A/B test
├── bandit_ope_simulation.py    # Bandit regret; IPS, SNIPS, DM, DR estimators
├── make_schematics.py          # Figures 16.1, 16.4, 16.6
├── stats_utils.py              # Welch, power, CUPED, SRM, delta method, bootstrap, mSPRT, BH
├── interleaving.py             # Team-draft and balanced interleaving, credit assignment
├── bandits.py                  # Epsilon-greedy, UCB1, Thompson sampling, LinUCB, environments
├── ope.py                      # Off-policy estimators with one shared signature
├── common.py                   # CLI flags, output writing, plot style, data-path convention
├── tests/                      # pytest suite with known-answer tests
├── docs/Chapter16_Design_Notes.md
├── CLOUD_RUN.md                # Full-run commands, hardware, runtimes, outputs
├── requirements.txt
└── README.md                   # This file
```

## Setup

```bash
python -m venv .venv            # create it outside any synced folder
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11, CPU only. No GPU, no network calls, no paid services.

## Data

Only `ab_test_simulation.py` reads real data: `listens.parquet` from the [Yandex Yambda dataset](https://huggingface.co/datasets/yandex/yambda), the 50M-event flat version (`flat/50m/listens.parquet`), as in Chapters 6 and 7. Point the script at the directory that contains it:

```bash
# Windows PowerShell
$env:YAMBDA_DATA_DIR = "C:\path\to\Dataset\Yandex\flat"
# macOS/Linux
export YAMBDA_DATA_DIR=/path/to/yambda/flat

# Or pass --data-dir on the command line
```

The script reads four columns (`uid`, `timestamp`, `is_organic`, `played_ratio_pct`). Timestamps are seconds from the first event in the dataset, rounded down to a multiple of 5 seconds (the "5-second bins" of the dataset card), so the day index is `timestamp // 86400`. The last 28 days form the experiment window and the 28 days before them form the pre-period. Users who never listen during the window are excluded, since they would never be exposed to either variant.

The Yambda 50M sample has about ten thousand users. Modes that need larger populations (power, SRM, novelty) resample users with replacement. Resampled copies of a user are independent draws for assignment, but they share that user's behaviour, so results at large simulated sizes are somewhat more optimistic than a truly larger population would give.

## Running

Every script accepts `--seed` (default 16), `--out-dir` (default `outputs/<script>`), and `--quick` (reduced sizes for a smoke run). Each mode draws from its own random stream derived from the seed, so running one mode on its own reproduces exactly what `--mode all` gives for that mode.

### A/B testing (Sections 16.3.5, 16.4, 16.6.1)

```bash
python ab_test_simulation.py --mode all              # Yambda
python ab_test_simulation.py --mode all --synthetic  # no dataset needed
python ab_test_simulation.py --mode cuped --quick    # one mode, fast
```

| Mode | What it shows | Main outputs |
|------|---------------|--------------|
| `aa` | A/A p-values are uniform; false positives sit near alpha | `aa_pvalues.csv`, `aa_pvalue_histogram.png` |
| `srm` | A logging bug that drops heavy users from one arm fakes an effect; the sample ratio mismatch check catches it | `summary.json` (`srm`) |
| `power` | Power against users for raw, winsorized, and CUPED analysis | `power_curves.csv`, `power_curves.png` |
| `cuped` | Pre-period data shrink the variance of the effect estimate without biasing it | `fig16_2_cuped.png` (Figure 16.2) |
| `ratio` | Per-listen standard errors are too small for completion rate; delta method and user bootstrap are right | `ratio_ci_coverage.png` |
| `peeking` | Daily looks inflate false positives; Bonferroni and the always-valid mSPRT do not | `fig16_3_peeking.png` (Figure 16.3) |
| `novelty` | Novelty and primacy effects make a two-week pooled estimate misleading | `fig16_5_novelty_primacy.png` (Figure 16.5) |

The synthetic panel (`--synthetic`) draws a heavy-tailed per-user listening rate from a Gamma distribution, a daily activity probability from a Beta distribution, and a pre-period rate correlated with the experiment-period rate. Daily listens are Poisson given the rate, which makes them negative-binomial-like across users. The parameters are illustrative and documented in `synthetic_panel`.

Injected effects keep the data valid: count metrics gain Poisson extra events in proportion to baseline (or lose events by binomial thinning for negative effects), and completion rate is raised by turning non-completed listens into completed ones, so rates stay within [0, 1]. A listen counts as completed when `played_ratio_pct >= --completion-threshold` (default 80), an illustrative definition.

### Interleaving (Section 16.7)

```bash
python interleaving_simulation.py --mode all
```

| Mode | What it shows | Main outputs |
|------|---------------|--------------|
| `sanity` | Identical rankers give about 50% wins and false positives near alpha for both methods | `summary.json` (`sanity`) |
| `sensitivity` | Interleaving reaches a correct decision with fewer sessions than an A/B test | `fig16_7_interleaving_sensitivity.png` (Figure 16.7) |
| `example` | One seeded impression: lists, coin flips, teams, clicks | `interleaving_example.json` (input to Figure 16.6) |

Each simulated user has a heavy-tailed activity level that scales how often they click on anything. That between-user variance is what makes A/B tests on clicks noisy, and interleaving cancels it because both rankers are compared within the same session. The ground-truth quality gap is confirmed by expected NDCG@10 under the true relevance, reported in `summary.json`.

### Bandits and off-policy evaluation (Sections 16.8, 16.9)

```bash
python bandit_ope_simulation.py --mode all
python bandit_ope_simulation.py --mode ope support   # Figure 16.9 only
```

| Mode | What it shows | Main outputs |
|------|---------------|--------------|
| `mab` | Regret of uniform, epsilon-greedy, UCB1, and Thompson sampling on Bernoulli arms | `regret_mab.csv` |
| `contextual` | Context-free policies keep paying regret; LinUCB's flattens | `regret_contextual.csv` |
| `ope` | IPS, clipped IPS, SNIPS, DM, and DR on epsilon-greedy logs, scored against the true value | `ope_estimates.csv` |
| `support` | Estimator error as the logging epsilon and log size vary; replay at uniform logging | `support_sweep.csv` |

`--mode` takes one or more modes. `--mode mab contextual` writes `fig16_8_bandit_regret.png` (Figure 16.8), `--mode ope support` writes `fig16_9_ope.png` (Figure 16.9), and `--mode all` writes both. A single mode writes only its own panel, as `fig16_8_panel_<mode>.png` or `fig16_9_panel_<mode>.png`.

The `ope` and `support` modes build their own logs: the logging policy is epsilon-greedy around a "production" ridge model and records the propensity of every logged action at decision time. The target policy is the greedy policy of a LinUCB model trained on separate traffic. The `mab` and `contextual` modes measure regret only and do not write logs. Replay is valid only under uniform logging, so it appears only in `support`, at logging epsilon 1.0. The direct method uses a per-arm linear reward model, which is deliberately misspecified against the logistic truth, so its bias does not disappear with more data, and its size and sign depend on where the logged data fall.

### Schematic figures (Sections 16.2.2, 16.5.2, 16.7.2)

```bash
python interleaving_simulation.py --mode example     # writes the Figure 16.6 input
python make_schematics.py                            # Figures 16.1, 16.4, 16.6
```

`make_schematics.py` reads `outputs/interleaving/interleaving_example.json`; if it is missing, it regenerates the example with the same seed.

## Figure Map

| Figure | Section | Produced by |
|--------|---------|-------------|
| 16.1 The evaluation ladder | 16.2.2 | `make_schematics.py` |
| 16.2 Variance reduction with pre-period data (CUPED) | 16.4.4 | `ab_test_simulation.py --mode cuped` |
| 16.3 Peeking inflates false positives | 16.4.5 | `ab_test_simulation.py --mode peeking` |
| 16.4 How experiment designs contain interference | 16.5.2 | `make_schematics.py` |
| 16.5 Novelty and primacy effects over time | 16.6.1 | `ab_test_simulation.py --mode novelty` |
| 16.6 Team-draft interleaving, one impression | 16.7.2 | `make_schematics.py` (from `--mode example`) |
| 16.7 Interleaving reaches correct decisions with fewer sessions | 16.7.3 | `interleaving_simulation.py --mode sensitivity` |
| 16.8 Regret of bandit policies | 16.8.3 | `bandit_ope_simulation.py --mode mab contextual` |
| 16.9 Off-policy estimators, bias, variance, and support | 16.9.3 | `bandit_ope_simulation.py --mode ope support` |

The book's Figures 16.1, 16.4, and 16.6 are house-style redraws of these schematics; the book's Figure 16.6 traces a different seeded impression.

## Tests

```bash
python -m pytest tests -q
```

The suite runs in well under a minute and checks the estimators against known or closed-form answers, for example: Welch's test matches SciPy; the sample-size formula gives the 16 sigma squared over delta squared rule of thumb; theta in CUPED equals the OLS slope and shrinks variance by exactly 1 minus rho squared in sample; CUPED is unbiased across A/A runs; the delta method is exact for a constant denominator and matches the user bootstrap; the SRM test flags a 50.5/49.5 split at a million users and passes 50/50; the mSPRT likelihood ratio matches numerical integration and keeps its false positive rate under continuous monitoring; Benjamini-Hochberg is a step-up procedure and agrees with SciPy's `false_discovery_control`; the Yambda loader builds the right windows and totals from a tiny hand-made file; team-draft matches a hand-worked example, gives balanced teams for identical rankers, and gives neither team an edge when clicks depend only on position; LinUCB's Sherman-Morrison updates equal the ridge closed form and its regret is sublinear; IPS equals the on-policy mean when target equals logger; DR equals DM when the reward model is exact; and IPS, SNIPS, and DR are unbiased against a closed-form policy value.

## Expected Runtime

With `--quick`, each script finishes in under a minute on a laptop CPU. On a 4-vCPU cloud VM, each full run takes 2 to 3 minutes, and all of them together take about 10 minutes. See `CLOUD_RUN.md` for exact commands, measured runtimes, hardware, and output files.

## References

- Kohavi, Tang, and Xu (2020), *Trustworthy Online Controlled Experiments*
- Deng, Xu, Kohavi, and Walker (2013), CUPED
- Deng, Knoblich, and Lu (2018), the delta method in metric analytics
- Fabijan et al. (2019), sample ratio mismatch
- Johari, Koomen, Pekelis, and Walsh (2017), always-valid inference and the mSPRT
- Radlinski, Kurup, and Joachims (2008), team-draft interleaving
- Auer, Cesa-Bianchi, and Fischer (2002), UCB1
- Chapelle and Li (2011), Thompson sampling
- Li, Chu, Langford, and Schapire (2010), LinUCB
- Li, Chu, Langford, and Wang (2011), replay evaluation
- Dudík, Langford, and Li (2011), doubly robust estimation
- Swaminathan and Joachims (2015), self-normalized IPS
- [Yandex Yambda Dataset](https://huggingface.co/datasets/yandex/yambda)
