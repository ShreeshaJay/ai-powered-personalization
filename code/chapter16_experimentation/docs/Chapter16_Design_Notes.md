# Chapter 16 Design Notes: Experimentation and Bandits

Implementation notes and run log for the Chapter 16 companion code. Every number in this file is a simulation output; the chapter describes these results qualitatively and quotes none of them.

## 1. Scope

Implemented: the three entry scripts of the companion-code plan (`ab_test_simulation.py`, `interleaving_simulation.py`, `bandit_ope_simulation.py`), the schematic script (`make_schematics.py`), the shared modules (`stats_utils.py`, `interleaving.py`, `bandits.py`, `ope.py`), and a pytest suite. Entry-point names, modes, and flags follow the plan.

Deferred: the Open Bandit Dataset extension and the interference simulation. Interference is taught through Figure 16.4 and Table 16.2 only.

## 2. Departures from the Plan

| Plan | Implementation | Reason |
|------|----------------|--------|
| Use the existing shared project environment | A dedicated environment, pinned in `requirements.txt` | The pinned versions are exactly the tested ones, independent of other chapters |
| Plain `run_tests.py` if pytest is missing | pytest suite in `tests/` | pytest is installed in the new environment |
| `design_notes_ch16.md` in the folder root | `docs/Chapter16_Design_Notes.md` | Mirrors the published Chapter 7 and 8 layout |
| No shared helper module named | `common.py` for CLI flags, output writing, plot style, and the data path | Avoids repeating the same code in four scripts |
| Interleaving simulator without per-user activity | Each user has a lognormal activity multiplier on click probability | Without between-user variance, the simulator lacks the noise source that interleaving removes, and the comparison would understate interleaving for the wrong reason |
| Group sequential alpha spending (optional) | Not implemented | Optional in the plan; Figure 16.3 needs only the four rules it lists |
| Linear Thompson sampling (optional) | Not implemented | Optional in the plan |
| Balanced interleaving (optional) | In `interleaving.py` with tests; not in the sensitivity figure | Figure 16.7 compares team-draft with A/B only |
| Figure 16.1 "candidates remaining" label above the staircase | Small key at the top left; dots above each rung | The label collided with the second rung |
| Figure 16.2(a) fitted line over all users | Line fitted on users with pre-period data; users without it stay visible at zero | About a tenth of Yambda users have no pre-period listens, which pulls a log-scale line off the main cloud. CUPED itself still uses all users |
| Figure 16.9(b) shaded region | Fixed shading for logging epsilon up to 0.03 | Simple and stable across seeds |

Additions not in the plan: `--metric rec_listens` lets the A/B script use recommendation-driven listens (`is_organic == 0`) as the count metric. The `aa`, `srm`, `power`, and `ratio` modes also write supplementary PNGs, which are not chapter figures.

## 3. Data Facts Verified for This Code

Probe of Yambda `flat/50m/listens.parquet` (SHA-256 `eed9cbd0…0592d1`, 369,110,332 bytes; the cloud run downloads the same file from Hugging Face and records its hash in `logs/yambda_file.txt`):

- 46,467,212 listens from 9,238 users.
- `timestamp` is in seconds from the first event in the dataset, rounded down to a multiple of 5 seconds. The Yambda paper defines it as T' = floor((T - T_start) / 5) x 5 (Section 3.3 of arXiv:2505.22238); the dataset card's phrase "binned into 5s units" means this 5-second grid, not a count of 5-second intervals. The data agree: every value is a multiple of 5; the maximum is 26,000,000 s, about 301 days, which matches the paper's 300-day train, 30-minute gap, and 1-day test split; for near-complete plays, the median gap to the user's next listen equals the track length in seconds; and hourly volume shows a clear daily cycle when timestamps are read as seconds and none when they are read as 5-second units. So `day = timestamp // 86400` is correct.
- `played_ratio_pct` can exceed 100 (maximum 159); the dataset card explains this as rewinding and replaying.
- 9,214 users are active in the last 28 days; 10.8% of them have no listens in the preceding 28 days. The Yambda paper describes sampling users who had at least one action in the final period of its observation window, which is consistent with almost every user being active in the experiment window.

## 4. Implementation Notes

### 4.1 `ab_test_simulation.py`

- **Random streams.** Each mode draws from its own generator, `common.mode_rng(seed, mode)`, seeded from the run seed and the mode name, so running one mode alone gives exactly the numbers and figure it gives inside `--mode all`. The synthetic panel has its own stream too. The other two scripts follow the same rule.
- **Assignment** hashes the unit ID with an experiment-specific salt (SHA-256 of the salt string, mixed into the ID with SplitMix64), so the same user always lands in the same arm and different salts give independent splits.
- **Effects** for count metrics: positive relative effects add Poisson(effect × y) events, negative ones thin events binomially; both give E[y'] = (1 + effect) y. For completion rate, each non-completed listen flips to completed with probability `--ratio-flip`.
- **Resampling.** `srm` (200,000 users), `power` (up to 100,000), and `novelty` (100,000) resample Yambda users with replacement. Copies of one user are assigned independently but share behaviour, which makes large-population results somewhat optimistic. `aa`, `cuped`, `ratio`, and `peeking` use the real users only.
- **SRM bug.** In the treatment arm, each user in the top decile of window listens loses their logs with probability `--srm-drop` (0.2). Quick mode keeps the full population size, because with too few users the chi-square check cannot see a bug of this size.
- **Winsorizing** caps at the pooled 99th percentile, a rule fixed before the analysis.
- **Peeking.** Users enter on their first active day. At each daily look, the metric is cumulative listens since entry among users who have entered. The mSPRT uses the Welch plug-in variance and a fixed mixing variance tau² = (tau_rel × final-day mean)², with tau_rel = 0.05. The always-valid guarantee assumes independent increments, which a cumulative per-user metric with arriving users only approximates. The simulation, not the theory, is the evidence that it holds here.
- **Novelty.** Relative effect delta(d) = delta_inf + (delta_0 - delta_inf) exp(-d / tau) by days since first exposure. Novelty scenario: 0.15, 0.03, 4. Primacy scenario: -0.08, 0.04, 5. The pooled two-week estimate uses each entered user's total over window days 0 to 13. After the effect is injected, user-days are winsorized at the pooled 99.9th percentile, a rule fixed in advance and blind to arm. Without it, one Yambda user who logs 13,712 listens on the final day of the data (about 6% of that day's listens), repeated about eleven times by the resampling, quadruples the width of the last day's interval and drags its estimate below zero in both panels.

### 4.2 `interleaving_simulation.py`

- Latent dimension 16, 5,000 users, 1,000 items, 100 candidates per session, lists of 10.
- True logit = 1.5 u·v + item bias + (-3); relevance = sigmoid(logit).
- Per-user activity ~ LogNormal(-0.5, 1), mean 1, multiplies examination probability 1/k (capped at 1).
- Ranker A noise 1.0; ranker B noise 1.1 (small gap) or 1.4 (large gap).
- Each pool simulates its sessions once, storing A-only clicks, B-only clicks, and the team-draft outcome; experiments subsample with replacement. The `sensitivity` pools hold 200,000 sessions. The `sanity` pools hold 1,000,000 sessions and each test draws 2,000, because every test from one pool shares that pool's chance imbalance. With 200,000 sessions and tests of 10,000, the equal-quality pool in an earlier version happened to sit 2.4 pool standard errors from a 50% win rate, which lifted the interleaving false positive rate to 0.08 by itself.
- The `example` mode searches seeded sessions for one that is useful to display: lists that share at least three items in different orders, each ranker winning at least one coin flip, and a non-tied outcome. `attempt` in the JSON records which session was used.

### 4.3 `bandit_ope_simulation.py`

- K = 10 arms, 8 context features plus an intercept; weights ~ N(0, 1/8); biases centred on logit(0.03) with spread 0.3.
- `mab` uses Bernoulli arm means drawn uniformly from [0.02, 0.10]. Epsilon-greedy uses epsilon 0.1; LinUCB uses alpha 0.5 and ridge 1.
- All policies in one seed see the same contexts and the same pre-drawn rewards for every arm.
- OPE production policy: greedy on per-arm ridge fitted to 2,000 uniform warm-up logs. Target: greedy LinUCB after 5,000 online rounds on separate traffic. The true value comes from Monte Carlo over 10⁶ contexts using the known probabilities.
- DM uses a per-arm ridge regression on the full context (intercept plus 8 features), misspecified against the logistic truth by design. It is a contextual model, only of the wrong functional form.
- The `ope` and `support` modes generate their own logs. The `mab` and `contextual` modes measure regret and log nothing. Replay is computed only in `support`, under uniform logging.
- `--mode` accepts several modes; `--mode mab contextual` and `--mode ope support` give the full Figures 16.8 and 16.9.
- At logging epsilon 0.1, the largest importance weight is K/epsilon = 100, so clipped IPS with M = 100 equals IPS. It is kept in the CSV and omitted from the box plot.

## 5. Qualitative Observations from the Full Run

These hold in the full cloud run recorded in Section 6, on Yambda unless stated otherwise.

- **CUPED:** the observed variance ratio closely matches 1 - rho² computed from the pre/post correlation, both estimators centre on the injected effect, and the CUPED estimate's standard deviation is about 30% smaller.
- **Ratio metrics:** per-listen confidence intervals undercover badly in A/A runs, because a user's listens are highly correlated. The delta method and the user bootstrap reach nominal coverage. The per-listen test's apparent "power" against a small injected effect is mostly its false positive rate; the correct delta-method test has little power against an effect that small.
- **SRM:** at 200,000 users, the chi-square check flags the bug in most replications but not all, and never flags a clean split. The expected flag rate for this bug is close to 88%: dropping a fifth of the top decile removes about 2,000 users from one arm, against a 0.001-level threshold of about 1,460 and a chance standard deviation of about 450. The naive metric comparison reports a significant, entirely spurious drop in every buggy replication.
- **Peeking:** daily looks with stop-at-first-significance more than triple the false positive rate by day 28. Bonferroni across 28 looks and the mSPRT both stay far below alpha and nearly coincide, so both are conservative at this horizon. The single fixed-horizon analysis sits at the nominal rate.
- **Novelty and primacy:** the pooled two-week estimate overstates the long-run effect under novelty, and under primacy it is close to zero, slightly negative, against a positive long-run effect. After winsorizing, late-day intervals are no wider than mid-window ones.
- **Interleaving:** identical rankers give 50% wins and false positive rates near alpha for both methods, and so do equal-quality rankers with independent noise. Team-draft interleaving reaches 80% correct decisions with roughly a third of the sessions the A/B test needs for the large gap, and with a little under half for the small gap.
- **Bandits:** in the context-free problem, epsilon-greedy and Thompson sampling end with far lower regret than UCB1, whose exploration bonus keeps it exploring at low reward rates. Thompson sampling's regret flattens the most; epsilon-greedy's has the widest spread across seeds. Under context, every context-free policy accrues close to linear regret against the context-aware oracle, while LinUCB's flattens.
- **Off-policy evaluation:** at logging epsilon 0.1, IPS, SNIPS, and DR are close to unbiased. SNIPS is barely tighter than IPS here, and DR only slightly tighter. DM has the smallest spread but a persistent bias, and clipping at M = 10 gives a large downward bias. As epsilon falls, IPS-family error grows and the effective sample size collapses; with 1,000 logs and epsilon of 0.03 or less, IPS is biased downward because the rare target actions mostly never appear. DM's error does not shrink with more exploration or more logs. Its bias is positive at every epsilon below 1 and every log size, and negative and largest in size under uniform logging, where its RMSE is the highest of the sweep. A likely mechanism, not tested separately: a linear model fitted over all contexts underpredicts the convex low-rate logistic reward where the target policy chooses each arm, while a model fitted mostly where production chose the arm overpredicts there. A ten-seed diagnostic during code review gave the same sign pattern in nearly every environment seed. Under uniform logging with a deterministic target, replay and SNIPS are the same estimator, and a test checks this identity.

## 6. Run Log

### 2026-11-08, local smoke runs (`--quick`, seed 16; checks only)

Environment: Windows 11 laptop, Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1, Polars 2.1.0, matplotlib 3.11.2.

Tests: 46 passed. Every script exits 0 with `--quick`, each in under 35 seconds of wall-clock time including start-up. These are the quick results of the final code, for comparison with a reader's own smoke run.

A/B on Yambda, quick:

- Panel: 9,214 users; mean 640 and median 395 listens per user over 28 days; 99th percentile 3,338; pre/post correlation 0.73.
- `aa` (200 sims): false positive rate 0.055 raw, 0.050 CUPED.
- `srm` (200,000 users, 10 sims): flagged 8 of 10 with the bug and 0 of 10 without.
- `cuped` (200 sims): rho 0.726; predicted variance ratio 0.473, observed 0.467; theta 0.757.
- `ratio` (200 sims, 30 bootstrap sims): A/A coverage 0.08 per-listen, 0.95 delta method, 0.97 bootstrap.
- `peeking` (300 sims): false positive rate by day 28 of 0.12 for naive daily looks, 0.00 Bonferroni, 0.003 mSPRT, 0.03 fixed horizon.
- `novelty` (20,000 resampled users): pooled two-week estimate +5.8% for novelty (long-run +3%) and -2.3% for primacy (long-run +4%).

Interleaving, quick: identical rankers win rate 0.500; large gap NDCG@10 0.811 against 0.713; small gap 0.811 against 0.785.

Bandits, quick (T = 3,000, 4 seeds): median final regret in the context-free problem was uniform 99, epsilon-greedy 44, UCB1 89, Thompson 56. In the contextual problem, uniform 296, epsilon-greedy 269, UCB1 293, Thompson 269, LinUCB 106.

OPE, quick (30 reps, 10,000 logs, epsilon 0.1): true value 0.1315; mean ESS 176. RMSE was IPS 0.026, SNIPS 0.026, DM 0.012 (bias +0.011), DR 0.025, clipped IPS (M = 10) 0.054 (bias -0.054).

### 2026-11-08, full cloud run `ch16-20261108-2341`

Platform: Google Compute Engine `e2-standard-4` (4 vCPU, 16 GB RAM), zone us-central1-a, Debian 12, Python 3.11.2, with the pinned `requirements.txt` (NumPy 2.4.6, SciPy 1.17.1, Polars 2.1.0, matplotlib 3.11.2, pytest 9.2.0). Yambda `flat/50m/listens.parquet` was downloaded from Hugging Face on the VM; its SHA-256 matched the probe in Section 3. Every command in `CLOUD_RUN.md` ran in order with seed 16 and exited 0.

Three earlier full runs on the same day (`ch16-20261108-2231`, `-2301`, `-2322`) validated intermediate versions of the code. From `-2301` onward the numbers are identical and only figure layout changed; `-2231` predates the novelty winsorizing and the larger sanity pool described in Section 4.

Runtimes on the VM, as wall-clock seconds, with peak resident memory from `/usr/bin/time -v`:

| Step | Command | Seconds | Peak memory (MB) |
|------|---------|---------|------------------|
| Setup | `apt-get install python3-venv python3-pip time zip` | 45 | |
| Setup | `pip install -q -r requirements.txt` into a fresh venv | 34 | |
| Setup | Yambda download with `wget` | 4 | |
| 1 | `python -m pytest tests -q` (46 passed) | 6.2 | 337 |
| 2 | `ab_test_simulation.py --mode all --data-dir ...` | 142.4 | 557 |
| 3 | `ab_test_simulation.py --mode all --synthetic` | 125.5 | 492 |
| 4 | `interleaving_simulation.py --mode all` | 148.1 | 341 |
| 5 | `bandit_ope_simulation.py --mode all` | 168.9 | 489 |
| 6 | `make_schematics.py --example-json ...` | 2.8 | 107 |
| 7 | `zip -r chapter16_outputs.zip outputs` | 0.2 | 2 |

The six chapter steps took 594 seconds (9.9 minutes) after 83 seconds of setup. The VM lived 14.5 minutes: about one minute of boot before the startup script began, 11.9 minutes of script including uploads to Cloud Storage, and 1.5 minutes of shutdown. The slowest modes were `power` (76 s on Yambda, 60 s on synthetic data), interleaving `sanity` (100 s), and bandit `support` (101 s). In run `-2322`, on an identical VM with identical outputs, steps 2 to 5 took 104, 99, 130, and 166 seconds; `e2` machines share physical cores, so step times can vary by about 40% between runs.

A/B on Yambda (9,214 users):

- `aa` (1,000 sims): false positive rate 0.053 raw, 0.043 CUPED; Kolmogorov-Smirnov p = 0.69 for uniform p-values.
- `srm` (200,000 users, 200 sims): flagged 85.5% of runs with the bug and none without; the naive test was significant in every run with the bug and in 4.5% without. Example run with the bug: arm sizes 98,241 and 99,876, SRM p = 0.0002, naive relative difference -5.2%.
- `power` (300 sims, effect 2%), power at 2,000 / 5,000 / 10,000 / 20,000 / 50,000 / 100,000 users: raw 0.06 / 0.11 / 0.13 / 0.20 / 0.52 / 0.78; winsorized 0.06 / 0.11 / 0.13 / 0.21 / 0.54 / 0.79; CUPED 0.10 / 0.14 / 0.25 / 0.43 / 0.81 / 0.96.
- `cuped` (1,000 sims): rho 0.726, theta 0.757; predicted variance ratio 0.473, observed 0.481. True effect 12.79 listens per user; mean estimate 12.68 raw and 12.74 CUPED; standard deviation 15.6 raw and 10.8 CUPED.
- `ratio` (1,000 sims; bootstrap on 200 of them with 200 replicates): completion rate 0.602 over 5.89 million listens. A/A coverage 0.072 per-listen, 0.950 delta method, 0.950 bootstrap; the delta-method standard error is a median 20.8 times the per-listen one. Rejection rate against a 1% flip: 0.906 per-listen, 0.070 delta method.
- `peeking` (2,000 sims, 28 daily looks): false positive rate by day 28 of 0.178 for naive daily looks, 0.0075 Bonferroni, 0.0075 mSPRT; 0.049 for the single fixed-horizon analysis.
- `novelty` (100,000 resampled users; user-day caps 379 and 371): pooled two-week estimate +8.0% under novelty (long-run +3%) and -0.2% under primacy (long-run +4%). The last day's estimates are +2.4% and +3.3%, with intervals as wide as mid-window ones.

A/B on synthetic data (9,429 users; mean 92 and median 45 listens; pre/post correlation 0.81): `aa` 0.056 raw and 0.046 CUPED; `srm` flagged 88.5% with the bug and none without; `cuped` predicted variance ratio 0.346, observed 0.349; `ratio` coverage 0.265 per-listen, 0.952 delta method, 0.955 bootstrap; `peeking` 0.151 naive, 0.009 Bonferroni, 0.007 mSPRT, 0.060 fixed horizon; `novelty` pooled +8.4% (long-run +3%) and -0.9% (long-run +4%).

Interleaving:

- `sanity` (pool of 1,000,000 sessions, 2,000 sessions per test, 500 tests): identical rankers, win rate 0.500 among non-ties, false positive rate 0.052 for the A/B test and 0.046 for interleaving. Equal-quality rankers with independent noise: win rate 0.499, false positive rates 0.044 and 0.030.
- `sensitivity` (pools of 200,000 sessions, 500 tests per size): large gap, NDCG@10 0.811 against 0.714, clicks per session 0.996 against 0.880, interleaving win rate 0.568. Share of correct significant decisions: interleaving 0.570 at 562 sessions and 0.828 at 1,000; A/B 0.600 at 1,778 and 0.834 at 3,162. Small gap, NDCG@10 0.811 against 0.785, clicks per session 0.996 against 0.957, win rate 0.518. Interleaving 0.678 at 10,000 and 0.902 at 17,782; A/B 0.616 at 17,782 and 0.812 at 31,622.
- `example`: session attempt 14; team B wins 2 clicks to 0 (Figure 16.6).

Bandits (T = 20,000, 20 seeds), median final regret, with the ratio of second-half to first-half regret in brackets:

- Context-free: uniform 660 (0.99), epsilon-greedy 117 (0.60), UCB1 499 (0.86), Thompson sampling 137 (0.29).
- Contextual (oracle value 0.142, best single arm 0.064, which is the best arm in 23% of contexts): uniform 1,957 (1.01), epsilon-greedy 1,661 (0.98), UCB1 1,910 (0.99), Thompson sampling 1,673 (0.94), LinUCB 264 (0.32).

Off-policy evaluation:

- `ope` (logging epsilon 0.1, 10,000 logs, 200 reps): true target value 0.1312; production policy value 0.1152; target and production agree on 43.5% of contexts; mean ESS 176. Bias, standard deviation, and RMSE: IPS -0.0036, 0.0221, 0.0223; SNIPS -0.0028, 0.0220, 0.0221; DM +0.0118, 0.0064, 0.0134; DR -0.0032, 0.0202, 0.0204; clipped IPS (M = 10) -0.0531, 0.0036, 0.0532; clipped IPS (M = 100) identical to IPS.
- `support` (100 reps), at logging epsilon 0.01 / 0.03 / 0.1 / 0.3 / 1.0. With 10,000 logs (Figure 16.9b): mean ESS 19 / 54 / 176 / 513 / 1,001; RMSE IPS 0.081 / 0.039 / 0.025 / 0.013 / 0.0098; SNIPS 0.078 / 0.036 / 0.023 / 0.013 / 0.0093; DM 0.012 / 0.010 / 0.014 / 0.011 / 0.021; DR 0.071 / 0.037 / 0.022 / 0.012 / 0.0092; replay 0.0093 at epsilon 1.0, equal to SNIPS. DM bias: +0.007 / +0.011 / +0.017 / +0.010 / -0.021 with 1,000 logs, +0.005 / +0.007 / +0.012 / +0.010 / -0.020 with 10,000, and +0.004 / +0.008 / +0.012 / +0.010 / -0.019 with 100,000. IPS bias with 1,000 logs is -0.029 at epsilon 0.01 and -0.018 at 0.03.
