# Chapter 15 Exercise: Design Notes and Run Log

This file records how the silent-degradation exercise is built, which Chapter 6 code it depends on, where it departs from that code, and every run made so far. Run-specific numbers live here and in the `outputs/` files, never in the chapter prose.

## 1. Purpose

Section 15.11 describes a reproducible silent failure. A ranker's important features arrive as schema-valid defaults (zero or the training mean), and the exercise records which monitors fire and which stay green, for all traffic and for a hashed 30 percent user segment.

## 2. Entry Points

| Entry point | What it demonstrates |
|---|---|
| `python run_exercise.py` | The full exercise on Yambda: train, corrupt, monitor, write `outputs/<run_name>/` |
| `python run_exercise.py --synthetic` | The same code path on generated data (smoke run; numbers say nothing about Yambda) |
| `silent_degradation.ipynb` | The reader walkthrough: the same experiment, with plots of the corrupted features, the KS gap under mean-fill, score distributions, reliability, and slices |
| `python -m pytest tests -q` | Known-answer tests of the estimators, corruption, windows, point-in-time counts, and an end-to-end synthetic run |

The library modules are `monitoring_signals.py` (estimators and thresholds), `corruption.py` (fills and the hashed segment), `ranker.py` (the retrained Chapter 6 ranker), and `synthetic_yambda.py`.

## 3. Dependency on Chapter 6

**Vendored copy (since 2026-11-09).** `ch6_polars_pipeline_vendored.py` is a verbatim copy of `code/chapter6_ranking/utils/polars_pipeline.py` at commit `8e8e71ab0a9ad9da387446b89f8432a23ad2d7fd` (2026-11-09), SHA-256 with LF endings `5eeb0a86dab5094d6c6b6dcffd3f4db858abebb04c4ba9150179f1e64d33e877`. Its header records both, and `tests/test_vendored_ch6.py` fails if the body no longer hashes to the recorded value. The exercise loads it by file path (no `sys.path` change, no torch import) unless `--ch6_dir` names another `chapter6_ranking` folder. Each run records the source commit, the recorded hash, the body hash, and whether they match in `run_manifest.json` under `provenance.chapter6`.

**What is used.** `PolarsPipeline` (with `PolarsFeatureEncoder` from the same file), `PolarsHistoricalFeatures`, `split_training_history`, and `POLARS_DTYPES`. The exercise does not use `load_yambda_polars`, `split_validation_window`, or any Chapter 6 training script.

**What `074886c` changed that matters here.** Relative to the `0c3a6c9` file used by the runs in sections 9.1 to 9.3:
- Time units: `load_yambda_polars`, `hour_of_day` and `day_of_week` now use seconds. The exercise already used seconds for its split and recomputed both calendar features, so their values do not change; `ranker.add_calendar_features` was removed and a smoke test now checks the Chapter 6 values.
- `previous_listen_count` is now a within-frame cumulative count (`int_range(len).over(pair, order_by=timestamp)`). The exercise still overwrites it with its cross-window, lookback-limited count (section 4.2), so its values do not change.
- `split_training_history` and `transform(history=...)` are new. The exercise uses both to fix its own training-row leak (section 4.3); this is the change that moves the numbers.
- `pl.count()` became `pl.len()`, so the exercise no longer suppresses a DeprecationWarning, and likes and dislikes are filtered to the listens' time span (unused here). The Chapter 6 Polars floor is now 1.9.

**Polars version.** `requirements.txt` pins Polars 1.44.2. On Polars 2.x the Chapter 6 transform does not keep row order, which in Chapter 6 makes tied `previous_listen_count` values and seeded results vary between runs. The exercise stays on 1.x, and `ranker.total_order` also sorts every transformed frame on all its columns, so the rows XGBoost sees do not depend on the order Polars returns. Only 1.44.2 has been tested.

**Earlier file states.** The cloud run in section 9.3 used a fresh clone of the public repository at HEAD `8c3463013ab467d112629904fdd69bda8cce08a5`, where `utils/polars_pipeline.py` was last committed in `0c3a6c9d357cd5205a20a071869f80a7ddcbe322` (git blob `7871b4e`). The laptop runs in sections 9.1 and 9.2 used the same commit plus an uncommitted local addition of `split_validation_window` (SHA-256 of that working file, with CRLF endings, `43b1911fbbf936dcd5c728d466813041320e4f7a90507fa9c7107da6cb775b54`). That addition was later committed in `5757e959a5c8a963c1e07f13666f5a1af8bb4fe7` (2026-11-08, git blob `71ef0cb`) as a single hunk that changes nothing the exercise imports. All three states give the same exercise. Every run records the file's SHA-256, its last commit, the repository HEAD, and whether it had uncommitted changes in `run_manifest.json` under `provenance.chapter6`. The SHA-256 depends on line endings: a Windows checkout with CRLF endings hashes differently from the same commit on Linux.

**Location before vendoring.** The runs in sections 9.1 to 9.3 imported Chapter 6 from `--ch6_dir`, then `CH6_RANKING_DIR`, then `../chapter6_ranking`. The cloud run in section 9.3 used the sibling folder; the laptop runs in sections 9.1 and 9.2 set `CH6_RANKING_DIR` to the repository's `code/chapter6_ranking`. `CH6_RANKING_DIR` is no longer read.

**Early stopping.** XGBoost early-stops on a validation window carved from the training data: the last 10 percent of the pre-reference rows by timestamp, the same rule as the new Chapter 6 `split_validation_window`. The test day is never passed to `fit`, whatever state the Chapter 6 training scripts are in.

## 4. Departures From the Chapter 6 Code

### 4.1 Yambda timestamps are seconds

Chapter 6 converts days to timestamp units at 17,280 units per day ("1 unit = 5 seconds"). The data say one unit is one second:

- The maximum timestamp is 26,000,000 and every value is a multiple of 5. In seconds that spans about 301 days, which matches the "300 days" in the Chapter 6 docstrings; in 5-second units it would be about 1,504 days.
- Listen counts binned at 360 units have an autocorrelation of +0.892 at a lag of 86,400 units and +0.914 at 7 × 86,400, against +0.201 at 17,280. The five largest periodogram peaks sit at periods of 86,379, 43,189, 86,083, 86,677 and 86,977 units.
- Read as seconds, the hour-of-day profile (`timestamp // 3600 % 24`) has a clear daily cycle: the busiest hour has 5.4 times the listens of the quietest. Read as 5-second units (`timestamp * 5 // 3600 % 24`), the ratio is 1.07, close to flat.

(Script: bin `timestamp // 360` over all 46,467,212 listens in `flat/50m`, detrend with a 2,401-bin moving average, then the autocorrelation and the FFT periodogram. It was run in scratch space and is not part of the exercise.)

Consequences in Chapter 6 before commit `074886c`, which fixed them: `train_days=30` covers 6 days, `test_days=1` covers 4.8 hours, the 30-minute gap is 6 minutes, `hour_of_day = (timestamp * 5 // 3600) % 24` cycles every 4.8 hours, and `day_of_week = (timestamp * 5 // 86400) % 7` steps every 4.8 hours and cycles every 1.4 days. The "about 1.3M rows for 30 days" in the Chapter 6 notes is therefore 6 days of data.

What the exercise does: its own global split in seconds (`ranker.load_global_split`, the same layout and boundary conventions as `load_yambda_polars`). Until 2026-11-09 it also recomputed `hour_of_day` and `day_of_week` from seconds after the Chapter 6 transform; Chapter 6 commit `074886c` made the same fix, so the vendored transform now produces those values directly. Yambda timestamps start at an undisclosed origin, so both are offsets from that origin, not local clock values.

### 4.2 `previous_listen_count` across windows

Before commit `074886c`, the Chapter 6 transform computed `pl.count().over(['uid', 'item_id'], order_by='timestamp') - 1`. `order_by` sorts the window but `count` still returns the size of the whole group, so every play of a pair gets the pair's total minus one, including later plays. Reproducer: three plays of one (user, item) pair at timestamps 10, 20, 30 all get 2; a point-in-time count gives 0, 1, 2. The comment above the expression and the module docstring describe a cumulative count, so the code and its documentation disagree.

What the exercise does instead: `ranker.add_pit_user_item_features` stacks all windows in time order, counts each pair's earlier plays with `cum_count`, and subtracts the plays older than the lookback (section 4.3), using an as-of join. `has_listened_before` is derived from it. `tests/test_ranker.py` covers both the cumulative and the windowed count. Since 2026-11-09 the stack also includes the training rows' history window (section 4.3), so training rows see the full lookback too. Before that the loaded history started at the fit window, and a training row saw only the plays since the fit window began.

Chapter 6 commit `074886c` replaced its whole-group count with a cumulative count within each transformed frame. That count still sees nothing from earlier windows, so the exercise keeps its own.

### 4.3 Daily lookup refresh and equal-length histories

Chapter 6 scores every later window with lookup tables fit once on the training window. Monitored that way, the test day is always one day staler than the reference day, so cold-start shares, join rates, and every lookup feature drift between the two days without any corruption. The exercise instead scores the reference day and the test day each with lookups rebuilt (`ranker.refresh_lookups`, a fresh `PolarsHistoricalFeatures`) from the fit-window-length history that ends where the day starts. That is how a daily feature-store job would serve them. The categorical encoders stay as fitted on the fit window, so every encoded input keeps the meaning the model was trained on. The re-listen count uses the same lookback.

Validation rows use the lookups fit on the fit window, which ends where the validation window starts. Training rows, since 2026-11-09, use lookups fit on the fit-window-length window before the fit window (`split_training_history` and `transform(history=...)` from Chapter 6, as the Chapter 6 trainers now do), so no training row's own label is in its user or item aggregates. `tests/test_smoke.py::test_training_aggregates_exclude_own_labels` checks each training row's `user_avg_completion` against the user's mean over that earlier window; it fails on the previous behaviour, where every one of the 6,679 synthetic training rows mismatched. If the data do not reach back a full fit window, Chapter 6 moves the earliest fit rows into the history; on Yambda the 30-day window starts about 270 days after the first event, so this does not happen.

Before 2026-11-09 the training rows took the fit-window lookups through `fit_transform`, so each training row's aggregates included its own label. The runs in sections 9.1 to 9.3 were made that way; section 9.4 records how the numbers moved when this was fixed.

### 4.4 Chapter 6 cold-start defaults are themselves silent fills

For users or items missing from the lookups, Chapter 6 fills the completion and organic-ratio features with global statistics (mean, median, or standard deviation), the count features with 0, `item_id_freq` with 0.0, and label codes with -1. The production pipeline therefore already applies a silent mean-fill and zero-fill to cold-start rows. The exercise keeps this behaviour; it is the reason zero-rate and mean monitors on count features move with the cold-start share.

## 5. Artifacts

The exercise loads no saved Chapter 6 model or `feature_encoder.pkl`: it retrains on a fixed, seeded window through the public Chapter 6 pipeline on every run. Each run saves `model/model.json` (XGBoost), `model/pipeline/` (the Chapter 6 encoders and fit-window lookups, written by `PolarsPipeline.save`), and `model/ch15_model_metadata.json`. The served-day lookups are rebuilt from the data and are not saved.

To reproduce a run exactly, pin:

| What | Value or source |
|---|---|
| Data | Yambda `flat/50m/listens.parquet`, 369,110,332 bytes, SHA-256 `eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1` (checked by every run) |
| Chapter 6 code | `ch6_polars_pipeline_vendored.py` (source commit and SHA-256 in its header and in `run_manifest.json`) |
| Packages | `requirements.txt` (exact versions tested) |
| Seeds | XGBoost `random_state` 42; segment salt 15; ranking tiebreak `splitmix64(row index + 42)` |
| Thresholds | `monitoring_signals.Thresholds` defaults, copied into `run_manifest.json` |

## 6. Monitoring Design

**Windows.** The global split, in days: [30-day train window][30-minute gap][test day], ending at the last event. The train window is carved, in time order, into the fit window, the validation window (early stopping only), and the reference day (the last day before the gap, never used for fitting or model selection).

**Reference and control.** Every monitor compares the scored test day with the scored reference day, the way a production monitor compares today with a trailing window. The training set, the other common reference, is not used because its lookup features come from an older window than any served day's (section 4.3) and its rows span weeks, so it would differ from a served day even without corruption. The uncorrupted test day (`clean`) is scored and monitored in exactly the same way, so every corrupted result can be read against what the same monitor reports with no corruption.

**Why one reference day.** A trailing multi-day reference would smooth some day-to-day noise in the extreme quantiles of heavy-tailed features. It would also mix in days served with older lookups, and it would not remove the largest clean-day shift in section 9.3, which comes from one account's behaviour on the test day itself. The clean control, scored on the same test day, is what separates the injected corruption from everything else that changed between the two days.

**Slices.** `aggregate` (all test-day rows), `segment` (users whose SplitMix64 hash, salt 15, falls below 0.3), and `rest`. Membership is per user, so the same users are in the segment on both days and on every machine.

**Targets.** The top two features by average gain, excluding identifier encodings (`uid_*`, `item_id_*`), label-encoded categories (`*_encoded`), and features with two or fewer distinct values. For a binary feature, a mean-fill rounded to the column's integer dtype would equal one of its two values and repeat the zero-fill or a one-fill. `--target_scope user` or `item` restricts the choice to the `user_*` or `item_*` lookup features.

**Fill values.** Zero, or the fit-window mean, rounded for integer columns so that each column keeps its dtype.

**Statistics.** Each monitor returns a statistic where larger means worse and fires when it exceeds its threshold:

| Monitor | Statistic | Threshold |
|---|---|---|
| Schema validation | Drop in the share of rows passing the type-and-range contract, plus 1 if any dtype changed | 0.001 |
| Rows per day | \|current / reference - 1\| | 0.25 |
| Join match rate | Drop in the share of user (item) keys found in the lookups | 0.02 |
| Null rate, zero rate | Rise in the share of rows that are null (exactly zero) | 0.01, 0.02 |
| Mean | \|mean shift\| / reference standard deviation | 0.2 |
| Quantiles | Largest \|shift\| among the 1, 5, 25, 50, 75, 95, 99th percentiles / reference SD | 0.2 |
| KS | Largest gap between the two empirical CDFs | 0.1 |
| PSI | Sum of (q - p) ln(q / p) over 10 reference-quantile bins plus a null bin, floored at 1e-4 | 0.25 |
| Score entropy | \|change\| in the entropy of a 20-bin histogram on [0, 1] | 0.1 |
| AUC, GAUC, NDCG@10 | Drop | 0.01 each |
| Log loss, ECE | Rise | 0.01 each |
| Calibration gap | Rise in \|mean prediction - observed rate\| | 0.01 |

The thresholds were fixed before any run and have not been tuned. PSI 0.25 is the common credit-scoring rule of thumb, which carries no stated error rate; the rest are round illustrative values. The KS p-value is recorded but never used to fire, because with hundreds of thousands of rows almost any difference is significant.

**Statuses.** FIRES (over threshold, clean not); FIRES, clean also over (both over, scenario worse than clean by more than the threshold); clean also over (both over, no material difference); green; false alarm (the clean control itself is over); exempt; n/a.

**Exemptions.** Computed but never alerted on: label-encoded identifiers (`uid_encoded`, and `item_id_encoded` when Chapter 6 label-encodes items), whose distribution tracks who is active that day; `day_of_week`, of which a one-day window holds at most two values; and quantile shift on two-valued features, whose quantiles jump between the two values whenever the share crosses a quantile level.

## 7. Expected by Construction

These follow from the design, before any data is seen:

- **Mean-fill.** Every row equals the fill value m, so the null and zero rates fall to zero. Both monitors alarm on a rise, so neither fires, but a feature with natural zeros on the reference day shows a drop that a two-sided monitor would catch. The mean monitor sees only |training mean - reference mean| / reference SD, which is small unless the served feature has drifted from training. The current CDF jumps from 0 to 1 at m, so the KS statistic is the larger of F_ref(m-) and 1 - F_ref(m). Those add up to 1 - P_ref(X = m), so KS ≥ (1 - P_ref(X = m)) / 2, which is at least 0.5 when no reference row equals m. PSI puts all current rows in the bin holding m, and each other bin with reference share p adds about p ln(p / 1e-4). The 1st and 99th percentiles collapse onto m, so the quantile monitor fires too. Mean-fill therefore escapes null-rate, zero-rate, and mean monitors but not quantile, KS, or PSI monitors.
- **Zero-fill.** The zero rate of each target rises by the corrupted share of rows. Null rate stays at zero.
- **Pipeline monitors.** The corruption overwrites values in two existing columns. Row counts, keys, and dtypes do not change, and zeros and means pass the range contract, so schema, rows-per-day, and join monitors report the same in every scenario as on the clean day.
- **Segment scenario.** The `rest` slice is identical to clean in every statistic (asserted in `tests/test_smoke.py`).
- **Ranking metrics.** GAUC and NDCG@10 compare scores within one user. A target that is constant across a user's rows (every `user_*` lookup feature) shifts all of that user's scores together, so these metrics move only through tree interactions with other features. AUC, log loss, and calibration compare scores across users and see the shift directly. Section 9.3 measures both the user-level default and `--target_scope item`.
- **Service metrics.** No serving code runs, so throughput, latency, and error rate are unaffected by construction. The exercise does not measure them.

## 8. Verification

`python -m pytest tests -q`: 78 passed (Windows, Python 3.11.9, the pinned packages; also on the cloud machine in section 9.4; 72 before the 2026-11-09 changes, section 9.3). Highlights:

- PSI of [0.5, 0.5] against [0.25, 0.75] is 0.2746530722, both from proportions and end to end on reference 0..9 with two bins.
- KS on [1,2,3] vs [4,5,6] is 1, on [1,2,3,4] vs [3,4,5,6] is 0.5, a point mass at 49.5 against 0..99 is 0.5, and it equals SciPy's statistic on tied integer samples. For a point mass at m against a reference with mass at m, KS equals the larger of P_ref(X < m) and P_ref(X > m), so the bound of section 7 holds; mean-fill lowers the zero rate by the reference share of zeros.
- Quantiles of [1,2,3,4] at 0.25, 0.5, 0.75 are 1.75, 2.5, 3.25; nulls and NaNs are ignored.
- ECE of predictions [0.1, 0.1, 0.9, 0.9] against labels [0, 1, 1, 1] is 0.25, and the calibration gap is -0.25. Log loss matches scikit-learn; GAUC matches size-weighted per-user scikit-learn AUC; NDCG matches the closed form, and the per-user mean of scikit-learn's `ndcg_score` over 40 users at k = 1, 3, and 10. Target selection honours `--target_scope`.
- SplitMix64 matches a pure-Python implementation, and its first output for seed 0 is 0xE220A8397B1DCDAF. A 0.3 segment over 100,000 IDs has 30 percent ± 1 point of the IDs.
- Point-in-time counts: three plays of one pair give 0, 1, 2 (where the Chapter 6 expression gives 2, 2, 2); ties across windows follow window order; the windowed count drops plays older than the lookback.
- End to end on synthetic data: the Chapter 6 pipeline still produces the 23 expected columns; windows are ordered and disjoint; training rows' `user_avg_completion` equals the user's mean over the window before the training rows (which fails on the previous in-sample behaviour); the Chapter 6 calendar features equal `timestamp // 3600 % 24` and `timestamp // 86400 % 7`; served days use refreshed lookups with the fit-window encoders; under mean-fill the null and zero rates of both targets are green and KS and PSI fire; pipeline monitors never fire.
- The vendored Chapter 6 file hashes to the SHA-256 in its header (LF endings), and the header names the source path and commit `8e8e71a`.
- `ranker.total_order` gives the same frame for any input row order, and plays in the training history window count towards training rows' `previous_listen_count`.

## 9. Run Log

Sections 9.1 and 9.2: Windows 11 laptop (AMD Ryzen 9 8945HS, 16 logical CPUs), Python 3.11.9, packages as in `requirements.txt`, Chapter 6 working copy as in section 3, outputs in scratch space. Section 9.3: the full run on a Google Compute Engine VM, with the committed Chapter 6 file, before the training-row leak fix. Section 9.4: the same run with the current code (vendored Chapter 6 at `074886c`, training rows' aggregates from the earlier window). Only section 9.4 describes the current exercise; section 9.3 is kept for comparison.

### 9.1 Synthetic smoke run (code-path check only; not for the chapter)

`python run_exercise.py --synthetic`, 2026-11-08. 400 synthetic users, 40 days. Rows: fit 46,051, validation 5,118, reference 1,712, test 1,800. Best iteration 32, validation AUC 0.8323. Targets: `item_avg_completion`, `user_median_completion`. Timings: data 1.1 s, training 7.7 s, monitors 5.4 s, total 14.2 s (23 s wall clock including imports). Under mean-fill both targets were green on null rate, zero rate, and mean (mean statistics 0.050 and 0.0003), and fired on KS (0.558, 0.517) and PSI (8.39, 8.46). Schema, rows-per-day, and join monitors were green in every scenario and slice.

### 9.2 Real-data smoke run (3 days, 20 trees; not for the chapter)

`python run_exercise.py --train_days 3 --n_estimators 20 --n_jobs 2 --run_name yambda_tiny_smoke`, 2026-11-08. This checks the Yambda path; a 3-day window and 20 trees make a deliberately weak model, so none of these numbers describe the full exercise.

- Data hash matched the published file. Items were frequency-encoded (96,649 distinct); users label-encoded.
- Rows: fit 386,978 (1.8 days), validation 43,004, reference day 217,014, test day 232,049. Best iteration 7, validation AUC 0.6807. Targets: `user_median_completion`, `user_avg_completion`. Segment: 30.8 percent of test-day users, 29.3 percent of rows.
- Timings: data 11.0 s, training 5.8 s, monitors 76.3 s, total 93.2 s (115 s wall clock). The monitor time depends on the test-day size, not the train window.
- Mean-fill: null rate, zero rate, and mean were green for both targets; KS was 0.515 and 0.712 and PSI 10.7 and 9.7, both firing. Quantile shift also fired (2.52, 1.97), over clean values already above threshold (0.88, 0.47).
- Zero-fill on all traffic: zero rate, mean, KS, PSI, and score monitors fired; quantile shift, AUC, log loss, ECE, and calibration gap fired over clean values already above threshold. GAUC and NDCG@10 stayed green in every scenario: both targets are user-level features, constant within a user, so corrupting them moves a user's scores together and barely changes the order within a user.
- Clean day: 75 monitor rows, across the three slices, were already over threshold. The test day contains one unusual user: uid 945800 logged 13,712 listens of 8 tracks (5.9 percent of all test-day rows; median track length 1,990 s), against 1,015 listens on the reference day, when no user had more than 0.5 percent of rows. With salt 15 this user falls in `rest`, and the largest clean-day shifts are there: in `previous_listen_count` the mean moved 14.9 SD and the largest quantile 356 SD in `rest`, against a mean shift under the 0.2 threshold and a quantile shift of 0.31 SD in `segment`. Over all test-day rows, the 99th percentile of track length was 1,990 s against 500 s on the reference day. On the aggregate, the clean day also crossed the AUC (drop 0.047), log loss, ECE, and calibration-gap thresholds.

An earlier version of the code (Chapter 6 time units, lookups frozen at the end of the fit window) flagged rows per day and join match rate on the clean day. After the changes in sections 4.1 and 4.3, both are green; the two changes were made together, so this run does not separate their effects.

### 9.3 Full run (Google Compute Engine, 2026-11-09)

The exercise as designed: 30-day window, Chapter 6 model settings, default thresholds, salt 15, seed 42, plus four robustness variants. Every number below comes from the run's result files (`signal_summary.md`, `signals_long.csv`, `quality_metrics.csv`, `clean_false_alarms.csv`, `run_manifest.json`) and its step logs.

**Machine.** `e2-standard-4`, on demand, `us-central1-a`: 4 vCPU (Intel Xeon at 2.20 GHz), 16 GB RAM, no swap, 30 GB boot disk, no GPU. Debian 12, Python 3.11.2, packages as pinned in `requirements.txt` (checked against `pip freeze`).

**Code and data.** The VM cloned the public repository at `8c3463013ab467d112629904fdd69bda8cce08a5` (committed 2026-11-08). There `chapter6_ranking/utils/polars_pipeline.py` was last committed in `0c3a6c9d357cd5205a20a071869f80a7ddcbe322`, with SHA-256 `f58ba80000dee0ecc92598147a99da2e46394316a74726d71697d6487c7e530d` (LF endings) and no uncommitted changes. The Chapter 15 folder, not yet in the repository at that commit, was copied in beside Chapter 6. Every Python file, test, and notebook code cell was byte-identical to the ones described here. Only documentation changed after the run: this file, `README.md`, `CLOUD_RUN.md`, and the wording of two Markdown cells in the notebook. The data file matched the published SHA-256, and `hf download` fetched it in 6 seconds.

**Commands, timings, memory.** Wall clock and peak resident memory are from `/usr/bin/time -v`. Package installation took 94 s.

| Step | Command | Result | Wall clock | Peak RSS |
|---|---|---|---|---|
| Tests | `python -m pytest tests -q` | 72 passed in 6.8 s | 7.9 s | 384 MiB |
| Full run | `python run_exercise.py` | completed | 88.5 s | 3,036 MiB |
| Notebook | `jupyter execute silent_degradation.ipynb` | 15 code cells, no error outputs | 95.5 s | 3,098 MiB |
| Salt 16 | `run_exercise.py --segment_salt 16` | completed | 95.9 s | 3,125 MiB |
| Salt 17 | `run_exercise.py --segment_salt 17` | completed | 91.9 s | 3,103 MiB |
| Seed 7 | `run_exercise.py --seed 7` | completed | 113.4 s | 3,122 MiB |
| Item targets | `run_exercise.py --target_scope item` | completed | 96.4 s | 3,066 MiB |

The main run timed itself at 85.9 s: 15.6 s to load data and build features, 17.8 s to train, and 52.3 s for the monitors, at 205 percent CPU. The notebook reproduced the main run exactly: across all 1,836 monitor rows, no status differed and no statistic differed by more than 1.2e-16.

**Windows.**

- Rows: fit 5,456,618 (26.1 days), validation 606,295 (2.9 days), reference day 217,014, test day 232,049.
- Users: 8,968 in the fit window, 4,849 on the reference day, 4,812 on the test day. Chapter 6 label-encoded the users and frequency-encoded the 339,788 items.
- Positive rate (played at least 50 percent): fit 0.638, validation 0.655, reference 0.662, test 0.680.
- The salt-15 segment holds 30.8 percent of test-day users and 29.3 percent of test-day rows.

**Model and targets.** The model's best iteration was 19, with validation AUC 0.7651. The targets were `user_avg_completion` and `user_median_completion`, ranked first and second by average gain over all features, so the eligibility rules skipped nothing above them. Their mean fills are 65.2171 and 72.5328 (percent played).

**Status by signal family** (from `signal_summary.md`):

| Signal | clean / aggregate | zero_fill_all / aggregate | zero_fill_segment / aggregate | zero_fill_segment / segment | zero_fill_segment / rest | mean_fill_all / aggregate |
|---|---|---|---|---|---|---|
| Schema validation pass rate | green | green | green | green | green | green |
| Rows per day | green | green | green | green | green | green |
| Join match rate (user, item) | green | green | green | green | green | green |
| Null rate | green | green | green | green | green | green |
| Zero rate | false alarm | FIRES | FIRES | FIRES | clean also over | green |
| Mean | green | FIRES | FIRES | FIRES | green | green |
| Quantiles | false alarm | FIRES, clean also over | FIRES, clean also over | FIRES | clean also over | FIRES, clean also over |
| KS statistic | green | FIRES | FIRES | FIRES | green | FIRES |
| PSI | green | FIRES | FIRES | FIRES | green | FIRES |
| Score mean | green | FIRES | FIRES | FIRES | green | FIRES |
| Score quantiles, KS, PSI, entropy | false alarm | FIRES | FIRES | FIRES | clean also over | FIRES |
| AUC | false alarm | FIRES, clean also over | FIRES, clean also over | FIRES | clean also over | FIRES, clean also over |
| GAUC | green | FIRES | green | FIRES | green | green |
| NDCG@10 | green | green | green | green | green | green |
| Log loss | false alarm | FIRES, clean also over | FIRES, clean also over | FIRES | clean also over | FIRES, clean also over |
| ECE | false alarm | FIRES, clean also over | FIRES, clean also over | FIRES | clean also over | FIRES, clean also over |
| Calibration gap | false alarm | FIRES, clean also over | FIRES, clean also over | FIRES | clean also over | FIRES, clean also over |

**Target features, aggregate slice.** The null rate is 0 in every scenario. Thresholds are 0.02 for the zero rate, 0.2 for the mean and quantile shifts, 0.1 for KS, and 0.25 for PSI.

| Scenario | Feature | Zero rate | Mean | Quantiles | KS | PSI |
|---|---|---|---|---|---|---|
| clean | `user_avg_completion` | 0.0003 | 0.136 | 0.563 | 0.061 | 0.032 |
| clean | `user_median_completion` | 0.0566 | 0.126 | 0.375 | 0.060 | 0.025 |
| zero_fill_all | `user_avg_completion` | 1.0000 | 2.979 | 4.449 | 1.000 | 8.28 |
| zero_fill_all | `user_median_completion` | 0.9928 | 2.007 | 2.682 | 0.993 | 9.21 |
| zero_fill_segment | `user_avg_completion` | 0.2929 | 1.019 | 2.249 | 0.353 | 0.597 |
| zero_fill_segment | `user_median_completion` | 0.3472 | 0.716 | 1.770 | 0.347 | 0.570 |
| mean_fill_all | `user_avg_completion` | -0.0000 | 0.067 | 2.258 | 0.569 | 8.28 |
| mean_fill_all | `user_median_completion` | -0.0072 | 0.062 | 1.918 | 0.684 | 8.59 |

**Scores, aggregate slice.**

| | Reference day | clean | zero_fill_all | zero_fill_segment | mean_fill_all |
|---|---|---|---|---|---|
| Mean score | 0.659 | 0.633 | 0.258 | 0.515 | 0.600 |
| Score entropy (nats, 20 bins) | 2.678 | 2.728 | 1.916 | 2.782 | 2.080 |

Relative to the reference day, score KS was 0.057 on clean, 0.806 under zero_fill_all, 0.288 under zero_fill_segment, and 0.357 under mean_fill_all; score PSI was 0.029, 4.56, 0.489, and 1.27.

**Quality and calibration, aggregate slice.**

| Metric | Reference day | clean | zero_fill_all | zero_fill_segment | mean_fill_all |
|---|---|---|---|---|---|
| AUC | 0.7748 | 0.7123 | 0.4572 | 0.5997 | 0.5429 |
| GAUC | 0.5678 | 0.5677 | 0.5512 | 0.5622 | 0.5598 |
| NDCG@10 | 0.7646 | 0.7622 | 0.7576 | 0.7607 | 0.7586 |
| Log loss | 0.5328 | 0.6004 | 1.0808 | 0.7557 | 0.6693 |
| ECE | 0.0187 | 0.0564 | 0.4242 | 0.1656 | 0.0896 |
| Calibration gap | -0.0025 | -0.0467 | -0.4215 | -0.1648 | -0.0798 |
| Positive rate | 0.6615 | 0.6796 | 0.6796 | 0.6796 | 0.6796 |

**AUC and ECE by slice.**

| Slice | AUC reference | AUC clean | AUC zero_fill_all | AUC zero_fill_segment | AUC mean_fill_all | ECE clean | ECE zero_fill_segment |
|---|---|---|---|---|---|---|---|
| aggregate | 0.7748 | 0.7123 | 0.4572 | 0.5997 | 0.5429 | 0.0564 | 0.1656 |
| segment | 0.7777 | 0.7858 | 0.4697 | 0.4697 | 0.5722 | 0.0227 | 0.4049 |
| rest | 0.7731 | 0.6829 | 0.4523 | 0.6829 | 0.5312 | 0.0720 | 0.0720 |

**Clean-day false alarms.** 45 monitor rows were over threshold with no corruption: 18 in the aggregate, 7 in the segment, and 20 in the rest. By layer, 35 were feature monitors, 2 score monitors, and 8 quality monitors. The segment's 7 are all quantile shifts, on `item_id_freq`, `user_active_span`, `item_total_plays`, `item_std_completion`, `item_unique_listeners`, `item_repeat_ratio`, and `previous_listen_count`. The aggregate and the rest also carry:

- mean and quantile shifts on `previous_listen_count` (aggregate 7.8 SD and 238 SD) and on `track_length_seconds` (0.35 SD and 8.6 SD);
- the zero rate of `user_median_completion` (+0.057);
- the score quantiles;
- AUC, log loss, ECE, and the calibration gap.

**The account behind most clean-day alarms.** User 945800 had 13,712 test-day plays, 5.9 percent of all test-day rows (the next-largest user had 685). The plays covered 8 tracks, 98.7 percent of them were completed, and they add up to 216 days of played audio within one day. On the reference day the same account had 1,015 plays of 8 tracks, only one of them shared with the test day, with 0.9 percent completed. In the fit and validation windows it had 3,115 plays of 260 tracks, with 17.5 percent completed. No listener produces 216 days of audio in a day, so this looks like an automated or looping client. Its test-day labels are almost all positive while its history says it rarely completes a track. Its lookup median completion is zero, and its rows make up 5.9 of the 6.4 percent of test-day rows where `user_median_completion` is zero. Together these explain the shifts in `previous_listen_count`, track length, the `user_median_completion` zero rate, AUC, and calibration. It hashes into `rest` with salts 15 and 17 and into `segment` with salt 16.

**Robustness runs.**

| Run | Targets | Best iteration, validation AUC | Account in | Clean false alarms (aggregate / segment / rest) | Status cells differing from the main run |
|---|---|---|---|---|---|
| Main (salt 15, seed 42) | `user_avg_completion`, `user_median_completion` | 19, 0.7651 | rest | 18 / 7 / 20 | |
| Salt 16 | same | 19, 0.7651 | segment | 18 / 44 / 3 | 16 of 102 |
| Salt 17 | same | 19, 0.7651 | rest | 18 / 7 / 22 | 1 of 102 |
| Seed 7 | same two, order swapped | 6, 0.7727 | rest | 17 / 7 / 19 | 9 of 102 |
| Item targets | `item_avg_completion`, `item_std_completion` | 19, 0.7651 | rest | 18 / 7 / 20 | 13 of 102 |

- **Salt 16.** With the account in the segment, the segment's clean AUC fell 0.182 below its reference day and the rest's rose 0.005, the reverse of the main run (segment +0.008, rest -0.090). All 16 differing cells are in the zero_fill_segment columns. In the segment, every FIRES became "FIRES, clean also over"; in the rest, every "clean also over" became green. Zero-fill on the segment still separated from clean in every family that fires in the main run.
- **Salt 17.** One cell differs: segment quantiles went from FIRES to "FIRES, clean also over".
- **Seed 7.** Early stopping chose iteration 6, and the smaller model's scores are narrower (reference entropy 2.211 nats). Under zero_fill_all, NDCG@10 now fires in the aggregate (a drop of 0.0158 from the reference day) and in the segment (0.0160). The clean day's ECE is no longer over threshold. Under mean_fill_all the score mean stays green (0.153).
- **Item targets.** The mean fills are 65.2171 and 39.786.
  - Zero-fill on all traffic lowers AUC by 0.079 from clean, GAUC by 0.038, and NDCG@10 by 0.015, so the per-user metrics move far more than with user targets and pooled AUC far less.
  - Mean-fill on all traffic raises pooled AUC by 0.009 over clean, while GAUC (0.025 below the reference day) and NDCG@10 (0.012) fire. The score mean (0.063) and score entropy (0.055) stay green, and the score-distribution family is no worse than clean.
  - Mean-fill lowers the zero rate of `item_std_completion` by 0.041, its share of natural zeros on the reference day. A two-sided zero-rate monitor at 0.02 would catch that.

**What the full run shows.**

- **Mean-fill.** It escapes the null-rate, zero-rate, and mean monitors and is caught by KS and PSI. The quantile monitor catches it over a clean value already above its threshold. The zero rates of the default targets fall by at most 0.0072, below any threshold used here.
- **Pipeline monitors.** Schema, rows per day, and join match rate are green in every scenario, slice, and run.
- **Segment dilution.** The aggregate dilutes the segment corruption: AUC falls 0.113 from clean in the aggregate against 0.316 in the segment. The rest stays identical to clean.
- **Pooled versus per-user metrics.** With user-level targets, zero-fill on all traffic lowers pooled AUC by 0.255 from clean, GAUC by 0.017 (just over its threshold), and NDCG@10 by 0.005, which fires only under seed 7. With item-level targets the per-user metrics move most.
- **Score entropy.** Corrupting all traffic compresses the score distribution (entropy falls). Corrupting the segment widens it (entropy rises), because the corrupted segment's scores form a second mode.
- **Clean-day alarms.** Most trace to one anomalous account. The slice without it has 3 to 7 clean alarms, almost all quantile shifts on heavy-tailed features. The clean control keeps every scenario attributable.

### 9.4 Full run after the leak fix (Google Compute Engine, 2026-11-09)

The run of section 9.3 repeated with the current code: the vendored Chapter 6 pipeline, training rows' aggregates from the window before the fit window, the training history in the point-in-time count, and the total row order. Same machine type, data, settings, and variants, plus one repeat of the main run. Results are in `outputs/cloud_ch15-20261109-1008/`.

**Code and data.** Run `ch15-20261109-1008`, `e2-standard-4` on demand in `us-central1-a`, Debian 12, Python 3.11.2. The VM cloned the public repository at `074886c3ee51a79dd1848d437b467d9be75843ca`; its `chapter6_ranking/utils/polars_pipeline.py` and the body of `ch6_polars_pipeline_vendored.py` both hashed to `cf532b51...` (`logs/repo_commit.txt`, `logs/vendored_body_sha256.txt`). The Chapter 15 folder was copied in from GCS (`logs/ch15_code_sha256.txt`). The data file matched the published SHA-256.

| Step | Result | Wall clock | Peak RSS |
|---|---|---|---|
| Tests | 78 passed in 4.4 s | 5.4 s | 390 MiB |
| Full run | completed | 126.6 s | 5,359 MiB |
| Repeat of the full run | completed | 124.5 s | 5,125 MiB |
| Notebook | 15 code cells, no error outputs | 128.3 s | 5,500 MiB |
| Salt 16, salt 17, seed 7, item targets | completed | 145.2, 129.9, 132.7, 128.5 s | 5,168 to 5,385 MiB |

The main run timed itself at 124.2 s: 24.4 s for data and features, 47.1 s to train, 52.6 s for the monitors. Against section 9.3, data and features take 9 s longer (one more window of aggregates and the sort), training 29 s longer (85 trees instead of 19), and peak memory rises from 3.0 to 5.2 GiB.

**Reproducibility.** The repeat and the notebook each matched the main run on all 1,836 monitor rows: no status differed and no statistic differed by more than 1.1e-16. A laptop run of the same code (16 threads instead of 4) stopped at iteration 98 instead of 85, and 8 of 204 summary cells differed from the cloud run. Seven are GAUC or NDCG@10 drops of 0.010 to 0.018 on the cloud machine against 0.005 to 0.009 on the laptop, either side of the 0.01 threshold; the eighth is a log-loss cell in the rest. A second laptop run with `--n_jobs 4`, the cloud machine's thread count, stopped at iteration 88 and still differed in 6 of 204 cells (the four borderline cells below stayed green), with quality metrics up to 0.027 apart, so the difference comes from the machine and not only the thread count. Results are exact on one machine but not across machines; section 9.4 on the Linux `e2-standard-4` is the reference.

**What did not change.** Window rows, users, positive rates, the targets (`user_avg_completion`, `user_median_completion`), the segment shares, the test-day feature statistics on the clean day and under zero-fill (the scored days' lookups were already built from earlier windows; mean-fill statistics move with the fill value), the pipeline monitors (green everywhere), and the clean-day false alarms (45: 18 aggregate, 7 segment, 20 rest; 35 feature, 2 score, 8 quality). The account of section 9.3 is the same.

**What moved, main run.**

| | Section 9.3 | Section 9.4 |
|---|---|---|
| Best iteration, validation AUC | 19, 0.7651 | 85, 0.7824 |
| Mean fills of the two targets | 65.2171, 72.5328 | 64.4416, 72.8854 |
| Reference-day AUC, GAUC, NDCG@10, ECE | 0.7748, 0.5678, 0.7646, 0.0187 | 0.7943, 0.5846, 0.7800, 0.0100 |
| AUC: clean, zero_fill_all, zero_fill_segment, mean_fill_all | 0.7123, 0.4572, 0.5997, 0.5429 | 0.7254, 0.4214, 0.5996, 0.5833 |
| GAUC, same order | 0.5677, 0.5512, 0.5622, 0.5598 | 0.5851, 0.5601, 0.5755, 0.5746 |
| NDCG@10, same order | 0.7622, 0.7576, 0.7607, 0.7586 | 0.7733, 0.7692, 0.7698, 0.7705 |
| ECE, same order | 0.0564, 0.4242, 0.1656, 0.0896 | 0.0538, 0.4353, 0.1724, 0.0874 |
| Log loss, same order | 0.6004, 1.0808, 0.7557, 0.6693 | 0.6007, 1.1711, 0.7838, 0.6440 |
| Score entropy: reference, clean, zero_fill_all, zero_fill_segment, mean_fill_all | 2.678, 2.728, 1.916, 2.782, 2.080 | 2.778, 2.841, 2.177, 2.934, 2.105 |
| Score PSI: clean, zero_fill_all, zero_fill_segment, mean_fill_all | 0.029, 4.56, 0.489, 1.27 | 0.023, 5.40, 0.427, 1.86 |
| Mean-fill, two targets: KS; mean shift; quantile shift | 0.569, 0.684; 0.067, 0.062; 2.258, 1.918 | 0.584, 0.684; 0.102, 0.052; 2.224, 1.928 |

Status table: 4 of the 102 cells of section 9.3's table changed, all ranking metrics going from green to FIRES: NDCG@10 under zero_fill_all (aggregate) and zero_fill_segment (aggregate and segment), and GAUC under mean_fill_all (aggregate). The three aggregate cells are drops from a higher reference day that pass the 0.01 threshold by at most 0.001 (NDCG@10 under zero_fill_all: 0.7800 to 0.7692); the segment's NDCG@10 drop is 0.018. The laptop run leaves all four green, so they should be read as borderline. The mean-fill result is unchanged: null rate, zero rate, and mean stay green, and KS and PSI fire.

**What moved, variants.**

| Run | Targets | Best iteration, validation AUC | Clean false alarms | Cells changed from section 9.3 | Cells differing from the new main run |
|---|---|---|---|---|---|
| Salt 16 | as main | 85, 0.7824 | 18 / 43 / 3 | 3 | 17 |
| Salt 17 | as main | 85, 0.7824 | 18 / 7 / 22 | 3 | 4 |
| Seed 7 | as main, same order | 89, 0.7827 | 18 / 7 / 20 | 9 | 4 |
| Item targets | `item_avg_completion`, `item_organic_ratio` | 85, 0.7824 | 18 / 7 / 20 | 11 | 17 |

- **Seed 7** no longer stops early at iteration 6, so it now behaves like the main run; its NDCG@10 alarm under zero_fill_all, noted in section 9.3, is gone.
- **Item targets.** The second target is now `item_organic_ratio` (mean fill 0.5058) instead of `item_std_completion`. Zero-fill on all traffic lowers AUC by 0.010 from clean (0.079 in section 9.3), GAUC by 0.031, and NDCG@10 by 0.015, so with item targets the pooled metrics barely move while the per-user metrics fire. Under mean-fill, AUC is 0.0007 below clean, GAUC and NDCG@10 fire, and the score-distribution family is green.

**What the run shows, against section 9.3.** Every qualitative finding of section 9.3 holds: mean-fill escapes the null, zero, and mean monitors and is caught by KS and PSI; the pipeline monitors are green everywhere; the aggregate dilutes the segment corruption (AUC falls 0.126 from clean in the aggregate against 0.376 in the segment, with the rest identical to clean); all-traffic corruption lowers score entropy and segment corruption raises it; and the clean-day alarms trace to the same account. With user-level targets, zero-fill on all traffic lowers pooled AUC by 0.304 from clean against 0.025 for GAUC and 0.004 for NDCG@10, so the per-user metrics still move far less than the pooled ones.
