# Chapter 13 Companion Code: Design Notes

These notes record the decisions behind the Chapter 13 code, where it differs from the plan (QA report Section 6) and from the chapter's figure and table specifications, what serving the published Chapter 5 to 8 code revealed, and what was measured. `README.md` explains how to run the code, and `CLOUD_RUN.md` lists every entry point with its hardware, runtime, inputs and outputs. The last section, the Cloud Run validation log, is appended to by `deploy/cloud_run/deploy.py`.

## Scope and Status

| Part | Status |
|---|---|
| Five FastAPI services under uvicorn, Docker Compose optional | Built and run on Windows with Python 3.11 |
| Published Chapter 5 to 8 models on CPU | Vendored byte for byte from public commit `074886c`, trained by the artifact build |
| Artifact build | Profiles `tiny`, `synthetic` and `sample` (Yambda) run; profile `full` written, not run |
| Contracts, deadlines, retries and the retry budget, failure-rate breakers, the six-rung degradation ladder, feature contract and staleness, policy pinning, health checks, config hot reload, popularity fallback | Built and covered by tests |
| pytest suite, smoke script, Locust closed-loop ramp | 135 tests pass after the re-vendoring at `074886c` and the row-order fix; smoke test and short Locust ramps run locally and on Cloud Run |
| `deploy/cloud_run/deploy.py` | Run three times on 2026-11-09 (Cloud Run Validation Log); the second and third runs passed the smoke test, and every run tore everything down |
| Container images | All five built with Cloud Build and served on Cloud Run. None was built locally, because Docker was not available on the build machine |

## Design Decisions

### Published code, vendored and pinned to the bundle

`vendor/` holds 20 files copied with `git show <commit>:<path>` from the public repository, so uncommitted edits in a checkout never reach the services. `vendor/MANIFEST.json` records the commit, each source path and each SHA-256, and `scripts/check_vendor.py` verifies them. `common/published.py` imports the files under chapter-specific aliases because Chapters 6 and 7 both name their package `models`. It skips package `__init__.py` files, which import PyTorch models that the lightweight ranker and the re-ranker do not need, and it contains Chapter 5's import-time `warnings.filterwarnings('ignore')` so that warnings stay visible in every service.

Every bundle records the SHA-256 of each vendored file it was built with (`published_code` in `bundle_manifest.json`). Each service records the vendored files it executes and compares them with that record before it reports ready. A changed feature function would otherwise compute serving features that differ from the training features without any error. This check was added after the public repository changed during this work (see Published-Code Observations).

### Immutable bundles behind a pointer

`pipeline_build/build_artifacts.py` writes one directory per bundle with the SHA-256 of every file in `bundle_manifest.json`, then switches `artifacts/ACTIVE` with an atomic file replace. Each service verifies the hashes of the files it loads. A missing file, a changed file, an index built for another user tower, index vectors that do not match the item tower, or vendored code that differs from the bundle's record stops the service at startup with a message that names the problem. No path falls back to random scores or an untrained model. Promotion and rollback rewrite `ACTIVE` and restart the services; on Cloud Run each bundle is baked into new images and a new revision.

### Retrieval and version pairing (Figure 13.4)

`retrieval/index_manifest.json` names the user-tower checkpoint (file, SHA-256 and model version), the index version, the vector dimension and the item count. Retrieval refuses to start when the checkpoint's model version differs from the manifest's, or when 32 items re-encoded by the loaded item tower differ from their stored index vectors by more than 1e-4. The index is a FAISS `IndexFlatIP` over L2-normalised item vectors, which gives exact cosine search; the catalogs here (20,000 items in the sample bundle) are small enough for exact search. Recent items and the request's `exclude_item_ids` are excluded inside the search with an ID selector. The user tower reads the user's last 20 items from the feature store, the length Chapter 5 uses. Retrieval is trained with Chapter 5's two-tower code on Yambda, using Yambda's audio embeddings (`normalized_embed`) as item content, so all five services serve one dataset.

### One feature snapshot for training and serving

Chapter 6's `HistoricalFeatureBuilder` is fitted on listens before the 0.80 time quantile, with the likes and dislikes given before that time. Both rankers train on listens from the 0.80 to the 0.95 quantile and are validated on the rest. The online store serves exactly the fitted snapshot, so a ranking request built from the store reproduces the training features. A test checks this equality and checks that both rankers score the rebuilt request as the published models score the training rows. SQLite (default), Parquet and Redis back ends share one interface.

Like and dislike labels on training rows count only likes and dislikes before the validation period starts, so a listen just before the boundary cannot take its label from a like given during validation. Chapter 7's labelling looks ahead `LIKE_WINDOW_TS` (24 hours) from each listen; the build passes the start of the validation period as its `label_end_ts` for training rows and no bound for validation rows, and sorts the labelled rows on every column so that the same seed trains the same models (Published-Code Observations, item 10).

Fitted with likes and dislikes, the builder adds two item counts, `item_likes` and `item_dislikes`, beside `item_like_ratio`. Chapter 6's `get_historical_feature_columns()` does not list them, but its XGBoost feature pipeline selects historical columns by name prefix and trains on both, so the feature contract declares them (`LIKE_COUNT_FEATURES` in `pipeline_build/feature_store_build.py`). Chapter 6 gives them no cold-start default; the build sets 0, which is what an item without plays in the feature window has. The MMoE reads none of the three like features.

### Feature contract and staleness (Sections 13.2.4 and 13.7.3)

The build writes `feature_store/feature_contract.json`, hashed into the bundle manifest. It declares each feature's group, type (float), encoding (raw), default and maximum age: 7 days for user and user-item features, 30 days for item statistics, and none for `track_length_seconds`, a fixed catalog attribute without a default. Each user and item row carries `updated_at`, and every lookup returns it. The store treats a value older than its maximum age at the request's time as missing, substitutes the contract default, and counts the substitution per feature, both in the response (`degradation.feature_defaults.stale` and `missing`) and process-wide in `GET /health`. An absent user-item pair is the true value zero (the user never played the item) and is not counted.

Ages are measured in the dataset's event-time units, and a request's time defaults to the bundle's reference time (the last listen in the data, also the default `context.timestamp`). `updated_at` is the materialisation time, so it equals that reference time, which lies after the end of the feature window (25,999,930 against 22,026,720 in the sample bundle). A fresh bundle therefore has no stale values at its default request time; the tests age the request time to exercise the path. Missing values are common: in the README example three of the user's candidate tracks had no plays in the feature window, so their nine item statistics (Chapter 6's seven and the two like counts) were replaced by defaults.

The defaults are the same cold-start values `HistoricalFeatureBuilder.transform` uses, applied by the feature-store client before the rankers see the request, so the training/serving consistency check still holds. Both rankers refuse to start when their model reads a feature the contract does not declare or when the contract's feature-set version differs from the bundle's.

### Rankers and the shared score

The lightweight ranker (Chapter 6 XGBoost, label `played_ratio_pct >= 50`) and the MMoE completion head (Chapter 7) both estimate the probability of completing a track, so when one ranking stage is skipped the other's score still estimates the same quantity. The value function combines the five MMoE task scores, so the ranking contract rejects an MMoE response in which any item lacks its task scores; the re-ranker would otherwise fall back to the raw score without a sign. The heavy ranker reproduces the published `MMoEFeatureProcessor.transform` with lookup tables built once at load (see Measured Results); a test asserts exact equality with the published transform, including for unseen IDs.

### Re-ranker and configuration

The re-ranker runs Chapter 8's value function, MMR diversity and artist and album pacing. MMR is vectorised; a test checks that it selects the same items as Chapter 8's implementation for four values of lambda, with duplicate embeddings forcing ties. Configurations live in `config/reranker/` with an `ACTIVE` pointer. They are validated before activation, polled every five seconds and reloadable through `POST /v1/config/reload`; an invalid configuration is refused and the last valid one keeps serving.

Every configuration file must name its `author` and the `reason` for the change (Section 13.6.2); the two shipped files name the companion code as author. A rejected configuration is logged as a warning and shown as `last_error` in `GET /v1/config` and `GET /health`; nothing sends an alert, which is monitoring's job (Chapter 15).

A request may pin a configuration with the optional `policy_version` field (Section 13.6), which the orchestrator passes to the re-ranker. The re-ranker loads and validates the pinned file once, keeps up to 16 pinned configurations in memory, and never changes `ACTIVE` for a pin. A pin that names no valid file serves the active configuration, reports `policy_pin: unavailable` and logs a warning, so a mistyped version degrades instead of failing. The response's `config_version`, `X-Model-Version` header and the orchestrator's `stage_versions.reranker_config` name the configuration actually used.

### Orchestrator

The deadline comes from `X-Request-Timeout-Ms` (default 1,000 ms, capped at 5,000 ms). Each call gets the smaller of its stage cap and the time left after reserving the minimum budgets of later stages; a stage whose budget would fall below its minimum is skipped. The callee receives its budget in the same header and returns 504 when the budget has passed. User features and retrieval are requested together. Lookups in an in-process store (SQLite, Parquet) run on the event loop, because in a worker thread they slowed down sharply under load (see "Item-feature lookup" under Measured Results). A Redis lookup runs in a worker thread, overlapping the retrieval call, under a timeout: the smaller of `ORCH_FEATURE_LOOKUP_CAP_MS` (100 ms) and the time the deadline leaves after reserving the later stages' minimum budgets. A lookup that times out counts as failed. Readiness of the orchestrator depends only on its own artifacts, so a failing stage degrades responses without removing the entry point from service. Each request produces one structured log record (logger `ch13.orchestrator`, WARNING when a stage was skipped) with the trace ID, rung, label, every skipped stage as `"<rung> <label>: <stage> <reason>"`, feature defaults, versions and timings.

### Retries, the retry budget and circuit breakers (Section 13.8.5)

Retries happen at one point only, the orchestrator's stage call; the services do not retry. A call is retried at most once, only after a connection error or HTTP 502 or 503, after a jittered pause (10 ms plus up to 50%), and only if the stage's time still allows; timeouts are never retried. A process-wide token bucket caps retries across all stages at one per second with a burst of ten, following the SRE book's example budget of 60 retries per minute. A retry that finds the bucket empty is skipped: the call counts as a failure and the request degrades. `GET /health` reports tokens, retries spent and retries dropped.

Each stage has a breaker judged on a failure rate over a sliding window, as Section 13.8.5 describes: it opens when more than half of at least 20 calls in the last 10 seconds failed. An open breaker skips the stage for 10 seconds, then lets one trial call through; a successful trial closes the breaker and clears the window, a failed one reopens it. A failure streak shorter than the minimum call count never opens the breaker, and a late success from a call that started before the breaker opened is ignored. Every call the breaker lets through reports back: an unexpected exception inside the call, such as a response body that cannot be decoded, counts as a failure, so a half-open trial call can never leave the breaker waiting for a result that will not come. A 422 from a stage is a contract violation by the caller: the breaker counts it as a successful call, because the stage answered, and it is never retried. The orchestrator logs it with the trace ID, both versions and the field-level detail and counts it per stage in `GET /health` (`contract_rejections`), the count Section 13.2.2 suggests alerting on. All thresholds are environment settings (README).

### Degradation ladder (Figure 13.6 and Section 13.8.5)

The rungs follow the figure's order and labels: 1 Full pipeline, 2 Skip heavy ranker, 3 Skip re-ranker, 4 Skip lightweight ranker, 5 Popularity candidates, 6 Cached list. Each fallback in `degradation.fallbacks` carries its stage, rung, label, reason and action. When several stages fail, the response reports the lowest rung reached (the highest number) and lists every skipped stage in figure order. Specifics:

- **Rung 3** sorts by the best score available (heavy ranker, else lightweight ranker, else retrieval), removes the user's recent items and applies Chapter 8 artist pacing.
- **Rung 4** keeps the top 100 candidates by retrieval similarity; the heavy ranker and re-ranker still run on them if they answer. A failed item-feature lookup also leads here, with reason `features_unavailable` for both rankers, because neither can score without item features.
- **Rung 5** ranks the in-memory popularity list (outside the user's recent items) if the rankers answer.
- **Rung 6** is returned when the deadline runs out, when every remote stage has failed, or when no candidate has a track length (reason `error` on stage `pipeline`).
- **Feature defaults are reported as a flag beside the rung.** A failed user-feature lookup serves the request with cold-start defaults and records `user_features` in `degradation.feature_defaults.lookup_failed`.
- Cold-start users are not degraded: retrieval returns the popularity list for a user with no usable history (`retrieval_source: popularity_cold_start`) on rung 1.

### Contract snapshot refresh

Round 2 changed `Fallback` (new required `rung` and `label`, new `stage` and `reason` values) and widened `Degradation.rung` to 1 to 6; these are breaking changes under the compatibility rules the code itself enforces. Because no client outside this repository has used v1 yet, `contracts/schema_snapshot_v1.json` was regenerated (`--update`) instead of publishing `/v2`. After publication the same change would need a new version. `policy_version`, `feature_defaults` and `policy_pin` are optional additions and compatible. The code QA of 2026-11-09 regenerated the snapshot once more after it corrected the description of `context.timestamp` (seconds on Yambda's clock); descriptions are outside the compatibility check, so this changed no rule.

### Service runtime

Each service is one uvicorn process. PyTorch, XGBoost and FAISS each use one thread, which matches Cloud Run's one vCPU and keeps five services from competing for cores on a laptop. Each service runs one inference before it reports ready. Contracts use Pydantic v2 with unknown fields ignored and non-finite numbers rejected; `contracts/schema_snapshot_v1.json` and a test guard against breaking changes.

### Cloud Run

Each service has its own image, holding only its code, the vendored code and its part of the bundle, and running as a non-root user. All services are deployed with `--no-allow-unauthenticated`. Each service runs as its own service account, named after the service (`ch13-orchestrator`, `ch13-retrieval` and so on), as Section 13.9.4 recommends. The orchestrator calls the stages with Google identity tokens from the metadata server, and only its account is granted `roles/run.invoker`, on the four stage services, so a stage cannot call another stage. No account holds any other role: the bundle is baked into the images, Cloud Run's own service agent pulls them and collects the logs, and minting an identity token for one's own account needs no grant. The tests call the orchestrator with the operator's identity token. The services need no secrets. Teardown removes the four invoker grants, deletes the services and the five accounts, and then checks that no `ch13` account and no project IAM binding naming one is left.

A new `roles/run.invoker` grant takes effect on Cloud Run only after a delay. Until it does, the stage answers the orchestrator's calls with 403 and requests degrade: in the first Cloud Run run the re-ranker refused all 20 smoke-test requests (rung 3) for about a minute after its grant. The script therefore polls the orchestrator after readiness and starts the smoke test only once a request is served on rung 1.

## Differences from the Plan and the Figure Specifications

1. **Figure 13.6 cannot show combined failures or the feature flag.** The code reports the lowest rung reached and lists every skipped stage; feature defaults are reported beside the rung. See "Proposed Changes to Figure 13.6" below.
2. **Rungs 4 and 5 still rank.** After a lightweight-ranker failure the heavy ranker and re-ranker run on the retrieval top 100, and after a retrieval failure the popularity candidates go through the rankers if they are healthy, so a rung-4 or rung-5 response can carry `score_source: value_function`.
3. **Cold-start users stay on rung 1.** Retrieval itself returns the popularity list for a user with no usable history (`retrieval_source: popularity_cold_start`); rung 5 is reserved for retrieval failures.
4. **Candidates without a track length are dropped before ranking.** Both rankers need it, and an item never played has no defensible default.
5. **`stage_versions` keys:** `feature_set`, `retrieval`, `index`, `retrieval_source`, `lightweight_ranker`, `heavy_ranker`, `reranker_config`, and `policy_pin` when the request pinned a configuration.
6. **No gateway or metadata service.** Figure 13.2's Gateway lifeline is Cloud Run's front end in the cloud and absent locally. Item metadata comes from the feature store's item table, and its time is reported as `hydrate`.
7. **Names.** Both rankers serve `POST /v1/rank`. Compose names the services `light-ranker` and `heavy-ranker`; Cloud Run names them `ch13-retrieval`, `ch13-light-ranker`, `ch13-heavy-ranker`, `ch13-reranker` and `ch13-orchestrator`.
8. **Additions:** warm-up before readiness, orchestrator readiness independent of the stages, the published-code check, `scripts/run_local.py --base-port`, `scripts/stage_latency.py`, one log line per POST request in every service (service, trace ID, path, status, duration, model version), the `sniffio` pin, peak build memory in the bundle manifest, and a Yambda loader that accepts both the Hugging Face layout (`flat/50m/`) and a flattened copy (`flat/`).
9. **Configuration changes in containers.** The re-ranker image bakes in `config/reranker/`; hot reload works locally and with a mounted `RERANKER_CONFIG_DIR`. On Cloud Run a configuration change is a new revision.
10. **Default budgets are illustrative** values for a laptop, as Figure 13.2 requires.
11. **Retrieval is trained on Yambda**, while Chapter 5 itself used Coveo.
12. **Not implemented:** progressive traffic shifting and rollback (Figure 13.8; `deploy.py` deploys one revision per service and deletes it), Table 13.2's refresh cadence and scaling signals beyond Cloud Run's `--max-instances` and `--concurrency`, sharded embedding lookup (Figure 13.5), infrastructure as code (no `infra/`), a secrets manager (Figure 13.7; nothing needs a secret, so the Cloud Run run does not exercise a secrets path), admission control (under overload, client latency exceeds the deadline), and the Streamlit demo.
13. **The `full` profile was not run**, so no full-size bundle exists.
14. **Cloud Run housekeeping.** Images are built with Cloud Build by default. The APIs the script enables stay enabled, together with the dependencies Google enables with them (Container Registry, IAM Credentials, Logging, Pub/Sub, Cloud Storage's API components and Telemetry), and so do the Google-managed service agents and their project roles that enabling creates. Cloud Build creates `gs://<project>_cloudbuild` when it is missing; the script records whether the bucket existed and, if this run created it, deletes it after removing the run's `ch13-source/<run_id>` objects (the delete fails on a bucket that still holds anything). Build records, build logs (in Cloud Build's Google-managed log bucket) and the services' Cloud Logging entries remain.
15. **The published code changed during this work.** The code was vendored at `8c34630`; the public repository then published `5757e95`, which changes Chapter 7's `train_mmoe.py` (an optional `thresholds` argument to `evaluate` and validation-based model selection in `main`) and adds functions to `utils/polars_pipeline.py`. The code was re-vendored at `5757e95`, the sample and synthetic bundles were rebuilt (`sample-20261109-090016`, `synthetic-v2`), and all tests and measurements were rerun. The earlier bundles `synthetic-v1` and `sample-20261109-075601` were kept but no longer load, because they predate the published-code record. Later on 2026-11-09 the public repository published `074886c`, which reads Yambda timestamps as seconds in Chapters 6 and 7 (Published-Code Observations, item 1), bounds Chapter 7's like and dislike labels with `label_end_ts`, and lets Chapter 5's `compute_popularity_scores` stop at a given time. The code was re-vendored at `074886c`, the build switched to `label_end_ts` and to fitting the feature builder with likes and dislikes, both bundles were rebuilt (`sample-20261109-192754` and `synthetic-20261109-193547`), and all tests and measurements were rerun, locally and on Cloud Run. The build then began sorting the labelled rows on every column (Published-Code Observations, item 10), and both bundles were rebuilt once more (`sample-20261109-202000`, now active, and `synthetic-20261109-202956`); the tests and the local smoke test were rerun, and the Cloud Run run on `sample-20261109-192754` was not repeated.
16. **Round 2 changed the bundle format.** Bundles now need `feature_store/feature_contract.json` and `updated_at` columns, so every bundle built before round 2 (`synthetic-v1`, `synthetic-v2`, `sample-20261109-075601`, `sample-20261109-090016`) no longer loads. `sample-20261109-162852`, built after round 2 at `5757e95`, no longer loads either, because the code it records was replaced by `074886c`. The old bundles were kept. The `full` profile was not rerun.
17. **No CI configuration.** Section 13.2 says the companion code's contracts are checked in CI, and Table 13.7 runs unit and contract tests in CI; this folder ships the test suite that CI would run, but no CI workflow.

## Published-Code Observations

1. **Timestamp unit (fixed in `074886c`).** Up to `5757e95`, published Chapters 6 and 7 treated Yambda timestamps as 5-second bins: Chapter 6's `add_derived_features` multiplied by 5, Chapter 7 set `LIKE_WINDOW_TS = 17,280`, and `utils/polars_pipeline.py` divided by 5. All 46,467,212 timestamps in `flat/listens.parquet` are multiples of 5, between 0 and 26,000,000; bin indices would take every integer value, so the data is seconds rounded down to 5. Under the old reading the hour-of-day and day-of-week features ran on a time axis stretched fivefold, and the 24-hour like and dislike window covered 4.8 hours. `074886c` reads seconds throughout (`LIKE_WINDOW_TS = 86,400`), and Chapter 13 vendors it. The longer window raised the like label's positive rate on the sample's validation rows from 0.67% to 0.77%; the dislike rate stayed at 0.12%.
2. **Per-call lookup cost in Chapter 7's `MMoEFeatureProcessor.transform`.** Each call rebuilds its class lookups (a pandas `isin` over the vocabulary and scikit-learn's `_map_to_integer`). With an item vocabulary of 18,175 entries this took 278 ms per 100 candidates, more than the heavy ranker's whole 300 ms budget.
3. **Missing dense columns in the same transform** are skipped without an error, which shifts the remaining columns into the wrong scaler positions. The heavy ranker checks that every feature is present before encoding and answers 422 when one is missing.
4. **Chapter 6's XGBoost parameters pass `use_label_encoder`**, which XGBoost 3.2 ignores with a warning; `pytest.ini` filters that warning.
5. **Chapter 5 calls `warnings.filterwarnings('ignore')` at import**, which would silence warnings in any process that imports it.
6. **Chapters 6 and 7 both name their package `models`**, and their `__init__.py` files import PyTorch models.
7. **Chapter 8's MMR makes one scikit-learn `cosine_similarity` call per item pair.** The re-ranker uses a vectorised equivalent that a test compares with the original.
8. **Chapter 8's artist mapping keeps the last row** for an item listed more than once; the build reproduces that.
9. **The two-tower model scores below the popularity baseline** on HR@10 in both bundles (sample 0.0125 against 0.0145; synthetic 0.0218 against 0.0515), with the short training these profiles use. Bundles built before the code QA of 2026-11-09 reported higher baselines (0.0155 and 0.0524) because the build ranked popularity on plays that included the held-out evaluation plays; Chapter 13's build now ranks it on plays before the evaluation period (the served cold-start list is unchanged).
10. **Chapter 7's labelling does not keep row order.** `_add_time_windowed_labels` left-joins the listens with the likes in a polars lazy query and groups by every listen column; polars 2.x does not keep row order through either step, so listens that share a timestamp, user and item come back in a different order on each call. Five calls on the sample bundle's 212,697 training rows returned the same rows and labels every time, with 1,905 to 1,933 rows at different positions. Training order changes the fitted models: sorted by timestamp, user and item only, two builds with the same seed produced byte-identical data, features, index and popularity lists but slightly different rankers. Chapter 13's `add_labels` now breaks those ties on every remaining column, and two builds with the same seed train identical models (Measured Results, Artifact builds); `tests/test_data.py` checks that a shuffled input gives the same output. The same grouping also merges listens that duplicate another listen in every column: 1,645 of 214,342 training listens and 612 of 71,447 validation listens in the sample bundle. Chapter 13 leaves this merging as published.
11. **Two like counts outside the declared feature list.** Fitted with likes, Chapter 6's `HistoricalFeatureBuilder` produces `item_likes` and `item_dislikes`, which its XGBoost pipeline trains on (it selects columns by name prefix), but `get_historical_feature_columns()` lists only `item_like_ratio` and the builder's cold-start defaults cover neither count. Chapter 13 declares both and defaults them to 0 (Design Decisions, One feature snapshot).

## Measured Results

All measurements ran on one laptop: AMD Ryzen 9 8945HS (8 cores, 16 logical processors), 15.3 GB RAM, Windows build 10.0.26200, Python 3.11.9, CPU only. Other workloads ran on the same machine throughout, and free memory ranged from about 0.3 to 1.5 GB. Results come from `outputs/local_run/<timestamp>/` and the bundle manifests.

### Tests

| Run | Result | Time |
|---|---|---|
| `pytest` (default), after the row-order fix | 135 passed | 71.6 s |
| `pytest` (default), after re-vendoring at `074886c` and the per-service accounts | 134 passed | 69.0 s; 123.7 s in a rerun under heavier load |
| `pytest` (default), after the code QA fixes of 2026-11-09 | 131 passed | 70.2 s |
| `pytest` (default), before the two deploy-script tests were added | 129 passed | 87.9 s |
| `pytest` (builds a `tiny` bundle in a temporary folder first), after round 2 | 123 passed | 69.5 s, peak working set 716 MB |
| `pytest` with `CH13_TEST_BUNDLE` set to `sample-20261109-162852`, after round 2 | 123 passed | 67.0 s, peak working set 771 MB |
| `pytest` (default), before round 2 | 101 passed | 65.3 s |
| `pytest` with `CH13_TEST_BUNDLE` set to `sample-20261109-090016`, before round 2 | 101 passed | 61.5 s |

An earlier default run of 98 of these tests took 127.9 s under heavier background load, 25 s of it building the `tiny` bundle.

### Artifact builds (4 build threads)

| Bundle | Data | Build time | Retrieval HR@10 (popularity) | XGBoost AUC | MMoE completion AUC |
|---|---|---|---|---|---|
| `synthetic-v1` | 300 users, 2,000 items, 43,339 listens | 101.0 s | 0.0218 (0.0524) | 0.6156 | 0.5805 |
| `synthetic-v2` | same as above | 32.8 s | 0.0218 (0.0524) | 0.6156 | 0.5805 |
| `sample-20261109-075601` | Yambda: 397 users, 20,000 items, 1,428,936 listens | 358.6 s | 0.0125 (0.0155) | 0.7415 | 0.7394 |
| `sample-20261109-090016` | same as above | 272.7 s | 0.0125 (0.0155) | 0.7415 | 0.7393 |
| `sample-20261109-162852` (round-2 format, `5757e95`) | same as above | 231.0 s | 0.0125 (0.0155) | 0.7415 | 0.7390 |
| `sample-20261109-192754` (`074886c`) | Yambda: 397 users, 20,000 items, 1,428,936 listens | 254.2 s | 0.0125 (0.0145) | 0.7445 | 0.7404 |
| `synthetic-20261109-193547` (`074886c`) | 300 users, 2,000 items, 43,339 listens | 25.7 s | 0.0218 (0.0515) | 0.6192 | 0.5990 |
| `sample-20261109-202000` (active, `074886c`, row-order fix) | Yambda: 397 users, 20,000 items, 1,428,936 listens | 266.6 s | 0.0125 (0.0145) | 0.7456 | 0.7400 |
| `synthetic-20261109-202956` (`074886c`, row-order fix) | 300 users, 2,000 items, 43,339 listens | 30.1 s | 0.0218 (0.0515) | 0.6192 | 0.5990 |

The popularity figures in brackets for bundles before `sample-20261109-192754` count the evaluation plays themselves and are too high (Published-Code Observations, item 9).

**Effect of re-vendoring at `074886c`** (sample, `sample-20261109-162852` against `sample-20261109-192754`). Data, splits and row counts are unchanged (212,697 training and 70,835 validation rows). The two like counts raised the XGBoost AUC from 0.7415 to 0.7445 (log loss 0.5592 to 0.5572, average precision 0.8375 to 0.8400), more than the 0.0006 by which two same-seed builds differed before the row-order fix (below). The MMoE reads neither the like counts nor any time feature, so its inputs are unchanged; its labels changed with the 24-hour like window and the `label_end_ts` bound. Completion AUC went from 0.7390 to 0.7404, engagement from 0.7196 to 0.7186, like from 0.6924 to 0.6738 (positive rate 0.67% to 0.77%) and dislike from 0.5038 to 0.5242 (positive rate 0.12% in both); play-ratio MSE from 0.1546 to 0.1543. All of these except the like AUC lie within the MMoE spread measured on earlier bundles (up to 0.0146). The dislike head stays near chance level. The build took 28.6 s to load data, 4.0 s for features, 10.5 s for XGBoost, 151.8 s for MMoE and 54.8 s for retrieval, with a peak build memory of 1,606 MB (round-2 bundle: 26.3, 3.5, 9.4, 129.7 and 57.7 s; 1,585 MB).

**Same-seed rebuild, before the row-order fix.** A second build of `sample-20261109-192754` with the same seed into a scratch folder reproduced the feature store, the feature encoders, the index, the item lists, the popularity lists and `sample_users.json` byte for byte, and the retrieval metrics exactly. The trained rankers differed: XGBoost AUC 0.7439 against 0.7445, MMoE completion 0.7402 against 0.7404, engagement 0.7171 against 0.7186, like 0.6788 against 0.6738 and dislike 0.5293 against 0.5242. Chapter 7's labelling returns tied training rows in a different order on each call (Published-Code Observations, item 10), and both XGBoost's row subsampling (Chapter 6 samples 80% of rows per tree by position) and the MMoE's seeded shuffle pick rows by position. That build took 210.2 s.

**Same-seed rebuild, after the row-order fix.** `sample-20261109-202000` and a second build with the same seed and four threads into a scratch folder (`sample-20261109-202431`, 261.5 s) trained identical models. XGBoost's `model.json` is byte-identical, and every tensor in `mmoe.pt` (150) and `two_tower.pt` (16) is equal; those two checkpoints and the JSON manifests differ only in the bundle version they record, Chapter 6's `metadata.json` only in its `trained_at` time. Every metric in `bundle_manifest.json` is equal apart from the timings: XGBoost AUC 0.7456 (log loss 0.5561, average precision 0.8403), MMoE completion 0.7400, engagement 0.7184, like 0.6741, dislike 0.5326 and play-ratio MSE 0.1546, retrieval HR@10 0.0125 (popularity 0.0145). Against `sample-20261109-192754` the XGBoost AUC moved by 0.0011, against 0.0006 between the two builds before the fix; the MMoE AUCs moved by 0.0002 to 0.0084 (dislike), within the MMoE spread of up to 0.0146 measured on earlier bundles, and the play-ratio MSE by 0.0003. The synthetic bundle has no tied listens that the sort reorders: `synthetic-20261109-202956` trained the same weights and metrics as `synthetic-20261109-193547`.

Build times vary with the load other work put on the machine; `synthetic-v1` and `synthetic-v2` trained identical models in 101.0 s and 32.8 s. The build records its peak memory from `synthetic-v2` on (631 MB for that bundle). The earlier `sample-20261109-090016` bundle took 29.8 s to load data, 3.3 s for features, 9.1 s for XGBoost, 155.1 s for MMoE (3 epochs, 571,393 parameters) and 70.9 s for retrieval. Its MMoE heads reached AUC 0.7178 for engagement, 0.6899 for like (positive rate 0.67%) and 0.4857 for dislike (positive rate 0.12%). Rebuilding with the same seed reproduced the retrieval and XGBoost metrics exactly; the MMoE metrics moved by up to 0.0146 AUC. With the row-order fix the MMoE trains identical weights on four threads, so that spread came from the order of the training rows rather than from PyTorch threading.

### Local smoke test after the row-order fix (bundle `sample-20261109-202000`, run `20261109T203305Z`)

`scripts/run_local.py --base-port 8100 --smoke`, without Locust. The smoke test passed 15 of 15 checks; 20 warm users were served on rung 1 at a client p50 of 107.1 ms (maximum 168.0 ms). The Locust ramp was not repeated on this bundle.

### Local stack after re-vendoring at `074886c` (bundle `sample-20261109-192754`, run `20261109T193840Z`)

Same command as below (`scripts/run_local.py --base-port 8100 --smoke --locust --ramp 20:2,40:5,60:10`). All five services were ready 7 to 11 s after launch. The smoke test passed 15 of 15 checks; 20 warm users were served on rung 1 at a client p50 of 110.3 ms (maximum 161.1 ms). The ramp completed 1,021 requests, all HTTP 200, at p50 320 ms, p95 600 ms and p99 720 ms (Locust's approximated percentiles); no breaker opened.

| Users | Requests | Throughput | p50 | p95 | p99 | Rungs |
|---|---|---|---|---|---|---|
| 2 | 246 | 12.3/s | 152.9 ms | 248.6 ms | 347.0 ms | 1: 244, 3: 1, 4: 1 |
| 5 | 365 | 18.1/s | 273.2 ms | 404.5 ms | 472.2 ms | 1: 338, 2: 1, 3: 2, 4: 24 |
| 10 | 410 | 20.5/s | 487.8 ms | 665.0 ms | 799.3 ms | 1: 297, 2: 3, 3: 23, 4: 87 |

Skipped stages were all timeouts: the lightweight ranker 112 times, the re-ranker 32 and the heavy ranker 6. Stage times as the orchestrator saw them (`scripts/stage_latency.py`, all 1,048 requests with stage timings):

| Stage | p50 | p95 | p99 |
|---|---|---|---|
| User features | 0.4 ms | 0.6 ms | 1.0 ms |
| Retrieval | 32.3 ms | 118.8 ms | 182.8 ms |
| Item and user-item features | 8.6 ms | 11.7 ms | 58.5 ms |
| Lightweight ranker | 89.9 ms | 210.1 ms | 251.3 ms |
| Heavy ranker | 71.2 ms | 170.9 ms | 261.4 ms |
| Re-ranker | 31.3 ms | 134.6 ms | 175.0 ms |
| Total | 287.7 ms | 554.5 ms | 647.4 ms |

Compared with the "after" run on `sample-20261109-162852` (Item-feature lookup, below), more requests landed on rung 4 after a lightweight-ranker timeout (112 against 71) and the stages were slower throughout, including the user-feature lookup and retrieval, which re-vendoring did not change; the laptop was shared with other work, so the difference is not attributed to the new bundle.

### Local stack after round 2 (bundle `sample-20261109-162852`, runs `20261109T163622Z` and `20261109T165503Z`)

Both runs used `scripts/run_local.py --base-port 8100 --smoke --locust --ramp 20:2,40:5,60:10`: 2, 5 and 10 concurrent users for 20 s each, zero think time, the default 1,000 ms deadline. The second run repeated the first after the per-request service log lines, configuration authorship and contract-rejection counting were added. Neither run opened a circuit breaker.

| Run | Smoke test | Requests (failures) | p50 / p95 / p99 | Rungs | Skipped stages |
|---|---|---|---|---|---|
| `20261109T163622Z` | 15 of 15 checks; 20 warm users on rung 1, client p50 107.6 ms, max 155.8 ms | 1,027 (0) | 260 / 760 / 870 ms | 1: 966, 3: 8, 4: 53 | lightweight ranker timeout 53, re-ranker timeout 8 |
| `20261109T165503Z` | 15 of 15 checks; 20 warm users on rung 1, client p50 90.9 ms, max 135.1 ms | 1,019 (0) | 260 / 790 / 930 ms | 1: 945, 2: 1, 3: 2, 4: 67, 6: 4 | lightweight ranker timeout 71, pipeline deadline 4, re-ranker timeout 2, heavy ranker timeout 1 |

Per step, second run:

| Users | Requests | Throughput | p50 | p95 | p99 | Rungs |
|---|---|---|---|---|---|---|
| 2 | 310 | 15.5/s | 116.4 ms | 198.4 ms | 288.5 ms | 1: 306, 3: 2, 4: 2 |
| 5 | 381 | 18.9/s | 254.5 ms | 413.9 ms | 466.6 ms | 1: 368, 4: 13 |
| 10 | 328 | 16.2/s | 617.8 ms | 878.4 ms | 1,026.2 ms | 1: 271, 2: 1, 4: 52, 6: 4 |

The 1 ms deadline check returned rung 6, Cached list, in both runs. `scripts/stage_latency.py` on the second run's orchestrator log (974 rung-1 requests) gives these stage times as the orchestrator saw them:

| Stage | p50 | p95 | p99 |
|---|---|---|---|
| User features | 3.3 ms | 16.0 ms | 27.1 ms |
| Retrieval | 14.8 ms | 46.3 ms | 69.7 ms |
| Item and user-item features | 52.6 ms | 466.6 ms | 531.4 ms |
| Lightweight ranker | 53.0 ms | 142.8 ms | 174.2 ms |
| Heavy ranker | 39.1 ms | 113.8 ms | 182.7 ms |
| Re-ranker | 17.8 ms | 48.5 ms | 85.8 ms |
| Total | 223.9 ms | 691.4 ms | 779.8 ms |

The item-feature lookup, which then ran in the orchestrator's thread pool and had no budget of its own, had the widest tail. On its own, outside the server, a lookup of 300 items from the SQLite store took a median of 4.6 ms with or without the staleness check. The cause and the fix are under "Item-feature lookup" below. The lookup's time comes out of the deadline before the lightweight ranker's budget is computed, so a slow lookup left less time for the lightweight ranker, the stage skipped most often. Chapter 14's latency analysis starts from this kind of breakdown.

### Item-feature lookup (code QA, 2026-11-09)

The tail came from Python's global interpreter lock (GIL), not from SQLite. `sqlite3` releases the GIL around each step of a query, so a lookup in a worker thread has to win the lock back from the event loop for every row. A busy event loop gives it up only after 5 ms (Python's switch interval), so a lookup that returns hundreds of rows (one item row per candidate, plus the user-item rows) waits behind the loop again and again. `gil_lookup_bench.py`, a scratch script that times a 300-item lookup (user, items and user-item pairs) on the sample bundle, shows the effect without any server:

| How the lookup runs | SQLite p50 / p95 | Parquet p50 / p95 |
|---|---|---|
| Inline on the main thread | 5.7 / 7.9 ms | 5.9 / 6.7 ms |
| Worker thread, main thread idle | 6.2 / 8.8 ms | 6.2 / 7.9 ms |
| Worker thread, main thread running Python code | 366.0 / 737.6 ms | 31.0 / 46.5 ms |
| Same, switch interval lowered to 0.5 ms | 28.2 / 52.2 ms | 36.9 / 46.1 ms |

The orchestrator now runs lookups in an in-process store (SQLite, Parquet) inline on the event loop, and only Redis lookups, which wait on the network, in a worker thread under a timeout (see Orchestrator above). Two local runs on `sample-20261109-162852` with the same command (`scripts/run_local.py --base-port 8100 --smoke --locust --ramp 20:2,40:5,60:10`), before and after the change, give these stage times as the orchestrator saw them (`scripts/stage_latency.py`, all requests):

| Stage | Before: p50 / p95 / p99 | After: p50 / p95 / p99 |
|---|---|---|
| User features | 3.1 / 12.2 / 20.7 ms | 0.3 / 0.5 / 1.0 ms |
| Retrieval | 12.8 / 36.9 / 70.3 ms | 24.8 / 101.1 / 175.5 ms |
| Item and user-item features | 56.8 / 399.0 / 461.0 ms | 6.7 / 10.2 / 52.4 ms |
| Lightweight ranker | 50.0 / 177.5 / 200.4 ms | 68.8 / 189.7 / 226.6 ms |
| Heavy ranker | 36.3 / 101.0 / 206.4 ms | 58.5 / 157.9 / 259.8 ms |
| Re-ranker | 16.3 / 58.9 / 94.4 ms | 23.5 / 101.3 / 145.6 ms |
| Total | 211.6 / 659.5 / 803.3 ms | 219.8 / 498.4 / 634.3 ms |

| Run | Smoke test | Requests (failures) | 10-user step | Rungs |
|---|---|---|---|---|
| Before (`codeqa_before`, started 17:18:58 UTC) | 15 of 15; 20 warm users on rung 1 | 1,164 (0) | 18.9/s, p95 746.4 ms | 1: 1,110, 4: 54 |
| After (`codeqa_after`, started 17:29:52 UTC) | 15 of 15; 20 warm users on rung 1 | 1,245 (0) | 22.9/s, p95 628.1 ms | 1: 1,156, 2: 9, 3: 7, 4: 71, 5: 2 |

The lookup's tail is gone. The other stages slowed because more requests went through the same laptop CPU, five services and the load generator sharing it, and the 10-user step still timed out the lightweight ranker (60 requests on rung 4). On Cloud Run, with one vCPU per service, the lookup took 5.8 ms at the median and 9.8 ms at p95 (Cloud Run Validation Log).

The ramp is shorter than the pre-round-2 ramp below (60 s and 10 users against 100 s and 30 users), so the two are not compared step by step. Under the earlier consecutive-failure breaker, the lightweight ranker's breaker opened 646 times in the longer ramp; with the failure-rate breaker, its timeouts (56 of 328 requests in the 10-user step) never reached half of the calls in a 10-second window, so no breaker opened and each timeout degraded only its own request.

### Local stack on bundle `sample-20261109-090016`, before round 2

All five services reported ready 7 to 9 s after launch. Loading took 0.07 s (re-ranker) to 2.55 s (heavy ranker), and warm-up 2.0 to 38.4 ms. The smoke test passed all 15 checks in 3.0 s: 20 warm users were all served on rung 1 with a client p50 of 93.8 ms and a maximum of 139.1 ms. One sequential client then sent 150 requests: p50 97.7 ms, 149 on rung 1 and one on rung 3 after a single re-ranker timeout.

| Service | CPU per request | Single-core ceiling | Peak resident memory |
|---|---|---|---|
| Orchestrator | 38.0 ms | about 26 requests/s | 87 MB |
| Lightweight ranker (300 candidates) | 29.6 ms | about 34 requests/s | 284 MB |
| Heavy ranker (100 candidates) | 21.9 ms | about 46 requests/s | 357 MB |
| Re-ranker | 8.3 ms | about 120 requests/s | 209 MB |
| Retrieval | 7.6 ms | about 132 requests/s | 300 MB |

### Locust closed-loop ramp before round 2 (run `20261109T090650Z`)

This run used the earlier five-rung ladder (2 any ranking or feature fallback, 3 re-ranker, 4 retrieval, 5 cached list) and a breaker that opened after five consecutive failures. The ramp ran 2, 5, 10, 20 and 30 concurrent users for 20 s each, with the default 1,000 ms deadline. It produced 1,986 requests, all HTTP 200, with no failures. Overall p50 was 720 ms, p95 1,200 ms and p99 1,300 ms (Locust's approximated percentiles). Rungs: 572 on rung 1, 636 on rung 2, 38 on rung 3, 1 on rung 4 and 739 on rung 5.

| Users | Requests | Throughput | p50 | p95 | p99 | Rungs |
|---|---|---|---|---|---|---|
| 2 | 244 | 12.3/s | 162.4 ms | 258.4 ms | 406.4 ms | 1: 242, 2: 2 |
| 5 | 422 | 21.0/s | 229.1 ms | 413.2 ms | 452.0 ms | 1: 171, 2: 249, 3: 2 |
| 10 | 387 | 18.9/s | 513.6 ms | 832.0 ms | 967.6 ms | 1: 138, 2: 246, 3: 2, 5: 1 |
| 20 | 407 | 19.8/s | 998.9 ms | 1,135.2 ms | 1,302.1 ms | 1: 21, 2: 139, 3: 34, 4: 1, 5: 212 |
| 30 | 526 | 27.1/s | 1,109.0 ms | 1,233.5 ms | 1,590.7 ms | 5: 526 |

The orchestrator log for the whole run (smoke test included) records these fallbacks: the pipeline deadline 754 times, the lightweight ranker's open circuit 646 times, its timeout 66 times and its budget 57 times, the heavy ranker's timeout 53 times and budget 49 times, retrieval's timeout 46 times, and the re-ranker's budget 34 times and timeout 4 times.

Throughput levelled off at about 19 to 21 requests per second from 5 to 20 users, below the orchestrator's single-core ceiling. The lightweight ranker, with the lowest ceiling of the four stages, missed its budget first, and its breaker then stayed open for much of the run, so the heavy ranker ranked retrieval's top 100 instead. From 20 users on, most requests reached the deadline and received the cached popular slate (rung 5); at 30 users all of them did, at a client p50 of 1,109 ms. Client latency exceeded the 1,000 ms deadline because the deadline clock starts when the orchestrator's handler runs, so time spent queued before it is not counted, and nothing limits admission.

### Fixes found by measuring

- **Heavy ranker encoding.** With Chapter 7's transform called per request, the heavy ranker's p50 was 341 ms on the sample bundle, above its 300 ms cap, so every request landed on rung 2. Lookup tables built once at load reduced encoding from 286 ms to 14 ms per request.
- **Orchestrator CPU.** httpcore 1.0.9 runs `import sniffio` on every call, and anyio 4.16 no longer installs sniffio, so every call repeated a failed import that searched `sys.path`. Pinning `sniffio==1.3.1` cut the orchestrator's CPU from about 65 ms to 37 ms per request. On the earlier sample bundle, the same Locust ramp completed 1,731 requests before the pin and 2,002 after it (runs `20261109T081739Z` and `20261109T082656Z`).

## Proposed Changes to Figure 13.6

The code follows the figure's rungs, order and labels. Four things the code must do cannot be read from the figure; each has a proposed change:

1. **Rung numbers.** The API returns `degradation.rung` from 1 to 6. Proposed: print 1 to 6 beside the rungs, top to bottom.
2. **Combined failures.** Several stages can fail in one request (for example the lightweight and heavy rankers). Proposed note under the ladder: "A response reports the lowest rung reached and lists every skipped stage."
3. **What counts as failing.** The code also falls back on an invalid response and when too little of the deadline is left for a stage. Proposed footer: "fails = timeout, error, open circuit, invalid response, or too little time left".
4. **Feature defaults.** A failed user-feature lookup and missing or stale values are reported beside the rung. Proposed note: "Missing or stale features: contract defaults, flagged on the response (not a rung)." A failed item-feature lookup leads to rung 4, because neither ranker can score without item features; the note can add "item features unavailable: rung 4".

One ordering point is left as drawn: a lightweight-ranker failure (rung 4) sits below a re-ranker failure (rung 3), although after a lightweight failure the heavy ranker and the value function still run. The rung order therefore describes which stage was lost; it does not rank the quality of the result.

## Files to Publish

Publish the folder as it is, except:

- `artifacts/` and `outputs/` (generated; both are gitignored);
- `services/reranker/artifacts/value_config.json`, `infra/` (empty) and `streamlit_demo/` (empty), which are leftovers from the earlier scaffold; nothing references them, and they are kept in this working copy only because files are not deleted without approval.

## Open Issues

- The measurements come from a machine shared with other workloads under memory pressure; they show behaviour on a laptop and are not benchmarks.
- `services/reranker/artifacts/value_config.json` and the empty `infra/` and `streamlit_demo/` folders are leftovers from the earlier scaffold. Nothing references them; they were kept because files are not deleted without approval, and they are excluded from the files to publish.
- Round 2 and the re-vendoring at `074886c` made every older bundle unloadable (differences 15 and 16); they are kept in `artifacts/` and excluded from publication.

### Resolved

- **Build repeatability.** Same-seed builds trained slightly different rankers because Chapter 7's labelling does not keep row order (Published-Code Observations, item 10). Chapter 13's build now sorts the labelled rows on every column, leaving the published code unchanged, and two same-seed builds train identical models (Measured Results, Artifact builds).

## Cloud Run Validation Log

`python deploy/cloud_run/deploy.py --execute --bundle-dir artifacts/<bundle_version>` appends a summary of each run at the end of this section; the full record is in `outputs/cloud_run/<run_id>/run_record.json`. The measurements below come from the first two runs of 2026-11-09, both on bundle `sample-20261109-162852` with one service account shared by all five services, in `us-central1` with Cloud Build's default worker, one vCPU per service and at most two instances per service. The Locust client ran on the laptop described under Measured Results, so client latency includes the round trip from the laptop to `us-central1`. The third run (`r20261109-194331`), on bundle `sample-20261109-192754` with one service account per service, is measured after its summary at the end of this section.

**What happened.** The first run (`r20261109-174503`) deployed and tore down cleanly, but its smoke test failed one check: all 20 warm-user slates came back on rung 3, Skip re-ranker. Cloud Run's request log shows the re-ranker refusing each of the orchestrator's calls with 403 ("the IAM principal lacks run.routes.invoke") from 18:04:30 to 18:04:35 UTC; its `roles/run.invoker` grant had been applied at 18:03:40, and its next call, at 18:04:47, succeeded. The other three stages had been granted 2 to 4 minutes earlier and answered. The orchestrator degraded as designed, and the run exercised the breaker on real failures: at 18:04:35 the re-ranker's breaker opened ("20 of 20 calls failed in the window"), the smoke test's next five requests skipped the re-ranker without calling it, and at 18:04:47, after the 10-second cool-down, the first Locust request went through as the half-open trial, succeeded and closed the breaker. One request that arrived while the trial was in flight was served without the re-ranker. The script now waits for a rung-1 response before the smoke test (Design Decisions, Cloud Run). In the second run (`r20261109-181455`) that wait took 17.4 s over 4 polls, during which the re-ranker refused 3 calls, and every check passed.

**Time per step**, from `run_record.json`:

| Step | First run | Second run |
|---|---|---|
| Enable the four APIs | 15.5 s (first enablement) | 7.7 s (already enabled) |
| Create the Artifact Registry repository; create the service account, then wait 15 s | 20.8 s; 4.6 s | 6.0 s; 3.9 s |
| `gcloud builds submit`: upload of the 35 MB context and the build | 705.6 s | 632.5 s |
| of which the Cloud Build build (five images in parallel, then the push) | 628.5 s | 560.0 s |
| `gcloud run deploy`: retrieval, lightweight ranker, heavy ranker, re-ranker, orchestrator | 125.8, 59.2, 83.5, 37.4, 31.5 s | 77.6, 46.9, 98.5, 38.5, 32.0 s |
| From the first `run deploy` to the orchestrator's URL, including describes and invoker grants | 382 s | 336 s |
| All five `/health/ready` checks pass | 3.7 s | 3.0 s |
| Wait for rung 1 | not in that version | 17.4 s |
| Smoke test; Locust ramp (90 s) | 12.1 s; 100.7 s | 11.3 s; 93.8 s |
| Teardown, five services to the final check | 95 s | 74 s |
| Whole run | 1,372.9 s (22.9 min) | 1,215.5 s (20.3 min) |

In the second run's build, the five image builds took 248.6 s (retrieval), 196.9 s (lightweight ranker), 255.1 s (heavy ranker) and 140.6 s each (re-ranker, orchestrator), and the push of all five took 298.4 s. The services existed for about 9.4 minutes in the first run and 8.3 minutes in the second. Entry point 6, the dry run, took 0.4 s. Entry point 8 on the already clean project took 44.7 s and wrongly reported five services and the service account as not deleted, because it did not recognise their "not found" replies; after the fix (and a test) it took 48.8 s and reported everything deleted.

**Cold start.** Cloud Run logs when it starts an instance and when the container's startup probe first succeeds; uvicorn opens the port only after the service has loaded and checked its bundle and run its warm-up inference, so the interval covers the image start, the model load and the warm-up:

| Service | Rollout start, first run | Rollout start, second run | Autoscaling start in the 10-user step |
|---|---|---|---|
| Retrieval | 18.7 s | 14.4 s | none |
| Lightweight ranker | 16.5 s | 10.7 s | 10.1 s (second run) |
| Heavy ranker | 20.2 s | 18.5 s | 28.5 s (second run, while the first instance served load) |
| Re-ranker | 12.1 s | 12.5 s | none |
| Orchestrator | 5.0 s | 5.2 s | 5.9 s (first run), 5.3 s (second run) |

A request routed to an instance that is still starting waits for it. In each run exactly one orchestrator request took over 1.5 s: 6.81 s in the first run and 5.62 s in the second, both on the second orchestrator instance during its start. These account for the maximum client latencies of 7,162 ms and 5,920 ms; `--min-instances 0` makes the first request after an idle period pay the same cost.

**Smoke test, second run:** 15 of 15 checks, 20 warm users on rung 1, client p50 277.3 ms (first run: 14 of 15, client p50 264.0 ms, all on rung 3).

**Locust closed-loop ramp** (`RAMP=30:2,60:5,90:10`, zero think time, the default 1,000 ms deadline):

| Run | Users | Requests | Throughput | p50 | p95 | p99 | Rungs |
|---|---|---|---|---|---|---|---|
| Second | 2 | 187 | 6.2/s | 309.5 ms | 431.5 ms | 561.9 ms | 1: 185, 5: 2 |
| Second | 5 | 418 | 13.9/s | 350.9 ms | 488.6 ms | 594.4 ms | 1: 413, 3: 1, 4: 4 |
| Second | 10 | 607 | 20.2/s | 457.7 ms | 692.6 ms | 848.4 ms | 1: 561, 2: 2, 3: 1, 4: 42, 5: 1 |
| First | 2 | 189 | 6.4/s | 311.6 ms | 417.0 ms | 592.3 ms | 1: 185, 3: 1, 4: 2, 5: 1 |
| First | 5 | 460 | 15.2/s | 313.3 ms | 440.5 ms | 573.9 ms | 1: 455, 2: 2, 3: 1, 4: 2 |
| First | 10 | 651 | 21.3/s | 399.9 ms | 626.2 ms | 4,355.6 ms | 1: 590, 2: 10, 4: 51 |

Both ramps returned HTTP 200 for every request (1,212 and 1,300). The skipped stages were timeouts: in the second run, the lightweight ranker 46 times, retrieval 3, the re-ranker 2 and the heavy ranker 2; in the first run, the lightweight ranker 55, the heavy ranker 12, the re-ranker 6 and retrieval 1, plus one request that found the re-ranker's breaker open. No breaker opened in the second run.

Stage times as the second run's orchestrator saw them (`scripts/stage_latency.py` on the 1,244 request records from Cloud Logging, saved as `outputs/cloud_run/r20261109-181455/orchestrator_requests.log`):

| Stage | p50 | p95 | p99 |
|---|---|---|---|
| User features | 0.2 ms | 0.4 ms | 0.9 ms |
| Retrieval | 32.9 ms | 85.8 ms | 126.7 ms |
| Item and user-item features | 5.8 ms | 9.8 ms | 50.0 ms |
| Lightweight ranker | 74.8 ms | 200.1 ms | 212.0 ms |
| Heavy ranker | 56.1 ms | 141.1 ms | 221.9 ms |
| Re-ranker | 32.4 ms | 83.7 ms | 142.8 ms |
| Total | 234.1 ms | 455.6 ms | 592.4 ms |

The lightweight ranker's p95 sits at its 200 ms cap, which is why it is the stage skipped most often. The gap between the orchestrator's total (p50 234.1 ms) and the client's p50 (390 ms over the whole ramp) is the time outside the orchestrator's handler: the network between the laptop and `us-central1`, Cloud Run's front end and any queueing.

**Resources.** Each of the first two runs created one Artifact Registry repository with five images, one service account (`ch13-runtime`) with four invoker grants, five Cloud Run services, the build source objects and the `gs://gen-lang-client-0843377880_cloudbuild` bucket, which did not exist before the first run. Teardown deleted all of them in both runs, and `gcloud` listings afterwards found no Cloud Run service in any region, no Artifact Registry repository, no `ch13-runtime` account and no `_cloudbuild` bucket. Cloud Build used about 20 build minutes across the two builds.

### Cloud Run run r20261109-174503
- Project gen-lang-client-0843377880, region us-central1, bundle sample-20261109-162852 (sample), builder cloud-build, max instances 2
- Started 2026-11-09T17:45:03Z, finished 2026-11-09T18:07:55Z
- Smoke test: FAILED or not run; rungs {'3': 20}; client p50 264.0 ms
- Locust: 1300 requests, 0 failures, p50 350 ms, p95 570 ms, rungs {'1': 1230, '2': 12, '3': 2, '4': 55, '5': 1}
- Teardown: complete
- Note (added by the code QA): the smoke test failed because the re-ranker's new invoker grant had not yet taken effect; see the start of this section.

### Cloud Run run r20261109-181455
- Project gen-lang-client-0843377880, region us-central1, bundle sample-20261109-162852 (sample), builder cloud-build, max instances 2
- Started 2026-11-09T18:14:55Z, finished 2026-11-09T18:35:10Z
- Full pipeline (rung 1) first served 17.4 s after all five services were ready
- Smoke test: passed; rungs {'1': 20}; client p50 277.3 ms
- Locust: 1212 requests, 0 failures, p50 390 ms, p95 640 ms, rungs {'1': 1159, '2': 2, '3': 2, '4': 46, '5': 3}
- Teardown: complete

### Cloud Run run r20261109-194331
- Project gen-lang-client-0843377880, region us-central1, bundle sample-20261109-192754 (sample), builder cloud-build, max instances 2
- Started 2026-11-09T19:43:31Z, finished 2026-11-09T20:05:17Z
- Full pipeline (rung 1) first served 54.7 s after all five services were ready
- Smoke test: passed; rungs {'1': 20}; client p50 292.1 ms
- Locust: 1174 requests, 0 failures, p50 370 ms, p95 610 ms, rungs {'1': 1130, '2': 1, '3': 6, '4': 37}
- Teardown: complete

#### Measurements of run r20261109-194331 (added by the re-vendoring run)

The first run with one service account per service and the first on code vendored at `074886c` (bundle `sample-20261109-192754`). Same region, worker, instance limits and Locust client as the two runs above.

**Time per step**, from `run_record.json`: enabling the APIs (already enabled) 7.0 s; the repository 5.7 s; the five service accounts 3.8 to 6.1 s each (23.1 s in all), then the 15 s wait; `gcloud builds submit` 586.3 s, of which the Cloud Build build took 531.9 s (image builds 250.1 s for retrieval, 200.6 s for the lightweight ranker, 255.2 s for the heavy ranker, 138.7 s and 138.9 s for the re-ranker and the orchestrator; the push of all five 269.8 s). `gcloud run deploy` took 82.7, 47.0, 124.8, 36.2 and 34.6 s for retrieval, lightweight ranker, heavy ranker, re-ranker and orchestrator, and 368.6 s passed from the first deploy to the orchestrator's URL. All five readiness checks passed within 3.6 s. The wait for rung 1 took 54.7 s over 11 polls: the re-ranker refused the orchestrator's calls with 403 ten times, from 20:00:35 to 20:01:23 UTC, while its new grant to `ch13-orchestrator` propagated; no other stage refused a call. Smoke test 11.3 s, Locust ramp 99.4 s. Teardown took 115.8 s from the first grant removal to the final check (four grant removals 20.9 s, five service deletes 38.3 s, the repository 5.2 s, five account deletes 17.2 s, the source objects and the bucket 22.7 s, the three checks 11.5 s). Whole run 1,306 s (21.8 min); the services existed for about 10 minutes.

**Cold start** (instance start to startup probe passed): rollout starts took 16.6 s for retrieval, 9.1 s for the lightweight ranker, 15.9 s for the heavy ranker, 12.8 s for the re-ranker and 9.5 s for the orchestrator. In the 10-user step Cloud Run started a second lightweight-ranker instance (14.6 s) and a second orchestrator instance (13.8 s). One orchestrator request, routed to that second instance as it started, took 14.4 s; it is the ramp's maximum client latency of 14,819 ms. No other request took over 1.5 s.

**Smoke test:** 15 of 15 checks, 20 warm users on rung 1, client p50 292.1 ms, maximum 357.7 ms.

**Locust closed-loop ramp** (`RAMP=30:2,60:5,90:10`): 1,174 requests, all HTTP 200, p50 370 ms, p95 610 ms, p99 740 ms.

| Users | Requests | Throughput | p50 | p95 | p99 | Rungs |
|---|---|---|---|---|---|---|
| 2 | 195 | 6.6/s | 291.4 ms | 415.7 ms | 732.2 ms | 1: 194, 4: 1 |
| 5 | 439 | 14.6/s | 325.8 ms | 479.6 ms | 543.3 ms | 1: 434, 3: 1, 4: 4 |
| 10 | 540 | 18.0/s | 457.6 ms | 659.5 ms | 790.1 ms | 1: 502, 2: 1, 3: 5, 4: 32 |

The skipped stages were timeouts: the lightweight ranker 37 times, the re-ranker 7 and the heavy ranker 1. No breaker opened.

**Stage times** as the orchestrator saw them (`scripts/stage_latency.py` on the 1,216 request records from Cloud Logging, saved as `outputs/cloud_run/r20261109-194331/orchestrator_requests.log`):

| Stage | p50 | p95 | p99 |
|---|---|---|---|
| User features | 0.2 ms | 0.3 ms | 0.5 ms |
| Retrieval | 30.6 ms | 87.2 ms | 137.0 ms |
| Item and user-item features | 7.2 ms | 11.1 ms | 43.1 ms |
| Lightweight ranker | 66.0 ms | 188.6 ms | 210.0 ms |
| Heavy ranker | 48.2 ms | 120.7 ms | 182.2 ms |
| Re-ranker | 35.4 ms | 95.5 ms | 140.9 ms |
| Total | 220.1 ms | 433.2 ms | 515.7 ms |

**Resources and teardown.** The run created one Artifact Registry repository with five images, five service accounts (`ch13-orchestrator`, `ch13-retrieval`, `ch13-light-ranker`, `ch13-heavy-ranker`, `ch13-reranker`), four `roles/run.invoker` grants to `ch13-orchestrator` (one on each stage service), five Cloud Run services, the build source objects and the `gs://gen-lang-client-0843377880_cloudbuild` bucket, which did not exist before the run. Teardown reported every item deleted. Independent `gcloud` listings afterwards found no Cloud Run service or job in any region, no Artifact Registry repository in `us-central1`, no service account other than the two that predate this work, no project IAM binding naming `ch13`, no `_cloudbuild` bucket and only the pre-existing `recsys-book-runs-0843377880` bucket. The APIs stay enabled. Cloud Build used about 9 build minutes.
