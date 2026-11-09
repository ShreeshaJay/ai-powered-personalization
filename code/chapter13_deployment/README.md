# Chapter 13: Online Deployment

Companion code for Chapter 13. It serves the models built in Chapters 5 to 8 as five web services that answer one recommendation request together, under a deadline, and keep answering when a stage is slow or down.

| Service | Port | Endpoints | What it runs | Published code it serves |
|---|---|---|---|---|
| Orchestrator | 8000 | `POST /v1/recommend` | feature lookups, deadline budgets, retries, circuit breakers, degradation ladder | (Chapter 13) |
| Retrieval | 8001 | `POST /v1/retrieve` | user tower and an in-process FAISS index | Chapter 5 two-tower model |
| Lightweight ranker | 8002 | `POST /v1/rank` | XGBoost on CPU, about 300 candidates | Chapter 6 `XGBoostRanker` |
| Heavy ranker | 8003 | `POST /v1/rank` | multi-task MMoE on CPU, about 100 candidates | Chapter 7 `MMoE` |
| Re-ranker | 8004 | `POST /v1/rerank`, `GET /v1/config`, `POST /v1/config/reload` | value function, MMR diversity, artist pacing | Chapter 8 |

Every service also serves `GET /health/live`, `GET /health/ready` and `GET /health`, and FastAPI's interactive documentation at `/docs`.

Everything runs on a laptop CPU with Python and uvicorn on Windows, macOS or Linux. Docker is optional. A separate script deploys the same five services to Google Cloud Run for a short test and deletes them again; see `CLOUD_RUN.md`.

## What Each Part Demonstrates

| Chapter topic | Code | How to see it |
|---|---|---|
| Contracts between stages (Figure 13.3) | `contracts/schemas.py`, used by every service | Every response carries `trace_id` and `model_version`; a malformed request gets HTTP 422; `tests/test_contracts.py` |
| Two-tower split and version pairing (Figure 13.4) | `pipeline_build/train_retrieval.py`, `services/retrieval/engine.py` | `retrieval/index_manifest.json` pairs the index with the user tower; retrieval refuses to start on a mismatched index (`tests/test_artifacts.py`) |
| Serving profiles of the five services (Table 13.4) | `services/*/engine.py`, `services/*/Dockerfile` | `GET /health` on each service lists what it loaded |
| Online features and training/serving consistency | `common/feature_store.py`, `pipeline_build/feature_store_build.py` | `tests/test_services.py` rebuilds a ranking request from the store and compares it with the training rows |
| Orchestration under a deadline (Figures 13.2 and 13.6) | `services/orchestrator/pipeline.py`, `common/resilience.py` | The `degradation` field of every response; `tests/test_orchestrator.py`; the Locust ramp |
| Value-function configuration and staged release | `config/reranker/`, `services/reranker/config_store.py`, `scripts/promote_config.py` | Validate, activate and hot-reload a configuration |
| Immutable artifacts, pointer switch, readiness | `artifacts/ACTIVE`, `common/artifacts.py`, `/health/ready` | Build a second bundle and restart the services |
| Validation activities (Table 13.7) | `scripts/test_endpoint.py`, `tests/`, `scripts/smoke_test.py`, `load/locustfile.py` | See "Validating the Pipeline" below |
| Containers and a cloud deployment (Figures 13.7 and 13.8) | `services/*/Dockerfile`, `docker-compose.yml`, `deploy/cloud_run/` | `CLOUD_RUN.md` |

## Quick Start

### 1. Install

Use Python 3.11 and create the virtual environment outside any synced folder (OneDrive, Dropbox, iCloud).

```bash
python -m venv ~/venvs/ch13
source ~/venvs/ch13/bin/activate          # Windows: C:\venvs\ch13\Scripts\activate
cd chapter13_deployment
pip install -r requirements.txt
```

`requirements.txt` installs CPU-only PyTorch wheels. Each service folder has its own smaller `requirements.txt` for its container image.

### 2. Build an Artifact Bundle

A bundle holds every model, index, feature table and list the five services load, with hashes. Build one from synthetic data (no download needed):

```bash
python -m pipeline_build.build_artifacts --profile synthetic
```

or from a random sample of Yambda users (400 users and their 20,000 most-played tracks):

```bash
python -m pipeline_build.build_artifacts --profile sample --data-dir /path/to/yambda
```

`/path/to/yambda` is a download of [Hugging Face, yandex/yambda](https://huggingface.co/datasets/yandex/yambda) holding `flat/50m/listens.parquet`, `flat/50m/likes.parquet`, `flat/50m/dislikes.parquet` (the 50M variant used in Chapters 6 to 8), `embeddings.parquet`, `artist_item_mapping.parquet` and `album_item_mapping.parquet`; a copy with the three interaction files directly under `flat/` also works. `CLOUD_RUN.md` shows a download command. You can also set `YAMBDA_DATA_DIR` instead of passing `--data-dir`.

The build writes `artifacts/<bundle_version>/` and points `artifacts/ACTIVE` at it. Section "Artifact Bundles" below covers the profiles and the path to full-size artifacts.

### 3. Run the Five Services

```bash
python scripts/run_local.py                   # start all five; Ctrl+C stops them
python scripts/run_local.py --smoke           # start, run the smoke test, stop
python scripts/run_local.py --smoke --locust  # ... then a closed-loop Locust ramp, stop
```

The runner refuses to start if a port is taken (use `--base-port 8100` to move all five), waits for every `/health/ready`, and stops every process it started, also after errors and Ctrl+C. Logs and reports go to `outputs/local_run/<timestamp>/`.

To run one service by hand:

```bash
uvicorn services.heavy_ranker.app:app --port 8003
```

### 4. Send a Request

```bash
python scripts/test_endpoint.py                          # first warm user of the active bundle
python scripts/test_endpoint.py --user-id 998000 --deadline-ms 300
```

or with curl (the user IDs here come from a Yambda sample bundle; each bundle lists its own):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/recommend \
  -H "Content-Type: application/json" -H "X-Request-Timeout-Ms: 1000" \
  -d '{"user_id": 10300, "num_results": 10}'
```

`tests/test_orchestrator.py` posts exactly the payload and headers of this curl command and checks the response against the contract.

Warm and cold-start user IDs for the active bundle are in its `sample_users.json` (the bundle folder is named in `artifacts/ACTIVE`). The response lists the items, the rung of the degradation ladder that served them, any stages skipped or substituted, the feature values replaced by defaults, the version of every stage, and per-stage timings. One response from a Yambda sample bundle, with `"num_results": 3` and the third item left out:

```json
{
  "trace_id": "8453a780b9124a68b0cade63e8299072",
  "model_version": "sample-20261109-202000",
  "user_id": 10300,
  "items": [
    {"rank": 1, "item_id": 3988618, "score": 1.0, "artist_id": 554106, "album_id": 747417, "track_length_seconds": 195},
    {"rank": 2, "item_id": 7253259, "score": 0.9773, "artist_id": 485948, "album_id": 3002276, "track_length_seconds": 200}
  ],
  "score_source": "value_function",
  "degradation": {
    "rung": 1, "label": "Full pipeline", "fallbacks": [],
    "feature_defaults": {
      "lookup_failed": [],
      "missing": {"item_total_plays": 3, "item_avg_completion": 3, "item_std_completion": 3, "item_unique_listeners": 3,
                  "item_organic_ratio": 3, "item_repeat_ratio": 3, "item_like_ratio": 3, "item_likes": 3,
                  "item_dislikes": 3},
      "stale": {}
    }
  },
  "stage_versions": {
    "feature_set": "fs-sample-20261109-202000",
    "retrieval": "two-tower-sample-20261109-202000",
    "index": "index-sample-20261109-202000",
    "retrieval_source": "two_tower",
    "lightweight_ranker": "xgboost-sample-20261109-202000",
    "heavy_ranker": "mmoe-sample-20261109-202000",
    "reranker_config": "vf-default-v1"
  },
  "timings_ms": {"user_features": 0.84, "retrieval": 15.54, "item_features": 7.79, "lightweight_ranker": 41.52,
                 "heavy_ranker": 27.39, "reranker": 10.37, "hydrate": 0.03, "total": 107.88},
  "deadline_ms": 1000.0
}
```

Here three candidate tracks had no plays in the feature window, so their nine item statistics were replaced by the contract defaults and counted under `missing`.

A degraded response names the rung and every skipped stage. With the re-ranker down, for example:

```json
"degradation": {
  "rung": 3, "label": "Skip re-ranker",
  "fallbacks": [{"stage": "reranker", "rung": 3, "label": "Skip re-ranker", "reason": "error",
                 "detail": "...", "action": "sorted by model score, removed the user's recent items and applied artist pacing"}],
  ...
}
```

A request that breaks the contract, for example `{"user_id": -1}`, gets HTTP 422:

```json
{"error": "contract_violation",
 "detail": [{"loc": ["body", "user_id"], "msg": "Input should be greater than or equal to 0", "type": "greater_than_equal"}],
 "trace_id": "55dfc27a1c9d41dfaf53e2db94c3200f", "model_version": "sample-20261109-202000"}
```

### 5. Run the Tests

```bash
pytest
```

The first test session builds a tiny synthetic bundle in a temporary folder, so the tests never touch `artifacts/`. To reuse a bundle you already built, set `CH13_TEST_BUNDLE`:

```bash
CH13_TEST_BUNDLE=artifacts/<bundle_version> pytest      # PowerShell: $env:CH13_TEST_BUNDLE="artifacts\<bundle_version>"; pytest
```

## Architecture

```
                         ┌────────────────────────────────────────────┐
  Client ── /v1/recommend│ Orchestrator :8000                         │
   X-Request-Timeout-Ms  │  deadline, stage budgets, retry, breakers  │
                         └──┬──────────┬───────────┬───────────┬──────┘
          feature store ◄───┤          │           │           │
          (user, item,      ▼          ▼           ▼           ▼
           user-item)   Retrieval  Lightweight   Heavy      Re-ranker
                        :8001      ranker :8002  ranker     :8004
                        user tower XGBoost       :8003      value function,
                        + FAISS    300 → 100     MMoE       MMR, pacing
                        → 300                    100 → 100  100 → 20
```

One request flows as follows (Figure 13.2):

1. The orchestrator reads the user's features and calls retrieval; with Redis, the feature lookup runs while the retrieval call is in flight.
2. Retrieval encodes the user's recent items with the user tower and searches the FAISS index, excluding those recent items. A user with no usable history gets the popularity list.
3. The orchestrator fetches item and user-item features for the candidates in one batch, drops candidates that lack the features the rankers need (items never played have no track length), and sends the rest to the lightweight ranker, which keeps the top 100.
4. The heavy ranker scores those 100 with five task heads (engagement, completion, play ratio, like, dislike).
5. The re-ranker combines the task scores with the Chapter 8 value function, applies MMR diversity and artist and album pacing, and returns the slate.
6. The orchestrator adds item metadata and returns the response.

## Contracts

`contracts/schemas.py` defines every request and response with Pydantic. Paths carry the version (`/v1/...`). Every response includes `trace_id` and `model_version`. A request that violates its contract gets HTTP 422 with an `ErrorResponse` body, never a guess. Unknown fields are ignored, so adding an optional field stays compatible; removing or retyping a field needs a new version. `contracts/schema_snapshot_v1.json` records the published v1 schemas, and `tests/test_contracts.py` fails if a change would break an existing client.

Request headers understood by every service:

| Header | Meaning |
|---|---|
| `X-Trace-Id` | Correlates the logs of one request across services; generated when absent and echoed back |
| `X-Request-Timeout-Ms` | Time the caller will wait; the service returns 504 when it cannot answer in time; an unparseable value is a 422 |

Responses also carry `X-Trace-Id`, `X-Model-Version` and a `Server-Timing` header.

## Orchestration Under a Deadline

The orchestrator turns the client's deadline (default 1,000 ms, capped at 5,000 ms) into a timeout for each call:

- **Budgets from the remaining time.** A stage gets the smaller of its cap and the time left after reserving the minimum budgets of the stages after it. If that is below the stage's own minimum, the stage is skipped (`deadline_budget`). The callee receives its timeout in `X-Request-Timeout-Ms`.
- **Bounded retries.** One retry point (the stage call), at most one retry, only for a connection error or HTTP 502 or 503, with a jittered pause, and only if the stage's time still allows it. Timeouts are never retried.
- **Retry budget.** All stages of one orchestrator process share a token bucket that refills at one retry per second and holds at most ten. A retry that finds the bucket empty is skipped: the call counts as failed and the request degrades. `GET /health` reports retries spent and dropped.
- **Circuit breakers.** One per stage, judged on a failure rate (Section 13.8.5): the breaker opens when more than half of at least twenty calls in the last ten seconds failed. An open breaker skips the stage for ten seconds; then one trial call is let through (half-open), and its success closes the breaker while its failure reopens it. A run of failures smaller than the minimum call count never opens it. A contract violation (HTTP 422) is the caller's error: it does not count against the stage and is never retried, but it is logged with the trace ID, the field-level detail and both versions, and counted per stage in `GET /health` (`contract_rejections`).
- **Degradation ladder (Figure 13.6).** Every response reports the rung that served it, with the figure's labels:

| Rung | Label | Reached when | What is served |
|---|---|---|---|
| 1 | Full pipeline | all five services answer | the re-ranked slate |
| 2 | Skip heavy ranker | the heavy ranker fails | the lightweight scores, re-ranked |
| 3 | Skip re-ranker | the re-ranker fails | candidates sorted by the best available score; the user's recent items removed and artist pacing applied |
| 4 | Skip lightweight ranker | the lightweight ranker fails, or item features cannot be read | the top candidates by retrieval similarity, then the heavy ranker and re-ranker if they answer |
| 5 | Popularity candidates | retrieval fails | the in-memory popularity list outside the user's recent items, ranked if the rankers answer |
| 6 | Cached list | the deadline runs out, or every stage fails | a precomputed popular slate, returned without ranking |

A stage fails when it times out, returns an error, has an open breaker, returns a response that breaks its contract, or would get less than its minimum budget. When several stages fail, the response reports the lowest rung reached and lists every skipped stage in `degradation.fallbacks`, each with its rung, label, reason (`timeout`, `error`, `circuit_open`, `deadline_budget`, `contract_violation`, `features_unavailable`) and the action taken.

- **Feature defaults are a flag beside the rung.** A failed user-feature lookup serves the request with the contract's cold-start defaults; missing or stale feature values are replaced by defaults too. `degradation.feature_defaults` lists any failed lookup and counts the substituted values per feature (`missing` and `stale`).

The orchestrator writes one structured log record per request (logger `ch13.orchestrator`, level WARNING when a stage was skipped) with the trace ID, rung, label, each skipped stage as `"<rung> <label>: <stage> <reason>"`, the feature defaults, the score source, the versions and the timings. Every service, the orchestrator included, also logs one line per POST request with the service name, trace ID, path, status, duration and model version, so searching all five logs for one trace ID shows each stage the request reached.

The default budgets are illustrative values for a laptop; Chapter 14 shows how to derive budgets from measurements. Every setting can be changed with an environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `ORCH_DEADLINE_MS` | 1000 | deadline when the client sends none |
| `ORCH_MAX_DEADLINE_MS` | 5000 | cap on a client-supplied deadline |
| `ORCH_RETRIEVAL_CANDIDATES` | 300 | candidates requested from retrieval |
| `ORCH_LIGHTWEIGHT_KEEP` | 100 | candidates passed to the heavy ranker |
| `ORCH_<STAGE>_CAP_MS`, `ORCH_<STAGE>_MIN_MS` | 250/20, 200/15, 300/25, 150/10 | budget cap and minimum for RETRIEVAL, LIGHTWEIGHT_RANKER, HEAVY_RANKER, RERANKER |
| `ORCH_RESPONSE_RESERVE_MS` | 10 | time kept for building the response |
| `ORCH_FEATURE_LOOKUP_CAP_MS` | 100 | timeout cap for one lookup in a networked feature store (Redis); SQLite and Parquet lookups have none |
| `ORCH_BREAKER_FAILURE_RATE`, `ORCH_BREAKER_MIN_CALLS`, `ORCH_BREAKER_WINDOW_S` | 0.5, 20, 10 | the breaker opens when the failure rate exceeds this share of at least this many calls in this many seconds |
| `ORCH_BREAKER_RESET_S` | 10 | cool-down before the half-open trial call |
| `ORCH_RETRY_MAX`, `ORCH_RETRY_BACKOFF_MS` | 1, 10 | retry limit and base pause (up to 50% jitter is added) |
| `ORCH_RETRY_BUDGET_PER_S`, `ORCH_RETRY_BUDGET_BURST` | 1, 10 | process-wide retry budget: refill rate and bucket size |
| `RETRIEVAL_URL`, `LIGHTWEIGHT_RANKER_URL`, `HEAVY_RANKER_URL`, `RERANKER_URL` | `http://127.0.0.1:8001` to `:8004` | stage addresses |
| `SERVICE_AUTH` | `none` | `gcp_id_token` on Cloud Run (the orchestrator then calls the stages with identity tokens) |

## Artifact Bundles

`pipeline_build/build_artifacts.py` trains every model with the published Chapter 5 to 8 code and writes one versioned bundle:

```
artifacts/<bundle_version>/
├── bundle_manifest.json       versions, data source, metrics, SHA-256 of every file
├── sample_users.json          warm and cold-start user IDs for tests and load tests
├── retrieval/                 two_tower.pt, item_index.faiss, item_ids.npy, content_embeddings.npy,
│                              popularity.json, index_manifest.json
├── lightweight_ranker/        Chapter 6 XGBoost model and feature list
├── heavy_ranker/              mmoe.pt, feature_processor.pkl, serving_manifest.json
├── reranker/                  item content embeddings for MMR, artist and album mappings
├── feature_store/             features.sqlite, the same tables as Parquet, feature_contract.json
└── orchestrator/              popularity.json
```

| Profile | Data | Use |
|---|---|---|
| `tiny` | synthetic, 120 users | built by the test suite in a temporary folder |
| `synthetic` | synthetic, 300 users, 2,000 items | runs anywhere without a download |
| `sample` | Yambda, 400 random users, their 20,000 most-played tracks with audio embeddings | real data on a laptop |
| `full` | all Yambda users and played tracks, the chapters' own epoch counts | the path to full-size artifacts (needs a large machine; see `CLOUD_RUN.md`) |

Bundles built before the feature contract was added (they have no `feature_store/feature_contract.json`) no longer load; rebuild them with the command above. `--users` and `--catalog-size` override a profile's sizes. Bundles are immutable: building one never changes another, and `artifacts/ACTIVE` is switched atomically after the build succeeds (`--no-activate` skips the switch). Set `CH13_BUNDLE_DIR` to serve a specific bundle, or `CH13_ARTIFACT_ROOT` to keep bundles elsewhere.

Services check the bundle before they report ready and stop with a clear message when anything is wrong: a missing file, a hash that does not match the manifest, an index built for another user tower, item vectors that do not match the item tower, or published code in `vendor/` that differs from the code the bundle was built with (the bundle manifest records the hash of every vendored file, so updating `vendor/` means rebuilding the bundle). No service ever falls back to random scores or an untrained model. Each service also runs one inference before it reports ready, so the first real request does not pay the model's first-call cost.

## Online Feature Store

`common/feature_store.py` serves the features through one interface with three back ends:

| `FEATURE_STORE_BACKEND` | Reads | Notes |
|---|---|---|
| `sqlite` (default) | `feature_store/features.sqlite` | read-only, one connection per thread |
| `parquet` | `feature_store/*.parquet` | loaded into memory at start-up |
| `redis` | a Redis server at `REDIS_URL` | load it first with `python scripts/load_redis.py` |

The tables are a snapshot of Chapter 6's `HistoricalFeatureBuilder`, fitted on the same time window the rankers trained on, and cold-start defaults are applied the way `HistoricalFeatureBuilder.transform` applies them. `tests/test_services.py` checks that a ranking request built from the store reproduces the offline training features exactly and that both rankers score it as the published models score the training rows.

`feature_store/feature_contract.json` declares every feature the rankers read: its group (user, item or user-item), type, encoding, default and maximum age. Each user and item row carries an `updated_at` timestamp, the time the row was materialised. The store treats a value older than its maximum age at the request's time as missing, substitutes the default, and counts the substitution; the counts appear in each response's `degradation.feature_defaults` and, summed over the process, in `GET /health`. Ages are in the dataset's event-time units (Yambda timestamps count seconds from the start of the dataset), and a request's time defaults to the bundle's reference time, the last listen in the data. Both rankers refuse to start if their model reads a feature the contract does not declare or if the contract belongs to another feature set.

SQLite and Parquet lookups read a local file or this process's memory, so the orchestrator runs them directly on its event loop (`DESIGN_NOTES.md` explains why a worker thread made them slow under load). A Redis lookup waits on another server, so it runs in a worker thread under a timeout: the smaller of `ORCH_FEATURE_LOOKUP_CAP_MS` and the time the deadline leaves after reserving the later stages' minimum budgets. A lookup that times out counts as failed: user features fall back to cold-start defaults, and missing item features send the request to rung 4.

## Value-Function Configuration

The re-ranker reads its value-function weights, MMR settings and pacing rules from `config/reranker/<config_version>.json`; the file named in `config/reranker/ACTIVE` is live. Each file records its `author` and the `reason` for the change, and a published version is never edited; a change is a new file. Configurations are validated (unknown keys, wrong types, out-of-range values, a missing author or reason, and a `config_version` that differs from the file name are all rejected) before they can become active.

```bash
python scripts/promote_config.py --list
python scripts/promote_config.py --validate config/reranker/vf-growth-v1.json
python scripts/promote_config.py --activate vf-growth-v1 --reload-url http://127.0.0.1:8004
curl -s http://127.0.0.1:8004/v1/config
```

The re-ranker also checks `ACTIVE` every five seconds (`RERANKER_CONFIG_POLL_S`). An invalid configuration is refused and the last good one keeps serving; every response names the `config_version` it used.

A request can pin a configuration with the optional `policy_version` field, for example to compare two policies on the same users:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/recommend -H "Content-Type: application/json" \
  -d '{"user_id": 10300, "num_results": 10, "policy_version": "vf-growth-v1"}'
```

If the pinned file exists and validates, the re-ranker uses it without changing `ACTIVE`; otherwise it serves the active configuration and logs a warning. The response's `stage_versions` names the configuration actually used (`reranker_config`) and whether the pin was `honoured` or `unavailable` (`policy_pin`).

## Validating the Pipeline

| Activity (Table 13.7) | Command | What it checks |
|---|---|---|
| Manual request | `python scripts/test_endpoint.py` or curl | one response, stage by stage |
| Contract tests | `pytest tests/test_contracts.py` | schemas, 422 on violations, compatibility with the v1 snapshot |
| Integration tests | `pytest` | each service against the published code; every rung of the ladder through fault injection; breaker, retry budget and deadline behaviour; feature staleness and the feature contract; policy pinning; artifact checks; feature consistency; the README curl example |
| Smoke test | `python scripts/smoke_test.py` | readiness of all five services, well-formed slates for warm users, popularity for cold-start users, 422 on each service, a 1 ms deadline that returns the cached list instead of failing |
| Load test | `python scripts/run_local.py --locust` or `locust -f load/locustfile.py --headless --host http://127.0.0.1:8000` | a closed-loop ramp of concurrent users; latency percentiles, throughput, ladder rungs and skipped stages per step |

The integration tests run the five services inside the test process and inject failures at each service boundary; the smoke and load tests run against services started by `scripts/run_local.py`, by Docker Compose, or on Cloud Run. The Locust ramp is set with `RAMP` as `end_second:users` pairs (default `20:2,40:5,60:10,80:20,100:30`). Each user has zero think time: it sends its next request as soon as the previous one returns. The summary is written to `LOCUST_SUMMARY` (the local runner puts it in `outputs/local_run/<timestamp>/locust_summary.json`). Chapter 14 starts from these results.

### Measuring One Stage's Latency

To measure the heavy ranker (or any stage) on your own machine, run a ramp and read the per-stage timings from the orchestrator's request log:

```bash
python scripts/run_local.py --locust --ramp 60:1
python scripts/stage_latency.py outputs/local_run/<timestamp>/orchestrator.log --rung 1
```

`stage_latency.py` prints the count, p50, p95 and p99 of each stage's time as the orchestrator saw it, network included. Each service also reports its own processing time in the `Server-Timing` response header (`app;dur=<ms>`). One user gives the latency without queueing; repeat with more users to see how it grows under load.

## Docker Compose (Optional)

```bash
BUNDLE=artifacts/<bundle_version> docker compose up --build
# PowerShell: $env:BUNDLE="artifacts/<bundle_version>"; docker compose up --build
```

Each image holds only its service's code, the vendored published code and the parts of the bundle it reads, and runs as a non-root user. To use Redis for the features:

```bash
docker compose --profile redis up -d redis
python scripts/load_redis.py --bundle-dir artifacts/<bundle_version>
BUNDLE=artifacts/<bundle_version> FEATURE_STORE_BACKEND=redis docker compose up --build
```

## Cloud Run (Optional)

`deploy/cloud_run/deploy.py` builds the five CPU images, deploys them to Cloud Run with minimum instances set to zero, runs the smoke test and a short Locust ramp, records the result in `outputs/cloud_run/<run_id>/` and in the Cloud Run validation log at the end of `DESIGN_NOTES.md`, and deletes everything it created. Without `--execute` it only prints the commands. Read `CLOUD_RUN.md` before running it; Cloud Run bills for use.

## Published Code

`vendor/` holds byte-identical copies of the published Chapter 5 to 8 files the services use, with their source paths and SHA-256 hashes in `vendor/MANIFEST.json`; `vendor/README.md` explains how they are loaded and refreshed. `python scripts/check_vendor.py` confirms the copies are unchanged.

## Code Structure

```
chapter13_deployment/
├── contracts/            Pydantic request and response models, v1 schema snapshot
├── common/               bundle checks, feature store, deadlines, budgets, breakers, retries, service skeleton
├── vendor/               published Chapter 5 to 8 code (unchanged copies) and MANIFEST.json
├── pipeline_build/       ★ artifact build: data loading, retrieval, rankers, feature store
├── services/
│   ├── orchestrator/     ★ /v1/recommend: pipeline, degradation ladder, settings
│   ├── retrieval/        ★ /v1/retrieve: user tower + FAISS
│   ├── lightweight_ranker/  ★ /v1/rank: XGBoost
│   ├── heavy_ranker/     ★ /v1/rank: MMoE
│   └── reranker/         ★ /v1/rerank: value function, MMR, pacing, config store
├── config/reranker/      value-function configurations and the ACTIVE pointer
├── scripts/              run_local, smoke_test, test_endpoint, stage_latency, promote_config, load_redis, vendor tools
├── load/locustfile.py    ★ closed-loop load test
├── tests/                pytest suite
├── deploy/cloud_run/     ★ Cloud Run deploy, test and teardown script
├── docker-compose.yml    optional containers (and Redis)
├── DESIGN_NOTES.md       decisions, departures from the plan, measured results
└── CLOUD_RUN.md          entry points, hardware, runtimes, inputs and outputs
```

Each service folder holds `app.py` (HTTP layer), `engine.py` (model loading and inference), `Dockerfile` and `requirements.txt`.

## References

- FastAPI: <https://fastapi.tiangolo.com/>
- Pydantic: <https://docs.pydantic.dev/>
- FAISS: Douze et al., "The Faiss library", 2024, <https://arxiv.org/abs/2401.08281>
- Locust: <https://docs.locust.io/>
- Cloud Run: <https://cloud.google.com/run/docs>
- Yambda: Ploshkin et al., "Yambda-5B: A Large-Scale Multi-modal Dataset for Ranking And Retrieval", 2025, <https://arxiv.org/abs/2505.22238>
