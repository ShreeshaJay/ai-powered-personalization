# Chapter 13: Entry Points, Hardware, Runtimes and Outputs

This file lists every entry point with the exact command, the hardware it needs, the expected runtime, the input data and how it arrives, and the output files to bring back. Measured times come from one laptop: AMD Ryzen 9 8945HS (8 cores, 16 logical processors), 15.3 GB RAM, Windows, Python 3.11.9, CPU only, shared with other work while the measurements ran. "Not measured" marks steps that have never run. `DESIGN_NOTES.md` holds the full measured results.

## Summary

| # | Entry point | What it demonstrates | Where to run it | Runtime |
|---|---|---|---|---|
| 1 | Synthetic bundle build | The artifact build without any download | Any laptop | 25.7 s to 101.0 s measured |
| 2 | Sample bundle build | The artifact build on real Yambda data | Laptop | 210.2 s to 358.6 s measured |
| 3 | Full bundle build | The path to full-size artifacts | Large CPU machine | Not measured |
| 4 | `pytest` | Contracts, integration, every ladder rung, breakers and retry budget, feature staleness, policy pinning, artifact checks, deploy plan | Laptop | 61.5 to 127.9 s measured |
| 5 | `scripts/run_local.py --smoke --locust` | Five services under a deadline, smoke test, closed-loop load ramp | Laptop | about 131 s measured |
| 6 | `deploy/cloud_run/deploy.py` (dry run) | Every gcloud command the deployment would run | Any machine | 0.4 s measured |
| 7 | `deploy/cloud_run/deploy.py --execute` | Cloud Run deployment, smoke test, Locust check, teardown | Operator machine, Cloud Build, Cloud Run | 20.3 to 22.9 min measured |
| 8 | `deploy/cloud_run/deploy.py --execute --teardown-only` | Deletes what a failed run left behind | Operator machine | 44.7 and 48.8 s measured |

## Setup for Every Entry Point

```bash
python -m venv ~/venvs/ch13               # outside any synced folder
source ~/venvs/ch13/bin/activate          # Windows: C:\venvs\ch13\Scripts\activate
cd chapter13_deployment
pip install -r requirements.txt           # CPU-only PyTorch wheels
```

## Getting Yambda (Entry Points 2 and 3)

The build reads six files from [yandex/yambda on Hugging Face](https://huggingface.co/datasets/yandex/yambda). `huggingface_hub` is in `requirements.txt`:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='yandex/yambda', repo_type='dataset', local_dir='yambda', allow_patterns=['flat/50m/listens.parquet', 'flat/50m/likes.parquet', 'flat/50m/dislikes.parquet', 'embeddings.parquet', 'artist_item_mapping.parquet', 'album_item_mapping.parquet'])"
```

| File | Size (Hugging Face listing) |
|---|---|
| `flat/50m/listens.parquet` | 369 MB |
| `flat/50m/likes.parquet` | 7.2 MB |
| `flat/50m/dislikes.parquet` | 1.0 MB |
| `embeddings.parquet` | 13.8 GB |
| `artist_item_mapping.parquet` | 38.3 MB |
| `album_item_mapping.parquet` | 54.5 MB |

That is about 14.3 GB in total; download time was not measured. A download of `flat/50m/dislikes.parquet` with this command was checked. The loader also accepts a copy that keeps the three interaction files directly under `flat/`.

## 1. Synthetic Bundle Build

- **Command:** `python -m pipeline_build.build_artifacts --profile synthetic`
- **Hardware:** any laptop CPU. Measured peak resident memory: 631 MB. Uses up to four threads (`BUILD_THREADS` overrides).
- **Runtime:** 30.1 s for the current bundle (`synthetic-20261109-202956`); earlier builds took 25.7 s, 32.8 s and 101.0 s on the same machine under different background load.
- **Input:** none; the data is generated with seed 42 (300 users, 2,000 items).
- **Outputs:** `artifacts/synthetic-<UTC timestamp>/` (about 5 MB) and `artifacts/ACTIVE` (skip with `--no-activate`). Bring back `bundle_manifest.json` and the console log.

## 2. Sample Bundle Build

- **Command:** `python -m pipeline_build.build_artifacts --profile sample --data-dir yambda`
- **Hardware:** a laptop CPU with 16 GB RAM ran it. Measured peak build memory: 1,838 MB (`build_host.peak_memory_mb`, bundle `sample-20261109-202000`); other builds measured 1,585 MB to 2,120 MB.
- **Runtime:** 266.6 s for the current bundle and 261.5 s for a same-seed rebuild; earlier builds took 210.2 s, 231.0 s, 254.2 s, 272.7 s and 358.6 s, all with four threads. MMoE training takes the largest share (119.7 s to 155.1 s), then retrieval (51.4 s to 70.9 s).
- **Input:** the Yambda download above. The build samples 400 users with seed 42, keeps their 20,000 most-played tracks that have an audio embedding, and reads `embeddings.parquet` one row group at a time, keeping only those tracks.
- **Outputs:** `artifacts/sample-<UTC timestamp>/` (34 MB) and `artifacts/ACTIVE`. Bring back `bundle_manifest.json` (versions, metrics, timings, data summary, hashes) and the console log.

## 3. Full Bundle Build (Path to Full-Size Artifacts)

This profile has not run. It uses every Yambda user and every played track that has an audio embedding, with the chapters' epoch counts: XGBoost with 100 trees, MMoE for 10 epochs, and the two-tower model for 10 epochs (2 of them warm-up) with no cap on training examples per user.

| Quantity | Sample bundle | Full Yambda 50M |
|---|---|---|
| Users | 397 | 9,238 |
| Listens | 1,428,936 | 46,467,212 |
| Played tracks | 20,000 kept | 877,168 (before the embedding filter) |
| Likes / dislikes | 22,247 / 1,462 | 881,456 / 107,776 |
| MMoE epochs | 3 | 10 |
| Two-tower examples per user | at most 300 | all |

Run it in two stages, on Linux, so the second stage can be sized from a measurement:

```bash
# Stage 1: 1,000 of the 9,238 users, every track they played
BUILD_THREADS=8 /usr/bin/time -v -o stage1_time.txt python -m pipeline_build.build_artifacts \
    --profile full --users 1000 --data-dir yambda --no-activate 2>&1 | tee stage1_build.log

# Stage 2: everything
BUILD_THREADS=16 /usr/bin/time -v -o full_time.txt python -m pipeline_build.build_artifacts \
    --profile full --data-dir yambda --no-activate 2>&1 | tee full_build.log
```

- **Hardware:** CPU only. The build trains on CPU, so a GPU does not help. For stage 1, 8 vCPUs and 32 GB RAM (for example a Compute Engine `n2-standard-8`). For stage 2, take the stage-1 `peak_memory_mb`, scale it by the ratio of listens (about 9 times for 1,000 users) and add headroom; expect a high-memory machine such as `n2-highmem-16` (16 vCPUs, 128 GB). Allow 50 GB of disk for the data, the environment and the bundle.
- **Runtime:** not measured. A linear extrapolation of the sample's MMoE time by rows (about 33 times) and epochs (10 against 3) gives roughly 17,000 s at four threads. Treat that only as an order of magnitude until stage 1 replaces it. The two-tower step builds one Python dictionary per training example and has no per-user cap in this profile, so its time and memory depend on the example count, which stage 1 reports as `metrics.retrieval.train_examples`.
- **Input:** the Yambda download, copied to the machine's local disk.
- **Outputs:** `artifacts/full-<UTC timestamp>/`. Its size has not been measured. The item content embeddings alone are about 0.45 GB (877,168 tracks by 128 floats), stored once for retrieval and once for the re-ranker, and the index adds about 0.18 GB (50 floats per track). Bring back `bundle_manifest.json`, the build logs (`stage1_build.log`, `full_build.log`) and the `/usr/bin/time` reports (`stage1_time.txt`, `full_time.txt`); bring back the bundle itself only if it is to be served. Check the Cloud Run memory settings before deploying a full bundle.

## 4. Test Suite

- **Command:** `pytest`, or `CH13_TEST_BUNDLE=artifacts/<bundle_version> pytest` to reuse a bundle. Add `--junitxml=outputs/pytest.xml` for a file to bring back.
- **Hardware:** a laptop CPU; measured peak working set 0.7 to 0.8 GB.
- **Runtime:** 71.6 s measured for the default run of the current 135 tests (69.0 s for 134, 70.2 s for 131, 69.5 s for the 123 of round 2), which first builds a `tiny` bundle in a temporary folder (up to 127.9 s on a busier machine with an earlier, smaller suite); 67.0 s with the sample bundle.
- **Input:** none by default; otherwise the bundle named in `CH13_TEST_BUNDLE`, which must be built with the current code (bundles without `feature_store/feature_contract.json` no longer load).
- **Outputs:** the console summary (135 tests) and, if requested, `outputs/pytest.xml`.

## 5. Local Stack, Smoke Test and Locust Ramp

- **Command:** `python scripts/run_local.py --smoke --locust`. Add `--base-port 8100` when ports 8000 to 8004 are taken; the runner refuses to start on a taken port.
- **Hardware:** a laptop CPU. Each service runs one process with one compute thread; measured peak resident memory per service was 87 MB (orchestrator) to 357 MB (heavy ranker), about 1.24 GB for all five.
- **Runtime:** about 131 s measured: 7 to 11 s until all five report ready, 3 s for the smoke test, 100 s for the default ramp (`RAMP=20:2,40:5,60:10,80:20,100:30`), then shutdown. With `--ramp 20:2,40:5,60:10` the whole run took about 94 s.
- **Input:** the active bundle (`artifacts/ACTIVE`), or `CH13_BUNDLE_DIR`.
- **Outputs:** `outputs/local_run/<UTC timestamp>/` with `run_local.json`, `smoke.json`, `locust_summary.json`, `locust_stats.csv`, `locust_stats_history.csv`, `locust_failures.csv`, `locust_exceptions.csv` and one log per service. Bring back `run_local.json`, `smoke.json`, `locust_summary.json`, `locust_stats_history.csv` and `orchestrator.log`.

## 6 to 8. Cloud Run Deployment

### Commands

```bash
# 6. Dry run (the default): prints every command, makes no cloud call
python deploy/cloud_run/deploy.py --bundle-dir artifacts/<bundle_version>

# 7. Build, deploy, test, record and tear down
python deploy/cloud_run/deploy.py --execute --bundle-dir artifacts/<bundle_version>

# 8. Delete anything an interrupted run left behind
python deploy/cloud_run/deploy.py --execute --teardown-only
```

Pass `--bundle-dir` explicitly; without it the script uses whatever `artifacts/ACTIVE` names. The bundle that tested cleanly here is `sample-20261109-192754`; bundles built before it no longer load. The active bundle was rebuilt afterwards as `sample-20261109-202000`, only to make the order of the training rows repeatable; the serving code is unchanged, and that bundle has passed the local smoke test but has not been deployed to Cloud Run. Other options: `--project` (default `gen-lang-client-0843377880`), `--region` (default `us-central1`), `--builder local-docker` (build and push from this machine instead of Cloud Build), `--max-instances` (default 2), `--ramp` (default `30:2,60:5,90:10`), `--ready-timeout-s` (default 600) and `--skip-apis`.

### Prerequisites

- The Google Cloud CLI on `PATH`, signed in with `gcloud auth login` as an account that can enable APIs and create Artifact Registry repositories, service accounts, Cloud Build builds and Cloud Run services in the project.
- The environment from "Setup for Every Entry Point" (the smoke test and Locust run on the operator machine).
- Docker only with `--builder local-docker`.

### What `--execute` Creates and Deletes

| Resource | Created | Deleted at the end (also after an error or Ctrl+C) |
|---|---|---|
| Artifact Registry repository `ch13-deploy` with five CPU images | yes | yes |
| One service account per service, named after it (`ch13-orchestrator`, `ch13-retrieval`, `ch13-light-ranker`, `ch13-heavy-ranker`, `ch13-reranker`); only `ch13-orchestrator` is granted `roles/run.invoker`, on the four stage services, and no account holds a project role | yes | yes: the grants, then the accounts; the script then lists any `ch13` account and any project IAM binding naming one that remains |
| Cloud Run services `ch13-retrieval`, `ch13-light-ranker`, `ch13-heavy-ranker`, `ch13-reranker`, `ch13-orchestrator` | yes | yes, then the script lists any service labelled `app=ch13` that remains |
| Build source under `gs://<project>_cloudbuild/ch13-source/<run_id>` | yes | yes |
| Bucket `gs://<project>_cloudbuild` | created by Cloud Build if absent | yes if this run created it (the delete refuses a bucket that still holds objects); otherwise it stays |
| APIs: Cloud Run, Artifact Registry, Cloud Build, IAM, and the APIs Google enables with them (Container Registry, IAM Credentials, Logging, Pub/Sub, Cloud Storage components, Telemetry) | enabled | stay enabled |
| Google-managed service agents for those APIs, with their project roles | created by Google when the APIs are enabled | stay |
| Build records and build logs (Cloud Build's Google-managed log bucket) | written by Cloud Build | stay |
| Service logs in Cloud Logging | written by the services | stay, under the project's retention settings |

Nothing else is used: no GKE, no GPU, no managed vector index and no Memorystore. The FAISS index and the SQLite feature store ship inside the images.

### Hardware and Settings

Every service runs with `--cpu=1`, `--min-instances=0`, `--max-instances=2`, `--timeout=60`, `--cpu-boost` and `--no-allow-unauthenticated`. Images are built on Cloud Build's default worker, five builds in parallel.

| Cloud Run service | Memory setting | Concurrency | Measured locally: peak memory | Measured locally: CPU per request |
|---|---|---|---|---|
| `ch13-retrieval` | 2 GiB | 8 | 300 MB | 7.6 ms |
| `ch13-light-ranker` | 1 GiB | 8 | 284 MB | 29.6 ms |
| `ch13-heavy-ranker` | 2 GiB | 8 | 357 MB | 21.9 ms |
| `ch13-reranker` | 1 GiB | 8 | 209 MB | 8.3 ms |
| `ch13-orchestrator` | 1 GiB | 20 | 87 MB | 38.0 ms |

The memory settings leave room above the sample bundle's measured peaks; smaller settings are possible but untested.

### Runtime

Measured on 2026-11-09 with the sample bundle in `us-central1` (three runs, the third with one service account per service; details in the Cloud Run Validation Log at the end of `DESIGN_NOTES.md`):

| Step | Measured |
|---|---|
| Entry point 6, dry run | 0.4 s |
| Entry point 7, whole run | 20.3, 21.8 and 22.9 min |
| of which `gcloud builds submit` (upload and the five parallel image builds and pushes) | 9.8 to 11.8 min |
| of which deploying the five services and granting the invoker role | 5.6 to 6.4 min |
| of which readiness, the wait until the orchestrator serves rung 1, the smoke test and the 90 s Locust ramp | about 2 to 3 min |
| of which teardown | 74 to 116 s (116 s with five accounts and the grant removals) |
| Cold start of one instance (Cloud Run instance start to startup probe passed) | 5.0 to 13.8 s for the orchestrator; 9.1 to 16.5 s for the lightweight ranker; 15.9 to 28.5 s for the heavy ranker, the slowest |
| Entry point 8 on a project with nothing left to delete | 44.7 and 48.8 s |

The script's fixed waits are 15 s for the new service accounts to propagate and up to 600 s each for readiness and for the first rung-1 response. A new `roles/run.invoker` grant took up to about a minute to take effect (in the third run the re-ranker refused calls for 48 s); until it does, the orchestrator's calls to that stage are refused and requests degrade, which is why the script waits for rung 1 before the smoke test.

### Input

The chapter's `contracts/`, `common/`, `vendor/`, `services/` and `config/` folders (under 1 MB) and the chosen bundle (34 MB for the sample bundle) are copied to a temporary staging folder on the operator machine and uploaded by `gcloud builds submit`. The smoke test and Locust read the bundle's `sample_users.json` locally. The tests authenticate with `gcloud auth print-identity-token`; the token is never printed or recorded.

### Outputs to Bring Back

`outputs/cloud_run/<run_id>/`:

- `run_record.json`: every command with its exit code and duration, service URLs, revisions and image references, time to readiness per service, the wait until the first rung-1 response, whether the run created the Cloud Build bucket, the smoke and Locust results, and the teardown result for each resource
- `summary.md`, also appended to the "Cloud Run Validation Log" at the end of `DESIGN_NOTES.md`
- `smoke.json`, `locust_summary.json` and the Locust CSV files

The script exits with 0 only when the smoke test passed and the teardown completed. If teardown is reported incomplete, run entry point 8 and check the console.

### Billable Items

No prices are given here. Each measured run used 9 to 11 Cloud Build minutes on the default worker, and its five services existed for 8 to 10 minutes. Cloud Build bills build time for the five images; Artifact Registry bills storage while the images exist; Cloud Run bills CPU and memory while instances run (zero minimum instances, at most two per service); Cloud Storage bills the uploaded source until it is deleted; Cloud Logging bills ingestion of the services' logs.
