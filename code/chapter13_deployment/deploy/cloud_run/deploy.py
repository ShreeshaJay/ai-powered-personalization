"""
Deploy the five Chapter 13 services to Cloud Run, test them, record the
result, and delete everything that was created.

DRY RUN IS THE DEFAULT. Without --execute the script makes no cloud calls; it
prints every gcloud command it would run, in order, with placeholders for
values only known at run time (service URLs).

    python deploy/cloud_run/deploy.py                             # print the plan
    python deploy/cloud_run/deploy.py --execute                   # build, deploy, test, tear down
    python deploy/cloud_run/deploy.py --execute --teardown-only   # delete what a failed run left

What --execute creates, and deletes again at the end (also after an error
or Ctrl+C):
  * Artifact Registry Docker repository ch13-deploy with five CPU images
  * one service account per service, named after it (Section 13.9.4); only the
    orchestrator's account is granted roles/run.invoker, on the four stage
    services, and no account holds any other role
  * Cloud Run services ch13-retrieval, ch13-light-ranker, ch13-heavy-ranker,
    ch13-reranker, ch13-orchestrator: --no-allow-unauthenticated,
    --min-instances 0, --max-instances 2, 1 vCPU each
  * the uploaded build source under gs://<project>_cloudbuild/ch13-source/<run_id>,
    and the bucket itself if it did not exist before the run
The APIs it enables (Cloud Run, Artifact Registry, Cloud Build, IAM) stay
enabled. Not used: GKE, GPUs, a managed vector index, Memorystore. The FAISS
index and the SQLite feature store ship inside the images.

After deployment it runs scripts/smoke_test.py and a short closed-loop Locust
ramp against the orchestrator, using your gcloud identity token, writes
outputs/cloud_run/<run_id>/run_record.json and summary.md, and appends the
summary to the Cloud Run validation log at the end of DESIGN_NOTES.md.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

CHAPTER_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHAPTER_DIR))

from common.artifacts import resolve_bundle_dir  # noqa: E402

DEFAULT_PROJECT = "gen-lang-client-0843377880"
DEFAULT_REGION = "us-central1"
REPOSITORY = "ch13-deploy"
APIS = ["run.googleapis.com", "artifactregistry.googleapis.com", "cloudbuild.googleapis.com", "iam.googleapis.com"]
# (Cloud Run service, image, memory, concurrency, local name for the smoke test)
STAGES = [
    ("ch13-retrieval", "retrieval", "2Gi", 8, "retrieval"),
    ("ch13-light-ranker", "lightweight-ranker", "1Gi", 8, "lightweight_ranker"),
    ("ch13-heavy-ranker", "heavy-ranker", "2Gi", 8, "heavy_ranker"),
    ("ch13-reranker", "reranker", "1Gi", 8, "reranker"),
]
ORCHESTRATOR = ("ch13-orchestrator", "orchestrator", "1Gi", 20)
SERVICE_NAMES = [ORCHESTRATOR[0], *(stage[0] for stage in STAGES)]
ORCHESTRATOR_URL_ENV = {"ch13-retrieval": "RETRIEVAL_URL", "ch13-light-ranker": "LIGHTWEIGHT_RANKER_URL",
                        "ch13-heavy-ranker": "HEAVY_RANKER_URL", "ch13-reranker": "RERANKER_URL"}
STAGING_ITEMS = ["contracts", "common", "vendor", "services", "config"]
DESIGN_NOTES = CHAPTER_DIR / "DESIGN_NOTES.md"


class DeployError(RuntimeError):
    pass


class Runner:
    """Prints every command; runs it only with --execute. Secrets are never printed or recorded."""

    def __init__(self, execute: bool, project: str):
        self.execute = execute
        self.project = project
        self.log = []
        self.gcloud = shutil.which("gcloud") if execute else "gcloud"
        if execute and not self.gcloud:
            raise DeployError("gcloud is not on PATH. Install the Google Cloud CLI and run `gcloud auth login`.")

    def gcloud_cmd(self, args, *, placeholder="", check=True, secret=False, project=True) -> str:
        args = list(args) + ([f"--project={self.project}"] if project else [])
        shown = "gcloud " + " ".join(shlex.quote(a) for a in args)
        return self._run([self.gcloud, *args], shown, placeholder, check, secret)

    def local_cmd(self, args, *, shown=None, env=None, check=True) -> str:
        shown = shown or " ".join(shlex.quote(str(a)) for a in args)
        return self._run([str(a) for a in args], shown, "", check, False, env=env, stream=True)

    def note(self, text: str) -> None:
        print(f"# {text}")
        self.log.append({"note": text})

    def _run(self, argv, shown, placeholder, check, secret, env=None, stream=False) -> str:
        print(("$ " if self.execute else "[dry-run] ") + shown)
        entry = {"command": shown, "started_at_utc": _now()}
        self.log.append(entry)
        if not self.execute:
            return placeholder
        started = time.time()
        if stream:
            result = subprocess.run(argv, cwd=CHAPTER_DIR, env=env)
            stdout, stderr = "", ""
        else:
            result = subprocess.run(argv, cwd=CHAPTER_DIR, env=env, capture_output=True, text=True)
            stdout, stderr = result.stdout.strip(), result.stderr.strip()
        entry.update({"exit_code": result.returncode, "seconds": round(time.time() - started, 1)})
        if not secret:
            entry["stderr_tail"] = stderr[-1500:]
        if check and result.returncode != 0:
            raise DeployError(f"Command failed ({result.returncode}): {shown}\n{stderr[-1500:]}")
        return stdout


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def account_email(service: str, project: str) -> str:
    """The runtime identity of one Cloud Run service; the account ID is the service name."""
    return f"{service}@{project}.iam.gserviceaccount.com"


def build_bucket(project: str) -> str:
    """Cloud Build's default staging bucket; `gcloud builds submit` creates it when it is missing."""
    return f"gs://{project}_cloudbuild"


def stage_build_context(bundle: Path, staging: Path) -> None:
    for item in STAGING_ITEMS:
        shutil.copytree(CHAPTER_DIR / item, staging / item,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "artifacts"))
    shutil.copytree(bundle, staging / "bundle")
    shutil.copy2(CHAPTER_DIR / ".dockerignore", staging / ".dockerignore")
    (staging / ".gcloudignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    shutil.copy2(Path(__file__).with_name("cloudbuild.yaml"), staging / "cloudbuild.yaml")


def wait_ready(urls: dict, token: str, timeout_s: float) -> dict:
    import httpx

    started, pending, ready = time.time(), dict(urls), {}
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(timeout=30.0, headers=headers) as client:
        while pending and time.time() - started < timeout_s:
            for name, url in list(pending.items()):
                try:
                    response = client.get(url + "/health/ready")
                    if response.status_code == 200:
                        ready[name] = {"seconds_to_ready": round(time.time() - started, 1),
                                       "model_version": response.json().get("model_version")}
                        del pending[name]
                except httpx.HTTPError:
                    pass
            time.sleep(3)
    if pending:
        raise DeployError(f"Services not ready after {timeout_s:.0f} s: {sorted(pending)}")
    return ready


def wait_full_pipeline(url: str, token: str, users_file: str, timeout_s: float) -> dict:
    """Poll the orchestrator until it serves a slate from the full pipeline (rung 1).

    A new roles/run.invoker grant can take minutes to reach Cloud Run. Until it does,
    the stage refuses the orchestrator's calls with 403 and the orchestrator serves a
    lower rung, which the smoke test would report as a failure.
    """
    import httpx

    user_id = json.loads(Path(users_file).read_text(encoding="utf-8"))["warm"][0]
    started, polls, last = time.time(), 0, None
    with httpx.Client(timeout=30.0, headers={"Authorization": f"Bearer {token}"}) as client:
        while time.time() - started < timeout_s:
            polls += 1
            try:
                response = client.post(url + "/v1/recommend", json={"user_id": user_id})
                last = response.json().get("degradation") if response.status_code == 200 else response.status_code
                if isinstance(last, dict) and last.get("rung") == 1:
                    return {"seconds": round(time.time() - started, 1), "polls": polls}
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(5)
    raise DeployError(f"Full pipeline not reached after {timeout_s:.0f} s; last degradation: {last}")


def teardown(runner: Runner, region: str, run_id: str, created_bucket: bool = False) -> list:
    """Delete everything; keep going past failures and report each one."""
    results = []
    accounts = [account_email(name, runner.project) for name in SERVICE_NAMES]
    caller = accounts[0]

    def attempt(label, args, still_exists=None):
        try:
            runner.gcloud_cmd(args)
            results.append({"resource": label, "deleted": True})
        except DeployError as exc:
            message = str(exc)
            lowered = message.lower()
            gone = any(text in lowered for text in ("not_found", "not found", "could not be found", "does not exist",
                                                    "matched no objects"))
            if not gone and still_exists is not None:
                gone = not still_exists()
            results.append({"resource": label, "deleted": gone, "detail": message[-400:]})

    def listed_accounts():
        listed = runner.gcloud_cmd(["iam", "service-accounts", "list", "--format=json"], placeholder="[]", check=False)
        try:
            return {account.get("email") for account in json.loads(listed or "[]")}
        except json.JSONDecodeError:
            return None

    def account_exists(email):
        # Deleting a missing service account fails with PERMISSION_DENIED, not NOT_FOUND.
        listed = listed_accounts()
        return True if listed is None else email in listed

    runner.note("Teardown: delete every resource this script creates")
    for name, *_ in STAGES:
        attempt(f"invoker binding for {caller} on {name}",
                ["run", "services", "remove-iam-policy-binding", name, f"--region={region}",
                 f"--member=serviceAccount:{caller}", "--role=roles/run.invoker"])
    for name in SERVICE_NAMES:
        attempt(f"run service {name}", ["run", "services", "delete", name, f"--region={region}", "--quiet"])
    attempt(f"artifact registry {REPOSITORY}",
            ["artifacts", "repositories", "delete", REPOSITORY, f"--location={region}", "--quiet"])
    for email in accounts:
        attempt(f"service account {email}", ["iam", "service-accounts", "delete", email, "--quiet"],
                lambda email=email: account_exists(email))
    staging = f"{build_bucket(runner.project)}/ch13-source/{run_id}" if run_id else \
        f"{build_bucket(runner.project)}/ch13-source"
    attempt(f"build source {staging}", ["storage", "rm", "--recursive", staging])
    if created_bucket:
        # `buckets delete` refuses a bucket that still holds objects, so nothing another run wrote is lost.
        attempt(f"build bucket {build_bucket(runner.project)} (created by this run)",
                ["storage", "buckets", "delete", build_bucket(runner.project)])

    runner.note("Verify nothing is left")
    leftover = runner.gcloud_cmd(["run", "services", "list", f"--region={region}", "--filter=metadata.labels.app=ch13",
                                  "--format=json"], placeholder="[]", check=False)
    try:
        remaining = [s["metadata"]["name"] for s in json.loads(leftover or "[]")]
    except (json.JSONDecodeError, KeyError, TypeError):
        remaining = ["<could not parse service list>"]
    results.append({"resource": "remaining ch13 Cloud Run services", "deleted": not remaining, "detail": remaining})
    listed = listed_accounts()
    left_accounts = ["<could not parse service account list>"] if listed is None else \
        sorted(email for email in listed if email in accounts)
    results.append({"resource": "remaining ch13 service accounts", "deleted": not left_accounts,
                    "detail": left_accounts})
    policy = runner.gcloud_cmd(["projects", "get-iam-policy", runner.project, "--format=json"],
                               placeholder="{}", check=False)
    try:
        left_bindings = [f"{binding['role']} {member}" for binding in json.loads(policy or "{}").get("bindings", [])
                         for member in binding.get("members", []) if any(email in member for email in accounts)]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        left_bindings = ["<could not parse the project IAM policy>"]
    results.append({"resource": "remaining project IAM bindings for ch13 accounts", "deleted": not left_bindings,
                    "detail": left_bindings})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="really run the commands (default: dry run)")
    parser.add_argument("--teardown-only", action="store_true", help="only delete resources left by earlier runs")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--bundle-dir", type=Path, default=None, help="default: artifacts/ACTIVE")
    parser.add_argument("--builder", choices=["cloud-build", "local-docker"], default="cloud-build",
                        help="where to build the images (local-docker needs Docker and pushes from this machine)")
    parser.add_argument("--max-instances", type=int, default=2)
    parser.add_argument("--ramp", default="30:2,60:5,90:10", help="Locust closed-loop ramp, end_second:users")
    parser.add_argument("--ready-timeout-s", type=float, default=600.0)
    parser.add_argument("--skip-apis", action="store_true", help="do not run `gcloud services enable`")
    args = parser.parse_args()

    run_id = "r" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    runner = Runner(args.execute, args.project)
    region, project = args.region, args.project
    caller = account_email(ORCHESTRATOR[0], project)
    registry = f"{region}-docker.pkg.dev/{project}/{REPOSITORY}"
    print(("EXECUTING" if args.execute else "DRY RUN (no cloud calls; add --execute to run)") +
          f": project {project}, region {region}, run {run_id}\n")

    if args.teardown_only:
        results = teardown(runner, region, "")
        print(json.dumps(results, indent=2))
        return 0 if all(r["deleted"] for r in results) else 1

    bundle = resolve_bundle_dir(args.bundle_dir)
    bundle_manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    out_dir = CHAPTER_DIR / "outputs" / "cloud_run" / run_id
    record = {"run_id": run_id, "project": project, "region": region, "started_at_utc": _now(),
              "bundle_version": bundle_manifest["bundle_version"], "bundle_profile": bundle_manifest["profile"],
              "builder": args.builder, "max_instances": args.max_instances, "dry_run": not args.execute,
              "services": {}, "smoke": None, "locust": None, "teardown": None, "error": None}
    labels = f"--labels=app=ch13,ch13-run={run_id}"
    staging_root = Path(tempfile.mkdtemp(prefix="ch13_cloud_run_")) if args.execute else Path("<temp dir>")
    staging = staging_root / "context"
    token = None
    try:
        runner.note(f"Bundle {bundle_manifest['bundle_version']} ({bundle_manifest['profile']}) from {bundle}")
        if args.execute and not runner.gcloud_cmd(["config", "get-value", "account"], project=False):
            raise DeployError("No gcloud account is configured. Run `gcloud auth login` first.")
        if not args.skip_apis:
            runner.gcloud_cmd(["services", "enable", *APIS])

        runner.note("Artifact Registry repository and one runtime service account per service")
        runner.gcloud_cmd(["artifacts", "repositories", "create", REPOSITORY, "--repository-format=docker",
                           f"--location={region}", "--description=Chapter 13 companion code (temporary)", labels])
        for name in SERVICE_NAMES:
            runner.gcloud_cmd(["iam", "service-accounts", "create", name,
                               f"--display-name=Chapter 13 {name} runtime (temporary)"])
        record["service_accounts"] = [account_email(name, project) for name in SERVICE_NAMES]
        if args.execute:
            time.sleep(15)  # let the new service accounts propagate before Cloud Run references them
        else:
            runner.note("wait 15 s for the service accounts to propagate")

        runner.note(f"Stage the build context: {', '.join(STAGING_ITEMS)} and the bundle as bundle/")
        if args.execute:
            stage_build_context(bundle, staging)
        tag = run_id
        if args.builder == "cloud-build":
            existing = runner.gcloud_cmd(["storage", "buckets", "describe", build_bucket(project), "--format=json"],
                                         placeholder="<bucket description, or empty if missing>", check=False)
            record["created_build_bucket"] = args.execute and not existing
            if not args.execute:
                runner.note(f"if {build_bucket(project)} was missing, the build creates it and teardown deletes it")
            runner.gcloud_cmd(["builds", "submit", str(staging), f"--config={staging / 'cloudbuild.yaml'}",
                               f"--substitutions=_REGION={region},_REPO={REPOSITORY},_TAG={tag}",
                               f"--gcs-source-staging-dir={build_bucket(project)}/ch13-source/{run_id}"])
        else:
            runner.gcloud_cmd(["auth", "configure-docker", f"{region}-docker.pkg.dev", "--quiet"], project=False)
            for dockerfile, image in [("retrieval", "retrieval"), ("lightweight_ranker", "lightweight-ranker"),
                                      ("heavy_ranker", "heavy-ranker"), ("reranker", "reranker"),
                                      ("orchestrator", "orchestrator")]:
                ref = f"{registry}/{image}:{tag}"
                runner.local_cmd(["docker", "build", "-f", staging / "services" / dockerfile / "Dockerfile",
                                  "--build-arg", "BUNDLE=bundle", "-t", ref, staging])
                runner.local_cmd(["docker", "push", ref])

        def deploy(name, image, memory, concurrency, env=None):
            command = ["run", "deploy", name, f"--image={registry}/{image}:{tag}", f"--region={region}",
                       "--no-allow-unauthenticated", "--min-instances=0", f"--max-instances={args.max_instances}",
                       "--cpu=1", f"--memory={memory}", f"--concurrency={concurrency}", "--timeout=60",
                       "--cpu-boost", f"--service-account={account_email(name, project)}", labels, "--quiet"]
            if env:
                env_file = staging_root / f"{name}.env.yaml"
                if args.execute:
                    env_file.write_text("".join(f"{k}: \"{v}\"\n" for k, v in env.items()), encoding="utf-8")
                else:
                    runner.note(f"{env_file.name}: " + ", ".join(f"{k}={v}" for k, v in env.items()))
                command.append(f"--env-vars-file={env_file}")
            runner.gcloud_cmd(command)
            described = runner.gcloud_cmd(["run", "services", "describe", name, f"--region={region}", "--format=json"],
                                          placeholder=json.dumps({"status": {"url": f"https://{name}-<hash>.run.app"}}))
            info = json.loads(described)
            status = info.get("status", {})
            record["services"][name] = {
                "url": status.get("url"), "revision": status.get("latestReadyRevisionName"),
                "image": ((info.get("spec", {}).get("template", {}).get("spec", {}).get("containers") or [{}])[0]
                          .get("image")),
            }
            return status["url"]

        runner.note("Deploy the four stage services")
        urls = {}
        for name, image, memory, concurrency, local_name in STAGES:
            urls[name] = deploy(name, image, memory, concurrency)
            runner.gcloud_cmd(["run", "services", "add-iam-policy-binding", name, f"--region={region}",
                               f"--member=serviceAccount:{caller}", "--role=roles/run.invoker"])

        runner.note("Deploy the orchestrator; it calls the stages with ID tokens from the metadata server")
        name, image, memory, concurrency = ORCHESTRATOR
        orchestrator_env = {ORCHESTRATOR_URL_ENV[n]: u for n, u in urls.items()}
        orchestrator_env["SERVICE_AUTH"] = "gcp_id_token"
        orchestrator_url = deploy(name, image, memory, concurrency, env=orchestrator_env)

        runner.note("Identity token for the tests (not printed)")
        token = runner.gcloud_cmd(["auth", "print-identity-token"], placeholder="<token>", secret=True, project=False)
        all_urls = {**{local: urls[n] for n, _, _, _, local in STAGES}, "orchestrator": orchestrator_url}
        users_file = str(bundle / "sample_users.json")
        if args.execute:
            record["ready"] = wait_ready(all_urls, token, args.ready_timeout_s)
            record["full_pipeline"] = wait_full_pipeline(orchestrator_url, token, users_file, args.ready_timeout_s)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            runner.note("poll /health/ready on all five services until 200")
            runner.note("poll /v1/recommend until a slate comes from the full pipeline (new invoker grants propagate)")

        test_env = dict(os.environ, AUTH_TOKEN=token or "", PYTHONIOENCODING="utf-8", USERS_FILE=users_file)
        smoke = [sys.executable, "scripts/smoke_test.py", "--url", orchestrator_url, "--timeout-s", "60",
                 "--users-file", users_file, "--out", str(out_dir / "smoke.json")]
        for name, _, _, _, local in STAGES:
            smoke += ["--service", f"{local}={urls[name]}"]
        record["smoke"] = {"exit_code": None}
        runner.local_cmd(smoke, env=test_env, check=False)
        if args.execute and (out_dir / "smoke.json").is_file():
            record["smoke"] = json.loads((out_dir / "smoke.json").read_text(encoding="utf-8"))

        test_env.update(RAMP=args.ramp, LOCUST_SUMMARY=str(out_dir / "locust_summary.json"))
        runner.local_cmd([sys.executable, "-m", "locust", "-f", "load/locustfile.py", "--headless",
                          "--host", orchestrator_url, "--csv", str(out_dir / "locust"), "--only-summary"],
                         shown=f"RAMP={args.ramp} AUTH_TOKEN=<token> python -m locust -f load/locustfile.py "
                               f"--headless --host {orchestrator_url} --csv {out_dir / 'locust'} --only-summary",
                         env=test_env, check=False)
        if args.execute and (out_dir / "locust_summary.json").is_file():
            record["locust"] = json.loads((out_dir / "locust_summary.json").read_text(encoding="utf-8"))
    except (Exception, KeyboardInterrupt) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"\nStopped: {record['error']}\n")
    finally:
        token = None
        record["teardown"] = teardown(runner, region, run_id, record.get("created_build_bucket", False))
        record["finished_at_utc"] = _now()
        record["commands"] = runner.log
        if args.execute:
            shutil.rmtree(staging_root, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            (out_dir / "summary.md").write_text(summary_markdown(record), encoding="utf-8")
            append_to_design_notes(summary_markdown(record))
            print(f"\nRecord: {out_dir / 'run_record.json'}")
            print(summary_markdown(record))
        else:
            runner.note(f"write outputs/cloud_run/{run_id}/run_record.json and summary.md, "
                        "and append the summary to DESIGN_NOTES.md")

    torn_down = all(r["deleted"] for r in record["teardown"])
    if not torn_down:
        print("WARNING: some resources may remain; run again with --execute --teardown-only and check the console.")
    smoke_ok = bool(record["smoke"] and record["smoke"].get("passed"))
    return 0 if (not args.execute or (smoke_ok and torn_down and not record["error"])) else 1


def append_to_design_notes(summary: str) -> None:
    """The Cloud Run validation log is the last section of DESIGN_NOTES.md."""
    if DESIGN_NOTES.is_file():
        with DESIGN_NOTES.open("a", encoding="utf-8") as handle:
            handle.write("\n" + summary)


def summary_markdown(record: dict) -> str:
    smoke = record.get("smoke") or {}
    locust = record.get("locust") or {}
    lines = [
        f"### Cloud Run run {record['run_id']}",
        f"- Project {record['project']}, region {record['region']}, bundle {record['bundle_version']} "
        f"({record['bundle_profile']}), builder {record['builder']}, max instances {record['max_instances']}",
        f"- Started {record['started_at_utc']}, finished {record.get('finished_at_utc')}",
        f"- Full pipeline (rung 1) first served {(record.get('full_pipeline') or {}).get('seconds')} s "
        f"after all five services were ready",
        f"- Smoke test: {'passed' if smoke.get('passed') else 'FAILED or not run'}; "
        f"rungs {smoke.get('rungs')}; client p50 {smoke.get('client_latency_ms', {}).get('p50')} ms",
        f"- Locust: {locust.get('requests')} requests, {locust.get('failures')} failures, "
        f"p50 {locust.get('p50_ms')} ms, p95 {locust.get('p95_ms')} ms, rungs {locust.get('rungs')}",
        f"- Teardown: {'complete' if all(r['deleted'] for r in record.get('teardown') or []) else 'INCOMPLETE'}",
    ]
    if record.get("error"):
        lines.append(f"- Error: {record['error'][:300]}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
