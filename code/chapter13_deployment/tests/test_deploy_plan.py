"""The Cloud Run script's dry run: no cloud calls, the agreed footprint, and a delete for everything it creates."""

import importlib.util
import json
import subprocess
import sys

import pytest

from common.artifacts import CHAPTER_DIR

PROJECT = "gen-lang-client-0843377880"
SERVICES = ["ch13-retrieval", "ch13-light-ranker", "ch13-heavy-ranker", "ch13-reranker", "ch13-orchestrator"]
STAGE_SERVICES = SERVICES[:4]
ACCOUNTS = {name: f"{name}@{PROJECT}.iam.gserviceaccount.com" for name in SERVICES}


@pytest.fixture(scope="module")
def plan(bundle_dir):
    result = subprocess.run([sys.executable, str(CHAPTER_DIR / "deploy" / "cloud_run" / "deploy.py"),
                             "--bundle-dir", str(bundle_dir)],
                            cwd=CHAPTER_DIR, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    commands = [line[len("[dry-run] "):] for line in lines if line.startswith("[dry-run] ")]
    return {"lines": lines, "gcloud": [c for c in commands if c.startswith("gcloud ")]}


def test_dry_run_is_the_default_and_executes_nothing(plan):
    assert plan["lines"][0].startswith("DRY RUN")
    assert not any(line.startswith("$ ") for line in plan["lines"])


def test_every_project_scoped_command_names_the_project(plan):
    scoped = [c for c in plan["gcloud"] if not c.startswith(("gcloud auth ", "gcloud config "))]
    assert scoped
    assert all(c.endswith(f"--project={PROJECT}") for c in scoped)


def test_services_scale_to_zero_on_cpu_and_require_authentication(plan):
    deploys = [c for c in plan["gcloud"] if c.startswith("gcloud run deploy ")]
    assert sorted(c.split()[3] for c in deploys) == sorted(SERVICES)
    for command in deploys:
        assert "--min-instances=0" in command
        assert "--cpu=1" in command
        assert "--no-allow-unauthenticated" in command
        assert "--gpu" not in command


def test_no_always_on_or_managed_services(plan):
    text = "\n".join(plan["gcloud"])
    for forbidden in ("container clusters", "redis instances", "memorystore", "ai indexes", "vector-search",
                      "compute instances", "--gpu"):
        assert forbidden not in text


def test_everything_created_is_deleted(plan):
    gcloud = plan["gcloud"]
    for name in SERVICES:
        assert any(c.startswith(f"gcloud run services delete {name} ") for c in gcloud)
    assert any(c.startswith("gcloud artifacts repositories delete ch13-deploy ") for c in gcloud)
    for email in ACCOUNTS.values():
        assert any(c.startswith(f"gcloud iam service-accounts delete {email} ") for c in gcloud)
    for name in STAGE_SERVICES:
        assert any(c.startswith(f"gcloud run services remove-iam-policy-binding {name} ")
                   and f"--member=serviceAccount:{ACCOUNTS['ch13-orchestrator']}" in c for c in gcloud)
    assert any(c.startswith(f"gcloud storage rm --recursive gs://{PROJECT}_cloudbuild/ch13-source/") for c in gcloud)
    first_delete = next(i for i, c in enumerate(gcloud) if " delete " in c or c.startswith("gcloud storage rm"))
    assert all(" create " not in c and not c.startswith("gcloud run deploy ") for c in gcloud[first_delete:])


def test_each_service_runs_under_its_own_account(plan):
    created = [c.split()[4] for c in plan["gcloud"] if c.startswith("gcloud iam service-accounts create ")]
    assert sorted(created) == sorted(SERVICES)
    for command in (c for c in plan["gcloud"] if c.startswith("gcloud run deploy ")):
        name = command.split()[3]
        assert f"--service-account={ACCOUNTS[name]}" in command


def test_only_the_orchestrator_may_call_the_stage_services(plan):
    grants = [c for c in plan["gcloud"] if "add-iam-policy-binding" in c]
    assert sorted(c.split()[4] for c in grants) == sorted(STAGE_SERVICES)
    for command in grants:
        assert command.startswith("gcloud run services add-iam-policy-binding ")
        assert f"--member=serviceAccount:{ACCOUNTS['ch13-orchestrator']}" in command
        assert "--role=roles/run.invoker" in command


def _load_deploy():
    spec = importlib.util.spec_from_file_location("ch13_deploy", CHAPTER_DIR / "deploy" / "cloud_run" / "deploy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _AlreadyTornDown:
    """gcloud's answers, as recorded on 2026-11-09, on a project where every resource is already gone."""

    project = PROJECT
    errors = {
        "run": "ERROR: (gcloud.run.services.delete) Service [ch13-orchestrator] could not be found.",
        "artifacts": "ERROR: (gcloud.artifacts.repositories.delete) NOT_FOUND: Requested entity was not found.",
        "iam": "PERMISSION_DENIED: Permission 'iam.serviceAccounts.delete' denied on resource (or it may not exist).",
        "storage": f"ERROR: (gcloud.storage.rm) gs://{PROJECT}_cloudbuild not found: 404.",
    }

    def __init__(self, deploy, accounts, bindings=()):
        self.deploy, self.accounts, self.bindings = deploy, accounts, bindings
        self.calls = []

    def note(self, text):
        pass

    def gcloud_cmd(self, args, *, placeholder="", check=True, secret=False, project=True):
        self.calls.append(args)
        if args[:3] == ["iam", "service-accounts", "list"]:
            return json.dumps([{"email": email} for email in self.accounts])
        if args[:3] == ["run", "services", "list"]:
            return "[]"
        if args[:2] == ["projects", "get-iam-policy"]:
            return json.dumps({"bindings": [{"role": role, "members": [member]} for role, member in self.bindings]})
        raise self.deploy.DeployError(self.errors[args[0]])


def test_teardown_of_an_already_clean_project_reports_everything_deleted():
    deploy = _load_deploy()
    runner = _AlreadyTornDown(deploy, accounts=["someone-else@example.iam.gserviceaccount.com"])
    results = deploy.teardown(runner, "us-central1", "")
    assert all(r["deleted"] for r in results), results
    deleted_accounts = [args[3] for args in runner.calls if args[:3] == ["iam", "service-accounts", "delete"]]
    assert sorted(deleted_accounts) == sorted(ACCOUNTS.values())
    removed = [args[3] for args in runner.calls if args[:3] == ["run", "services", "remove-iam-policy-binding"]]
    assert sorted(removed) == sorted(STAGE_SERVICES)


def test_teardown_does_not_count_a_service_account_that_still_exists_as_deleted():
    deploy = _load_deploy()
    left = ACCOUNTS["ch13-reranker"]
    results = deploy.teardown(_AlreadyTornDown(deploy, accounts=[left]), "us-central1", "")
    assert [r["resource"] for r in results if not r["deleted"]] == [f"service account {left}",
                                                                     "remaining ch13 service accounts"]


def test_teardown_reports_a_project_binding_left_for_a_ch13_account():
    deploy = _load_deploy()
    member = f"deleted:serviceAccount:{ACCOUNTS['ch13-orchestrator']}?uid=123"
    results = deploy.teardown(_AlreadyTornDown(deploy, accounts=[], bindings=[("roles/viewer", member)]),
                              "us-central1", "")
    [left] = [r for r in results if not r["deleted"]]
    assert left["resource"] == "remaining project IAM bindings for ch13 accounts"
    assert left["detail"] == [f"roles/viewer {member}"]
