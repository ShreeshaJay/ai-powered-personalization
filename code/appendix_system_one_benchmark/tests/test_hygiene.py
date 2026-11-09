"""Release hygiene for this folder: no machine-specific paths, no private folder names,
no secrets file, no large files.

Files are those git would track here (`git ls-files -co --exclude-standard`); outside
a git checkout, a directory walk that skips local-only folders is used instead.
Text files and the decompressed `.jsonl.gz` pack files are scanned. The Colab
runtime path `/root/` in the notebook is allowed.
"""

from __future__ import annotations

import gzip
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 50 * 1024 * 1024
TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".ipynb", ".txt", ".csv", ".tsv", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".sh", ".ps1"}
TEXT_NAMES = {".gitignore", ".gitattributes", ".env.example"}
LOCAL_ONLY_DIRS = {".git", ".venvs", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "references", "data"}
SEPARATOR = r"(?:\\\\|\\|/)"
WINDOWS_PATH = re.compile(rf"(?<![A-Za-z0-9])[A-Za-z]:{SEPARATOR}[^\\/\s\"'<>|*?:]+{SEPARATOR}")
# The Colab mount points /content/drive and /content/drive/MyDrive are allowed; a folder inside MyDrive is not.
COLAB_MY_DRIVE = "/content" + "/drive/MyDrive/"
POSIX_HOME = re.compile(
    r"(?:(?<![A-Za-z0-9_.~-])|(?<=/mnt/[a-z]))/(?:home|Users)/[A-Za-z0-9_.-]+/"
    + "|" + re.escape(COLAB_MY_DRIVE) + r"[^/\"'\s]+/"
)
# Private workspace folder names, split so this file does not match itself.
PRIVATE_NAMES = re.compile("|".join(["RecSys" + r"\s+book\s+code", "Appendix" + r"\s+Hybrid\s+Search", "esci_llm" + "_enrichment"]), re.IGNORECASE)


def tracked_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-co", "--exclude-standard", "-z", "--", "."],
            capture_output=True,
            check=True,
        )
        files = [ROOT / name for name in result.stdout.decode("utf-8").split("\0") if name]
        if files:
            return [path for path in files if path.is_file()]
    except (OSError, subprocess.CalledProcessError):
        pass
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not LOCAL_ONLY_DIRS.intersection(path.relative_to(ROOT).parts[:-1])
    ]


FILES = tracked_files()
SCANNED_FILES = [
    path
    for path in FILES
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES or path.name.endswith(".jsonl.gz")
]


def file_text(path: Path) -> str:
    data = gzip.decompress(path.read_bytes()) if path.name.endswith(".gz") else path.read_bytes()
    return data.decode("utf-8", errors="replace")


def machine_paths(text: str) -> list[str]:
    return [match.group(0) for pattern in (WINDOWS_PATH, POSIX_HOME, PRIVATE_NAMES) for match in pattern.finditer(text)]


def test_patterns_catch_machine_paths_and_private_names() -> None:
    drive = "C" + ":"
    assert machine_paths(drive + "\\Users\\someone\\file.txt")
    assert machine_paths(drive + "\\\\Users\\\\someone\\\\file.json")
    assert machine_paths(drive + "/work/project/file.py")
    assert machine_paths("/home/" + "someone/project")
    assert machine_paths("/Users/" + "someone/project")
    assert machine_paths("/mnt/c" + "/Users/" + "someone/project")
    assert machine_paths(COLAB_MY_DRIVE + "project/run.zip")
    assert not machine_paths(COLAB_MY_DRIVE + "bundle.zip and " + COLAB_MY_DRIVE.rstrip("/"))
    assert machine_paths("RecSys" + " book code/x")
    assert machine_paths("appendix" + " hybrid search")
    assert machine_paths("outputs/esci_llm" + "_enrichment/a.csv")
    assert not machine_paths("https://github.com/amazon-science/esci-data and /root/.local/bin")
    assert not machine_paths("../appendix_hybrid_search/README.md")
    assert not machine_paths('"Step x:\\n"')


def test_files_were_found() -> None:
    assert (ROOT / "README.md") in FILES
    assert any(path.name.endswith(".jsonl.gz") for path in SCANNED_FILES)


@pytest.mark.parametrize("path", SCANNED_FILES, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_no_machine_paths_or_private_names(path: Path) -> None:
    hits = machine_paths(file_text(path))
    assert not hits, hits[:5]


def test_no_env_file() -> None:
    names = [path.relative_to(ROOT).as_posix() for path in FILES if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example")]
    assert names == []


def test_no_file_over_50_mb() -> None:
    large = [path.relative_to(ROOT).as_posix() for path in FILES if path.stat().st_size > MAX_BYTES]
    assert large == []
