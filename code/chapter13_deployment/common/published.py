"""
Import the book's published Chapter 5 to 8 code from vendor/.

Chapters 6 and 7 both call their model package `models`, so they cannot both
sit on sys.path. Each vendored package is imported under a chapter-specific
alias (ch6_models, ch7_models, ch8_value_functions), and only the submodules
a caller asks for are executed. Package __init__ files are skipped because
they import sibling models (DeepFM, DCN, DLRM, the Chapter 8 pipeline) that
pull in PyTorch; the lightweight ranker and the re-ranker do not need it.

Every vendored file a process executes is recorded, so a service can check
that it runs the same published code its artifact bundle was built with
(`loaded_files`, `vendored_files`, `common.artifacts.check_published_code`).
"""

import importlib.util
import json
import sys
import types
import warnings
from pathlib import Path
from typing import Dict

from .artifacts import sha256_file

VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor"
_LOADED: Dict[str, Path] = {}

_PACKAGES = {
    "ch6_models": VENDOR_DIR / "chapter6_ranking" / "models",
    "ch7_models": VENDOR_DIR / "chapter7_advanced_ranking" / "models",
    "ch7_utils": VENDOR_DIR / "chapter7_advanced_ranking" / "utils",
    "ch8_value_functions": VENDOR_DIR / "chapter8_value_functions",
}


def _package(alias: str) -> types.ModuleType:
    if alias in sys.modules:
        return sys.modules[alias]
    directory = _PACKAGES[alias]
    if not directory.is_dir():
        raise ImportError(f"Vendored package missing: {directory}. Run scripts/vendor_published_code.py.")
    module = types.ModuleType(alias)
    module.__path__ = [str(directory)]
    module.__package__ = alias
    sys.modules[alias] = module
    return module


def _load_file(name: str, path: Path) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    if not path.is_file():
        raise ImportError(f"Vendored file missing: {path}. Run scripts/vendor_published_code.py.")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Chapter 5 calls warnings.filterwarnings('ignore') at import time; keep
    # that from silencing warnings for the whole service.
    saved_filters = list(warnings.filters)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        warnings.filters[:] = saved_filters
    _LOADED[name] = path
    return module


def loaded_files() -> Dict[str, str]:
    """{vendor-relative path: SHA-256} of every vendored file executed in this process."""
    return {path.relative_to(VENDOR_DIR).as_posix(): sha256_file(path) for path in set(_LOADED.values())}


def vendored_files() -> Dict[str, str]:
    """{vendor-relative path: SHA-256} of every file listed in vendor/MANIFEST.json, hashed from disk."""
    manifest = json.loads((VENDOR_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    return {entry["vendored_path"]: sha256_file(VENDOR_DIR / entry["vendored_path"])
            for entry in manifest["files"]}


def _submodule(alias: str, name: str) -> types.ModuleType:
    parent = _package(alias)
    module = _load_file(f"{alias}.{name}", Path(parent.__path__[0]) / f"{name}.py")
    setattr(parent, name, module)
    return module


def chapter5_two_tower() -> types.ModuleType:
    """code/chapter5_retrieval/supervised_two_tower.py"""
    return _load_file("ch5_supervised_two_tower", VENDOR_DIR / "chapter5_retrieval" / "supervised_two_tower.py")


def chapter6(name: str) -> types.ModuleType:
    """code/chapter6_ranking/models/<name>.py, for example chapter6('xgboost_ranker')."""
    return _submodule("ch6_models", name)


def chapter7_mmoe() -> types.ModuleType:
    """code/chapter7_advanced_ranking/models/mmoe.py"""
    return _submodule("ch7_models", "mmoe")


def chapter8(name: str) -> types.ModuleType:
    """code/chapter8_value_functions/<name>.py, for example chapter8('value_function')."""
    return _submodule("ch8_value_functions", name)


def chapter7_train_mmoe() -> types.ModuleType:
    """
    code/chapter7_advanced_ranking/train_mmoe.py: the artifact build trains with it,
    and the heavy ranker loads its MMoEFeatureProcessor.

    The script imports `models.mmoe` and `utils.polars_pipeline` as top-level
    names and inserts its own folder into sys.path. Those names are pointed at
    the Chapter 7 aliases for the duration of the import and restored after.
    """
    name = "ch7_train_mmoe"
    if name in sys.modules:
        return sys.modules[name]
    mmoe = chapter7_mmoe()
    polars_pipeline = _submodule("ch7_utils", "polars_pipeline")
    overrides = {
        "models": _package("ch7_models"),
        "models.mmoe": mmoe,
        "utils": _package("ch7_utils"),
        "utils.polars_pipeline": polars_pipeline,
    }
    saved_modules = {key: sys.modules.get(key) for key in overrides}
    saved_path = list(sys.path)
    sys.modules.update(overrides)
    try:
        return _load_file(name, VENDOR_DIR / "chapter7_advanced_ranking" / "train_mmoe.py")
    finally:
        sys.path[:] = saved_path
        for key, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous
