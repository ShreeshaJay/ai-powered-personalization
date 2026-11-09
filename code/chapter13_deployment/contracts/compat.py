"""
Detect breaking contract changes by comparing JSON Schemas with a snapshot.

Rules (the footer of Figure 13.3):
  * adding an optional field is compatible;
  * removing a field, changing a field's type or constraints, or adding a
    required field is breaking and needs a new path version (/v2/).

The snapshot for /v1/ is contracts/schema_snapshot_v1.json.
`python -m contracts.compat` exits non-zero on a breaking change, and
tests/test_contracts.py makes the same comparison; a CI job would run either. After an intended compatible change, refresh
the snapshot with `python -m contracts.compat --update`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .schemas import ALL_CONTRACTS, CONTRACT_VERSION

SNAPSHOT_PATH = Path(__file__).with_name(f"schema_snapshot_{CONTRACT_VERSION}.json")
_IGNORED_KEYS = {"description", "title", "default", "examples"}


def current_schemas() -> Dict[str, dict]:
    return {name: model.model_json_schema() for name, model in sorted(ALL_CONTRACTS.items())}


def load_snapshot() -> Dict[str, dict]:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _strip(node):
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if k not in _IGNORED_KEYS}
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


def _objects(schema: dict, root: str) -> Dict[str, dict]:
    objects = {root: schema}
    for name, sub in schema.get("$defs", {}).items():
        objects[f"{root}.{name}"] = sub
    return objects


def compare(old: Dict[str, dict], new: Dict[str, dict]) -> Tuple[List[str], List[str]]:
    """Return (breaking changes, compatible changes) between two schema sets."""
    breaking, compatible = [], []
    for contract in sorted(set(old) | set(new)):
        if contract not in new:
            breaking.append(f"{contract}: contract removed")
            continue
        if contract not in old:
            compatible.append(f"{contract}: contract added")
            continue
        old_objects, new_objects = _objects(old[contract], contract), _objects(new[contract], contract)
        for name in sorted(set(old_objects) | set(new_objects)):
            if name not in old_objects or name not in new_objects:
                continue  # nested object added or removed; its parent field change reports it
            old_props = old_objects[name].get("properties", {})
            new_props = new_objects[name].get("properties", {})
            old_required = set(old_objects[name].get("required", []))
            new_required = set(new_objects[name].get("required", []))
            for field in sorted(set(old_props) | set(new_props)):
                where = f"{name}.{field}"
                if field not in new_props:
                    breaking.append(f"{where}: field removed")
                elif field not in old_props:
                    if field in new_required:
                        breaking.append(f"{where}: required field added")
                    else:
                        compatible.append(f"{where}: optional field added")
                elif _strip(old_props[field]) != _strip(new_props[field]):
                    breaking.append(f"{where}: type or constraints changed")
                elif field in new_required and field not in old_required:
                    breaking.append(f"{where}: field became required")
                elif field in old_required and field not in new_required:
                    if "Response" in contract:
                        breaking.append(f"{where}: response field became optional")
                    else:
                        compatible.append(f"{where}: request field became optional")
    return breaking, compatible


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="Rewrite the snapshot from the current contracts")
    args = parser.parse_args()
    if args.update:
        SNAPSHOT_PATH.write_text(json.dumps(current_schemas(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {SNAPSHOT_PATH.name}")
        return 0
    breaking, compatible = compare(load_snapshot(), current_schemas())
    for change in compatible:
        print(f"compatible  {change}")
    for change in breaking:
        print(f"BREAKING    {change}")
    if breaking:
        print(f"{len(breaking)} breaking change(s): publish them under a new path version, not /{CONTRACT_VERSION}/.")
        return 1
    print(f"No breaking changes against {SNAPSHOT_PATH.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
