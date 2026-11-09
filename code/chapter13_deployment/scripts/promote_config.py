"""
Promote or revert a value-function config (Section 13.6).

    python scripts/promote_config.py --list
    python scripts/promote_config.py --validate config/reranker/vf-growth-v1.json
    python scripts/promote_config.py --activate vf-growth-v1 --reload-url http://127.0.0.1:8004
    python scripts/promote_config.py --activate vf-default-v1           # revert

--activate validates the target with the same code the re-ranker uses, then
rewrites the ACTIVE pointer atomically. Replicas pick it up on their next
poll; --reload-url asks one replica to reload now and prints its status.
Config files are never edited in place: a change is a new versioned file.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.reranker.config_store import (DEFAULT_CONFIG_DIR, ConfigError, RerankerConfig,  # noqa: E402
                                            build_active, read_config, read_pointer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", type=Path,
                        default=Path(os.environ.get("RERANKER_CONFIG_DIR", DEFAULT_CONFIG_DIR)))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--validate", type=Path, metavar="FILE")
    group.add_argument("--activate", metavar="VERSION")
    parser.add_argument("--reload-url", default=None, help="re-ranker base URL to reload immediately")
    args = parser.parse_args()
    directory = args.config_dir

    if args.list:
        active = read_pointer(directory)
        for path in sorted(directory.glob("*.json")):
            try:
                read_config(directory, path.stem)
                state = "valid"
            except ConfigError as exc:
                state = f"INVALID: {exc}"
            print(f"{'*' if path.stem == active else ' '} {path.stem:24s} {state}")
        return 0

    if args.validate:
        try:
            config = RerankerConfig.model_validate(json.loads(args.validate.read_text(encoding="utf-8")))
            build_active(config)
        except Exception as exc:  # report every failure the service would report
            print(f"INVALID {args.validate.name}: {exc}")
            return 1
        print(f"valid: {config.config_version}")
        return 0

    try:
        build_active(read_config(directory, args.activate))
    except ConfigError as exc:
        print(f"Not promoted: {exc}")
        return 1
    previous = read_pointer(directory)
    tmp = directory / "ACTIVE.tmp"
    tmp.write_text(args.activate + "\n", encoding="utf-8")
    os.replace(tmp, directory / "ACTIVE")
    print(f"ACTIVE: {previous} -> {args.activate}")

    if args.reload_url:
        import httpx

        response = httpx.post(args.reload_url.rstrip("/") + "/v1/config/reload", timeout=10.0)
        print(f"reload {response.status_code}: {response.json()}")
        return 0 if response.status_code == 200 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
