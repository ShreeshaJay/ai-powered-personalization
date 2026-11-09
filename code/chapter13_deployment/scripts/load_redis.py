"""
Load the active bundle's feature store into Redis (optional backend).

    docker compose --profile redis up -d redis
    python scripts/load_redis.py                       # REDIS_URL defaults to redis://localhost:6379/0
    FEATURE_STORE_BACKEND=redis python scripts/run_local.py
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.artifacts import resolve_bundle_dir  # noqa: E402
from common.feature_store import load_into_redis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", type=Path, default=None, help="default: artifacts/ACTIVE")
    parser.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    args = parser.parse_args()

    import redis

    client = redis.Redis.from_url(args.redis_url)
    client.ping()
    directory = resolve_bundle_dir(args.bundle_dir) / "feature_store"
    counts = load_into_redis(directory, client)
    print(f"Loaded {counts} from {directory} into {args.redis_url}")


if __name__ == "__main__":
    main()
