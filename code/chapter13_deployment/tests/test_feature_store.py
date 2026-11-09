"""The three feature-store backends return identical records, including cold-start defaults."""

import shutil

import fakeredis
import pytest

from common.artifacts import ArtifactError
from common.feature_store import load_into_redis, open_feature_store


@pytest.fixture(scope="module")
def stores(bundle_dir):
    directory = bundle_dir / "feature_store"
    client = fakeredis.FakeRedis()
    load_into_redis(directory, client, batch_size=100)
    return {
        "sqlite": open_feature_store(directory, backend="sqlite"),
        "parquet": open_feature_store(directory, backend="parquet"),
        "redis": open_feature_store(directory, backend="redis", redis_client=client),
    }


def test_backends_agree(stores, sample_users):
    users = sample_users["warm"][:30] + sample_users["cold"][:3]
    item_ids = list(range(0, 400, 3)) + [10**9]
    reference = stores["sqlite"]
    for name, store in stores.items():
        for user in users:
            assert store.get_user(user) == reference.get_user(user), (name, user)
            assert store.get_user_items(user, item_ids) == reference.get_user_items(user, item_ids), name
        assert store.get_items(item_ids) == reference.get_items(item_ids), name


def test_unknown_keys_get_the_builders_defaults(stores, sample_users):
    store = stores["sqlite"]
    user = store.get_user(sample_users["cold"][0])
    assert not user.found and user.recent_item_ids == []
    assert user.features == {k: float(v) for k, v in store.defaults["user"].items() if k in user.features}
    item = store.get_items([10**9])[10**9]
    assert not item.found and item.artist_id is None
    assert store.get_user_items(1, [10**9])[10**9] == {"previous_listen_count": 0.0, "has_listened_before": 0.0}


def test_history_is_chronological_and_bounded(stores, sample_users):
    store = stores["sqlite"]
    for user in sample_users["warm"][:20]:
        history = store.get_user(user).recent_item_ids
        assert 0 < len(history) <= store.manifest["history_length"]


def test_contract_declares_every_feature_with_default_and_max_age(stores):
    store = stores["sqlite"]
    declared = store.contract["features"]
    for group, names in (("user", store.user_feature_names), ("item", store.item_feature_names),
                         ("user_item", store.user_item_feature_names)):
        for name in names:
            assert declared[name]["group"] == group and declared[name]["default"] == store.defaults[group][name]
            assert declared[name]["max_age"] > 0
    assert declared["track_length_seconds"]["max_age"] is None


@pytest.mark.parametrize("backend", ["sqlite", "parquet", "redis"])
def test_a_stale_value_becomes_missing_and_is_counted(stores, sample_users, backend):
    from common.feature_store import FeatureSubstitutions

    store = stores[backend]
    user_id = sample_users["warm"][0]
    fresh = store.get_user(user_id)
    assert fresh.updated_at == store.reference_timestamp
    user_age = store.max_age[store.user_feature_names[0]]
    item_age = store.max_age[store.item_feature_names[0]]
    assert user_age < item_age

    report = FeatureSubstitutions()
    assert store.get_user(user_id, as_of=fresh.updated_at + user_age, report=report).features == fresh.features
    assert not report.stale                                          # exactly the maximum age is still fresh

    before = dict(store.describe()["substitutions"]["stale"])
    as_of = fresh.updated_at + user_age + 1
    stale = store.get_user(user_id, as_of=as_of, report=report)
    assert stale.features == {k: float(store.defaults["user"][k]) for k in store.user_feature_names}
    assert stale.recent_item_ids == fresh.recent_item_ids             # history is not a model feature
    assert dict(report.stale) == {name: 1 for name in store.user_feature_names}

    history = fresh.recent_item_ids
    stored_pairs = sum(p["has_listened_before"] == 1.0 for p in store.get_user_items(user_id, history).values())
    assert stored_pairs > 0
    pairs = store.get_user_items(user_id, history, as_of=as_of, updated_at=stale.updated_at, report=report)
    assert all(p == {"previous_listen_count": 0.0, "has_listened_before": 0.0} for p in pairs.values())
    assert report.stale["previous_listen_count"] == stored_pairs

    items = store.get_items(history, as_of=as_of, report=report)       # items allow 30 days
    assert items == store.get_items(history)
    assert all(r.updated_at == store.reference_timestamp for r in items.values() if r.found)
    assert not any(name in report.stale for name in store.item_feature_names)
    late = FeatureSubstitutions()
    items = store.get_items(history, as_of=fresh.updated_at + item_age + 1, report=late)
    served = {i for i in history if items[i].track_length_seconds is not None}
    name = store.item_feature_names[0]
    assert late.stale[name] > 0 and late.stale[name] + late.missing[name] == len(served)

    after = store.describe()["substitutions"]["stale"]
    assert after[store.user_feature_names[0]] == before.get(store.user_feature_names[0], 0) + 1


def test_cold_start_defaults_are_counted_as_missing(stores, sample_users):
    from common.feature_store import FeatureSubstitutions

    report = FeatureSubstitutions()
    stores["sqlite"].get_user(sample_users["cold"][0], as_of=stores["sqlite"].reference_timestamp, report=report)
    assert dict(report.missing) == {name: 1 for name in stores["sqlite"].user_feature_names}
    assert not report.stale


def test_ranker_startup_check_rejects_an_undeclared_feature(bundle_dir):
    from common.feature_store import check_model_features, load_contract

    contract = load_contract(bundle_dir / "feature_store")
    names = list(contract["features"])
    check_model_features(contract, names, contract["feature_set_version"], "ranker")
    with pytest.raises(ArtifactError, match="does not declare"):
        check_model_features(contract, names + ["user_mood"], contract["feature_set_version"], "ranker")
    with pytest.raises(ArtifactError, match="trained on"):
        check_model_features(contract, names, "fs-other", "ranker")


def test_redis_refuses_to_serve_a_partial_load(bundle_dir):
    client = fakeredis.FakeRedis()
    with pytest.raises(ArtifactError, match="load_redis"):
        open_feature_store(bundle_dir / "feature_store", backend="redis", redis_client=client)


def test_missing_or_altered_store_files_fail_loudly(bundle_dir, tmp_path):
    copy = tmp_path / "feature_store"
    shutil.copytree(bundle_dir / "feature_store", copy)
    with open(copy / "features.sqlite", "ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ArtifactError, match="features.sqlite"):
        open_feature_store(copy, backend="sqlite")
    (copy / "items.parquet").unlink()
    with pytest.raises(ArtifactError, match="items.parquet"):
        open_feature_store(copy, backend="parquet")
    with pytest.raises(ArtifactError):
        open_feature_store(copy, backend="cassandra")
