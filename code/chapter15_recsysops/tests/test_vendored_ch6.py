"""The vendored Chapter 6 pipeline is byte-identical to its recorded source."""

import hashlib

import ranker as rk


def test_vendored_body_matches_recorded_hash():
    fields, body = rk.split_vendored()
    assert fields["source_sha256"] == "5eeb0a86dab5094d6c6b6dcffd3f4db858abebb04c4ba9150179f1e64d33e877"
    assert hashlib.sha256(body).hexdigest() == fields["source_sha256"]


def test_header_records_source():
    fields, _ = rk.split_vendored()
    assert fields["source_path"] == "code/chapter6_ranking/utils/polars_pipeline.py"
    assert len(fields["source_commit"]) == 40 and fields["source_commit"].startswith("8e8e71a")


def test_vendored_copy_is_the_default():
    assert rk.locate_ch6_pipeline() == rk.VENDORED_CH6
    prov = rk.ch6_provenance(rk.VENDORED_CH6)
    assert prov["body_matches_source"] is True
