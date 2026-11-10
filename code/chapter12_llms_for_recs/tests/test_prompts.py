import numpy as np

from utils.prompt_templates import (
    engagement_prompt,
    listwise_ranking_prompt,
    parse_listwise_ranking,
    short_title,
)


def parse(text, n=4, seed=0):
    return parse_listwise_ranking(text, n, np.random.default_rng(seed))


def test_parses_a_complete_ranking():
    r = parse("Ranking: C, A, D, B")
    assert r.order == [2, 0, 3, 1]
    assert r.complete(4) and r.found_ranking_line and not r.filled


def test_reads_only_the_ranking_line():
    r = parse("I think A is a good fit.\nRanking: B > D\nBecause I said so")
    assert r.order[:2] == [1, 3]
    assert r.valid_labels == 2
    assert sorted(r.filled) == [0, 2]


def test_ignores_letters_in_prose_without_marker():
    r = parse("[D], [B]\nI would pick A later")
    assert r.order[:2] == [3, 1]
    assert r.valid_labels == 2 and not r.found_ranking_line


def test_counts_duplicates_and_out_of_range():
    r = parse("Ranking: A, A, Z, B")
    assert r.valid_labels == 2 and r.duplicate_labels == 1 and r.invalid_labels == 1


def test_unparseable_output_is_filled_randomly_not_in_display_order():
    orders = {tuple(parse("no idea", n=6, seed=s).order) for s in range(5)}
    assert all(sorted(o) == list(range(6)) for o in orders)
    assert len(orders) > 1
    assert not parse("no idea").parsed


def test_prompts_label_candidates_and_truncate_titles():
    prompt = listwise_ranking_prompt(["h1", "h2"], ["c1", "c2", "c3"])
    assert "[A] c1" in prompt and "[C] c3" in prompt and "A to C" in prompt
    assert "Yes or No" in engagement_prompt(["h"], "c")
    assert len(short_title("x" * 500)) == 100
