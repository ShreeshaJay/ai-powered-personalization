"""Figure C.2 series, matched to the seeded simulation behind the published chart."""

from __future__ import annotations

import pytest

from simulate_best_of_n import SIM_KS, SIM_REPS, SIM_SEED, SIM_SIGMA_Q, simulate

# Values written by figures_src/appD_figures.py with the same seed and draws.
EXPECTED = [
    (1, 0.004851535349370218, -0.005829748562705896, -0.005829748562705896),
    (2, 0.5852681716430356, 0.037558596450108346, 0.013260466845794685),
    (4, 1.062258385317669, 0.07803598773644076, -0.005924417458985288),
    (8, 1.4830467692214504, 0.12236071848478845, 0.003973189607114832),
    (16, 1.8480829903187406, 0.14089374304930644, -0.00014595747317436744),
    (32, 2.162170908093125, 0.17903355740773258, 0.0003467554920623157),
    (64, 2.433981741269081, 0.2004843602775498, 0.002283646408183386),
    (128, 2.7097155033649725, 0.21615232016861602, -0.000586638064615288),
    (256, 2.9572168087558786, 0.24369183266887404, -0.00024253052893964328),
]


def test_figure_constants_match_the_figure_script():
    assert SIM_SEED == 20261109
    assert SIM_REPS == 5000
    assert SIM_KS == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert SIM_SIGMA_Q == pytest.approx(0.3)


def test_series_matches_figure_c2():
    rows = simulate()
    assert [int(row["K"]) for row in rows] == [item[0] for item in EXPECTED]
    for row, expected in zip(rows, EXPECTED):
        assert row["val_selected"] == pytest.approx(expected[1], abs=1e-9)
        assert row["test_selected"] == pytest.approx(expected[2], abs=1e-9)
        assert row["test_average_candidate"] == pytest.approx(expected[3], abs=1e-9)
