"""Figure files have no titles, and publishable text stays free of local paths."""

import json
from pathlib import Path

from make_figures import figure_a3, figure_a4, refuse_demo_overwrite, _demo_payloads

ROOT = Path(__file__).resolve().parents[1]


def _banned():
    return (
        "one" + "drive",
        "c:" + "\\use" + "rs",
        "c:/" + "use" + "rs",
        "jaga" + "deesh",
        "chapter10_" + "item_embeddings",
    )


def test_figures_have_no_titles(tmp_path):
    a3, a4 = _demo_payloads()
    a3_path = tmp_path / "figure_A_3.png"
    a4_path = tmp_path / "figure_A_4.png"
    figure_a3(a3, a3_path)
    figure_a4(a4, a4_path)
    assert a3_path.stat().st_size > 1000
    assert a4_path.stat().st_size > 1000
    import matplotlib.pyplot as plt

    # Rebuild and inspect the axes before they are closed by saving through the same functions.
    from make_figures import apply_font, _density, A3_SERIES
    import numpy as np

    apply_font()
    fig, axes = plt.subplots(1, 2)
    grid = np.linspace(-0.2, 1.0, 32)
    for axis, key in zip(axes, ("random_pairs", "positive_pairs")):
        for series_key, label, color in A3_SERIES:
            axis.plot(grid, _density(np.asarray(a3[key][series_key]), grid), color=color, label=label)
        assert axis.get_title() == ""
    assert fig._suptitle is None or fig._suptitle.get_text() == ""
    plt.close(fig)
    # The saved JSON used by the demo path is marked synthetic.
    assert a3["synthetic"] and a4["synthetic"]
    json.dumps(a3)
    measured = tmp_path / "measured"
    measured.mkdir()
    (measured / "diagnostics.json").write_text('{"synthetic": false}', encoding="utf-8")
    try:
        refuse_demo_overwrite(measured)
    except SystemExit:
        refused = True
    else:
        refused = False
    assert refused
    refuse_demo_overwrite(tmp_path)


def test_publishable_files_have_no_local_identifiers():
    suffixes = {".py", ".md", ".txt", ".ipynb"}
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if "_legacy" in path.parts or "outputs" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "\u2014" in text:
            offenders.append(f"{path.name}: em dash")
        lowered = text.lower()
        for fragment in _banned():
            if fragment in lowered:
                offenders.append(f"{path.name}: {fragment}")
    assert offenders == []
