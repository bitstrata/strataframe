from __future__ import annotations

from pathlib import Path

from strataframe.graph.las_utils import _choose_curve_index, _iter_las_ascii_rows


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "sample.las"
    p.write_text(text, encoding="utf-8")
    return p


def test_iter_las_ascii_rows_unwrapped(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "\n".join(
            [
                "~C",
                "DEPT.M : Depth",
                "GR.API : Gamma",
                "RHOB.G/CC : Density",
                "~A",
                "1000 50 2.5",
                "1000.5,55,2.6",
                "# comment",
                "1001 60 2.7",
            ]
        ),
    )

    rows = list(_iter_las_ascii_rows(p, n_curves=3, wrapped=False))
    assert rows == [
        ["1000", "50", "2.5"],
        ["1000.5", "55", "2.6"],
        ["1001", "60", "2.7"],
    ]


def test_iter_las_ascii_rows_wrapped(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "\n".join(
            [
                "~C",
                "DEPT.M : Depth",
                "GR.API : Gamma",
                "RHOB.G/CC : Density",
                "~A",
                "1000 50",
                "2.5 1000.5 55",
                "2.6 1001 60 2.7",
                "1002 70",  # incomplete row (ignored)
            ]
        ),
    )

    rows = list(_iter_las_ascii_rows(p, n_curves=3, wrapped=True))
    assert rows == [
        ["1000", "50", "2.5"],
        ["1000.5", "55", "2.6"],
        ["1001", "60", "2.7"],
    ]


def test_choose_curve_index_primary_then_fallback() -> None:
    curves = ["CGR", "DT", "SGR"]
    idx = _choose_curve_index(curves, primary=("GR",), fallback=("SGR",))
    assert idx == 0  # CGR canonicalizes to GR, chosen by primary

    curves2 = ["DT", "SGR", "NPHI"]
    idx2 = _choose_curve_index(curves2, primary=("GR",), fallback=("SGR",))
    assert idx2 == 1
