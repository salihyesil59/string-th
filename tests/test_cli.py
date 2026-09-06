"""The command-line summary."""

from __future__ import annotations

import pytest

from stringsim.cli import main, summary
from stringsim.units import Conventions


def test_summary_reports_the_headline_numbers():
    text = summary(Conventions(alpha_prime=1.0, dim=26))
    for expected in ("D =  26", "graviton (299)", "Kalb-Ramond (276)", "dilaton (1)", "U(3)"):
        assert expected in text


def test_summary_tracks_a_different_alpha_prime():
    text = summary(Conventions(alpha_prime=4.0, dim=26))
    assert "sqrt(alpha') = 2.0000" in text


def test_main_runs_and_writes_figures(tmp_path, capsys):
    assert main(["--figures", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Veneziano amplitude" in out
    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "hagedorn.png",
        "open_spectrum.png",
        "regge_trajectory.png",
    ]


def test_main_rejects_an_impossible_dimension():
    with pytest.raises(ValueError):
        main(["--dim", "2"])
