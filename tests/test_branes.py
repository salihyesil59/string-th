"""D-branes: tension scaling, stretched strings, and the gauge group."""

from __future__ import annotations

import math

import pytest

from stringsim.branes.dbrane import (
    BraneStack,
    dp_brane_tension,
    gauge_group,
    stretched_spectrum,
    tachyon_free_separation,
)
from stringsim.units import Conventions

CONV = Conventions(alpha_prime=1.0, dim=26)


def test_tension_scales_as_one_over_g_s():
    """The single power of ``1/g_s`` is what separates a D-brane from a soliton."""
    for p in (0, 2, 5):
        t1 = dp_brane_tension(p, 1.0, CONV)
        t2 = dp_brane_tension(p, 0.5, CONV)
        assert t2 / t1 == pytest.approx(2.0)


def test_tension_matches_the_closed_form():
    for p in (0, 1, 3, 7):
        for g_s in (0.2, 1.0):
            expected = 1.0 / ((2 * math.pi) ** p * g_s * CONV.alpha_prime ** ((p + 1) / 2))
            assert dp_brane_tension(p, g_s, CONV) == pytest.approx(expected)


def test_tension_scales_with_alpha_prime():
    a = dp_brane_tension(3, 1.0, Conventions(alpha_prime=1.0))
    b = dp_brane_tension(3, 1.0, Conventions(alpha_prime=4.0))
    assert a / b == pytest.approx(4.0 ** ((3 + 1) / 2))


def test_tension_rejects_bad_input():
    with pytest.raises(ValueError):
        dp_brane_tension(-1, 1.0, CONV)
    with pytest.raises(ValueError):
        dp_brane_tension(3, 0.0, CONV)


def test_coincident_branes_give_massless_vectors():
    levels = stretched_spectrum(0.0, 2, CONV)
    assert levels[1].mass_squared == pytest.approx(0.0)
    assert levels[1].degeneracy == CONV.transverse_dim
    assert levels[0].mass_squared < 0  # the open bosonic tachyon survives


def test_separating_branes_gives_mass_in_proportion_to_distance():
    """``M^2 = (d / 2 pi alpha')^2`` at level 1: the Higgs mechanism, geometrically."""
    for d in (0.5, 2.0, 7.0):
        level1 = stretched_spectrum(d, 1, CONV)[1]
        assert level1.mass_squared == pytest.approx((d / (2 * math.pi * CONV.alpha_prime)) ** 2)


def test_mass_squared_is_monotonic_in_separation():
    masses = [stretched_spectrum(d, 1, CONV)[1].mass_squared for d in (0.0, 1.0, 2.0, 5.0)]
    assert masses == sorted(masses)


def test_tachyon_disappears_at_the_predicted_separation():
    d_crit = tachyon_free_separation(CONV)
    assert d_crit == pytest.approx(2 * math.pi * math.sqrt(CONV.alpha_prime))
    assert stretched_spectrum(d_crit, 0, CONV)[0].mass_squared == pytest.approx(0.0, abs=1e-12)
    assert stretched_spectrum(d_crit * 1.01, 0, CONV)[0].mass_squared > 0
    assert stretched_spectrum(d_crit * 0.99, 0, CONV)[0].mass_squared < 0


def test_stretched_level_mass_sign_convention():
    tachyon = stretched_spectrum(0.0, 0, CONV)[0]
    assert tachyon.mass == pytest.approx(-1.0)  # negative root flags M^2 < 0


def test_stretched_spectrum_rejects_negative_separation():
    with pytest.raises(ValueError):
        stretched_spectrum(-1.0, 2, CONV)


@pytest.mark.parametrize(
    "positions, group, vectors",
    [
        ([0.0], "U(1)", 1),
        ([0.0, 0.0], "U(2)", 4),
        ([0.0, 0.0, 0.0], "U(3)", 9),
        ([0.0, 0.0, 1.0], "U(2) x U(1)", 5),
        ([0.0, 1.0, 2.0], "U(1) x U(1) x U(1)", 3),
        ([0.0, 0.0, 1.0, 1.0], "U(2) x U(2)", 8),
    ],
)
def test_gauge_group_from_brane_positions(positions, group, vectors):
    assert gauge_group(positions) == group
    assert BraneStack(tuple(positions)).massless_vectors == vectors


def test_breaking_a_stack_never_increases_the_gauge_group():
    """``U(N) -> U(k) x U(N-k)``: ``N^2 >= k^2 + (N-k)^2`` for every split."""
    n = 6
    full = BraneStack(tuple([0.0] * n)).massless_vectors
    for k in range(1, n):
        split = BraneStack(tuple([0.0] * k + [3.0] * (n - k))).massless_vectors
        assert split <= full
