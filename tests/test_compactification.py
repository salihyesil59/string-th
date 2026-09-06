"""A closed string on a circle: winding, T-duality, enhanced symmetry."""

from __future__ import annotations

import math

import pytest

from stringsim.compactification.circle import (
    extra_massless_states,
    kaluza_klein_mass,
    lightest_mass,
    massless_states,
    self_dual_radius,
    spectrum,
    spectrum_is_t_dual,
    t_dual_radius,
    winding_mass,
)
from stringsim.units import Conventions

CONV = Conventions(alpha_prime=1.0, dim=26)
RANGE = dict(n_max=3, w_max=3, level_max=2)


def test_self_dual_radius_is_its_own_dual():
    r0 = self_dual_radius(CONV)
    assert r0 == pytest.approx(math.sqrt(CONV.alpha_prime))
    assert t_dual_radius(r0, CONV) == pytest.approx(r0)


def test_t_duality_is_an_involution():
    for r in (0.3, 1.0, 4.2):
        assert t_dual_radius(t_dual_radius(r, CONV), CONV) == pytest.approx(r)


@pytest.mark.parametrize("radius", [0.25, 0.7, 1.0, 1.9, 5.0])
@pytest.mark.parametrize("alpha_prime", [0.5, 1.0, 3.0])
def test_spectrum_is_invariant_under_t_duality(radius, alpha_prime):
    conv = Conventions(alpha_prime=alpha_prime, dim=26)
    assert spectrum_is_t_dual(radius, conv, **RANGE)


def test_t_duality_exchanges_momentum_and_winding():
    """State ``(n, w)`` at ``R`` has the same mass as ``(w, n)`` at ``alpha'/R``."""
    r = 2.4
    dual = t_dual_radius(r, CONV)
    here = {(s.n, s.w, s.level, s.level_tilde): s.alpha_m2 for s in spectrum(r, CONV, **RANGE)}
    there = {(s.n, s.w, s.level, s.level_tilde): s.alpha_m2 for s in spectrum(dual, CONV, **RANGE)}
    for (n, w, lvl, lvl_t), m2 in here.items():
        assert there[(w, n, lvl, lvl_t)] == pytest.approx(m2)


def test_level_matching_is_enforced():
    for s in spectrum(1.3, CONV, **RANGE):
        assert s.level - s.level_tilde == s.n * s.w


def test_towers_swap_at_the_self_dual_radius():
    r0 = self_dual_radius(CONV)
    assert kaluza_klein_mass(1, r0) == pytest.approx(winding_mass(1, r0, CONV))
    assert kaluza_klein_mass(1, 0.5 * r0) > winding_mass(1, 0.5 * r0, CONV)
    assert kaluza_klein_mass(1, 2.0 * r0) < winding_mass(1, 2.0 * r0, CONV)


def test_winding_has_no_point_particle_analogue():
    """Winding energy grows with the radius; Kaluza-Klein energy falls."""
    assert winding_mass(1, 10.0, CONV) > winding_mass(1, 1.0, CONV)
    assert kaluza_klein_mass(1, 10.0) < kaluza_klein_mass(1, 1.0)


def test_generic_radius_has_only_the_uncharged_massless_states():
    for r in (0.6, 1.4, 3.1):
        assert extra_massless_states(r, CONV, **RANGE) == []
        total = sum(s.degeneracy for s in massless_states(r, CONV, **RANGE))
        assert total == (CONV.dim - 2) ** 2


def test_self_dual_radius_has_extra_massless_states():
    r0 = self_dual_radius(CONV)
    extra = extra_massless_states(r0, CONV, **RANGE)
    assert len(extra) == 8
    gauge = [s for s in extra if (s.level, s.level_tilde) in ((1, 0), (0, 1))]
    assert len(gauge) == 4
    assert {(s.n, s.w) for s in gauge} == {(1, 1), (-1, -1), (1, -1), (-1, 1)}
    # The remaining four are the tachyon's momentum/winding tower crossing zero.
    tower = [s for s in extra if (s.level, s.level_tilde) == (0, 0)]
    assert {(s.n, s.w) for s in tower} == {(2, 0), (-2, 0), (0, 2), (0, -2)}
    total = sum(s.degeneracy for s in massless_states(r0, CONV, **RANGE))
    assert total > (CONV.dim - 2) ** 2


def test_enhancement_happens_only_at_the_self_dual_radius():
    r0 = self_dual_radius(CONV)
    counts = {
        r: sum(s.degeneracy for s in massless_states(r, CONV, **RANGE))
        for r in (0.5, 0.9, r0, 1.1, 2.0)
    }
    assert counts[r0] == max(counts.values())
    assert all(counts[r] < counts[r0] for r in counts if r != r0)


def test_lightest_state_is_kaluza_klein_when_large_and_winding_when_small():
    big = lightest_mass(4.0, CONV, **RANGE)
    small = lightest_mass(0.25, CONV, **RANGE)
    assert (big.n, big.w) == (-1, 0) or (abs(big.n), big.w) == (1, 0)
    assert (abs(small.w), small.n) == (1, 0)
    # T-duality again: the two masses agree because 4.0 and 0.25 are dual.
    assert big.alpha_m2 == pytest.approx(small.alpha_m2)


def test_radius_must_be_positive():
    with pytest.raises(ValueError):
        spectrum(0.0, CONV)
    with pytest.raises(ValueError):
        t_dual_radius(-1.0, CONV)
