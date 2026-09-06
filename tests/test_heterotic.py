"""Heterotic strings: the two lattices, and what level matching forces.

The central fact -- that there are exactly two heterotic strings -- is a
statement about even self-dual lattices in sixteen dimensions, so the checks
here are lattice checks: evenness, unimodularity, root counts, and the
decomposition that tells ``e8 + e8`` from ``so(32)`` when counting cannot.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from stringsim.compactification.torus import identify_algebra
from stringsim.heterotic.lattice import (
    GAUGE_DIMENSION,
    RootLattice,
    _d_roots,
    _integer_row_basis,
    d16_plus,
    e8,
    e8_squared,
    even_self_dual_dimension_rule,
    heterotic_lattices,
)
from stringsim.heterotic.spectrum import (
    LEFT_DIMENSION,
    RIGHT_DIMENSION,
    anomaly_free_dimension,
    central_charges,
    has_tachyon,
    internal_dimension,
    left_mass,
    level_matched_masses,
    massless_content,
    right_mass,
)
from stringsim.quantum.zeta import central_charge, critical_dimension
from stringsim.superstring.rns import Sector

# -- the integer basis helper ------------------------------------------------


def test_integer_row_basis_recovers_a_known_lattice():
    """``Z^2`` from a redundant generating set, and a sublattice of index 2."""
    assert _integer_row_basis([[1, 0], [0, 1], [3, 5]]) == [[1, 0], [0, 1]]
    basis = _integer_row_basis([[2, 0], [0, 1]])
    assert abs(np.linalg.det(np.array(basis, dtype=float))) == pytest.approx(2.0)


def test_integer_row_basis_handles_an_empty_input():
    assert _integer_row_basis([]) == []


# -- the lattices ------------------------------------------------------------


def test_e8_is_even_self_dual_with_240_roots():
    lattice = e8()
    assert lattice.dim == 8
    assert lattice.n_roots == 240
    assert lattice.roots_have_norm_two()
    assert lattice.is_integral()
    assert lattice.is_even()
    assert lattice.is_unimodular()
    assert lattice.covolume() == pytest.approx(1.0)
    assert lattice.algebra == "e8"
    assert lattice.algebra_dimension == 248


def test_e8_roots_split_into_the_two_standard_families():
    """112 vectors ``+/-e_i +/- e_j`` and 128 half-integer ones."""
    roots = e8().roots
    integral = [r for r in roots if np.allclose(r, np.rint(r))]
    half = [r for r in roots if not np.allclose(r, np.rint(r))]
    assert len(integral) == 112
    assert len(half) == 128
    # every half-integer root has an even number of minus signs
    assert all(sum(1 for x in r if x < 0) % 2 == 0 for r in half)


@pytest.mark.parametrize("name", ["E8 + E8", "Spin(32)/Z2"])
def test_both_heterotic_lattices_are_even_and_self_dual(name):
    lattice = heterotic_lattices()[name]
    assert lattice.dim == 16
    assert lattice.n_roots == 480
    assert lattice.roots_have_norm_two()
    assert lattice.is_even_self_dual()
    assert lattice.covolume() == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["E8 + E8", "Spin(32)/Z2"])
def test_both_give_496_gauge_bosons(name):
    """480 roots plus 16 Cartan generators, which is what anomaly cancellation wants."""
    lattice = heterotic_lattices()[name]
    assert lattice.algebra_dimension == GAUGE_DIMENSION == 496
    assert lattice.n_roots + lattice.dim == 496


def test_counting_cannot_tell_the_two_apart():
    """Rank 16 with 480 roots is genuinely ambiguous, and the code says so."""
    assert identify_algebra(16, 480) == ("e8 + e8", "so(32)")


def test_the_root_geometry_can_tell_them_apart():
    """Two orthogonal families of 240 versus one connected system of 480."""
    assert e8_squared().components == [(8, 240), (8, 240)]
    assert e8_squared().algebra == "e8 + e8"
    assert d16_plus().components == [(16, 480)]
    assert d16_plus().algebra == "so(32)"


def test_the_two_lattices_are_inequivalent():
    """Same rank, same root count, different root systems -- so different theories."""
    a, b = e8_squared(), d16_plus()
    assert (a.dim, a.n_roots) == (b.dim, b.n_roots)
    assert a.components != b.components


def test_the_spinor_class_is_what_makes_d16_self_dual():
    """``D_16`` alone has covolume 2; adding the half-integer coset halves it."""
    bare = RootLattice("D16", _d_roots(16))
    assert bare.covolume() == pytest.approx(2.0)
    assert not bare.is_unimodular()
    assert bare.is_even()  # even but not self-dual
    assert d16_plus().covolume() == pytest.approx(1.0)


def test_the_spinor_vectors_are_not_roots():
    """They have norm 4, so they contribute no gauge boson -- only self-duality."""
    extra = d16_plus().extra
    assert extra.shape == (1, 16)
    assert float(extra[0] @ extra[0]) == pytest.approx(4.0)


def test_even_self_dual_lattices_need_a_multiple_of_eight():
    assert [d for d in range(33) if even_self_dual_dimension_rule(d)] == [0, 8, 16, 24, 32]
    assert even_self_dual_dimension_rule(internal_dimension())
    with pytest.raises(ValueError):
        even_self_dual_dimension_rule(-1)


def test_lattice_validates_its_input():
    with pytest.raises(ValueError):
        RootLattice("bad", np.zeros(5))
    with pytest.raises(ValueError):
        RootLattice("thirds", np.array([[1 / 3, 0.0], [0.0, 1 / 3]])).basis()


# -- heterosis ---------------------------------------------------------------


def test_the_internal_dimension_is_the_difference_of_the_two_critical_dimensions():
    assert LEFT_DIMENSION == critical_dimension("bosonic") == 26
    assert RIGHT_DIMENSION == critical_dimension("superstring") == 10
    assert internal_dimension() == 16


def test_central_charges_differ_between_the_two_sides():
    left, right = central_charges()
    assert left == 26.0
    assert right == 15.0
    # each side's own anomaly cancels against its own ghosts
    assert central_charge(26, "bosonic") == pytest.approx(0.0)
    assert central_charge(10, "superstring") == pytest.approx(0.0)


# -- level matching ----------------------------------------------------------


def test_the_left_side_is_always_an_integer():
    """Because the lattice is even, ``p^2/2`` is an integer and so is the mass."""
    for level in range(4):
        for p_squared in (0, 2, 4, 6):
            assert left_mass(level, p_squared).denominator == 1
    assert left_mass(0, 0) == -1


def test_left_mass_rejects_an_odd_norm():
    """An odd ``p^2`` would mean the lattice was not even."""
    with pytest.raises(ValueError):
        left_mass(0, 3)
    with pytest.raises(ValueError):
        left_mass(-1, 0)


def test_right_mass_uses_the_sector_intercepts():
    """The ground-state masses are minus the intercepts the superstring module gives."""
    from stringsim.superstring.rns import intercept
    from stringsim.units import Conventions

    ten = Conventions(dim=10)
    for sector in (Sector.NS, Sector.R):
        assert float(right_mass(0, sector)) == pytest.approx(-intercept(sector, ten))
    assert right_mass(0, Sector.NS) == Fraction(-1, 2)
    assert right_mass(Fraction(1, 2), Sector.NS) == 0
    assert right_mass(0, Sector.R) == 0
    with pytest.raises(ValueError):
        right_mass(Fraction(1, 3), Sector.NS)  # NS levels are multiples of 1/2
    with pytest.raises(ValueError):
        right_mass(Fraction(1, 2), Sector.R)  # R levels are integers
    with pytest.raises(ValueError):
        right_mass(-1, Sector.R)


def test_there_is_no_tachyon_and_gso_is_not_what_removes_it():
    """The lightest matched state is massless whether or not GSO is applied."""
    for gso in (True, False):
        states = level_matched_masses(3, 6, gso=gso)
        assert states, "no matched states found"
        assert states[0].alpha_m2 == 0
        assert not has_tachyon(gso=gso)


def test_the_would_be_tachyons_have_nothing_to_pair_with():
    """Left ``-1`` is an integer, right ``-1/2`` is not; neither finds a partner."""
    assert left_mass(0, 0) == -1
    assert right_mass(0, Sector.NS) == Fraction(-1, 2)
    matched = {state.alpha_m2 for state in level_matched_masses(3, 6, gso=False)}
    assert -4 not in matched  # 4 x (-1)
    assert min(matched) == 0


def test_the_massless_level_is_reached_two_ways():
    """One oscillator with no lattice momentum, or a root with no oscillator."""
    massless = [s for s in level_matched_masses(2, 4) if s.is_massless]
    left_options = {(s.left_level, s.p_squared) for s in massless}
    assert left_options == {(1, 0), (0, 2)}
    assert {s.sector for s in massless} == {Sector.NS, Sector.R}


def test_matched_masses_are_sorted_and_quantised():
    states = level_matched_masses(3, 6)
    assert [s.alpha_m2 for s in states] == sorted(s.alpha_m2 for s in states)
    assert all(s.alpha_m2 % 4 == 0 for s in states)  # alpha'M^2 = 4 x integer


def test_level_matched_masses_validates_its_cutoffs():
    with pytest.raises(ValueError):
        level_matched_masses(-1)
    with pytest.raises(ValueError):
        level_matched_masses(2, -2)


# -- the massless spectrum ---------------------------------------------------


@pytest.mark.parametrize("name", ["E8 + E8", "Spin(32)/Z2"])
def test_massless_content_adds_up(name):
    content = massless_content(heterotic_lattices()[name])
    assert content.left_oscillator_states == 24
    assert content.left_lattice_states == 480
    assert content.left_states == 504
    assert content.right_states == 16
    assert content.total == 8064
    assert content.supergravity_states == 128
    assert content.gauge_states == 496 * 16 == 7936
    assert content.supergravity_states + content.gauge_states == content.total


def test_the_supergravity_multiplet_is_the_familiar_one():
    """Graviton, ``B``, dilaton, gravitino, dilatino -- the same reps as type II."""
    content = massless_content(e8_squared())
    dimensions = [field.dimension for field in content.supergravity]
    assert dimensions == [35, 28, 1, 56, 8]
    bosons = sum(f.dimension for f in content.supergravity if f.is_boson)
    fermions = sum(f.dimension for f in content.supergravity if not f.is_boson)
    assert bosons == fermions == 64


def test_the_gauge_part_carries_the_lattice_algebra():
    for name, expected in (("E8 + E8", "e8 + e8"), ("Spin(32)/Z2", "so(32)")):
        content = massless_content(heterotic_lattices()[name])
        assert content.gauge_algebra == expected
        assert content.gauge_dimension == anomaly_free_dimension() == 496
