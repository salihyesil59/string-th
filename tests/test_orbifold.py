"""Toroidal orbifolds: the projection, twisted sectors and what constrains them.

Three independent routes are made to agree here. The twisted-sector intercept is
computed from a closed form *and* from cut-off mode sums; the untwisted
projection is computed by the character projector *and* by splitting the
transverse index by hand; and the allowed orbifold orders are found by searching
integer matrices rather than quoted.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from stringsim.compactification.orbifold import (
    Orbifold,
    crystallographic_orders,
    massless_summary,
    matrix_order,
    oscillator_trace_series,
    transverse_phases,
    twisted_spectrum,
    untwisted_degeneracy,
    untwisted_massless_content,
)
from stringsim.compactification.torus import TorusBackground
from stringsim.quantum.partition import oscillator_degeneracies
from stringsim.quantum.zeta import regularised_shifted_sum
from stringsim.units import Conventions

CONV = Conventions(alpha_prime=1.0, dim=26)


def _circle_z2() -> Orbifold:
    return Orbifold.inversion(TorusBackground.from_radii([1.7], CONV))


def _all_standard() -> dict[str, Orbifold]:
    return {
        "S1/Z2": _circle_z2(),
        "T2/Z2": Orbifold.inversion(TorusBackground.from_radii([1.3, 2.1], CONV)),
        "T4/Z2": Orbifold.inversion(TorusBackground.from_radii([1.3, 2.1, 0.9, 1.5], CONV)),
        "T2/Z3": Orbifold.z3_hexagonal(CONV),
        "T2/Z4": Orbifold.z4_square(CONV),
        "T2/Z6": Orbifold.z6_hexagonal(CONV),
    }


# -- which orbifolds are allowed ---------------------------------------------


def test_crystallographic_restriction_in_two_dimensions():
    """Only ``N = 1, 2, 3, 4, 6`` -- found by search, not asserted from memory."""
    assert crystallographic_orders(2, bound=2) == {1, 2, 3, 4, 6}
    assert crystallographic_orders(1, bound=2) == {1, 2}


def test_crystallographic_search_is_stable_under_widening_the_box():
    """A wider search must not turn up a new order, or the bound was too small."""
    assert crystallographic_orders(2, bound=3) == crystallographic_orders(2, bound=2)


def test_matrix_order():
    assert matrix_order(np.eye(2)) == 1
    assert matrix_order(-np.eye(2)) == 2
    assert matrix_order(np.array([[0.0, -1.0], [1.0, 0.0]])) == 4
    assert matrix_order(np.array([[1.0, 1.0], [0.0, 1.0]])) is None  # infinite order


def test_rotation_must_preserve_the_moduli():
    """A lattice automorphism that changes ``G`` is not a symmetry to quotient by."""
    square = TorusBackground(np.eye(2), conventions=CONV)
    rectangle = TorusBackground.from_radii([1.0, 2.0], CONV)
    quarter_turn = np.array([[0.0, -1.0], [1.0, 0.0]])
    Orbifold(square, quarter_turn)  # fine: the square lattice has a 90 degree symmetry
    with pytest.raises(ValueError):
        Orbifold(rectangle, quarter_turn)  # the rectangle does not


def test_rotation_validation():
    background = TorusBackground.from_radii([1.3, 2.1], CONV)
    with pytest.raises(ValueError):
        Orbifold(background, np.eye(3))  # wrong shape
    with pytest.raises(ValueError):
        Orbifold(background, 0.5 * np.eye(2))  # not integer
    with pytest.raises(ValueError):
        Orbifold(background, np.array([[1.0, 1.0], [0.0, 1.0]]))  # infinite order


@pytest.mark.parametrize(
    "factory, order",
    [(Orbifold.z3_hexagonal, 3), (Orbifold.z4_square, 4), (Orbifold.z6_hexagonal, 6)],
)
def test_standard_orbifolds_have_the_advertised_order(factory, order):
    assert factory(CONV).order == order


def test_inversion_works_in_any_dimension():
    for d in (1, 2, 3, 5):
        orbifold = Orbifold.inversion(TorusBackground.self_dual(d, CONV))
        assert orbifold.order == 2
        assert np.allclose(orbifold.twist_phases(1), 0.5)


# -- fixed points ------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [("S1/Z2", 2), ("T2/Z2", 4), ("T4/Z2", 16), ("T2/Z3", 3), ("T2/Z4", 2), ("T2/Z6", 1)],
)
def test_fixed_point_counts(name, expected):
    """``|det(1 - theta)|`` -- the standard table for the ``T^2`` orbifolds."""
    assert _all_standard()[name].fixed_points(1) == expected


def test_inversion_has_two_to_the_d_fixed_points():
    for d in (1, 2, 3, 4, 5):
        orbifold = Orbifold.inversion(TorusBackground.self_dual(d, CONV))
        assert orbifold.fixed_points(1) == 2**d


def test_z4_squared_is_the_z2_sector():
    """``theta^2 = -1`` on the square lattice, with its own four fixed points."""
    z4 = Orbifold.z4_square(CONV)
    assert np.allclose(z4.power(2), -np.eye(2))
    assert z4.fixed_points(2) == 4
    assert z4.fixed_points(1) == z4.fixed_points(3) == 2


def test_fixed_points_refuses_the_untwisted_and_the_non_isolated_case():
    z4 = Orbifold.z4_square(CONV)
    with pytest.raises(ValueError):
        z4.fixed_points(0)
    with pytest.raises(ValueError):
        z4.fixed_points(4)
    # a rotation acting on only one of two directions leaves a fixed line
    background = TorusBackground.from_radii([1.3, 2.1], CONV)
    partial = Orbifold(background, np.diag([-1.0, 1.0]))
    assert not partial.has_isolated_fixed_points(1)
    with pytest.raises(ValueError):
        partial.fixed_points(1)


# -- the twisted ground-state energy -----------------------------------------


@pytest.mark.parametrize("shift", [1.0, 0.5, 1 / 3, 2 / 3, 0.25])
def test_hurwitz_zeta_from_a_cutoff_sum(shift):
    """``zeta(-1, a) = -B_2(a)/2``, extracted the same way ``-1/12`` was."""
    estimate = regularised_shifted_sum(shift)
    assert estimate.error < 1e-7


def test_shifted_sum_validates_its_argument():
    with pytest.raises(ValueError):
        regularised_shifted_sum(0.0)
    with pytest.raises(ValueError):
        regularised_shifted_sum(1.5)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("S1/Z2", Fraction(15, 16)),
        ("T2/Z2", Fraction(7, 8)),
        ("T4/Z2", Fraction(3, 4)),
        ("T2/Z3", Fraction(8, 9)),
        ("T2/Z4", Fraction(29, 32)),
        ("T2/Z6", Fraction(67, 72)),
    ],
)
def test_twisted_intercepts_match_the_closed_forms(name, expected):
    assert _all_standard()[name].intercept(1) == pytest.approx(float(expected))


def test_intercept_agrees_with_the_regularised_mode_sums():
    r"""The independent route: ``a = -sum_j (1/2) zeta(-1, phi_j)`` over all transverse bosons.

    The closed form ``a = 1 - (1/4) sum phi(1-phi)`` is a rearrangement of that,
    so agreement is a genuine check of the rearrangement and of the phases.
    """
    for orbifold in _all_standard().values():
        for k in range(1, orbifold.order):
            phases = transverse_phases(orbifold, k)
            # a phase of 0 means an ordinary periodic boson, i.e. shift 1
            shifts = np.where(phases < 1e-12, 1.0, phases)
            zero_point = sum(0.5 * regularised_shifted_sum(float(s)).value for s in shifts)
            assert -zero_point == pytest.approx(orbifold.intercept(k), abs=1e-6)


def test_untwisted_intercept_is_one():
    for orbifold in _all_standard().values():
        assert orbifold.intercept(0) == pytest.approx(1.0)


def test_more_twisting_lowers_the_intercept():
    """Every twisted direction raises the ground-state energy, so ``a`` falls."""
    previous = 1.0
    for d in (1, 2, 3, 4):
        value = Orbifold.inversion(TorusBackground.self_dual(d, CONV)).intercept(1)
        assert value < previous
        assert value == pytest.approx(1.0 - d / 16.0)
        previous = value


# -- the untwisted projection ------------------------------------------------


def test_trace_series_at_k_zero_is_the_ordinary_degeneracy():
    orbifold = _circle_z2()
    series = oscillator_trace_series(transverse_phases(orbifold, 0), 6)
    assert np.allclose(series.real, oscillator_degeneracies(6, 24))
    assert np.max(np.abs(series.imag)) < 1e-9


def test_z2_trace_series_matches_the_sign_flipped_product():
    r"""For ``theta = -1`` the trace is the ``(1-q^n)^{-23}(1+q^n)^{-1}`` product."""
    orbifold = _circle_z2()
    series = oscillator_trace_series(transverse_phases(orbifold, 1), 4).real
    # build the same product independently
    expected = np.zeros(5)
    expected[0] = 1.0
    for n in range(1, 5):
        for sign, count in ((1.0, 23), (-1.0, 1)):
            for _ in range(count):
                for level in range(n, 5):
                    expected[level] += sign * expected[level - n]
    assert np.allclose(series, expected)


@pytest.mark.parametrize("name", ["S1/Z2", "T2/Z2", "T4/Z2"])
def test_character_projection_matches_the_index_counting(name):
    """``(D-2-d)^2 + d^2`` by hand, versus the projector's trace formula."""
    orbifold = _all_standard()[name]
    content = untwisted_massless_content(orbifold)
    assert untwisted_degeneracy(orbifold, 1, 1) == content.surviving


def test_untwisted_content_breakdown_adds_up():
    for name in ("S1/Z2", "T2/Z2", "T4/Z2"):
        content = untwisted_massless_content(_all_standard()[name])
        assert content.graviton_sector + content.moduli == content.surviving
        assert content.surviving + content.projected_out == content.torus_states
        assert content.removed == content.projected_out


def test_kaluza_klein_vectors_are_projected_out():
    """The physical point: ``S^1/Z_2`` has no massless untwisted vectors."""
    content = untwisted_massless_content(_circle_z2())
    assert content.torus_states == 576
    assert content.surviving == 530
    assert content.projected_out == 46  # 2 x 23, the two U(1) gauge bosons
    assert content.moduli == 1  # the radius survives


def test_untwisted_content_refuses_a_non_reflection():
    with pytest.raises(ValueError):
        untwisted_massless_content(Orbifold.z3_hexagonal(CONV))


def test_rotation_orbifolds_keep_only_the_invariant_compact_pair():
    r"""For a rotation with eigenvalues ``lambda, lambda*`` only ``lambda lambda* = 1`` survives."""
    for name in ("T2/Z3", "T2/Z4", "T2/Z6"):
        # 22 non-compact transverse squared, plus the two mixed compact states
        assert untwisted_degeneracy(_all_standard()[name], 1, 1) == 22**2 + 2


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_projected_counts_are_non_negative_integers(level):
    """A sign or conjugation slip in the character sum breaks exactly this."""
    for orbifold in _all_standard().values():
        count = untwisted_degeneracy(orbifold, level, level)
        assert isinstance(count, int)
        assert count >= 0


def test_projection_never_increases_the_state_count():
    unprojected = oscillator_degeneracies(3, 24)
    for orbifold in _all_standard().values():
        for level in range(4):
            assert untwisted_degeneracy(orbifold, level, level) <= unprojected[level] ** 2


def test_untwisted_degeneracy_rejects_negative_levels():
    with pytest.raises(ValueError):
        untwisted_degeneracy(_circle_z2(), -1, 0)


# -- twisted sectors ---------------------------------------------------------


def test_twisted_ground_state_mass_and_multiplicity():
    orbifold = _circle_z2()
    ground = twisted_spectrum(orbifold, 1, n_levels=1)[0]
    assert ground.level == Fraction(0)
    assert ground.alpha_m2 == pytest.approx(4.0 * (0.0 - 15.0 / 16.0))
    assert ground.oscillator_states == 1
    assert ground.fixed_points == 2
    assert ground.degeneracy == 2
    assert ground.is_tachyonic


def test_twisted_levels_are_spaced_by_one_over_the_order():
    for orbifold in _all_standard().values():
        levels = twisted_spectrum(orbifold, 1, n_levels=2)
        denominators = {lv.level.denominator for lv in levels}
        assert all(orbifold.order % denominator == 0 for denominator in denominators)


def test_circle_z2_twisted_oscillator_counts():
    """Half-integer modes in one direction, ordinary ones in the other 23."""
    levels = {lv.level: lv.oscillator_states for lv in twisted_spectrum(_circle_z2(), 1, 2)}
    assert levels[Fraction(0)] == 1
    assert levels[Fraction(1, 2)] == 1  # the single twisted mode at 1/2
    assert levels[Fraction(1)] == 24  # 23 ordinary at n=1, plus two halves
    assert levels[Fraction(3, 2)] == 25


def test_twisted_masses_increase_with_level():
    for orbifold in _all_standard().values():
        levels = twisted_spectrum(orbifold, 1, n_levels=2)
        masses = [lv.alpha_m2 for lv in levels]
        assert masses == sorted(masses)


def test_the_bosonic_orbifolds_have_no_massless_twisted_states():
    """Honest negative result: ``a_k`` is never on the level lattice here."""
    for orbifold in _all_standard().values():
        for k in range(1, orbifold.order):
            if not orbifold.has_isolated_fixed_points(k):
                continue
            for level in twisted_spectrum(orbifold, k, n_levels=3):
                assert abs(level.alpha_m2) > 1e-9


def test_every_twisted_sector_starts_tachyonic():
    """The bosonic string's ground state stays tachyonic in every twisted sector."""
    for orbifold in _all_standard().values():
        for k in range(1, orbifold.order):
            if orbifold.has_isolated_fixed_points(k):
                assert twisted_spectrum(orbifold, k, n_levels=1)[0].is_tachyonic


def test_z4_second_sector_reproduces_the_z2_orbifold():
    """``theta^2`` of the ``Z_4`` is the ``Z_2`` inversion, with the same data."""
    z4 = Orbifold.z4_square(CONV)
    z2 = Orbifold.inversion(TorusBackground(np.eye(2), conventions=CONV))
    assert z4.intercept(2) == pytest.approx(z2.intercept(1))
    assert z4.fixed_points(2) == z2.fixed_points(1)
    a = twisted_spectrum(z4, 2, n_levels=2)
    b = twisted_spectrum(z2, 1, n_levels=2)
    assert [lv.level for lv in a] == [lv.level for lv in b]
    assert [lv.oscillator_states for lv in a] == [lv.oscillator_states for lv in b]


def test_conjugate_sectors_agree():
    """``theta^k`` and ``theta^{N-k}`` are complex conjugates, so their sectors match."""
    for orbifold in (Orbifold.z3_hexagonal(CONV), Orbifold.z4_square(CONV)):
        n = orbifold.order
        for k in range(1, n):
            if not orbifold.has_isolated_fixed_points(k):
                continue
            assert orbifold.intercept(k) == pytest.approx(orbifold.intercept(n - k))
            assert orbifold.fixed_points(k) == orbifold.fixed_points(n - k)


def test_twisted_spectrum_refuses_the_untwisted_sector_and_fixed_lines():
    orbifold = _circle_z2()
    with pytest.raises(ValueError):
        twisted_spectrum(orbifold, 0)
    partial = Orbifold(
        TorusBackground.from_radii([1.3, 2.1], CONV), np.diag([-1.0, 1.0])
    )
    with pytest.raises(ValueError):
        twisted_spectrum(partial, 1)


def test_transverse_phases_pad_with_the_untouched_directions():
    orbifold = _circle_z2()
    phases = transverse_phases(orbifold, 1)
    assert len(phases) == CONV.transverse_dim
    assert np.count_nonzero(phases) == 1
    assert phases[-1] == pytest.approx(0.5)


def test_summary_mentions_each_sector():
    text = massless_summary(Orbifold.z4_square(CONV))
    assert "T^2/Z_4" in text
    assert text.count("sector k=") == 3
    assert math.isfinite(Orbifold.z4_square(CONV).intercept(1))
