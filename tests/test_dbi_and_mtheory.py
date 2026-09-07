"""The brane's own action, and the eleventh dimension.

Both halves are checkable two ways, which is why they are worth having.

The DBI determinant has a closed form in the literature; here the matrix is
built and ``numpy`` takes its determinant, and the closed form is asserted
against that. The Hamiltonian is written down in closed form *and* obtained by
a numerical Legendre transform of the Lagrangian, and the two must agree. The
BIon spike's energy per unit height is a numerical integral over the brane, and
it must equal the fundamental string tension times an integer that came from a
completely separate flux calculation.

The M-theory dictionary is nothing but consistency conditions: five reductions
whose right-hand sides come from ``dbrane.dp_brane_tension`` and the string
tension, neither of which knows about eleven dimensions, plus the Dirac
condition tying the two membrane tensions to Newton's constant.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.branes.dbi import (
    bion_charge,
    bion_flux,
    bion_profile,
    bion_tension,
    bogomolny_gap,
    critical_field,
    dbi_determinant,
    dbi_determinant_closed,
    dbi_matrix,
    displacement,
    electric_series,
    energy_density,
    lagrangian_density,
    reduced_displacement,
    sphere_area,
)
from stringsim.branes.dbrane import dp_brane_tension
from stringsim.branes.mtheory import (
    dirac_residual,
    gravitational_coupling,
    kaluza_klein_check,
    m2_tension,
    m5_tension,
    m_theory_radius,
    planck_length,
    reduction_table,
    string_coupling_from_radius,
)
from stringsim.units import Conventions

CONV = Conventions()
OTHER = Conventions(alpha_prime=0.37)
G_S = 0.4
TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------
# the determinant
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
def test_the_closed_form_matches_the_determinant(p):
    rng = np.random.default_rng(11 + p)
    for _ in range(120):
        grad = rng.normal(size=p) * 0.8
        field = rng.normal(size=p) * 0.05
        assert dbi_determinant(grad, field, CONV.alpha_prime) == pytest.approx(
            dbi_determinant_closed(grad, field, CONV.alpha_prime), abs=1e-12
        )


def test_the_matrix_has_the_right_shape_and_signature():
    matrix = dbi_matrix(np.zeros(3), np.zeros(3), CONV.alpha_prime)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, np.diag([-1.0, 1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="must match"):
        dbi_matrix(np.zeros(3), np.zeros(2))


def test_a_flat_brane_with_no_field_is_just_its_tension():
    value = lagrangian_density(np.zeros(3), np.zeros(3), 3, G_S, CONV)
    assert value == pytest.approx(-dp_brane_tension(3, G_S, CONV))


def test_the_gauge_field_only_enters_through_two_pi_alpha_prime():
    """Doubling alpha' and halving E leaves the determinant alone."""
    grad, field = np.array([0.3, -0.2]), np.array([0.01, 0.04])
    first = dbi_determinant(grad, field, 1.0)
    second = dbi_determinant(grad, field / 2.0, 2.0)
    assert first == pytest.approx(second)


# --------------------------------------------------------------------------
# the critical field
# --------------------------------------------------------------------------


def test_the_critical_field_is_the_string_tension():
    for conv in (CONV, OTHER):
        assert critical_field(conv) == pytest.approx(1.0 / (TWO_PI * conv.alpha_prime))


def test_the_determinant_vanishes_at_the_critical_field():
    crit = critical_field(CONV)
    assert dbi_determinant(np.zeros(3), np.array([crit, 0.0, 0.0]), CONV.alpha_prime) == (
        pytest.approx(0.0, abs=1e-12)
    )
    with pytest.raises(ValueError, match="critical"):
        lagrangian_density(np.zeros(3), np.array([crit, 0.0, 0.0]), 3, G_S, CONV)
    with pytest.raises(ValueError, match="critical"):
        lagrangian_density(np.zeros(3), np.array([1.5 * crit, 0.0, 0.0]), 3, G_S, CONV)


def test_the_expansion_is_the_binomial_series():
    """Measured from the function by Cauchy's formula, not written down."""
    measured = electric_series(8)
    # sqrt(1 - x) = sum_k C(1/2, k) (-x)^k, so the signs alternate into the
    # binomial recursion and every term after the first is negative.
    exact = [1.0]
    for k in range(1, 8):
        exact.append(-exact[-1] * (0.5 - (k - 1)) / k)
    assert np.allclose(measured, exact, atol=1e-12)
    assert measured[0] == pytest.approx(1.0)
    assert measured[1] == pytest.approx(-0.5)
    with pytest.raises(ValueError, match="at least two"):
        electric_series(1)


# --------------------------------------------------------------------------
# the Legendre transform and the BPS bound
# --------------------------------------------------------------------------


def test_the_hamiltonian_is_the_legendre_transform():
    """The closed form and the numerical transform have to be the same function."""
    rng = np.random.default_rng(4)
    for p in (2, 3, 4):
        for _ in range(60):
            grad = rng.normal(size=p) * 0.6
            field = rng.normal(size=p) * 0.02
            conjugate = displacement(grad, field, p, G_S, CONV)
            reduced = reduced_displacement(grad, field, p, G_S, CONV)
            transformed = conjugate @ field - lagrangian_density(grad, field, p, G_S, CONV)
            assert transformed == pytest.approx(
                energy_density(grad, reduced, p, G_S, CONV), rel=1e-8
            )


def test_the_reduced_displacement_is_the_raw_one_scaled():
    grad, field = np.array([0.4, 0.1, -0.3]), np.array([0.01, 0.0, 0.02])
    scale = TWO_PI * CONV.alpha_prime * dp_brane_tension(3, G_S, CONV)
    assert np.allclose(
        reduced_displacement(grad, field, 3, G_S, CONV),
        displacement(grad, field, 3, G_S, CONV) / scale,
    )


def test_the_bogomolny_gap_is_non_negative_and_vanishes_only_at_bps():
    rng = np.random.default_rng(9)
    for p in (2, 3, 4):
        for _ in range(200):
            grad = rng.normal(size=p) * 0.8
            disp = rng.normal(size=p) * 0.8
            assert bogomolny_gap(grad, disp, p, G_S, CONV) >= -1e-12
        grad = rng.normal(size=p) * 0.8
        assert bogomolny_gap(grad, grad, p, G_S, CONV) == pytest.approx(0.0, abs=1e-12)


def test_at_bps_the_energy_is_the_brane_plus_the_string():
    grad = np.array([0.5, -0.2, 0.7])
    tension = dp_brane_tension(3, G_S, CONV)
    assert energy_density(grad, grad, 3, G_S, CONV) == pytest.approx(
        tension * (1.0 + grad @ grad)
    )


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="shapes must match"):
        energy_density(np.zeros(3), np.zeros(2), 3, G_S, CONV)


# --------------------------------------------------------------------------
# the spike
# --------------------------------------------------------------------------


def test_sphere_areas():
    assert sphere_area(1) == pytest.approx(2.0)
    assert sphere_area(2) == pytest.approx(TWO_PI)
    assert sphere_area(3) == pytest.approx(4.0 * math.pi)
    assert sphere_area(4) == pytest.approx(2.0 * math.pi**2)
    with pytest.raises(ValueError, match="at least 1"):
        sphere_area(0)


@pytest.mark.parametrize("p", [3, 4, 5])
@pytest.mark.parametrize("n", [1, 2, 5])
def test_the_flux_counts_the_strings(p, n):
    """Quantised, and independent of where the sphere is drawn."""
    for radius in (0.05, 0.4, 3.0):
        assert bion_flux(radius, n, p, G_S, CONV) == pytest.approx(float(n), abs=1e-9)


@pytest.mark.parametrize("p", [3, 4, 5])
@pytest.mark.parametrize("n", [1, 2, 5])
def test_the_spike_weighs_what_the_strings_weigh(p, n):
    """A numerical integral over the brane against the string tension."""
    expected = n / (TWO_PI * CONV.alpha_prime)
    assert bion_tension(n, p, G_S, CONV) == pytest.approx(expected, rel=1e-7)


def test_the_answer_does_not_depend_on_the_cutoff():
    first = bion_tension(2, 3, G_S, CONV, inner=1e-2)
    second = bion_tension(2, 3, G_S, CONV, inner=1e-4)
    assert first == pytest.approx(second, rel=1e-9)


def test_the_spike_tension_does_not_depend_on_the_coupling():
    """More charge is needed at weak coupling, and the two effects cancel."""
    weak = bion_tension(1, 3, 0.05, CONV)
    strong = bion_tension(1, 3, 2.0, CONV)
    assert weak == pytest.approx(strong, rel=1e-9)
    assert bion_charge(1, 3, 0.05, CONV) < bion_charge(1, 3, 2.0, CONV)


def test_the_profile_is_harmonic_and_scales_with_n():
    radii = np.array([0.2, 0.5, 1.0])
    one = bion_profile(radii, 1, 3, G_S, CONV)
    three = bion_profile(radii, 3, 3, G_S, CONV)
    assert np.allclose(three, 3.0 * one)
    assert np.allclose(one * radii, one[0] * radii[0])  # X = q/r
    assert bion_profile(radii, 0, 3, G_S, CONV).tolist() == [0.0, 0.0, 0.0]


def test_the_spike_is_bps_where_it_matters():
    """D = grad X by construction, so the gap vanishes along the whole profile."""
    for radius in (0.1, 0.5, 2.0):
        charge = bion_charge(1, 3, G_S, CONV)
        slope = np.array([-charge / radius**2, 0.0, 0.0])
        assert bogomolny_gap(slope, slope, 3, G_S, CONV) == pytest.approx(0.0, abs=1e-12)


def test_low_dimensional_spikes_are_refused():
    with pytest.raises(ValueError, match="logarithm"):
        bion_charge(1, 2, G_S, CONV)
    with pytest.raises(ValueError, match="logarithm"):
        bion_profile([1.0], 1, 2, G_S, CONV)
    with pytest.raises(ValueError, match="radii must be positive"):
        bion_profile([0.0], 1, 3, G_S, CONV)


# --------------------------------------------------------------------------
# eleven dimensions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("g_s", [0.02, 0.4, 1.0, 5.0])
@pytest.mark.parametrize("conv", [CONV, OTHER])
def test_the_d0_brane_is_a_kaluza_klein_mode(g_s, conv):
    """The two numbers come from unrelated formulas and must be the same."""
    assert kaluza_klein_check(g_s, conv) == pytest.approx(0.0, abs=1e-14)
    assert dp_brane_tension(0, g_s, conv) == pytest.approx(1.0 / m_theory_radius(g_s, conv))


@pytest.mark.parametrize("g_s", [0.02, 0.4, 1.0, 5.0])
@pytest.mark.parametrize("conv", [CONV, OTHER])
def test_every_reduction_lands_on_the_ten_dimensional_answer(g_s, conv):
    table = reduction_table(g_s, conv)
    assert [row.result for row in table] == ["F1", "D2", "D4", "NS5", "D0"]
    for row in table:
        assert row.residual < 1e-13, row


@pytest.mark.parametrize("g_s", [0.02, 0.4, 1.0, 5.0])
@pytest.mark.parametrize("conv", [CONV, OTHER])
def test_the_dirac_condition_holds(g_s, conv):
    assert dirac_residual(g_s, conv) == pytest.approx(0.0, abs=1e-12)
    product = gravitational_coupling(g_s, conv) * m2_tension(g_s, conv) * m5_tension(g_s, conv)
    assert product == pytest.approx(TWO_PI)


def test_the_ns5_is_not_a_d_brane():
    """Its tension goes like 1/g_s^2, which no dp_brane_tension does."""
    ratios = []
    for g_s in (0.1, 0.2, 0.4):
        ns5 = next(row for row in reduction_table(g_s, CONV) if row.result == "NS5")
        ratios.append(ns5.tension)
    assert ratios[0] / ratios[1] == pytest.approx(4.0)
    assert ratios[1] / ratios[2] == pytest.approx(4.0)
    d5 = [dp_brane_tension(5, g_s, CONV) for g_s in (0.1, 0.2)]
    assert d5[0] / d5[1] == pytest.approx(2.0)


def test_the_circle_grows_with_the_coupling():
    radii = [m_theory_radius(g_s, CONV) for g_s in (0.01, 0.1, 1.0, 10.0)]
    assert radii == sorted(radii)
    assert string_coupling_from_radius(m_theory_radius(0.7, CONV), CONV) == pytest.approx(0.7)


def test_planck_length_relations():
    for g_s in (0.05, 1.0, 3.0):
        for conv in (CONV, OTHER):
            length = planck_length(g_s, conv)
            assert length**3 == pytest.approx(g_s * conv.alpha_prime**1.5)
            assert m2_tension(g_s, conv) == pytest.approx(1.0 / ((TWO_PI) ** 2 * length**3))
            assert m5_tension(g_s, conv) == pytest.approx(1.0 / ((TWO_PI) ** 5 * length**6))


def test_bad_couplings_are_refused():
    for function in (m_theory_radius, planck_length, m2_tension, m5_tension):
        with pytest.raises(ValueError, match="positive"):
            function(-1.0, CONV)
    with pytest.raises(ValueError, match="positive"):
        string_coupling_from_radius(0.0, CONV)
