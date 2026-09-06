"""Veneziano and Virasoro-Shapiro: poles, residues, and high-energy behaviour."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.amplitudes.veneziano import (
    closed_mandelstam_sum,
    hard_scattering,
    regge_alpha,
    regge_asymptotic,
    veneziano,
    veneziano_log_abs,
    veneziano_pole_positions,
    veneziano_residue,
    virasoro_shapiro,
    virasoro_shapiro_pole_positions,
)
from stringsim.quantum.spectrum import closed_bosonic_spectrum, open_bosonic_spectrum
from stringsim.units import Conventions

CONV = Conventions(alpha_prime=1.0, dim=26)
# alpha(t) must stay away from the non-negative integers, or the amplitude has
# a pole in t too and the residue in s cannot be extracted.
T_SAMPLE = np.array([-0.7, 0.35, 0.9, 2.3])


def test_amplitude_is_symmetric_in_s_and_t():
    for s, t in ((-2.5, -1.5), (-0.3, -4.1), (-1.1, -1.1)):
        assert veneziano(s, t) == pytest.approx(veneziano(t, s))


def test_amplitude_matches_the_beta_function():
    """``A(-2.5, -1.5) = B(1.5, 0.5) = pi/2``, an independent closed form."""
    assert veneziano(-2.5, -1.5) == pytest.approx(math.pi / 2)


def test_poles_sit_exactly_on_the_open_string_masses():
    poles = veneziano_pole_positions(5, CONV.alpha_prime)
    levels = open_bosonic_spectrum(5, CONV)
    assert np.allclose(poles, [lv.alpha_m2 / CONV.alpha_prime for lv in levels])


@pytest.mark.parametrize("alpha_prime", [0.5, 1.0, 2.0])
def test_poles_track_alpha_prime(alpha_prime):
    poles = veneziano_pole_positions(3, alpha_prime)
    assert np.allclose(regge_alpha(poles, alpha_prime), np.arange(4))


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_numerical_residue_matches_the_closed_form(n):
    eps = 1e-7
    numeric = eps * veneziano((n - 1) + eps, T_SAMPLE, CONV.alpha_prime)
    exact = veneziano_residue(n, T_SAMPLE, CONV.alpha_prime)
    assert np.allclose(numeric, exact, rtol=1e-4)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_residue_is_a_polynomial_of_degree_n(n):
    """Degree ``n`` means spin at most ``n`` is exchanged at level ``n``."""
    t = np.linspace(-3.0, 3.0, 40)
    coeffs = np.polyfit(t, veneziano_residue(n, t, CONV.alpha_prime), n)
    assert abs(coeffs[0]) > 1e-8  # the leading coefficient is genuinely there
    residual = np.polyval(coeffs, t) - veneziano_residue(n, t, CONV.alpha_prime)
    assert np.max(np.abs(residual)) < 1e-8


def test_residue_rejects_a_negative_pole_index():
    with pytest.raises(ValueError):
        veneziano_residue(-1, 0.5)


def test_regge_limit_is_approached():
    """``|A| / |Gamma(-alpha_t)(-alpha_s)^{alpha_t}| -> 1`` at fixed ``t``."""
    t = -0.4
    ratios = [
        float(np.exp(veneziano_log_abs(s, t) - np.log(abs(regge_asymptotic(s, t)))))
        for s in (-50.0, -500.0, -5000.0, -50000.0)
    ]
    assert all(abs(r - 1) < 0.02 for r in ratios)
    # and the approach is monotone
    assert all(abs(ratios[i + 1] - 1) < abs(ratios[i] - 1) for i in range(len(ratios) - 1))


@pytest.mark.parametrize("ratio", [0.4, 1.0, 2.5])
def test_hard_scattering_falls_off_exponentially(ratio):
    """The measured slope matches Stirling, so the fall-off is exponential in ``s``."""
    hs = hard_scattering(np.linspace(-4000.0, -2000.0, 60), ratio=ratio)
    assert hs.log_slope == pytest.approx(hs.predicted_log_slope, rel=1e-3)
    assert hs.predicted_log_slope > 0  # decay as s -> -inf


def test_hard_scattering_validates_its_kinematics():
    with pytest.raises(ValueError):
        hard_scattering(np.linspace(1.0, 10.0, 5))
    with pytest.raises(ValueError):
        hard_scattering(np.linspace(-10.0, -1.0, 5), ratio=-1.0)


def test_log_abs_agrees_with_the_amplitude_where_both_are_finite():
    s = np.linspace(-6.0, -1.2, 25)
    t = -1.7
    assert np.allclose(np.log(np.abs(veneziano(s, t))), veneziano_log_abs(s, t), atol=1e-10)


def test_amplitude_is_finite_far_beyond_where_gamma_would_overflow():
    """``gamma(2001)`` overflows; the log form does not."""
    value = veneziano_log_abs(-2000.0, -2000.0)
    assert np.isfinite(value) and value < 0


# -- closed string -----------------------------------------------------------


def test_virasoro_shapiro_poles_match_the_closed_spectrum():
    poles = virasoro_shapiro_pole_positions(4, CONV.alpha_prime)
    levels = closed_bosonic_spectrum(4, CONV)
    assert np.allclose(poles, [lv.alpha_m2 / CONV.alpha_prime for lv in levels])


def test_closed_poles_are_four_times_as_widely_spaced_as_open_ones():
    open_gap = np.diff(veneziano_pole_positions(3, CONV.alpha_prime))
    closed_gap = np.diff(virasoro_shapiro_pole_positions(3, CONV.alpha_prime))
    assert np.allclose(closed_gap, 4 * open_gap)


def test_mandelstam_sum_for_four_closed_tachyons():
    assert closed_mandelstam_sum(1.0) == pytest.approx(-16.0)
    assert closed_mandelstam_sum(4.0) == pytest.approx(-4.0)


def test_virasoro_shapiro_has_a_genuine_pole_at_each_level():
    t = -2.3
    for s_pole in virasoro_shapiro_pole_positions(3, CONV.alpha_prime):
        eps = 1e-7
        residue = eps * virasoro_shapiro(s_pole + eps, t, alpha_prime=CONV.alpha_prime)
        assert abs(float(residue)) > 1e-3
        # halving eps must not change the residue: that is what "simple pole" means
        residue2 = 0.5 * eps * virasoro_shapiro(s_pole + 0.5 * eps, t, alpha_prime=CONV.alpha_prime)
        assert float(residue2) == pytest.approx(float(residue), rel=1e-4)


def test_virasoro_shapiro_is_symmetric_in_its_three_invariants():
    s, t = -5.0, -6.0
    u = closed_mandelstam_sum(1.0) - s - t
    base = virasoro_shapiro(s, t)
    assert virasoro_shapiro(t, s) == pytest.approx(base)
    assert virasoro_shapiro(s, u, t) == pytest.approx(base)
    assert virasoro_shapiro(u, t, s) == pytest.approx(base)


def test_a_pole_of_the_denominator_gives_a_clean_zero_not_a_nan():
    """``a + b = -1`` sends one gamma in the denominator to a pole."""
    value = virasoro_shapiro(-1.3, -2.7)
    assert value == 0.0
