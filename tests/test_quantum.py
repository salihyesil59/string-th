"""The quantised spectrum: counting, regularisation, and particle content."""

from __future__ import annotations

import cmath
import math

import pytest

from stringsim.quantum.partition import (
    dedekind_eta,
    fit_hagedorn,
    hagedorn_temperature,
    jacobi_identity_residual,
    oscillator_degeneracies,
    superstring_degeneracies,
    theta_series,
)
from stringsim.quantum.spectrum import (
    closed_bosonic_spectrum,
    leading_trajectory_spin,
    open_bosonic_spectrum,
    open_superstring_spectrum,
)
from stringsim.quantum.states import (
    antisym_dim,
    closed_massless_content,
    little_group,
    open_level_content,
    sym_traceless_dim,
)
from stringsim.quantum.zeta import (
    central_charge,
    critical_dimension,
    normal_ordering_constant,
    regularised_sum,
)
from stringsim.units import Conventions

CONV = Conventions(alpha_prime=1.0, dim=26)


# -- degeneracies ------------------------------------------------------------


def test_known_open_string_degeneracies():
    """``1, 24, 324, 3200, 25650, ...`` -- the standard table in ``D = 26``."""
    assert oscillator_degeneracies(8, 24) == [
        1, 24, 324, 3200, 25650, 176256, 1073720, 5930496, 30178575,
    ]


def test_single_species_degeneracies_are_the_partition_numbers():
    """One oscillator tower counts integer partitions."""
    assert oscillator_degeneracies(10, 1) == [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42]


def test_degeneracies_are_exact_integers():
    degen = oscillator_degeneracies(80, 24)
    assert all(isinstance(d, int) for d in degen)
    assert len(str(degen[80])) > 25  # far past the reach of float64's mantissa


def test_degeneracies_reject_bad_input():
    with pytest.raises(ValueError):
        oscillator_degeneracies(-1, 24)
    with pytest.raises(ValueError):
        oscillator_degeneracies(4, 0)


# -- modular functions -------------------------------------------------------


def test_jacobi_abstruse_identity_holds_exactly():
    """``theta_3^4 = theta_2^4 + theta_4^4``, in integer arithmetic.

    Physically: the GSO-projected NS and R sectors have equally many states at
    every level, so the one-loop vacuum amplitude vanishes.
    """
    assert all(c == 0 for c in jacobi_identity_residual(80))


def test_theta_series_have_the_right_leading_terms():
    assert theta_series(3, 9)[:10] == [1, 2, 0, 0, 2, 0, 0, 0, 0, 2]
    assert theta_series(4, 9)[:10] == [1, -2, 0, 0, 2, 0, 0, 0, 0, -2]
    assert theta_series(2, 9)[:7] == [1, 0, 1, 0, 0, 0, 1]
    with pytest.raises(ValueError):
        theta_series(5, 4)


def test_dedekind_eta_is_modular():
    """``eta(-1/tau) = sqrt(-i tau) eta(tau)``, which makes the torus amplitude finite."""
    for tau in (0.31 + 0.87j, -0.2 + 1.4j, 0.5 + 0.6j):
        lhs = dedekind_eta(-1.0 / tau)
        rhs = cmath.sqrt(-1j * tau) * dedekind_eta(tau)
        assert abs(lhs - rhs) < 1e-13


def test_dedekind_eta_is_periodic_under_tau_shift():
    """``eta(tau + 1) = e^{i pi / 12} eta(tau)``."""
    tau = 0.23 + 1.1j
    assert abs(dedekind_eta(tau + 1) - cmath.exp(1j * cmath.pi / 12) * dedekind_eta(tau)) < 1e-13


def test_dedekind_eta_needs_the_upper_half_plane():
    with pytest.raises(ValueError):
        dedekind_eta(0.3 - 0.2j)


def test_eta_reproduces_the_degeneracy_generating_function():
    """``q^{1/24}/eta(tau) = sum_N p(N) q^N`` for a single species."""
    q = 0.07
    tau = cmath.log(q) / (2j * cmath.pi)
    series = sum(d * q**n for n, d in enumerate(oscillator_degeneracies(60, 1)))
    assert abs(series - q ** (1 / 24) / dedekind_eta(tau)) < 1e-12


def test_superstring_degeneracies_match_the_standard_table():
    assert superstring_degeneracies(6) == [8, 128, 1152, 7680, 42112, 200448, 855552]


# -- zeta regularisation and the critical dimension --------------------------


def test_regularised_sum_gives_minus_one_twelfth():
    est = regularised_sum()
    assert est.error < 1e-9


def test_critical_dimensions():
    assert critical_dimension("bosonic") == 26
    assert critical_dimension("superstring") == 10
    with pytest.raises(ValueError):
        critical_dimension("membrane")


def test_normal_ordering_constant_at_the_critical_dimension():
    assert normal_ordering_constant(26, "bosonic") == pytest.approx(1.0)
    assert normal_ordering_constant(10, "superstring") == pytest.approx(0.5)
    with pytest.raises(ValueError):
        normal_ordering_constant(26, "membrane")


def test_central_charge_vanishes_at_the_critical_dimension():
    """The anomaly route and the normal-ordering route give the same ``D``."""
    for theory in ("bosonic", "superstring"):
        assert central_charge(critical_dimension(theory), theory) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        central_charge(26, "membrane")


# -- spectra -----------------------------------------------------------------


def test_open_bosonic_spectrum_masses_and_counts():
    levels = open_bosonic_spectrum(4, CONV)
    assert [lv.alpha_m2 for lv in levels] == [-1.0, 0.0, 1.0, 2.0, 3.0]
    assert [lv.degeneracy for lv in levels] == [1, 24, 324, 3200, 25650]
    assert levels[0].is_tachyonic
    assert levels[1].is_massless
    assert not levels[2].is_massless


def test_closed_spectrum_is_the_square_of_the_open_counting():
    open_levels = open_bosonic_spectrum(3, CONV)
    closed_levels = closed_bosonic_spectrum(3, CONV)
    for o, c in zip(open_levels, closed_levels, strict=True):
        assert c.degeneracy == o.degeneracy**2
        assert c.alpha_m2 == pytest.approx(4 * o.alpha_m2)


def test_superstring_has_no_tachyon():
    levels = open_superstring_spectrum(4, Conventions(alpha_prime=1.0, dim=10))
    assert all(not lv.is_tachyonic for lv in levels)
    assert levels[0].is_massless and levels[0].degeneracy == 8


def test_mass_scales_with_alpha_prime():
    for ap in (0.25, 1.0, 4.0):
        levels = open_bosonic_spectrum(3, Conventions(alpha_prime=ap, dim=26))
        for lv in levels:
            assert lv.mass_squared == pytest.approx(lv.alpha_m2 / ap)


def test_leading_trajectory_matches_the_max_spin():
    for lv in open_bosonic_spectrum(5, CONV):
        assert leading_trajectory_spin(lv.alpha_m2) == pytest.approx(lv.max_spin)


# -- particle content --------------------------------------------------------


def test_representation_dimensions():
    assert sym_traceless_dim(25, 2) == 324
    assert sym_traceless_dim(25, 3) == 2900
    assert sym_traceless_dim(24, 2) == 299
    assert sym_traceless_dim(7, 1) == 7
    assert sym_traceless_dim(7, 0) == 1
    assert antisym_dim(24, 2) == 276
    assert antisym_dim(25, 2) == 300


def test_representation_dimensions_reject_bad_rank():
    with pytest.raises(ValueError):
        sym_traceless_dim(10, -1)
    with pytest.raises(ValueError):
        antisym_dim(4, 5)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_named_open_content_adds_up_to_the_degeneracy(level):
    """The named irreps must exhaust the counted states -- nothing left over."""
    counted = open_bosonic_spectrum(level, CONV)[level].degeneracy
    named = sum(p.dimension for p in open_level_content(level, CONV.dim))
    assert named == counted


def test_open_content_stops_where_it_stops_being_derived():
    with pytest.raises(NotImplementedError):
        open_level_content(4, CONV.dim)


def test_closed_massless_level_is_graviton_plus_b_field_plus_dilaton():
    pieces = closed_massless_content(CONV.dim)
    names = {p.name: p.dimension for p in pieces}
    assert names == {"graviton": 299, "Kalb-Ramond": 276, "dilaton": 1}
    assert sum(names.values()) == (CONV.dim - 2) ** 2
    assert closed_bosonic_spectrum(1, CONV)[1].degeneracy == sum(names.values())
    graviton = next(p for p in pieces if p.name == "graviton")
    assert graviton.spin == "2"


def test_closed_massless_content_tracks_the_dimension():
    for dim in (6, 10, 26):
        assert sum(p.dimension for p in closed_massless_content(dim)) == (dim - 2) ** 2


def test_little_group_distinguishes_massless_massive_and_tachyonic():
    assert little_group(0.0, 26) == ("SO(24)", 24)
    assert little_group(3.0, 26) == ("SO(25)", 25)
    assert little_group(-1.0, 26) == ("SO(24,1)", 25)


# -- Hagedorn ----------------------------------------------------------------


def test_hagedorn_slope_approaches_four_pi():
    fit = fit_hagedorn(n_max=300, n_species=24, n_fit=120)
    assert fit.predicted_beta == pytest.approx(4 * math.pi)
    assert fit.beta_hagedorn == pytest.approx(fit.predicted_beta, rel=5e-3)
    assert fit.temperature == pytest.approx(1 / fit.beta_hagedorn)
    assert fit.residual < 1e-3


def test_hagedorn_prediction_scales_with_alpha_prime():
    """``beta_H = 2 pi sqrt(c alpha'/6)`` -- a longer string is harder to heat."""
    for ap in (0.25, 1.0, 4.0):
        assert hagedorn_temperature(24, ap) == pytest.approx(4 * math.pi * math.sqrt(ap))


def test_hagedorn_fit_validates_its_window():
    with pytest.raises(ValueError):
        fit_hagedorn(n_max=50, n_fit=60)
    with pytest.raises(ValueError):
        fit_hagedorn(n_max=50, n_fit=2)
