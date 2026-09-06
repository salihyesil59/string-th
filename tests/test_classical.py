"""Classical worldsheet dynamics.

The discipline here is the same one the rest of the package follows: every
claim that can be checked by two independent routes is checked that way.  The
mode expansion is compared against a finite-difference solution of the same
equation; the light-cone mass formula against the momentum integral; the
rotating string's Regge slope against the Noether charges.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.classical.constraints import (
    angular_momentum,
    total_momentum,
    virasoro_residual,
)
from stringsim.classical.evolve import evolve, mode_spectrum
from stringsim.classical.lightcone import LightconeOpenString, oscillator_amplitudes
from stringsim.classical.modes import ClosedString, OpenString
from stringsim.classical.rotating import regge_trajectory, rigid_rotator
from stringsim.units import Conventions, dot

CONV = Conventions(alpha_prime=1.0, dim=26)


# -- mode expansions ---------------------------------------------------------


def test_open_string_obeys_the_wave_equation():
    rng = np.random.default_rng(0)
    alphas = {n: rng.normal(size=CONV.dim) + 1j * rng.normal(size=CONV.dim) for n in (1, 2, 4)}
    s = OpenString(conventions=CONV, p=rng.normal(size=CONV.dim), alphas=alphas)

    tau = np.linspace(0.3, 1.1, 9)[:, None]
    sigma = np.linspace(0.2, math.pi - 0.2, 11)[None, :]
    h = 1e-4
    d2_tau = (s.position(tau + h, sigma) - 2 * s.position(tau, sigma) + s.position(tau - h, sigma))
    d2_sigma = (
        s.position(tau, sigma + h) - 2 * s.position(tau, sigma) + s.position(tau, sigma - h)
    )
    assert np.max(np.abs(d2_tau - d2_sigma)) / h**2 < 1e-4


def test_open_string_derivatives_match_finite_differences():
    rng = np.random.default_rng(1)
    s = OpenString(
        conventions=CONV,
        p=rng.normal(size=CONV.dim),
        alphas={1: rng.normal(size=CONV.dim) + 1j * rng.normal(size=CONV.dim)},
    )
    tau, sigma, h = 0.7, 1.3, 1e-6
    v = (s.position(tau + h, sigma) - s.position(tau - h, sigma)) / (2 * h)
    sl = (s.position(tau, sigma + h) - s.position(tau, sigma - h)) / (2 * h)
    assert np.allclose(v, s.velocity(tau, sigma), atol=1e-7)
    assert np.allclose(sl, s.slope(tau, sigma), atol=1e-7)


def test_neumann_boundary_conditions_hold():
    rng = np.random.default_rng(2)
    s = OpenString(
        conventions=CONV,
        alphas={n: rng.normal(size=CONV.dim) + 1j * rng.normal(size=CONV.dim) for n in (1, 3)},
    )
    tau = np.linspace(0, 2 * math.pi, 17)
    for end in (0.0, math.pi):
        assert np.max(np.abs(s.slope(tau, np.full_like(tau, end)))) < 1e-12


def test_closed_string_is_periodic():
    rng = np.random.default_rng(3)
    s = ClosedString(
        conventions=CONV,
        p=rng.normal(size=CONV.dim),
        alphas={1: rng.normal(size=CONV.dim) + 1j * rng.normal(size=CONV.dim)},
        alphas_tilde={2: rng.normal(size=CONV.dim) + 1j * rng.normal(size=CONV.dim)},
    )
    tau = np.linspace(0, 3, 11)
    a = s.position(tau, np.zeros_like(tau))
    b = s.position(tau, np.full_like(tau, 2 * math.pi))
    assert np.allclose(a, b, atol=1e-12)


def test_mode_amplitudes_must_have_the_right_shape():
    with pytest.raises(ValueError):
        OpenString(conventions=CONV, alphas={1: np.zeros(3)})
    with pytest.raises(ValueError):
        OpenString(conventions=CONV, alphas={0: np.zeros(CONV.dim)})


# -- light-cone gauge --------------------------------------------------------


EXCITATIONS = [
    {(1, 0): 1},
    {(1, 0): 1, (1, 1): 1},
    {(2, 0): 1},
    {(1, 0): 2, (3, 5): 1},
    {(1, 0): 1, (2, 1): 1, (3, 2): 1},
]


@pytest.mark.parametrize("excitation", EXCITATIONS)
def test_lightcone_solves_the_virasoro_constraints(excitation):
    amps = oscillator_amplitudes(excitation, CONV.transverse_dim, {(1, 1): math.pi / 2})
    lc = LightconeOpenString(conventions=CONV, p_plus=1.3, transverse_modes=amps)
    report = virasoro_residual(lc.to_open_string(), n_tau=17, n_sigma=23)
    assert report.satisfied(1e-11), str(report)


@pytest.mark.parametrize("excitation", EXCITATIONS)
def test_lightcone_mass_equals_the_level(excitation):
    """``alpha' M^2 = N`` classically, with ``M^2`` taken from ``2p^+p^- - p^i p^i``."""
    amps = oscillator_amplitudes(excitation, CONV.transverse_dim)
    lc = LightconeOpenString(conventions=CONV, p_plus=0.8, transverse_modes=amps)
    level = sum(n * k for (n, _), k in excitation.items())
    assert lc.level() == pytest.approx(level)
    assert CONV.alpha_prime * lc.mass_squared() == pytest.approx(level, rel=1e-12)
    assert lc.mass_squared_quantum() == pytest.approx((level - 1) / CONV.alpha_prime)


def test_lightcone_mass_is_independent_of_p_plus():
    """``p^+`` is a boost choice, so the invariant mass must not depend on it."""
    amps = oscillator_amplitudes({(1, 0): 1, (2, 3): 1}, CONV.transverse_dim)
    masses = [
        LightconeOpenString(conventions=CONV, p_plus=pp, transverse_modes=amps).mass_squared()
        for pp in (0.25, 1.0, 7.5)
    ]
    assert masses[0] == pytest.approx(masses[1]) == pytest.approx(masses[2])


def test_lightcone_mass_survives_transverse_boosts():
    amps = oscillator_amplitudes({(2, 0): 1}, CONV.transverse_dim)
    rest = LightconeOpenString(conventions=CONV, p_plus=1.0, transverse_modes=amps)
    moving = LightconeOpenString(
        conventions=CONV,
        p_plus=1.0,
        p_transverse=np.eye(CONV.transverse_dim)[4] * 1.7,
        transverse_modes=amps,
    )
    assert moving.mass_squared() == pytest.approx(rest.mass_squared())


def test_alpha_minus_is_real_analytic_continuation():
    """``alpha_{-n}^- = conj(alpha_n^-)``, required for ``X^-`` to be real."""
    amps = oscillator_amplitudes({(1, 0): 1, (2, 1): 1}, CONV.transverse_dim, {(2, 1): 0.9})
    lc = LightconeOpenString(conventions=CONV, p_plus=1.1, transverse_modes=amps)
    for n in (1, 2, 3, 4):
        assert lc.alpha_minus(-n) == pytest.approx(np.conj(lc.alpha_minus(n)))


def test_lightcone_requires_nonzero_p_plus():
    with pytest.raises(ValueError):
        LightconeOpenString(conventions=CONV, p_plus=0.0)


def test_oscillator_amplitudes_rejects_bad_input():
    with pytest.raises(ValueError):
        oscillator_amplitudes({(0, 0): 1}, CONV.transverse_dim)
    with pytest.raises(ValueError):
        oscillator_amplitudes({(1, 99): 1}, CONV.transverse_dim)
    with pytest.raises(ValueError):
        oscillator_amplitudes({(1, 0): 0}, CONV.transverse_dim)


# -- conserved charges -------------------------------------------------------


def test_momentum_integral_reproduces_p():
    amps = oscillator_amplitudes({(1, 0): 1, (2, 2): 1}, CONV.transverse_dim)
    lc = LightconeOpenString(conventions=CONV, p_plus=1.4, transverse_modes=amps)
    full = lc.to_open_string()
    assert np.allclose(total_momentum(full), lc.momentum(), atol=1e-10)


def test_momentum_is_conserved_in_tau():
    amps = oscillator_amplitudes({(1, 0): 1, (3, 1): 1}, CONV.transverse_dim)
    full = LightconeOpenString(
        conventions=CONV, p_plus=1.0, transverse_modes=amps
    ).to_open_string()
    p0 = total_momentum(full, tau=0.0)
    p1 = total_momentum(full, tau=1.7)
    assert np.allclose(p0, p1, atol=1e-9)


# -- the rotating string -----------------------------------------------------


@pytest.mark.parametrize("alpha_prime", [0.5, 1.0, 2.0])
def test_rotating_string_lies_on_the_regge_trajectory(alpha_prime):
    conv = Conventions(alpha_prime=alpha_prime, dim=26)
    for point in regge_trajectory(conventions=conv):
        assert point.spin == pytest.approx(point.alpha_m2, rel=1e-9, abs=1e-12)


def test_rotating_string_satisfies_the_constraints():
    report = virasoro_residual(rigid_rotator(CONV, 1.7))
    assert report.satisfied(1e-12), str(report)


def test_rotating_endpoints_move_at_the_speed_of_light():
    s = rigid_rotator(CONV, 2.3)
    tau = np.linspace(0.0, 2.0, 4001)
    X = s.position(tau, np.zeros_like(tau))
    dX = np.gradient(X, tau, axis=0)
    speed = np.linalg.norm(dX[:, 1:], axis=1) / np.abs(dX[:, 0])
    assert speed.max() == pytest.approx(1.0, abs=1e-4)
    assert speed.min() == pytest.approx(1.0, abs=1e-4)


def test_rotating_string_energy_and_spin_match_closed_forms():
    a, conv = 2.0, CONV
    s = rigid_rotator(conv, a)
    p = total_momentum(s)
    j = angular_momentum(s)
    assert p[0] == pytest.approx(a / (2 * conv.alpha_prime))
    assert j[1, 2] == pytest.approx(a**2 / (4 * conv.alpha_prime))
    assert j[1, 2] == pytest.approx(-j[2, 1])


def test_rotating_string_is_the_level_one_state():
    """The rotator is what ``oscillator_amplitudes`` builds for ``alpha_{-1}^1 alpha_{-1}^2``."""
    conv = Conventions(alpha_prime=1.0, dim=26)
    amps = oscillator_amplitudes(
        {(1, 0): 1, (1, 1): 1}, conv.transverse_dim, {(1, 1): math.pi / 2}
    )
    lc = LightconeOpenString(conventions=conv, p_plus=1.0, transverse_modes=amps)
    # Same invariant mass: level 2 in both descriptions.  M = A/(2 alpha') inverts
    # to A = 2 alpha' M.
    a = 2 * conv.alpha_prime * math.sqrt(lc.mass_squared())
    rot = rigid_rotator(conv, a)
    assert -dot(rot.p, rot.p) == pytest.approx(lc.mass_squared(), rel=1e-12)


def test_rotating_string_rejects_a_nonpositive_amplitude():
    with pytest.raises(ValueError):
        rigid_rotator(CONV, 0.0)


# -- finite-difference evolution ---------------------------------------------


def _analytic(tau, sigma, coefficients):
    return sum(a * np.cos(n * tau) * np.cos(n * sigma) for n, a in coefficients.items())


def test_evolver_is_second_order_accurate():
    coefficients = {1: 0.7, 3: -0.25}
    errors = []
    for n_cells in (64, 128, 256):
        sigma = np.linspace(0.0, math.pi, n_cells + 1)
        X0 = _analytic(0.0, sigma, coefficients)[:, None]
        ev = evolve(X0, boundary="neumann", n_steps=int(2.0 / (0.5 * math.pi / n_cells)))
        exact = _analytic(ev.tau[:, None], sigma[None, :], coefficients)
        errors.append(np.max(np.abs(ev.X[:, :, 0] - exact)))
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(3.6 < r < 4.4 for r in ratios), ratios


def test_courant_one_is_exact_for_the_wave_equation():
    """At ``dtau = dsigma`` the scheme is d'Alembert's solution, exact to round-off."""
    coefficients = {2: 0.4}
    n_cells = 128
    sigma = np.linspace(0.0, math.pi, n_cells + 1)
    X0 = _analytic(0.0, sigma, coefficients)[:, None]
    ev = evolve(X0, boundary="neumann", n_steps=200, courant=1.0)
    exact = _analytic(ev.tau[:, None], sigma[None, :], coefficients)
    assert np.max(np.abs(ev.X[:, :, 0] - exact)) < 1e-12


def test_dirichlet_ends_stay_put():
    n_cells = 128
    sigma = np.linspace(0.0, math.pi, n_cells + 1)
    X0 = np.column_stack([np.sin(sigma), 0.3 * np.sin(2 * sigma)])
    ev = evolve(X0, boundary="dirichlet", n_steps=300)
    assert np.max(np.abs(ev.X[:, 0, :] - X0[0])) < 1e-14
    assert np.max(np.abs(ev.X[:, -1, :] - X0[-1])) < 1e-14


def test_periodic_evolution_stays_periodic():
    n_cells = 128
    sigma = np.arange(n_cells) * (2 * math.pi / n_cells)
    X0 = np.column_stack([np.cos(sigma), np.sin(2 * sigma)])
    ev = evolve(X0, boundary="periodic", sigma_max=2 * math.pi, n_steps=250)
    # A closed string's centre of mass does not move when it starts at rest.
    com = ev.X.mean(axis=1)
    assert np.max(np.abs(com - com[0])) < 1e-12


def test_evolver_rejects_bad_arguments():
    X0 = np.zeros((16, 2))
    with pytest.raises(ValueError):
        evolve(X0, boundary="wibble")
    with pytest.raises(ValueError):
        evolve(X0, courant=1.5)
    with pytest.raises(ValueError):
        evolve(np.zeros(16))


def test_mode_spectrum_recovers_a_pure_harmonic():
    sigma = np.linspace(0.0, math.pi, 513)
    X = np.column_stack([1.4 * np.cos(3 * sigma)])
    coeffs = mode_spectrum(X, boundary="neumann", n_modes=6)[:, 0]
    assert coeffs[3] == pytest.approx(1.4, abs=1e-6)
    assert np.max(np.abs(np.delete(coeffs, 3))) < 1e-6


def test_triangular_pluck_has_the_classical_harmonics():
    """A midpoint pluck keeps only ``n = 2, 6, 10, ...`` and falls off like ``1/n^2``."""
    sigma = np.linspace(0.0, math.pi, 2049)
    triangle = np.where(sigma < math.pi / 2, sigma, math.pi - sigma)[:, None]
    coeffs = mode_spectrum(triangle, boundary="neumann", n_modes=12)[:, 0]
    assert coeffs[0] == pytest.approx(math.pi / 4, abs=1e-5)
    for n in (1, 3, 4, 5, 7, 8, 9, 11):
        assert abs(coeffs[n]) < 1e-5
    assert coeffs[6] / coeffs[2] == pytest.approx(1 / 9, rel=1e-3)


def test_mixed_boundary_holds_one_end_and_frees_the_other():
    """Neumann at ``sigma = 0``, Dirichlet at ``sigma = pi``: one end on a brane."""
    n_cells = 128
    sigma = np.linspace(0.0, math.pi, n_cells + 1)
    X0 = np.column_stack([np.cos(0.5 * sigma), 0.2 * np.cos(1.5 * sigma)])
    ev = evolve(X0, boundary="mixed", n_steps=250)
    assert np.max(np.abs(ev.X[:, -1, :] - X0[-1])) < 1e-14  # pinned
    assert np.max(np.abs(ev.X[:, 0, :] - X0[0])) > 1e-3  # free


def test_mode_spectrum_bases_match_their_boundary_conditions():
    sigma = np.linspace(0.0, math.pi, 1025)
    dirichlet = (0.8 * np.sin(2 * sigma))[:, None]
    coeffs = mode_spectrum(dirichlet, boundary="dirichlet", n_modes=5)[:, 0]
    assert coeffs[1] == pytest.approx(0.8, abs=1e-6)
    assert np.max(np.abs(np.delete(coeffs, 1))) < 1e-6

    mixed = (0.6 * np.cos(2.5 * sigma))[:, None]
    coeffs = mode_spectrum(mixed, boundary="mixed", n_modes=5)[:, 0]
    assert coeffs[2] == pytest.approx(0.6, abs=1e-6)
    assert np.max(np.abs(np.delete(coeffs, 2))) < 1e-6


def test_periodic_mode_spectrum_is_the_fourier_transform():
    n_points = 512
    sigma = np.arange(n_points) * (2 * math.pi / n_points)
    X = (1.3 * np.cos(3 * sigma))[:, None]
    coeffs = mode_spectrum(X, sigma_max=2 * math.pi, boundary="periodic", n_modes=6)[:, 0]
    assert abs(coeffs[3]) == pytest.approx(0.65, abs=1e-9)  # split between +/-3
    assert abs(coeffs[1]) < 1e-9
