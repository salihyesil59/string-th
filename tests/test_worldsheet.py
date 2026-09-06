"""Numerical evolution of the worldsheet fermions.

The mode numbers, the period doubling and the Ramond zero mode are all
properties ``rns.py`` states algebraically.  Here they have to come out of a
grid being stepped forward instead, which is only a check because the two
routes share no code: the evolution knows one thing, the sign in
``psi(sigma + 2 pi) = eta psi(sigma)``, and everything else is measured.

The evolution is also required to reproduce the exact mode solution it was
never told about, at the second-order rate the scheme promises, and to keep the
supercurrent chiral against a bosonic run from ``classical/evolve.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.classical.evolve import evolve
from stringsim.superstring.rns import Sector, fermion_mode_numbers
from stringsim.superstring.worldsheet import (
    SPIN_STRUCTURES,
    boundary_residual,
    evolve_closed_fermion,
    evolve_open_fermion,
    fermion_mode_spectrum,
    fold,
    norm,
    open_fermion_modes,
    reconstruct_from_modes,
    sector_twist,
    supercurrent,
    supercurrent_residual,
    susy_variation,
    unfold,
    zero_mode,
)

TWO_PI = 2.0 * math.pi

MODES = {
    Sector.NS: {0.5: [1.0, 0.3j], 1.5: [0.2, -0.4], 2.5: [0.1j, 0.05]},
    Sector.R: {0.0: [0.7, 0.0], 1.0: [0.5, 0.2j], 2.0: [-0.3, 0.1]},
}


def half_grid(n_points: int) -> np.ndarray:
    return np.linspace(0.0, math.pi, n_points // 2 + 1)


def open_data(sector: Sector, n_points: int, tau: float = 0.0):
    return open_fermion_modes(MODES[sector], sector, tau, half_grid(n_points))


# --------------------------------------------------------------------------
# folding the open string open
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sector", [Sector.NS, Sector.R])
def test_analytic_data_satisfies_both_boundary_conditions(sector):
    twist = sector_twist(sector)
    at_zero, at_pi = boundary_residual(*open_data(sector, 64), twist)
    assert at_zero < 1e-12
    assert at_pi < 1e-12


@pytest.mark.parametrize("sector", [Sector.NS, Sector.R])
def test_wrong_sector_fails_the_condition_at_pi(sector):
    """The condition at sigma = 0 is sector-blind; the one at pi is not."""
    twist = sector_twist(sector)
    at_zero, at_pi = boundary_residual(*open_data(sector, 64), -twist)
    assert at_zero < 1e-12
    assert at_pi > 0.1


@pytest.mark.parametrize("sector", [Sector.NS, Sector.R])
def test_fold_inverts_unfold(sector):
    twist = sector_twist(sector)
    minus, plus = open_data(sector, 64)
    back_minus, back_plus = fold(unfold(minus, plus, twist), twist)
    assert np.max(np.abs(back_minus - minus)) < 1e-14
    assert np.max(np.abs(back_plus - plus)) < 1e-14


def test_unfolded_field_has_the_twisted_periodicity():
    """psi(sigma_N) = eta psi(0) is what folding is for; check it directly."""
    for sector in (Sector.NS, Sector.R):
        twist = sector_twist(sector)
        n_points = 64
        doubled = unfold(*open_data(sector, n_points), twist)
        sigma = np.arange(n_points) * (TWO_PI / n_points)
        exact = open_fermion_modes(MODES[sector], sector, 0.0, sigma + TWO_PI)[0]
        assert np.max(np.abs(exact - twist * doubled)) < 1e-12


def test_unfold_rejects_inadmissible_data():
    minus, plus = open_data(Sector.NS, 64)
    with pytest.raises(ValueError, match="boundary conditions"):
        unfold(minus, plus, +1)


def test_unfold_accepts_inadmissible_data_when_told_to():
    minus, plus = open_data(Sector.NS, 64)
    assert unfold(minus, plus, +1, tolerance=10.0).shape[0] == 64


def test_fold_needs_an_even_grid():
    with pytest.raises(ValueError, match="even number"):
        fold(np.zeros((7, 2)), 1)


# --------------------------------------------------------------------------
# the evolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sector", [Sector.NS, Sector.R])
def test_courant_one_reproduces_the_exact_solution(sector):
    n_points = 128
    run = evolve_open_fermion(
        *open_data(sector, n_points), sector=sector, n_steps=n_points, courant=1.0
    )
    assert run.tau[-1] == pytest.approx(TWO_PI)
    for index in (0, 17, 64, n_points):
        exact = open_data(sector, n_points, float(run.tau[index]))
        assert np.max(np.abs(run.psi_minus[index] - exact[0])) < 1e-13
        assert np.max(np.abs(run.psi_plus[index] - exact[1])) < 1e-13


@pytest.mark.parametrize("sector", [Sector.NS, Sector.R])
def test_a_round_trip_multiplies_by_the_twist(sector):
    """tau -> tau + 2 pi returns eta psi, so NS needs 4 pi to come home."""
    twist = sector_twist(sector)
    n_points = 96
    run = evolve_open_fermion(
        *open_data(sector, n_points), sector=sector, n_steps=n_points, courant=1.0
    )
    assert np.max(np.abs(run.unfolded[-1] - twist * run.unfolded[0])) < 1e-13
    if twist == -1:
        assert np.max(np.abs(run.unfolded[-1] - run.unfolded[0])) > 1.0


def test_the_scheme_is_second_order():
    errors = []
    for n_points in (32, 64, 128, 256):
        run = evolve_open_fermion(
            *open_data(Sector.NS, n_points),
            sector=Sector.NS,
            n_steps=2 * n_points,
            courant=0.5,
        )
        exact = open_data(Sector.NS, n_points, float(run.tau[-1]))
        errors.append(float(np.max(np.abs(run.psi_minus[-1] - exact[0]))))
    ratios = [a / b for a, b in zip(errors[:-1], errors[1:], strict=True)]
    assert all(3.6 < r < 4.2 for r in ratios), ratios


def test_norm_is_exact_at_courant_one_and_only_damped_below_it():
    n_points = 64
    exact = evolve_open_fermion(
        *open_data(Sector.NS, n_points), sector=Sector.NS, n_steps=n_points, courant=1.0
    )
    assert norm(exact.unfolded[-1]) == pytest.approx(norm(exact.unfolded[0]), abs=1e-13)
    damped = evolve_open_fermion(
        *open_data(Sector.NS, n_points), sector=Sector.NS, n_steps=n_points, courant=0.5
    )
    drift = norm(damped.unfolded[-1]) - norm(damped.unfolded[0])
    assert -1e-2 < drift < 0.0


def test_courant_must_be_in_range():
    minus, plus = open_data(Sector.NS, 32)
    with pytest.raises(ValueError, match="courant"):
        evolve_open_fermion(minus, plus, sector=Sector.NS, courant=1.5)


def test_evolution_metadata():
    n_points = 64
    run = evolve_open_fermion(
        *open_data(Sector.R, n_points), sector=Sector.R, n_steps=10, courant=0.5
    )
    assert run.courant == pytest.approx(0.5)
    assert run.sector is Sector.R
    assert run.boundary == "open"
    assert run.unfolded.shape == (11, n_points, 2)
    minus, plus = run.at(3)
    assert minus.shape == (n_points // 2 + 1, 2)
    assert np.allclose(plus, run.psi_plus[3])


# --------------------------------------------------------------------------
# mode content read off the grid
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sector", [Sector.NS, Sector.R])
def test_measured_mode_numbers_are_the_ones_rns_asserts(sector):
    twist = sector_twist(sector)
    n_points = 128
    doubled = unfold(*open_data(sector, n_points), twist)
    coefficients = fermion_mode_spectrum(doubled, twist, 6)
    measured = [
        r
        for r, c in zip(
            np.arange(6) + (0.0 if twist == 1 else 0.5), coefficients, strict=True
        )
        if np.max(np.abs(c)) > 1e-9
    ]
    quoted = [float(r) for r in fermion_mode_numbers(sector, 6)]
    assert set(measured) <= set(quoted)
    assert set(measured) == {float(r) for r in MODES[sector]}


@pytest.mark.parametrize("sector", [Sector.NS, Sector.R])
def test_amplitudes_come_back_at_half_size(sector):
    """A real field splits its amplitude between +r and -r; r = 0 does not."""
    twist = sector_twist(sector)
    n_points = 128
    doubled = unfold(*open_data(sector, n_points), twist)
    coefficients = fermion_mode_spectrum(doubled, twist, 6)
    for r, amplitude in MODES[sector].items():
        index = int(r) if twist == 1 else int(r - 0.5)
        expected = np.asarray(amplitude, complex) * (1.0 if r == 0.0 else 0.5)
        assert np.max(np.abs(coefficients[index] - expected)) < 1e-12


@pytest.mark.parametrize("sector", [Sector.NS, Sector.R])
def test_the_wrong_mode_numbers_are_a_different_vector_space(sector):
    twist = sector_twist(sector)
    n_points = 128
    doubled = unfold(*open_data(sector, n_points), twist)
    right = reconstruct_from_modes(fermion_mode_spectrum(doubled, twist, 8), twist, n_points)
    wrong = reconstruct_from_modes(fermion_mode_spectrum(doubled, -twist, 8), -twist, n_points)
    assert np.max(np.abs(right - doubled)) < 1e-13
    assert np.max(np.abs(wrong - doubled)) > 0.5


def test_only_ramond_has_a_zero_mode():
    n_points = 64
    ramond = unfold(*open_data(Sector.R, n_points), +1)
    neveu = unfold(*open_data(Sector.NS, n_points), -1)
    assert np.allclose(zero_mode(ramond, +1), np.array([0.7, 0.0]), atol=1e-12)
    assert np.allclose(zero_mode(neveu, -1), 0.0)


def test_the_ramond_zero_mode_never_moves():
    n_points = 64
    run = evolve_open_fermion(
        *open_data(Sector.R, n_points), sector=Sector.R, n_steps=50, courant=0.5
    )
    first = zero_mode(run.unfolded[0], +1)
    for index in (10, 30, 50):
        assert np.max(np.abs(zero_mode(run.unfolded[index], +1) - first)) < 1e-13


def test_open_fermion_modes_rejects_the_other_sector():
    sigma = half_grid(32)
    with pytest.raises(ValueError, match="half-integral"):
        open_fermion_modes({1.0: [1.0]}, Sector.NS, 0.0, sigma)
    with pytest.raises(ValueError, match="integral"):
        open_fermion_modes({0.5: [1.0]}, Sector.R, 0.0, sigma)
    with pytest.raises(ValueError, match="neither integral nor half-integral"):
        open_fermion_modes({0.3: [1.0]}, Sector.R, 0.0, sigma)
    with pytest.raises(ValueError, match="at least one mode"):
        open_fermion_modes({}, Sector.R, 0.0, sigma)


# --------------------------------------------------------------------------
# the closed string
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SPIN_STRUCTURES))
def test_each_spin_structure_returns_with_its_own_sign(name):
    twist_minus, twist_plus = SPIN_STRUCTURES[name]
    n_points = 96
    sigma = np.arange(n_points) * (TWO_PI / n_points)

    def data(twist):
        rate = 1.0 if twist == 1 else 0.5
        return np.stack([np.cos(rate * sigma), 0.3 * np.sin(rate * sigma)], axis=-1)

    run = evolve_closed_fermion(
        data(twist_minus), data(twist_plus), spin_structure=name,
        n_steps=n_points, courant=1.0,
    )
    assert np.max(np.abs(run.psi_minus[-1] - twist_minus * run.psi_minus[0])) < 1e-13
    assert np.max(np.abs(run.psi_plus[-1] - twist_plus * run.psi_plus[0])) < 1e-13


def test_the_two_closed_string_sides_move_in_opposite_directions():
    n_points = 64
    sigma = np.arange(n_points) * (TWO_PI / n_points)
    profile = np.exp(-(((sigma - math.pi) / 0.3) ** 2))[:, None]
    run = evolve_closed_fermion(
        profile, profile, spin_structure="R-R", n_steps=8, courant=1.0
    )
    # At Courant 1 a step is an exact one-cell shift, opposite ways round.
    assert np.max(np.abs(run.psi_minus[8] - np.roll(profile, 8, axis=0))) < 1e-13
    assert np.max(np.abs(run.psi_plus[8] - np.roll(profile, -8, axis=0))) < 1e-13


def test_closed_string_rejects_nonsense():
    field = np.zeros((16, 2))
    with pytest.raises(ValueError, match="unknown spin structure"):
        evolve_closed_fermion(field, field, spin_structure="NS-X")
    with pytest.raises(ValueError, match="twist must be"):
        evolve_closed_fermion(field, field, spin_structure=(1, 0))
    with pytest.raises(ValueError, match="same shape|shape"):
        evolve_closed_fermion(field, np.zeros((16, 3)))


def test_a_closed_string_has_no_single_sector():
    field = np.zeros((16, 2))
    run = evolve_closed_fermion(field, field, n_steps=2)
    assert run.unfolded is None
    with pytest.raises(ValueError, match="spin structure"):
        _ = run.sector


# --------------------------------------------------------------------------
# supersymmetry and the supercurrent
# --------------------------------------------------------------------------


def bosonic_run(n_points: int):
    sigma = half_grid(n_points)
    shape = np.stack([0.3 * np.cos(sigma), 0.2 * np.cos(2.0 * sigma)], axis=-1)
    return evolve(shape, boundary="neumann", n_steps=2 * n_points, courant=0.5)


def susy_data(n_points: int):
    """The variation at a generic time -- at tau = 0 the string is at rest."""
    bosons = bosonic_run(n_points)
    index = max(1, int(round(0.7 / float(bosons.tau[1] - bosons.tau[0]))))
    return bosons, susy_variation(bosons, index=index)


def test_susy_variation_of_a_neumann_boson_is_ramond_data():
    """It satisfies R at both ends, and misses NS at pi by an amount that stays."""
    ramond, neveu = [], []
    for n_points in (32, 64, 128, 256):
        _, (minus, plus) = susy_data(n_points)
        ramond.append(max(boundary_residual(minus, plus, +1)))
        neveu.append(boundary_residual(minus, plus, -1)[1])
    assert ramond[0] > ramond[-1] * 100.0
    assert ramond[-1] < 1e-5
    assert all(0.3 < value < 0.5 for value in neveu), neveu


def test_susy_variation_needs_an_interior_time():
    bosons = bosonic_run(32)
    with pytest.raises(ValueError, match="interior"):
        susy_variation(bosons, index=0)
    with pytest.raises(ValueError, match="interior"):
        susy_variation(bosons, index=len(bosons.tau) - 1)


def test_the_supercurrent_is_chiral_to_second_order():
    residuals = []
    for n_points in (32, 64, 128, 256):
        bosons, (minus, plus) = susy_data(n_points)
        fermions = evolve_open_fermion(
            minus, plus, sector=Sector.R, n_steps=2 * n_points,
            courant=0.5, tolerance=1e-1,
        )
        residuals.append(supercurrent_residual(bosons, fermions))
    ratios = [a / b for a, b in zip(residuals[:-1], residuals[1:], strict=True)]
    assert residuals[-1] < 1e-4
    assert all(3.4 < r < 4.4 for r in ratios), ratios


def test_the_supercurrent_has_the_shape_of_the_shared_grid():
    n_points = 32
    bosons, (minus, plus) = susy_data(n_points)
    fermions = evolve_open_fermion(
        minus, plus, sector=Sector.R, n_steps=2 * n_points, courant=0.5, tolerance=1e-1
    )
    current = supercurrent(bosons, fermions)
    assert current.shape == (len(bosons.tau) - 2, n_points // 2 + 1)


def test_the_supercurrent_needs_matching_grids():
    bosons, (minus, plus) = susy_data(32)
    fermions = evolve_open_fermion(
        minus, plus, sector=Sector.R, n_steps=64, courant=0.5, tolerance=1e-1
    )
    coarse = evolve(bosons.X[0][::2], boundary="neumann", n_steps=8, courant=0.5)
    with pytest.raises(ValueError, match="disagree about sigma"):
        supercurrent(coarse, fermions)
    closed = evolve_closed_fermion(np.zeros((16, 2)), np.zeros((16, 2)), n_steps=2)
    with pytest.raises(ValueError, match="open string"):
        supercurrent(bosons, closed)
