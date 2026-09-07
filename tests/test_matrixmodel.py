"""The matrix model, integrated: what it conserves and what it does not.

Two conserved quantities behave differently on purpose, and the tests say so.
The energy drift falls like ``h^2`` -- a property of velocity Verlet. The Gauss
constraint stays at round-off whatever the step, because the equations conserve
it exactly and the integrator inherits that. A test that only checked "both are
small" would miss the distinction, so both are checked for the right behaviour.

The physics is then pinned against things computed elsewhere: the potential is
``myers_potential`` at zero flux, the off-diagonal frequency is the brane
separation -- the stretched string of ``dbrane.py`` with the tension scaled out
-- and the Lyapunov exponent scales like ``E^(1/4)``, which follows from a
symmetry of the equations that is itself verified.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.branes.matrixmodel import (
    acceleration,
    commutator_potential,
    energy,
    evolve,
    gauss_constraint,
    kinetic_energy,
    lyapunov_exponent,
    lyapunov_scaling,
    project_gauss,
    random_state,
    scaling_residual,
    separated_branes,
    stretched_mode,
)
from stringsim.branes.myers import fuzzy_sphere, myers_potential, su2_generators


def state(seed: int = 11, size: int = 4, dim: int = 3, scale: float = 1.0):
    return random_state(np.random.default_rng(seed), size, dim, scale)


# --------------------------------------------------------------------------
# the potential and the force
# --------------------------------------------------------------------------


def test_the_potential_is_myers_at_zero_flux():
    """Written once, in myers.py, and checked here rather than duplicated."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        raw = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
        matrices = np.array([(m + m.conj().T) / 2.0 for m in raw])
        assert commutator_potential(matrices) == pytest.approx(
            myers_potential(matrices, 0.0), abs=1e-12
        )


def test_the_potential_is_non_negative_and_vanishes_only_on_commuting_matrices():
    rng = np.random.default_rng(5)
    for _ in range(30):
        raw = rng.normal(size=(4, 3, 3)) + 1j * rng.normal(size=(4, 3, 3))
        matrices = np.array([(m + m.conj().T) / 2.0 for m in raw])
        assert commutator_potential(matrices) > 0.0
    diagonal = np.array([np.diag(rng.normal(size=5)).astype(complex) for _ in range(3)])
    assert commutator_potential(diagonal) == pytest.approx(0.0, abs=1e-12)


def test_the_force_is_minus_the_gradient():
    """Checked along random Hermitian directions, not entry by entry."""
    rng = np.random.default_rng(7)
    step = 1e-6
    for _ in range(10):
        raw = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
        matrices = np.array([(m + m.conj().T) / 2.0 for m in raw]) * 0.6
        raw = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
        direction = np.array([(m + m.conj().T) / 2.0 for m in raw])
        derivative = (
            commutator_potential(matrices + step * direction)
            - commutator_potential(matrices - step * direction)
        ) / (2.0 * step)
        predicted = -sum(
            np.trace(acceleration(matrices)[i] @ direction[i]).real for i in range(3)
        )
        assert derivative == pytest.approx(predicted, rel=1e-5, abs=1e-6)


def test_a_fuzzy_sphere_is_not_a_solution_without_the_flux():
    """The Myers minimum needs the cubic term; here nothing holds it up."""
    matrices = fuzzy_sphere(5, 1.7)
    assert np.max(np.abs(acceleration(matrices))) > 0.1
    assert np.max(np.abs(acceleration(su2_generators(1)))) == pytest.approx(0.0)


def test_shapes_are_validated():
    with pytest.raises(ValueError, match="stack of square matrices"):
        commutator_potential(np.zeros((3, 4, 5)))
    with pytest.raises(ValueError, match="shapes must match"):
        gauss_constraint(np.zeros((3, 4, 4)), np.zeros((2, 4, 4)))


# --------------------------------------------------------------------------
# the constraint
# --------------------------------------------------------------------------


def test_starting_from_rest_satisfies_the_constraint():
    matrices, velocities = separated_branes([1.0, -1.0, 3.0])
    assert np.max(np.abs(gauss_constraint(matrices, velocities))) == 0.0


def test_the_projection_is_exact_at_any_scale():
    """One least-squares solve, so it does not care how big the matrices are."""
    rng = np.random.default_rng(4)
    for scale in (0.1, 1.0, 8.0):
        matrices, velocities = random_state(rng, 4, 3, scale)
        assert np.max(np.abs(gauss_constraint(matrices, velocities))) < 1e-10 * max(1.0, scale**2)


def test_the_projection_removes_only_gauge():
    """Projecting twice changes nothing the second time."""
    matrices, velocities = state()
    again = project_gauss(matrices, velocities)
    assert np.allclose(again, velocities, atol=1e-10)


def test_the_constraint_is_anti_hermitian():
    matrices, velocities = state()
    residual = gauss_constraint(matrices, velocities + matrices)
    assert np.allclose(residual, -residual.conj().T)


# --------------------------------------------------------------------------
# the integrator
# --------------------------------------------------------------------------


def test_the_energy_drift_is_second_order():
    matrices, velocities = state()
    drifts = []
    for step in (0.02, 0.01, 0.005):
        run = evolve(matrices, velocities, step, int(6.0 / step), stride=20)
        drifts.append(run.energy_drift)
    ratios = [a / b for a, b in zip(drifts[:-1], drifts[1:], strict=True)]
    assert all(3.0 < ratio < 5.0 for ratio in ratios), ratios


def test_the_constraint_does_not_care_about_the_step():
    """Unlike the energy: it is conserved by the equations, not by the scheme."""
    matrices, velocities = state()
    for step in (0.02, 0.005):
        run = evolve(matrices, velocities, step, int(6.0 / step), stride=20)
        assert run.constraint_drift < 1e-11


def test_the_trajectory_reports_what_it_holds():
    matrices, velocities = state()
    run = evolve(matrices, velocities, 0.01, 100, stride=10)
    assert run.matrices.shape == (11, 3, 4, 4)
    assert run.times.shape == (11,)
    assert run.times[-1] == pytest.approx(1.0)
    assert run.eigenvalues(0).shape == (11, 4)
    assert run.element(1, 0, 1).shape == (11,)
    with pytest.raises(ValueError, match="need dt"):
        evolve(matrices, velocities, -0.01, 10)


def test_the_scaling_symmetry_holds():
    matrices, velocities = state()
    for scale in (0.6, 1.4, 2.2):
        assert scaling_residual(matrices, velocities, scale) < 1e-10


# --------------------------------------------------------------------------
# flat directions and stretched strings
# --------------------------------------------------------------------------


def test_commuting_matrices_move_freely_for_ever():
    positions = np.array([-2.0, 0.0, 2.0])
    velocity = np.array([-0.3, 0.0, 0.3])
    matrices, velocities = separated_branes(positions)
    velocities[0] = np.diag(velocity).astype(complex)
    run = evolve(matrices, velocities, 0.01, 500)
    assert commutator_potential(run.matrices[-1]) == pytest.approx(0.0, abs=1e-20)
    assert run.energy_drift == pytest.approx(0.0, abs=1e-14)
    assert np.allclose(run.eigenvalues(0)[-1], np.sort(positions + 5.0 * velocity))


@pytest.mark.parametrize("separation", [0.25, 0.5, 1.0, 2.5, 4.0])
def test_the_off_diagonal_frequency_is_the_separation(separation):
    """The stretched string of dbrane.py, with the tension scaled out."""
    # omega = r holds for an infinitesimal amplitude; at a finite one the string
    # slowly pulls the branes together and the frequency drifts with them, so
    # the amplitude is kept small and the run to a dozen periods.
    matrices, velocities = stretched_mode(separation, amplitude=1e-5)
    # the step tracks the period, so both the accuracy and the cost are the
    # same at every separation; a fixed step would make small r quadratically
    # more work for no more precision
    step = 0.006 / separation
    duration = 24.0 * math.pi / separation
    run = evolve(matrices, velocities, step, int(duration / step))
    signal = run.element(1, 0, 1).real
    crossings = np.where(np.diff(np.sign(signal)) != 0)[0]
    assert len(crossings) > 20
    period = 2.0 * float(np.mean(np.diff(crossings))) * step
    assert 2.0 * math.pi / period == pytest.approx(separation, rel=1e-4)


def test_the_string_pulls_the_branes_together_at_second_order():
    """The back-reaction is quadratic in the amplitude, and it is attractive.

    A string stretched between two branes costs energy proportional to the
    separation, so it pulls.  The effect is ``O(a^2)``: quadrupling the
    amplitude multiplies the drift by sixteen, and the branes move *toward*
    each other, never apart.
    """
    drifts = []
    for amplitude in (1e-3, 2e-3, 4e-3):
        matrices, velocities = stretched_mode(2.0, amplitude=amplitude)
        run = evolve(matrices, velocities, 0.002, 5000)
        gaps = run.eigenvalues(0)[:, 1] - run.eigenvalues(0)[:, 0]
        assert np.all(gaps <= gaps[0] + 1e-12)  # attractive, never repulsive
        drifts.append(gaps[0] - gaps[-1])
    ratios = [b / a for a, b in zip(drifts[:-1], drifts[1:], strict=True)]
    assert all(3.5 < ratio < 4.5 for ratio in ratios), ratios


def test_stretched_mode_needs_a_positive_separation():
    with pytest.raises(ValueError, match="separation must be positive"):
        stretched_mode(0.0)


# --------------------------------------------------------------------------
# chaos
# --------------------------------------------------------------------------


def test_nearby_configurations_separate_exponentially():
    matrices, velocities = state()
    fit = lyapunov_exponent(matrices, velocities, dt=0.004, steps=8_000, renormalise=400)
    assert fit.exponent > 0.3
    assert fit.growth[-1] > fit.growth[0]
    assert np.all(np.diff(fit.growth) > 0.0)


def test_a_flat_direction_is_not_chaotic():
    """Free motion has no exponential separation; the exponent is ~ 0."""
    matrices, velocities = separated_branes([-4.0, 0.0, 4.0])
    velocities[0] = np.diag([-0.4, 0.0, 0.4]).astype(complex)
    fit = lyapunov_exponent(matrices, velocities, dt=0.004, steps=6_000, renormalise=400)
    assert abs(fit.exponent) < 0.05


def test_the_exponent_scales_like_the_fourth_root_of_the_energy():
    matrices, velocities = state()
    power, energies, exponents = lyapunov_scaling(
        matrices, velocities, scales=(0.5, 1.0, 2.0), dt=0.004, steps=8_000, renormalise=400
    )
    assert energies[-1] / energies[0] > 200.0  # a real range, not a nudge
    assert power == pytest.approx(0.25, abs=0.03)
    assert np.all(np.diff(exponents) > 0.0)


def test_lyapunov_input_is_validated():
    matrices, velocities = state()
    with pytest.raises(ValueError, match="too few renormalisations"):
        lyapunov_exponent(matrices, velocities, steps=100, renormalise=500)


def test_kinetic_energy_of_rest_is_zero():
    assert kinetic_energy(np.zeros((3, 2, 2))) == 0.0
    matrices, velocities = state()
    assert energy(matrices, velocities) == pytest.approx(
        kinetic_energy(velocities) + commutator_potential(matrices)
    )
