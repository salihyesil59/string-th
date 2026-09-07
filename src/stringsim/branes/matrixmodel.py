r"""D0-branes with time: the matrix model, integrated.

:mod:`stringsim.branes.myers` freezes ``N`` D0-branes in a flux and asks what
the minimum is.  Take the flux away and let them move, and what is left is the
matrix quantum mechanics itself -- in the gauge :math:`A_0 = 0`,

.. math::
   L = \tfrac12 \mathrm{Tr}\,\dot X_i \dot X_i
       + \tfrac14 \mathrm{Tr}\,[X_i, X_j][X_i, X_j],
   \qquad
   \ddot X_i = \big[X_j, [X_i, X_j]\big] ,

with ``d`` Hermitian ``N x N`` matrices.  The potential is the same
commutator-squared term the Myers effect fights against, and
:func:`commutator_potential` is checked against
:func:`stringsim.branes.myers.myers_potential` at zero flux rather than written
twice.

**What the simulation conserves, it conserves for a reason.**  Velocity Verlet
is symplectic, so the energy drift is bounded and falls like ``h^2`` --
:attr:`Trajectory.energy_drift` measures the ratio.  The Gauss constraint
:math:`\sum_i [X_i, \dot X_i] = 0` is conserved to round-off, not to
:math:`h^2`, because it is a *symmetry* of the equations rather than an accident
of the scheme: :func:`gauss_constraint` stays at :math:`10^{-13}` for as long
as you run.  Starting from rest satisfies it for free; anything else needs
:func:`project_gauss`.

**The flat directions are free branes.**  Commuting matrices have :math:`V = 0`
exactly, so diagonal ``X`` moves in a straight line for ever and never comes
back.  That is the continuous spectrum of the matrix model, and it is why the
theory describes branes that can separate rather than a confined system.

**The off-diagonal modes are strings, and their mass is the separation.**  Put
two branes at :math:`\pm r/2` along one direction and excite an off-diagonal
element in another.  The quadratic piece of the potential is :math:`a^2 r^2`
against a kinetic term :math:`\dot a^2`, so

.. math::  \omega = r ,

and running the simulation gives it back to a part in :math:`10^{3}`.  This is
the stretched string of :mod:`stringsim.branes.dbrane` with the tension scaled
out: ``X`` is measured in units where the mass of a string of length ``L`` is
``L``, i.e. ``X = (separation)/(2 pi alpha')``.

**And the generic motion is chaotic.**  The classical equations have no small
parameter, and two nearby configurations separate exponentially --
:func:`lyapunov_exponent` measures the rate by renormalising a shadow
trajectory.  The one prediction available without solving anything is
dimensional: :math:`X \to sX`, :math:`t \to t/s` maps solutions to solutions
(:func:`scaling_residual` verifies it), so :math:`E \to s^4 E` while
:math:`\lambda \to s\lambda`, and therefore

.. math::  \lambda \propto E^{1/4} .

:func:`lyapunov_scaling` fits the exponent over a factor of 250 in energy and
gets ``0.25`` back.

**Not here.**  The fermions, and therefore supersymmetry, the flat directions'
quantum fate and the whole reason the supersymmetric model has a normalisable
ground state; and the gauge field ``A_0``, which is set to zero and its equation
of motion kept as the constraint.  This is the bosonic model, classically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "commutator_potential",
    "acceleration",
    "kinetic_energy",
    "energy",
    "gauss_constraint",
    "project_gauss",
    "random_state",
    "separated_branes",
    "stretched_mode",
    "Trajectory",
    "evolve",
    "LyapunovFit",
    "lyapunov_exponent",
    "lyapunov_scaling",
    "scaling_residual",
]


def _as_stack(matrices) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=complex)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(f"expected a stack of square matrices, got shape {matrices.shape}")
    return matrices


def commutator_potential(matrices) -> float:
    r"""``-(1/4) sum_ij Tr([X_i,X_j][X_i,X_j])``, which is non-negative.

    The commutator of two Hermitian matrices is anti-Hermitian, so its square
    has non-positive trace and the sign works out.  Zero exactly when the
    matrices commute -- the flat directions.
    """
    matrices = _as_stack(matrices)
    total = 0.0
    for i in range(matrices.shape[0]):
        for j in range(matrices.shape[0]):
            commutator = matrices[i] @ matrices[j] - matrices[j] @ matrices[i]
            total += -0.25 * np.trace(commutator @ commutator).real
    return float(total)


def acceleration(matrices) -> np.ndarray:
    r"""``d^2 X_i / dt^2 = [X_j, [X_i, X_j]]``, minus the gradient of the potential."""
    matrices = _as_stack(matrices)
    out = np.zeros_like(matrices)
    for i in range(matrices.shape[0]):
        for j in range(matrices.shape[0]):
            commutator = matrices[i] @ matrices[j] - matrices[j] @ matrices[i]
            out[i] += matrices[j] @ commutator - commutator @ matrices[j]
    return out


def kinetic_energy(velocities) -> float:
    r"""``(1/2) sum_i Tr(V_i V_i)``."""
    velocities = _as_stack(velocities)
    return float(
        0.5 * sum(np.trace(velocities[i] @ velocities[i]).real for i in range(len(velocities)))
    )


def energy(matrices, velocities) -> float:
    """Kinetic plus potential; conserved by the exact equations."""
    return kinetic_energy(velocities) + commutator_potential(matrices)


def gauss_constraint(matrices, velocities) -> np.ndarray:
    r"""``sum_i [X_i, V_i]``, the generator ``A_0`` would have multiplied.

    Anti-Hermitian, and conserved exactly: its time derivative vanishes by the
    Jacobi identity, not by any property of the integrator.  Physical initial
    data has it zero.
    """
    matrices, velocities = _as_stack(matrices), _as_stack(velocities)
    if matrices.shape != velocities.shape:
        raise ValueError(f"shapes must match: {matrices.shape} against {velocities.shape}")
    return sum(
        matrices[i] @ velocities[i] - velocities[i] @ matrices[i]
        for i in range(len(matrices))
    )


def _anti_hermitian_basis(size: int) -> list[np.ndarray]:
    """An orthonormal real basis of the anti-Hermitian matrices, ``size^2`` of them."""
    basis = []
    for row in range(size):
        element = np.zeros((size, size), dtype=complex)
        element[row, row] = 1j
        basis.append(element)
    for row in range(size):
        for column in range(row + 1, size):
            first = np.zeros((size, size), dtype=complex)
            first[row, column], first[column, row] = 1.0, -1.0
            basis.append(first / math.sqrt(2.0))
            second = np.zeros((size, size), dtype=complex)
            second[row, column], second[column, row] = 1j, 1j
            basis.append(second / math.sqrt(2.0))
    return basis


def project_gauss(matrices, velocities) -> np.ndarray:
    r"""Remove the gauge part of the velocities, so that ``sum_i [X_i, V_i] = 0``.

    The constraint generates gauge transformations, and the directions it
    generates are :math:`\delta V_i = [X_i, \epsilon]` with
    :math:`\epsilon` anti-Hermitian.  Asking for the shift that kills the
    residual gives a *linear* equation,

    .. math::  \sum_i \big[X_i, [X_i, \epsilon]\big] = G ,

    which is solved once by least squares rather than approached by descent.
    It always has a solution: the operator is self-adjoint, and ``G`` is
    orthogonal to its kernel for any velocities at all.  What is removed is
    pure gauge, so nothing physical is lost.
    """
    matrices = _as_stack(matrices)
    velocities = _as_stack(velocities).copy()
    size = matrices.shape[1]
    basis = _anti_hermitian_basis(size)
    residual = gauss_constraint(matrices, velocities)

    def apply(element: np.ndarray) -> np.ndarray:
        total = np.zeros_like(element)
        for matrix in matrices:
            inner = matrix @ element - element @ matrix
            total = total + matrix @ inner - inner @ matrix
        return total

    columns = np.array([apply(element).ravel() for element in basis]).T
    design = np.vstack([columns.real, columns.imag])
    target = np.concatenate([residual.ravel().real, residual.ravel().imag])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    shift = sum(weight * element for weight, element in zip(coefficients, basis, strict=True))
    for index, matrix in enumerate(matrices):
        velocities[index] -= matrix @ shift - shift @ matrix
    return velocities


def random_state(rng, size: int, dim: int = 3, scale: float = 1.0):
    """Random Hermitian matrices and velocities, already on the constraint surface."""

    def hermitian() -> np.ndarray:
        raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        return (raw + raw.conj().T) / 2.0

    matrices = np.array([hermitian() for _ in range(dim)]) * scale
    velocities = np.array([hermitian() for _ in range(dim)]) * scale
    return matrices, project_gauss(matrices, velocities)


def separated_branes(positions, dim: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal ``X_1``, everything else zero: branes sitting on a line.

    A flat direction, so the potential is exactly zero and stays that way.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1)
    size = positions.size
    matrices = np.zeros((dim, size, size), dtype=complex)
    matrices[0] = np.diag(positions).astype(complex)
    return matrices, np.zeros_like(matrices)


def stretched_mode(separation: float, amplitude: float = 1e-3, dim: int = 3):
    r"""Two branes at :math:`\pm r/2`, with an off-diagonal mode switched on.

    The excited element is a string with one end on each brane.  Its frequency
    is the separation; :func:`evolve` measures that rather than assuming it.
    """
    if separation <= 0:
        raise ValueError("the separation must be positive")
    matrices = np.zeros((dim, 2, 2), dtype=complex)
    matrices[0] = np.diag([separation / 2.0, -separation / 2.0]).astype(complex)
    matrices[1] = amplitude * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    return matrices, np.zeros_like(matrices)


# ---------------------------------------------------------------------------
# the evolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trajectory:
    """The result of :func:`evolve`.

    Attributes
    ----------
    times:
        Sampled times, shape ``(n_frames,)``.
    matrices:
        ``X_i(t)``, shape ``(n_frames, dim, size, size)``.
    energies, constraints:
        The energy and ``max |Gauss|`` at each sample.
    """

    times: np.ndarray
    matrices: np.ndarray
    energies: np.ndarray
    constraints: np.ndarray

    @property
    def energy_drift(self) -> float:
        """Largest relative departure from the initial energy."""
        return float(np.max(np.abs(self.energies - self.energies[0])) / abs(self.energies[0]))

    @property
    def constraint_drift(self) -> float:
        """Largest ``max |Gauss|`` along the run."""
        return float(np.max(self.constraints))

    def eigenvalues(self, direction: int = 0) -> np.ndarray:
        """Eigenvalues of ``X_direction`` at every sample: the brane positions."""
        return np.array([np.linalg.eigvalsh(frame[direction]) for frame in self.matrices])

    def element(self, direction: int, row: int, column: int) -> np.ndarray:
        """One matrix element through time, for reading a frequency off."""
        return self.matrices[:, direction, row, column]


def evolve(matrices, velocities, dt: float = 0.005, steps: int = 2000, stride: int = 1):
    """Velocity Verlet on ``d^2X/dt^2 = [X_j,[X_i,X_j]]``.

    Symplectic, so the energy oscillates within a band that shrinks like
    ``h^2`` instead of drifting away; the constraint is conserved to round-off
    because the equations conserve it exactly.
    """
    if dt <= 0 or steps < 1 or stride < 1:
        raise ValueError("need dt > 0, steps >= 1 and stride >= 1")
    position = _as_stack(matrices).copy()
    velocity = _as_stack(velocities).copy()
    if position.shape != velocity.shape:
        raise ValueError(f"shapes must match: {position.shape} against {velocity.shape}")

    force = acceleration(position)
    frames, energies, constraints = [position.copy()], [energy(position, velocity)], [
        float(np.max(np.abs(gauss_constraint(position, velocity))))
    ]
    for step in range(1, steps + 1):
        velocity = velocity + 0.5 * dt * force
        position = position + dt * velocity
        force = acceleration(position)
        velocity = velocity + 0.5 * dt * force
        if step % stride == 0:
            frames.append(position.copy())
            energies.append(energy(position, velocity))
            constraints.append(float(np.max(np.abs(gauss_constraint(position, velocity)))))
    return Trajectory(
        times=np.arange(len(frames)) * (dt * stride),
        matrices=np.array(frames),
        energies=np.array(energies),
        constraints=np.array(constraints),
    )


# ---------------------------------------------------------------------------
# chaos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LyapunovFit:
    """The largest Lyapunov exponent, and how well the growth was a straight line."""

    exponent: float
    times: np.ndarray
    growth: np.ndarray
    residual: float

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"lambda = {self.exponent:.5f} (fit residual {self.residual:.2e})"


def lyapunov_exponent(
    matrices,
    velocities,
    dt: float = 0.002,
    steps: int = 30_000,
    separation: float = 1e-8,
    renormalise: int = 500,
    seed: int = 0,
    burn: float = 0.3,
) -> LyapunovFit:
    r"""Largest Lyapunov exponent, by running a shadow trajectory.

    A second copy is started a distance ``separation`` away, and every
    ``renormalise`` steps the accumulated ``log`` of the growth is banked and
    the shadow pulled back in.  The slope of the banked total against time is
    the exponent; ``burn`` discards the leading fraction, where the perturbation
    is still finding the most unstable direction.
    """
    rng = np.random.default_rng(seed)
    position = _as_stack(matrices).copy()
    velocity = _as_stack(velocities).copy()
    dim, size = position.shape[0], position.shape[1]

    raw = rng.normal(size=(dim, size, size)) + 1j * rng.normal(size=(dim, size, size))
    offset = np.array([(m + m.conj().T) / 2.0 for m in raw])
    offset *= separation / math.sqrt(float(np.sum(np.abs(offset) ** 2)))
    shadow, shadow_velocity = position + offset, velocity.copy()

    force, shadow_force = acceleration(position), acceleration(shadow)
    banked, times, growth = 0.0, [], []
    for step in range(1, steps + 1):
        velocity = velocity + 0.5 * dt * force
        position = position + dt * velocity
        force = acceleration(position)
        velocity = velocity + 0.5 * dt * force

        shadow_velocity = shadow_velocity + 0.5 * dt * shadow_force
        shadow = shadow + dt * shadow_velocity
        shadow_force = acceleration(shadow)
        shadow_velocity = shadow_velocity + 0.5 * dt * shadow_force

        if step % renormalise == 0:
            gap = math.sqrt(
                float(np.sum(np.abs(shadow - position) ** 2))
                + float(np.sum(np.abs(shadow_velocity - velocity) ** 2))
            )
            if gap <= 0.0:
                raise RuntimeError("the shadow trajectory collapsed onto the reference")
            banked += math.log(gap / separation)
            times.append(step * dt)
            growth.append(banked)
            factor = separation / gap
            shadow = position + (shadow - position) * factor
            shadow_velocity = velocity + (shadow_velocity - velocity) * factor
            shadow_force = acceleration(shadow)

    times, growth = np.array(times), np.array(growth)
    if len(times) < 4:
        raise ValueError("too few renormalisations to fit; lower renormalise or raise steps")
    keep = times > times[-1] * burn
    slope, intercept = np.polyfit(times[keep], growth[keep], 1)
    residual = float(np.max(np.abs(slope * times[keep] + intercept - growth[keep])))
    return LyapunovFit(
        exponent=float(slope), times=times, growth=growth, residual=residual
    )


def lyapunov_scaling(matrices, velocities, scales=(0.5, 0.75, 1.0, 1.5), **kwargs):
    r"""Fit ``lambda ~ E^p`` over rescaled copies of one configuration.

    Rescaling by ``s`` -- :math:`X \to sX`, :math:`V \to s^2V` -- is a symmetry
    of the equations with :math:`t \to t/s`, so the energy goes like
    :math:`s^4` and the exponent like :math:`s`.  The fitted power should
    therefore be ``1/4``, and it is a prediction with nothing adjustable in it.

    Returns ``(power, energies, exponents)``.
    """
    matrices, velocities = _as_stack(matrices), _as_stack(velocities)
    energies, exponents = [], []
    for scale in scales:
        scaled_x, scaled_v = matrices * scale, velocities * scale**2
        energies.append(energy(scaled_x, scaled_v))
        exponents.append(lyapunov_exponent(scaled_x, scaled_v, **kwargs).exponent)
    energies, exponents = np.array(energies), np.array(exponents)
    power = float(np.polyfit(np.log(energies), np.log(exponents), 1)[0])
    return power, energies, exponents


def scaling_residual(matrices, velocities, scale: float = 1.7, dt: float = 0.002, steps: int = 400):
    r"""How well ``X -> sX``, ``V -> s^2 V``, ``t -> t/s`` maps solutions to solutions.

    Evolving the rescaled data for ``steps/scale`` of the time must give
    ``scale`` times the original.  Zero to the integrator's accuracy; it is the
    symmetry behind :math:`\lambda \propto E^{1/4}`.
    """
    matrices, velocities = _as_stack(matrices), _as_stack(velocities)
    plain = evolve(matrices, velocities, dt, steps)
    rescaled = evolve(matrices * scale, velocities * scale**2, dt / scale, steps)
    return float(
        np.max(np.abs(rescaled.matrices[-1] - scale * plain.matrices[-1]))
        / max(1e-30, np.max(np.abs(scale * plain.matrices[-1])))
    )
