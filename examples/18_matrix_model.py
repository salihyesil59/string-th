"""D0-branes with time: the matrix model, integrated.

Run:  python examples/18_matrix_model.py

``examples/16`` freezes N D0-branes in a flux and asks what the minimum is.
Take the flux away and let them move, and the matrices become a dynamical
system: separated branes drift freely, strings stretched between them oscillate
at the separation, and the generic motion is chaotic.

Writes ``figures/matrix_worldlines.png``, ``figures/lyapunov.png`` and
``figures/matrix_scattering.gif``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.branes.matrixmodel import (  # noqa: E402
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
from stringsim.branes.myers import myers_potential  # noqa: E402
from stringsim.viz.animate import animate_matrix_eigenvalues  # noqa: E402
from stringsim.viz.plots import plot_lyapunov, plot_matrix_worldlines  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"


def what_is_conserved() -> None:
    print("=" * 72)
    print("The equations, and what holding on to them costs")
    print("=" * 72)
    print("  L = (1/2) Tr(Xdot_i Xdot_i) + (1/4) Tr([X_i,X_j][X_i,X_j])")
    print("  Xddot_i = [X_j, [X_i, X_j]]")
    print()
    rng = np.random.default_rng(11)
    matrices, velocities = random_state(rng, 4, 3)
    print(f"  the potential is myers_potential at zero flux, to "
          f"{abs(commutator_potential(matrices) - myers_potential(matrices, 0.0)):.1e}")
    print(f"  starting energy {energy(matrices, velocities):.6f}, "
          f"Gauss constraint {np.max(np.abs(gauss_constraint(matrices, velocities))):.1e}")
    print()
    print(f"  {'dt':>8s} {'energy drift':>14s} {'ratio':>7s} {'max |Gauss|':>13s}")
    previous = None
    for step in (0.02, 0.01, 0.005, 0.0025):
        run = evolve(matrices, velocities, step, int(10.0 / step), stride=20)
        ratio = f"{previous / run.energy_drift:.2f}" if previous else "   -"
        previous = run.energy_drift
        print(f"  {step:>8.4f} {run.energy_drift:>14.3e} {ratio:>7s} {run.constraint_drift:>13.1e}")
    print()
    print("  Halving the step quarters the energy drift: velocity Verlet is")
    print("  second order and symplectic, so the error stays in a band rather")
    print("  than growing.  The Gauss constraint is different -- it sits at 1e-13")
    print("  whatever the step, because the equations conserve it exactly and the")
    print("  integrator inherits that.  One is a property of the scheme, the other")
    print("  of the physics, and they behave differently on purpose.")
    print()
    print(f"  the scaling symmetry X -> sX, V -> s^2 V, t -> t/s holds to "
          f"{scaling_residual(matrices, velocities):.1e}")
    print()


def flat_directions() -> None:
    print("=" * 72)
    print("Commuting matrices cost nothing, and that is the whole point")
    print("=" * 72)
    matrices, velocities = separated_branes([-2.0, 0.0, 2.0])
    velocities[0] = np.diag([-0.3, 0.0, 0.3]).astype(complex)
    run = evolve(matrices, velocities, 0.01, 500)
    print("  three branes on a line, drifting apart at 0.3 each")
    print(f"    V at the start: {commutator_potential(run.matrices[0]):.2e}")
    print(f"    V at t = 5:     {commutator_potential(run.matrices[-1]):.2e}")
    print(f"    positions:      {np.round(run.eigenvalues(0)[-1], 6)}")
    print(f"    expected:       {np.round([-3.5, 0.0, 3.5], 6)}")
    print(f"    energy drift:   {run.energy_drift:.1e}")
    print()
    print("  Nothing brings them back.  These flat directions are why the matrix")
    print("  model describes branes that can separate rather than a bound system,")
    print("  and they are exactly what the flux term of the Myers effect lifts.")
    print()


def strings_between_branes() -> None:
    print("=" * 72)
    print("An off-diagonal entry is a string, and its mass is the separation")
    print("=" * 72)
    print("  Two branes at +/- r/2 along X_1, one off-diagonal element of X_2")
    print("  switched on.  The potential gives it a^2 r^2 against a kinetic")
    print("  adot^2, so omega = r -- measured here, not assumed:")
    print()
    print(f"  {'r':>6s} {'measured omega':>16s} {'ratio':>10s}")
    for separation in (0.25, 0.5, 1.0, 2.5, 4.0):
        matrices, velocities = stretched_mode(separation, amplitude=1e-5)
        step = 0.006 / separation  # the step tracks the period, so the cost does not
        run = evolve(matrices, velocities, step, int(24.0 * math.pi / separation / step))
        signal = run.element(1, 0, 1).real
        crossings = np.where(np.diff(np.sign(signal)) != 0)[0]
        period = 2.0 * float(np.mean(np.diff(crossings))) * step
        measured = 2.0 * math.pi / period
        print(f"  {separation:>6.2f} {measured:>16.5f} {measured / separation:>10.6f}")
    print()
    print()
    print("  The amplitude is kept at 1e-5 on purpose: at a finite one the string")
    print("  pulls the branes together and the frequency drifts down with them.")
    print("  omega = r is the statement for an infinitesimal excitation.")
    print()
    print("  X is measured in units where a string of length L weighs L, that is")
    print("  X = (separation)/(2 pi alpha').  In those units this is the stretched")
    print("  string of examples/05, arrived at from a matrix equation of motion")
    print("  rather than from a mode expansion.")
    print()


def chaos() -> None:
    print("=" * 72)
    print("The generic motion, and the one thing predictable about it")
    print("=" * 72)
    rng = np.random.default_rng(11)
    matrices, velocities = random_state(rng, 4, 3)
    fit = lyapunov_exponent(matrices, velocities, dt=0.002, steps=30_000)
    print(f"  N = 4, d = 3, E = {energy(matrices, velocities):.4f}:  {fit}")
    print()
    print("  Two configurations a part in 1e8 apart separate exponentially.  There")
    print("  is no small parameter in the equations to make that surprising.")
    print()
    print("  What *is* predictable is the scaling.  X -> sX with t -> t/s is a")
    print("  symmetry, so E goes like s^4 and lambda like s:  lambda ~ E^(1/4).")
    print()
    power, energies, exponents = lyapunov_scaling(
        matrices, velocities, scales=(0.5, 0.75, 1.0, 1.5, 2.0), dt=0.002, steps=20_000
    )
    print(f"  {'scale':>7s} {'E':>12s} {'lambda':>10s} {'lambda / E^(1/4)':>18s}")
    for scale, value, exponent in zip((0.5, 0.75, 1.0, 1.5, 2.0), energies, exponents, strict=True):
        print(f"  {scale:>7.2f} {value:>12.4f} {exponent:>10.5f} {exponent / value**0.25:>18.5f}")
    print()
    print(f"  fitted power over a factor of {energies[-1]/energies[0]:.0f} in energy: "
          f"{power:.4f}, against 1/4")
    print()


def figures() -> None:
    rng = np.random.default_rng(4)
    matrices, velocities = separated_branes([-6.0, -2.0, 2.0, 6.0])
    raw = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
    velocities = np.array([(m + m.conj().T) / 2.0 for m in raw]) * 0.28
    velocities[0] += np.diag([1.1, 0.4, -0.4, -1.1]).astype(complex)
    velocities = project_gauss(matrices, velocities)
    run = evolve(matrices, velocities, 0.004, 6000, stride=6)

    plot_matrix_worldlines(run.times, run.eigenvalues(0), FIG / "matrix_worldlines.png")
    print(f"  wrote {FIG / 'matrix_worldlines.png'}")

    potential = np.array([commutator_potential(frame) for frame in run.matrices])
    kinetic = run.energies - potential
    animate_matrix_eigenvalues(
        run.times, run.eigenvalues(0), kinetic, potential,
        FIG / "matrix_scattering.gif", stride=4,
    )
    print(f"  wrote {FIG / 'matrix_scattering.gif'}")
    print("    While the branes are apart the potential sits on the floor.  Where")
    print("    they meet, energy moves into the strings between them and they")
    print("    leave in directions the incoming state did not determine.")

    base_x, base_v = random_state(np.random.default_rng(11), 4, 3)
    fit = lyapunov_exponent(base_x, base_v, dt=0.002, steps=30_000)
    power, energies, exponents = lyapunov_scaling(
        base_x, base_v, scales=(0.5, 0.75, 1.0, 1.5, 2.0), dt=0.002, steps=20_000
    )
    plot_lyapunov(fit, energies, exponents, power, FIG / "lyapunov.png")
    print(f"  wrote {FIG / 'lyapunov.png'}")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    what_is_conserved()
    flat_directions()
    strings_between_branes()
    chaos()
    figures()
    print(f"  (kinetic energy of a state at rest: {kinetic_energy(np.zeros((3, 2, 2))):.1f})")
    print("done.")
