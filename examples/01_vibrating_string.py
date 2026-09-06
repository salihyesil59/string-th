"""A string vibrating, two ways: from its amplitudes, and from a pluck.

Run:  python examples/01_vibrating_string.py

Produces, in ``figures/``:

* ``rotating_string.gif``   -- the rigidly rotating solution, endpoints at c
* ``excited_string.gif``    -- a three-mode excited state, level N = 6
* ``snapshots.png``         -- the same motion as still frames
* ``plucked_string.gif``    -- a triangle released from rest
* ``plucked_modes.png``     -- its normal-mode content

and prints the Virasoro residuals, which is the number that says whether what
is being drawn is a physical string or merely a wiggling curve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.classical.constraints import total_momentum, virasoro_residual  # noqa: E402
from stringsim.classical.evolve import evolve, mode_spectrum  # noqa: E402
from stringsim.classical.lightcone import (  # noqa: E402
    LightconeOpenString,
    oscillator_amplitudes,
)
from stringsim.classical.rotating import regge_trajectory, rigid_rotator  # noqa: E402
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.animate import animate_evolution, animate_modes, snapshot_grid  # noqa: E402
from stringsim.viz.plots import plot_mode_spectrum, plot_regge_trajectory  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
CONV = Conventions(alpha_prime=1.0, dim=26)


def light_cone_states() -> None:
    """Build states from transverse oscillators and check they solve the constraints."""
    print("=" * 72)
    print("Light-cone states: transverse amplitudes in, physical string out")
    print("=" * 72)
    cases = {
        "alpha_{-1}^1 |0>": ({(1, 0): 1}, None),
        "alpha_{-1}^1 alpha_{-1}^2 |0>": ({(1, 0): 1, (1, 1): 1}, {(1, 1): np.pi / 2}),
        "alpha_{-2}^1 |0>": ({(2, 0): 1}, None),
        "alpha_{-1}^1 alpha_{-2}^2 |0>": ({(1, 0): 1, (2, 1): 1}, {(2, 1): 0.4}),
    }
    for label, (excitation, phases) in cases.items():
        amps = oscillator_amplitudes(excitation, CONV.transverse_dim, phases)
        lc = LightconeOpenString(conventions=CONV, p_plus=1.0, transverse_modes=amps)
        full = lc.to_open_string()
        report = virasoro_residual(full, n_tau=25, n_sigma=41)
        p_err = np.max(np.abs(total_momentum(full) - lc.momentum()))
        print(
            f"{label:<32s} N = {lc.level():>4.1f}   "
            f"alpha'M^2 (classical) = {lc.mass_squared():>5.2f}   "
            f"(quantum) = {lc.mass_squared_quantum():>5.2f}"
        )
        print(
            f"{'':<32s} Virasoro residual {report.relative:.2e}, "
            f"momentum recovered to {p_err:.2e}"
        )
    print()


def rotating() -> None:
    """The rotating solution and the Regge trajectory it traces."""
    print("=" * 72)
    print("Rigidly rotating string: J = alpha' M^2")
    print("=" * 72)
    for point in regge_trajectory(conventions=CONV):
        print(
            f"  A = {point.amplitude:>4.1f}   alpha'M^2 = {point.alpha_m2:>8.5f}   "
            f"J = {point.spin:>8.5f}   difference {abs(point.alpha_m2 - point.spin):.2e}"
        )
    plot_regge_trajectory(regge_trajectory(conventions=CONV), FIG / "regge_trajectory.png")

    string = rigid_rotator(CONV, amplitude=2.0)
    animate_modes(
        string,
        FIG / "rotating_string.gif",
        projection=(1, 2, 3),
        n_frames=90,
        title="Rotating open string (endpoints move at the speed of light)",
    )
    snapshot_grid(
        string,
        FIG / "snapshots.png",
        projection=(1, 2),
        title="Rigidly rotating open string",
    )
    print(f"  wrote {FIG / 'rotating_string.gif'}")
    print(f"  wrote {FIG / 'snapshots.png'}\n")


def superposed_state() -> None:
    """A three-mode state, which is where the shape stops being a straight line."""
    amps = oscillator_amplitudes(
        {(1, 0): 1, (2, 1): 1, (3, 2): 1},
        CONV.transverse_dim,
        phases={(2, 1): np.pi / 3, (3, 2): np.pi / 5},
    )
    lc = LightconeOpenString(conventions=CONV, p_plus=1.0, transverse_modes=amps)
    animate_modes(
        lc.to_open_string(),
        FIG / "excited_string.gif",
        projection=(1, 2, 3),
        n_frames=120,
        title=r"$\alpha_{-1}\alpha_{-2}\alpha_{-3}|0\rangle$, level $N = 6$",
    )
    print(f"  wrote {FIG / 'excited_string.gif'}  (N = {lc.level():.0f})\n")


def plucked() -> None:
    """Pluck a string, release it, and read its harmonics back off."""
    print("=" * 72)
    print("Plucked string: modes measured, not assumed")
    print("=" * 72)
    n_points = 257
    sigma = np.linspace(0.0, np.pi, n_points)
    triangle = np.where(sigma < np.pi / 2, sigma, np.pi - sigma)
    # The second component is a clean n = 3 harmonic; both satisfy X' = 0 at the ends,
    # which the Neumann boundary condition requires of the *initial data* too.
    X0 = np.column_stack([triangle, 0.35 * np.cos(3 * sigma)])

    ev = evolve(X0, boundary="neumann", n_steps=600, courant=0.5)
    animate_evolution(ev, FIG / "plucked_string.gif", stride=4, title="plucked open string")

    coeffs = mode_spectrum(ev.X[0], boundary="neumann", n_modes=10)
    plot_mode_spectrum(
        coeffs, FIG / "plucked_modes.png", title="Harmonics of a triangular pluck"
    )
    for n, (a, b) in enumerate(coeffs):
        print(f"  n = {n}:  triangle {a:+.5f}   harmonic {b:+.5f}")
    print("  the triangle keeps only n = 0, 2, 6, 10, ... and falls off like 1/n^2;")
    print("  the second component is the single mode it was built from.")
    print(f"  wrote {FIG / 'plucked_string.gif'}")
    print(f"  wrote {FIG / 'plucked_modes.png'}\n")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    light_cone_states()
    rotating()
    superposed_state()
    plucked()
    print("done.")
