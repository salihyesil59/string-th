"""D-branes: tensions, stretched strings, and gauge symmetry from geometry.

Run:  python examples/05_dbranes.py

Prints Dp-brane tensions and their coupling dependence, follows the mass of a
stretched string as two branes separate, and reads off the gauge group of a
stack.  Writes ``figures/brane_separation.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.branes.dbrane import (  # noqa: E402
    BraneStack,
    dp_brane_tension,
    gauge_group,
    stretched_spectrum,
    tachyon_free_separation,
)
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.plots import plot_brane_separation  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
CONV = Conventions(alpha_prime=1.0, dim=26)


def tensions() -> None:
    print("=" * 72)
    print("Dp-brane tension:  T_p = 1 / ((2 pi)^p g_s alpha'^{(p+1)/2})")
    print("=" * 72)
    header = "  ".join(f"{name:>12s}" for name in ("g_s = 1", "g_s = 0.1", "g_s = 0.01"))
    print(f"  {'p':>3s} {header}")
    for p in range(0, 10, 3):
        row = "  ".join(f"{dp_brane_tension(p, g, CONV):>12.6g}" for g in (1.0, 0.1, 0.01))
        print(f"  {p:>3d} {row}")
    print("  The single power of 1/g_s is the point: a D-brane is heavy at weak")
    print("  coupling (invisible to perturbation theory) and light at strong coupling.")
    print("  Compare a fundamental string, whose tension does not depend on g_s at all,")
    print("  and a soliton in field theory, which would scale as 1/g_s^2.")
    print()


def separation() -> None:
    print("=" * 72)
    print("A string stretched between two parallel branes")
    print("=" * 72)
    print(f"  {'d':>8s} {'N=0':>12s} {'N=1':>12s} {'N=2':>12s}")
    seps = [0.0, 1.0, 3.0, tachyon_free_separation(CONV), 8.0]
    for d in seps:
        levels = stretched_spectrum(d, 2, CONV)
        row = "  ".join(f"{lv.mass_squared:>10.4f}" for lv in levels)
        print(f"  {d:>8.4f} {row}")
    print(f"  the level-0 state stops being tachyonic at d = 2 pi sqrt(alpha') = "
          f"{tachyon_free_separation(CONV):.4f}")
    print("  At d = 0 the level-1 states are massless vectors; pulling the branes")
    print("  apart gives them a mass proportional to the distance.  That is the")
    print("  Higgs mechanism with the vacuum expectation value read as a length.")
    print()

    grid = np.linspace(0.0, 10.0, 200)
    plot_brane_separation(grid, [stretched_spectrum(d, 2, CONV) for d in grid],
                          FIG / "brane_separation.png")
    print(f"  wrote {FIG / 'brane_separation.png'}\n")


def stacks() -> None:
    print("=" * 72)
    print("Gauge group of a stack")
    print("=" * 72)
    for positions in ([0, 0, 0, 0], [0, 0, 0, 2.5], [0, 0, 2.5, 2.5], [0, 1, 2, 3]):
        stack = BraneStack(tuple(float(x) for x in positions))
        print(
            f"  positions {str(positions):<16s} -> {gauge_group(positions):<20s} "
            f"{stack.massless_vectors:>3d} massless vectors"
        )
    print("  N coincident branes carry U(N): the massless vectors are the N^2 ways")
    print("  of choosing which brane each end of the string lands on.  Separating")
    print("  them breaks the group, and the broken generators are exactly the")
    print("  strings that now have to stretch.")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    tensions()
    separation()
    stacks()
    print("done.")
