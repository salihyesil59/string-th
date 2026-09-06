"""Which particle is which vibration.

Run:  python examples/02_particle_spectrum.py

Prints the open and closed bosonic spectra with their degeneracies and named
content, checks that the named pieces add up to the counted states, and writes
``figures/open_spectrum.png`` and ``figures/closed_spectrum.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.quantum.spectrum import (  # noqa: E402
    closed_bosonic_spectrum,
    leading_trajectory_spin,
    open_bosonic_spectrum,
    open_superstring_spectrum,
)
from stringsim.quantum.states import closed_massless_content, open_level_content  # noqa: E402
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.plots import plot_mass_spectrum  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
CONV = Conventions(alpha_prime=1.0, dim=26)


def open_string() -> None:
    print("=" * 72)
    print("Open bosonic string, D = 26:  alpha' M^2 = N - 1")
    print("=" * 72)
    levels = open_bosonic_spectrum(6, CONV)
    for lv in levels:
        print("  ", lv)
    print()
    print("  Named content of the low levels (little group SO(24) or SO(25)):")
    for n in range(4):
        pieces = open_level_content(n, CONV.dim)
        total = sum(p.dimension for p in pieces)
        counted = levels[n].degeneracy
        flag = "OK" if total == counted else "MISMATCH"
        print(f"    N = {n}  [{pieces[0].little_group}]  counted {counted}, named {total}  {flag}")
        for p in pieces:
            print(f"        {p}")
    print()
    plot_mass_spectrum(levels, FIG / "open_spectrum.png", "Open bosonic string")
    print(f"  wrote {FIG / 'open_spectrum.png'}\n")


def closed_string() -> None:
    print("=" * 72)
    print("Closed bosonic string:  alpha' M^2 = 4(N - 1),  N = Ntilde")
    print("=" * 72)
    levels = closed_bosonic_spectrum(4, CONV)
    for lv in levels:
        print("  ", lv)
    print()
    print("  The massless level is 24 x 24 = 576 states, and it splits into:")
    pieces = closed_massless_content(CONV.dim)
    for p in pieces:
        print(f"        {p}")
    total = sum(p.dimension for p in pieces)
    print(f"    total {total} = (D-2)^2 = {(CONV.dim - 2) ** 2}")
    print("    The spin-2 piece is a graviton.  It is not optional: every closed")
    print("    string has this state, so any theory of closed strings contains gravity.")
    print()
    plot_mass_spectrum(levels, FIG / "closed_spectrum.png", "Closed bosonic string")
    print(f"  wrote {FIG / 'closed_spectrum.png'}\n")


def superstring() -> None:
    print("=" * 72)
    print("Open superstring after the GSO projection, D = 10:  alpha' M^2 = N")
    print("=" * 72)
    for lv in open_superstring_spectrum(5, Conventions(alpha_prime=1.0, dim=10)):
        print("  ", lv)
    print("  No tachyon: the projection removes the NS ground state, and the")
    print("  8 massless states are the transverse polarisations of a gauge boson.")
    print()


def regge() -> None:
    print("=" * 72)
    print("Leading Regge trajectory")
    print("=" * 72)
    print("  Open string, intercept 1:")
    for lv in open_bosonic_spectrum(4, CONV):
        print(
            f"    alpha'M^2 = {lv.alpha_m2:>5.1f}   J_max = {lv.max_spin}   "
            f"alpha'M^2 + 1 = {leading_trajectory_spin(lv.alpha_m2):.1f}"
        )
    print("  The classical rotating string gave J = alpha' M^2 exactly;")
    print("  the +1 is the quantum normal-ordering constant, seen from a second side.")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    open_string()
    closed_string()
    superstring()
    regge()
    print("done.")
