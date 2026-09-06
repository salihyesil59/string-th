"""The Veneziano amplitude, and what its poles know about the spectrum.

Run:  python examples/06_amplitudes.py

Locates the poles, takes their residues numerically and compares with the
closed-form polynomial, checks the Regge limit, measures the exponential
fall-off at high energy against Stirling, and does the same pole check for the
closed-string Virasoro-Shapiro amplitude.  Writes ``figures/veneziano.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.amplitudes.veneziano import (  # noqa: E402
    hard_scattering,
    regge_asymptotic,
    veneziano,
    veneziano_log_abs,
    veneziano_pole_positions,
    veneziano_residue,
    virasoro_shapiro,
    virasoro_shapiro_pole_positions,
)
from stringsim.quantum.spectrum import open_bosonic_spectrum  # noqa: E402
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.plots import plot_veneziano  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
CONV = Conventions(alpha_prime=1.0, dim=26)


def poles() -> None:
    print("=" * 72)
    print("Poles of A(s,t) versus the open-string mass levels")
    print("=" * 72)
    amp_poles = veneziano_pole_positions(5, CONV.alpha_prime)
    levels = open_bosonic_spectrum(5, CONV)
    for lv, s_pole in zip(levels, amp_poles, strict=True):
        print(f"  N = {lv.n}:  spectrum alpha'M^2 = {lv.alpha_m2:>5.1f}   "
              f"amplitude pole at alpha's = {s_pole:>5.1f}")
    print("  Same numbers.  The amplitude was written down before the spectrum existed.")
    print()


def residues() -> None:
    print("=" * 72)
    print("Residues: how much spin each level exchanges")
    print("=" * 72)
    # alpha(t) must avoid the non-negative integers, or the amplitude has a
    # pole in t as well and the residue in s cannot be taken numerically.
    t = np.array([-0.7, 0.35, 0.9, 2.3])
    eps = 1e-7
    print(f"  t = {t}")
    for n in range(5):
        numeric = eps * veneziano((n - 1) + eps, t, CONV.alpha_prime)
        exact = veneziano_residue(n, t, CONV.alpha_prime)
        err = float(np.max(np.abs(numeric - exact)))
        print(
            f"  n = {n}: residue {np.round(exact, 4)}  "
            f"(degree {n} in t)  numeric error {err:.1e}"
        )
    print("  Degree n means spin up to n and no higher: the leading Regge trajectory,")
    print("  now read off the amplitude rather than the spectrum.")
    print()


def regge() -> None:
    print("=" * 72)
    print("Regge limit: fixed t, large |s|")
    print("=" * 72)
    t = -0.4
    print(f"  t = {t};  comparing A(s,t) with Gamma(-alpha_t) (-alpha_s)^alpha_t")
    for s in (-50.0, -500.0, -5000.0, -50000.0):
        ratio = float(np.exp(veneziano_log_abs(s, t) - np.log(abs(regge_asymptotic(s, t)))))
        print(f"    s = {s:>10.0f}:  |A| / |asymptotic| = {ratio:.6f}")
    print("  The power of s slides with t.  A field-theory exchange would give a")
    print("  fixed power; a whole trajectory of states gives a moving one.")
    print()


def hard() -> None:
    print("=" * 72)
    print("Hard scattering: both invariants large")
    print("=" * 72)
    s = np.linspace(-4000.0, -2000.0, 80)
    for ratio in (0.4, 1.0, 2.5):
        hs = hard_scattering(s, ratio=ratio, alpha_prime=CONV.alpha_prime)
        print(
            f"  t/s = {ratio:>4.1f}:  measured d log|A|/ds = {hs.log_slope:.6f}, "
            f"Stirling {hs.predicted_log_slope:.6f}, "
            f"relative error {abs(hs.log_slope / hs.predicted_log_slope - 1):.2e}"
        )
    print("  Exponential in the energy, not a power.  Extended objects have no hard core.")
    print()


def closed() -> None:
    print("=" * 72)
    print("Virasoro-Shapiro: the closed-string counterpart")
    print("=" * 72)
    poles_s = virasoro_shapiro_pole_positions(4, CONV.alpha_prime)
    print(f"  poles at alpha' s = {poles_s}  i.e. alpha' M^2 = 4(N-1)")
    eps = 1e-7
    t = -2.3
    for n, s_pole in enumerate(poles_s):
        residue = eps * virasoro_shapiro(s_pole + eps, t, alpha_prime=CONV.alpha_prime)
        print(f"    N = {n}: pole at s = {s_pole:>5.1f}, residue {float(residue):+.5f}")
    print("  The closed spectrum has four times the spacing of the open one, and so")
    print("  does its amplitude.")
    print()


def figure() -> None:
    s = np.linspace(-1.6, 4.6, 4000)
    a = veneziano(s, -0.35, CONV.alpha_prime)
    plot_veneziano(s, a, veneziano_pole_positions(5, CONV.alpha_prime), FIG / "veneziano.png")
    print(f"  wrote {FIG / 'veneziano.png'}\n")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    poles()
    residues()
    regge()
    hard()
    closed()
    figure()
    print("done.")
