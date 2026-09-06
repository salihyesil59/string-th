"""A closed string on a circle: winding, T-duality, enhanced symmetry.

Run:  python examples/04_tduality.py

Shows that the spectrum at radius ``R`` and at ``alpha'/R`` is identical, finds
the extra massless states at the self-dual radius by enumeration, and writes
``figures/tduality.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.compactification.circle import (  # noqa: E402
    extra_massless_states,
    kaluza_klein_mass,
    lightest_mass,
    massless_states,
    self_dual_radius,
    spectrum,
    spectrum_is_t_dual,
    t_dual_radius,
    winding_mass,
)
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.plots import plot_tduality  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
CONV = Conventions(alpha_prime=1.0, dim=26)
RANGE = dict(n_max=3, w_max=3, level_max=2)


def towers() -> None:
    print("=" * 72)
    print("Two towers a point particle does not have")
    print("=" * 72)
    print(f"  {'R':>8s} {'KK n=1':>12s} {'winding w=1':>14s}   lighter")
    for r in (0.2, 0.5, 1.0, 2.0, 5.0):
        kk = kaluza_klein_mass(1, r)
        wd = winding_mass(1, r, CONV)
        which = "winding" if wd < kk else ("KK" if kk < wd else "degenerate")
        print(f"  {r:>8.2f} {kk:>12.4f} {wd:>14.4f}   {which}")
    print("  Shrinking the circle makes momentum expensive and winding cheap.")
    print("  That exchange is the whole content of T-duality.")
    print()


def duality() -> None:
    print("=" * 72)
    print("T-duality:  R  <->  alpha'/R,  n <-> w")
    print("=" * 72)
    for r in (0.35, 0.8, 1.6, 4.0):
        dual = t_dual_radius(r, CONV)
        ok = spectrum_is_t_dual(r, CONV, **RANGE)
        print(f"  R = {r:.3f}  <->  R' = {dual:.3f}   spectra identical: {ok}")
    r = 2.5
    a = sorted(s.alpha_m2 for s in spectrum(r, CONV, **RANGE))[:6]
    b = sorted(s.alpha_m2 for s in spectrum(t_dual_radius(r, CONV), CONV, **RANGE))[:6]
    print(f"  lightest six at R = {r}:      {np.round(a, 6)}")
    print(f"  lightest six at R = {t_dual_radius(r, CONV):.3f}:   {np.round(b, 6)}")
    print("  A circle smaller than sqrt(alpha') is a circle you have already seen.")
    print()


def enhancement() -> None:
    print("=" * 72)
    print("The self-dual radius")
    print("=" * 72)
    r0 = self_dual_radius(CONV)
    print(f"  R = sqrt(alpha') = {r0:.4f}")
    for r in (0.6, r0, 1.4):
        total = sum(s.degeneracy for s in massless_states(r, CONV, **RANGE))
        extra = extra_massless_states(r, CONV, **RANGE)
        print(
            f"  R = {r:.4f}:  {total:>4d} massless states, "
            f"{len(extra):>2d} of them carrying (n,w)"
        )
    print()
    print("  The extra states at the self-dual radius:")
    for s in extra_massless_states(r0, CONV, **RANGE):
        kind = "gauge boson" if (s.level, s.level_tilde) in ((1, 0), (0, 1)) else "tachyon tower"
        print(f"    {s}   [{kind}]")
    print("  The four oscillator states enlarge U(1)_L x U(1)_R to SU(2)_L x SU(2)_R.")
    print("  The four without oscillators are the bosonic tachyon passing through M^2 = 0,")
    print("  which is a feature of the bosonic string, not of the compactification.")
    print()


def figure() -> None:
    radii = np.geomspace(0.15, 6.0, 400)
    kk = [kaluza_klein_mass(1, r) for r in radii]
    wd = [winding_mass(1, r, CONV) for r in radii]
    plot_tduality(radii, kk, wd, self_dual_radius(CONV), FIG / "tduality.png")
    print(f"  wrote {FIG / 'tduality.png'}")
    print("  lightest massive state:")
    for r in (0.3, 1.0, 3.0):
        print(f"    R = {r:.2f}:  {lightest_mass(r, CONV, **RANGE)}")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    towers()
    duality()
    enhancement()
    figure()
    print("done.")
