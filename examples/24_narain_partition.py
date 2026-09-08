"""The Narain partition function: why the lattice has to be even and self-dual.

Run:  python examples/24_narain_partition.py

``examples/07`` builds the Narain lattice and reads a spectrum off it;
``examples/15`` integrates a modular-invariant density over the fundamental
domain.  Neither mentions the other.  The lattice theta series is the object
that joins them, and it turns a docstring's claim -- that even self-duality is
what makes the one-loop amplitude modular invariant -- into a computation.

The two halves of that claim fail separately, and this shows them failing.

Writes ``figures/modular_invariance.png``, ``figures/narain_levels.png`` and
``figures/narain_radius.gif``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.amplitudes.narain import (  # noqa: E402
    charge_vectors,
    compactified_integrand,
    duality_residual,
    evenness,
    level_degeneracies,
    modular_residual,
    root_multiplicity,
    theta_series,
    truncation_gap,
)
from stringsim.amplitudes.oneloop import torus_integrand  # noqa: E402
from stringsim.compactification.torus import (  # noqa: E402
    TorusBackground,
    b_shift,
    basis_change,
    factorized_duality,
    gauge_algebra,
    root_vectors,
)
from stringsim.viz.animate import animate_narain_levels  # noqa: E402
from stringsim.viz.plots import plot_modular_invariance, plot_narain_levels  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
TAU = 0.3 + 1.1j
CUTOFF = 24.0


def circle(radius: float) -> TorusBackground:
    """A circle of the given radius in units of ``sqrt(alpha')``."""
    return TorusBackground(metric=np.array([[radius**2]]))


SELF_DUAL_2 = TorusBackground(metric=np.eye(2))


def generic() -> TorusBackground:
    rng = np.random.default_rng(5)
    a = rng.normal(size=(2, 2))
    return TorusBackground(
        metric=a @ a.T + 1.5 * np.eye(2),
        b_field=np.array([[0.0, 0.4], [-0.4, 0.0]]),
    )


def the_sum() -> None:
    """A lattice sum that actually converges."""
    print("The theta series")
    print("-" * 70)
    print("  Theta = sum over (w, n) of q^(l_L^2/2) qbar^(l_R^2/2).")
    print("  Each term has modulus exp(-pi tau_2 (l_L^2 + l_R^2)), and that")
    print("  combination is positive definite -- the generalized metric -- even")
    print("  though l_L^2 - l_R^2 is not.  So the sum converges and the cutoff")
    print("  is a precision knob, not an approximation of unknown quality.")
    print()
    print("      background          charges   Theta(tau)          truncation")
    for label, background in (
        ("self-dual T^1", circle(1.0)),
        ("self-dual T^2", SELF_DUAL_2),
        ("generic T^2 + B", generic()),
    ):
        value = theta_series(background, TAU, CUTOFF)
        print(f"    {label:<18s} {len(charge_vectors(background, CUTOFF)):6d}   "
              f"{value.real:10.6f}          "
              f"{truncation_gap(background, TAU, CUTOFF):.1e}")
    print()


def the_two_conditions() -> list:
    """T needs even; S needs self-dual."""
    print("What modular invariance needs")
    print("-" * 70)
    print("  T: tau -> tau + 1 multiplies each term by exp(2 pi i n.w).  That is")
    print("     1 because the lattice is EVEN, and nothing else is used.")
    print("  S: tau -> -1/tau resums the lattice against its dual, so it is the")
    print("     same series only because the lattice is SELF-DUAL.")
    print()
    print("  Restricting the momenta to multiples of s gives a sublattice of")
    print("  index s^d: still even, no longer self-dual.")
    print()
    print("      lattice                    even?        T residual   S residual")
    rows = []
    for scale in (1, 2, 3, 4):
        label = "Gamma_{2,2}" if scale == 1 else f"momenta in {scale}Z"
        shift, invert = modular_residual(SELF_DUAL_2, TAU, CUTOFF, momentum_scale=scale)
        gap = truncation_gap(SELF_DUAL_2, TAU, CUTOFF, momentum_scale=scale)
        even = evenness(SELF_DUAL_2, CUTOFF, scale)
        print(f"    {label:<24s} {even:.1e}      {shift:.2e}     {invert:.2e}")
        rows.append((label, shift, invert, gap))
    print()
    print("  The T column never moves.  The S column goes from round-off to a")
    print("  tenth, and the lattice sums have converged, so that is the lattice")
    print("  and not the arithmetic.")
    print()
    print("  The S column does not keep growing, which is worth saying: from")
    print("  s = 3 on it is flat.  Removing momentum modes only matters while")
    print("  they contribute, and at tau_2 = 1.1 the lightest one already")
    print("  carries exp(-4 pi tau_2) ~ 1e-6.  Lowering tau_2 raises the plateau.")
    print()
    return rows


def the_weight() -> list:
    """The theta series is not invariant on its own."""
    print("Where the invariance comes from")
    print("-" * 70)
    print("  Theta is not invariant: it picks up |tau|^d under S, and |eta|^{2d}")
    print("  picks up exactly the same.  The invariance is that cancellation.")
    print()
    print("      |tau|    |Theta(-1/tau)/Theta(tau)|     |tau|^d")
    samples = []
    for tau in (0.2 + 0.8j, 0.3 + 1.1j, 0.1 + 1.6j, 0.4 + 2.2j):
        here = theta_series(SELF_DUAL_2, tau, CUTOFF)
        there = theta_series(SELF_DUAL_2, -1.0 / tau, CUTOFF)
        ratio = abs(there / here)
        expected = abs(tau) ** SELF_DUAL_2.dim
        print(f"    {abs(tau):6.4f}   {ratio:22.10f}   {expected:12.10f}")
        samples.append((abs(tau), ratio, expected))
    print()
    return samples


def the_bridge() -> None:
    """Both modules, joined."""
    print("The two modules it joins")
    print("-" * 70)
    empty = TorusBackground(metric=np.zeros((0, 0)))
    here = compactified_integrand(empty, TAU, CUTOFF).real
    there = torus_integrand(TAU)
    print("  With no compact directions the theta series is 1 and the integrand")
    print("  collapses onto oneloop.torus_integrand:")
    print(f"    d = 0:  {here:.8f}   vs   {there:.8f}   "
          f"(relative {abs(here / there - 1):.1e})")
    print()
    print("  O(d,d;Z) moves the moduli and the charges together, so the")
    print("  partition function of two T-dual backgrounds is one number:")
    for label, omega in (
        ("factorized duality", factorized_duality(2, 0)),
        ("B-shift", b_shift(np.array([[0.0, 1.0], [-1.0, 0.0]]))),
        ("basis change", basis_change(np.array([[1, 1], [0, 1]]))),
    ):
        print(f"    {label:<20s} residual {duality_residual(generic(), omega, TAU, CUTOFF):.2e}")
    print()
    print("  And the roots read off the theta expansion are the ones")
    print("  root_vectors solves for exactly:")
    for label, background in (
        ("self-dual T^1", circle(1.0)),
        ("self-dual T^2", SELF_DUAL_2),
        ("R = 1.3 circle", circle(1.3)),
        ("generic T^2 + B", generic()),
    ):
        exact = sum(len(side) for side in root_vectors(background))
        print(f"    {label:<18s} theta {root_multiplicity(background, CUTOFF):2d}   "
              f"exact {exact:2d}   {gauge_algebra(background).name}")
    print()


def figures(rows: list, samples: list) -> None:
    print("Figures")
    print("-" * 70)

    plot_modular_invariance(rows, samples, FIG / "modular_invariance.png")
    print(f"  wrote {FIG / 'modular_invariance.png'}")

    panels = [
        ("self-dual circle", level_degeneracies(circle(1.0), 16.0)),
        ("$R = 1.3$", level_degeneracies(circle(1.3), 16.0)),
        ("self-dual $T^2$", level_degeneracies(SELF_DUAL_2, 16.0)),
    ]
    plot_narain_levels(panels, FIG / "narain_levels.png")
    print(f"  wrote {FIG / 'narain_levels.png'}")

    radii = np.exp(np.linspace(np.log(0.45), np.log(1 / 0.45), 61))
    radii = np.sort(np.concatenate([radii, [1.0]]))
    frames = [(float(r), level_degeneracies(circle(float(r)), 16.0)) for r in radii]
    animate_narain_levels(frames, FIG / "narain_radius.gif")
    print(f"  wrote {FIG / 'narain_radius.gif'}")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_sum()
    rows = the_two_conditions()
    samples = the_weight()
    the_bridge()
    figures(rows, samples)
    print("done.")
