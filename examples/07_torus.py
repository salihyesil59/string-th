"""Toroidal compactification: the Narain lattice, O(d,d;Z), and gauge symmetry.

Run:  python examples/07_torus.py

Checks that d = 1 reproduces the circle module exactly, verifies that every
O(d,d;Z) generator leaves the spectrum alone, and reads the gauge algebra off
the root system at several points in moduli space.  Writes
``figures/roots_su2.png``, ``figures/roots_su3.png`` and
``figures/torus_enhancement.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.compactification import circle  # noqa: E402
from stringsim.compactification.torus import (  # noqa: E402
    TorusBackground,
    b_shift,
    basis_change,
    decompose_roots,
    factorized_duality,
    gauge_algebra,
    identify_algebra,
    is_odd_integer,
    narain_gram_matrix,
    narain_momenta,
    odd_metric,
    root_vectors,
    spectrum,
    spectrum_is_dual,
    transform,
)
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.plots import plot_enhancement_map, plot_root_system  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
CONV = Conventions(alpha_prime=1.0, dim=26)


def reduces_to_the_circle() -> None:
    print("=" * 72)
    print("d = 1 is the circle module, level by level")
    print("=" * 72)
    for radius in (0.35, 1.0, 1.7, 3.0):
        bg = TorusBackground.from_radii([radius], CONV)
        mine = sorted(round(s.alpha_m2, 10) for s in spectrum(bg, charge_max=2, level_max=2))
        theirs = sorted(
            round(s.alpha_m2, 10)
            for s in circle.spectrum(radius, CONV, n_max=2, w_max=2, level_max=2)
        )
        print(
            f"  R = {radius:.2f}: {len(mine):>3d} states, "
            f"identical to circle.py: {mine == theirs}"
        )
    print("  Two independent routes to the same spectrum -- the torus code is not")
    print("  a generalisation on paper only.")
    print()


def the_two_forms() -> None:
    print("=" * 72)
    print("Two quadratic forms on one lattice")
    print("=" * 72)
    bg = TorusBackground(
        np.array([[1.7, 0.35], [0.35, 2.3]]), np.array([[0.0, 0.4], [-0.4, 0.0]]), CONV
    )
    h = bg.generalized_metric()
    eta = odd_metric(2)
    print(f"  moduli: G symmetric + B antisymmetric = d^2 = {bg.moduli_count}")
    print(f"  H is symmetric: {np.allclose(h, h.T)}, positive definite: "
          f"{np.all(np.linalg.eigvalsh(h) > 0)}, and lies in O(2,2): "
          f"{np.allclose(h @ eta @ h, eta)}")
    worst_mass, worst_match = 0.0, 0.0
    for n in [(1, 0), (0, 1), (2, -1), (3, 2)]:
        for w in [(1, 1), (0, -2), (1, 0), (-1, 3)]:
            left, right = narain_momenta(bg, n, w)
            z = np.concatenate([w, n]).astype(float)
            worst_mass = max(worst_mass, abs(left @ left + right @ right - z @ h @ z))
            worst_match = max(worst_match, abs(left @ left - right @ right - 2 * np.dot(n, w)))
    print(f"  max |l_L^2 + l_R^2 - Z^T H Z|  = {worst_mass:.2e}   (the mass)")
    print(f"  max |l_L^2 - l_R^2 - 2 n.w|    = {worst_match:.2e}   (level matching)")
    gram = narain_gram_matrix(2)
    eigenvalues = np.linalg.eigvalsh(gram)
    print(f"  Narain lattice: |det| = {abs(np.linalg.det(gram)):.0f} (self-dual), "
          f"signature ({int(sum(eigenvalues > 0))}, {int(sum(eigenvalues < 0))}), "
          "norms 2 n.w (even)")
    print("  Even self-duality is what makes the one-loop amplitude modular invariant,")
    print("  and it holds for every value of the moduli -- H moves, eta does not.")
    print()


def duality_group() -> None:
    print("=" * 72)
    print("O(d,d;Z): the circle's R -> alpha'/R is one generator among many")
    print("=" * 72)
    bg = TorusBackground(
        np.array([[1.7, 0.35], [0.35, 2.3]]), np.array([[0.0, 0.4], [-0.4, 0.0]]), CONV
    )
    generators = {
        "basis change": basis_change(np.array([[1, 1], [0, 1]])),
        "B-shift by 1": b_shift(np.array([[0.0, 1.0], [-1.0, 0.0]])),
        "duality on X^1": factorized_duality(2, 0),
        "duality on X^2": factorized_duality(2, 1),
    }
    generators["composition"] = (
        generators["duality on X^1"] @ generators["B-shift by 1"] @ generators["basis change"]
    )
    for name, omega in generators.items():
        moved = transform(bg, omega)
        print(
            f"  {name:<16s} in O(2,2;Z): {str(is_odd_integer(omega)):<5s} "
            f"spectrum unchanged: {str(spectrum_is_dual(bg, omega)):<5s} "
            f"det G: {np.linalg.det(bg.metric):.3f} -> {np.linalg.det(moved.metric):.3f}"
        )
    print()
    print("  A truncated charge box is not preserved by a basis change, so comparing")
    print("  the two enumerations as multisets would give a false negative.  The test")
    print("  above follows each state through the charge map instead.")
    print()
    for radius in (0.3, 0.8, 2.5):
        bg1 = TorusBackground.from_radii([radius], CONV)
        bg2 = transform(bg1, factorized_duality(1, 0))
        print(
            f"  d = 1: R = {radius:.2f} -> {float(np.sqrt(bg2.metric[0, 0])):.4f}, "
            f"alpha'/R = {circle.t_dual_radius(radius, CONV):.4f}"
        )
    print()


def gauge_symmetry() -> None:
    print("=" * 72)
    print("The gauge group is a root system")
    print("=" * 72)
    points = [
        ("generic T^2", TorusBackground.from_radii([1.3, 2.1], CONV)),
        ("one leg self-dual", TorusBackground.from_radii([1.0, 2.1], CONV)),
        ("self-dual T^1", TorusBackground.self_dual(1, CONV)),
        ("self-dual T^2", TorusBackground.self_dual(2, CONV)),
        ("self-dual T^3", TorusBackground.self_dual(3, CONV)),
        ("self-dual T^4", TorusBackground.self_dual(4, CONV)),
        ("A_2 point of T^2", TorusBackground.su3_point(CONV)),
    ]
    for label, bg in points:
        algebra = gauge_algebra(bg)
        print(f"  {label:<18s} {algebra}")
    print()
    print("  Every root has squared length 2, so only simply-laced algebras can appear:")
    for label, bg in points[3:]:
        left, _ = root_vectors(bg)
        if len(left):
            print(f"    {label:<18s} max |l^2 - 2| = "
                  f"{np.max(np.abs(np.sum(left * left, axis=1) - 2)):.1e}, "
                  f"components {decompose_roots(left)}")
    print()
    print("  Counting alone is not always enough -- rank 3 with six roots is ambiguous:")
    print(f"    identify_algebra(3, 6) = {identify_algebra(3, 6)}")
    print("    the root geometry settles it: three orthogonal pairs, so su(2)^3.")
    print()


def figures() -> None:
    self_dual = TorusBackground.self_dual(2, CONV)
    left, _ = root_vectors(self_dual)
    plot_root_system(
        left, FIG / "roots_su2.png", "Self-dual $T^2$", "four roots at right angles: su(2) + su(2)"
    )
    su3 = TorusBackground.su3_point(CONV)
    left, _ = root_vectors(su3)
    plot_root_system(
        left, FIG / "roots_su3.png", "$A_2$ point of $T^2$", "six roots at 60 degrees: su(3)"
    )
    print(f"  wrote {FIG / 'roots_su2.png'}")
    print(f"  wrote {FIG / 'roots_su3.png'}")

    # A rational grid, so the integrality condition is decided exactly.
    denominator = 24
    steps = np.arange(-denominator, denominator + 1) / denominator
    counts = np.zeros((len(steps), len(steps)), dtype=int)
    for i, g12 in enumerate(steps):
        for j, b12 in enumerate(steps):
            metric = np.array([[1.0, g12], [g12, 1.0]])
            if np.linalg.eigvalsh(metric)[0] <= 1e-9:
                continue
            bg = TorusBackground(metric, np.array([[0.0, b12], [-b12, 0.0]]), CONV)
            counts[i, j] = len(root_vectors(bg)[0])
    plot_enhancement_map(steps, steps, counts, FIG / "torus_enhancement.png")
    print(f"  wrote {FIG / 'torus_enhancement.png'}")
    print(f"  enhanced points found: {int(np.sum(counts > 0))} of {counts.size} grid points; "
          f"largest root count {int(counts.max())}")
    print("  The enhanced loci are lines, not regions: enhancement needs an exact")
    print("  integrality condition, so it happens on a measure-zero set.")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    reduces_to_the_circle()
    the_two_forms()
    duality_group()
    gauge_symmetry()
    figures()
    print("done.")
