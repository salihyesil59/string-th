"""The heterotic string on a torus: Gamma_{16+d,d} and Wilson lines.

Run:  python examples/11_heterotic_compactified.py

Shows the charge lattice growing to signature (16+d, d), checks the two
reductions the module has to satisfy -- to the uncompactified heterotic string
at d = 0 and to the plain torus at zero gauge charge -- and works through the
standard Wilson-line breakings of both ten-dimensional gauge groups, then adds
the winding states that put the symmetry back at special radii.  Writes
``figures/wilson_breaking.png`` and ``figures/wilson_enhancement.png``.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.compactification.torus import TorusBackground, narain_momenta  # noqa: E402
from stringsim.heterotic.compactified import (  # noqa: E402
    GAUGE_RANK,
    HeteroticBackground,
    charge_lattice_gram,
    enhanced_algebra,
    enhancement_radii,
    gauge_algebra,
    massless_vectors,
    narain_form,
    unbroken_roots,
    wilson_boost,
)
from stringsim.heterotic.lattice import d16_plus, e8_squared  # noqa: E402
from stringsim.heterotic.spectrum import left_mass  # noqa: E402
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.plots import (  # noqa: E402
    plot_root_connectivity,
    plot_wilson_enhancement,
)

FIG = Path(__file__).resolve().parents[1] / "figures"
CONV = Conventions()
CIRCLE = TorusBackground(np.array([[1.7]]), conventions=CONV)


def line(values, dim: int = 1) -> np.ndarray:
    """A Wilson line with the given leading gauge components."""
    a = np.zeros((GAUGE_RANK, dim))
    a[: len(values), 0] = values
    return a


def the_lattice_grows() -> None:
    print("=" * 72)
    print("The charge lattice becomes Gamma_{16+d,d}")
    print("=" * 72)
    print(f"  {'d':>2s} {'spacetime':>10s} {'signature':>12s} {'rank':>5s} {'moduli':>7s}"
          f"  {'|det eta|':>10s}")
    for dim in range(5):
        torus = None if dim == 0 else TorusBackground(np.eye(dim), conventions=CONV)
        bg = HeteroticBackground(e8_squared(), torus)
        eta = bg.narain_form()
        print(
            f"  {dim:>2d} {bg.spacetime_dimension:>10d} {str(bg.signature):>12s} "
            f"{bg.rank:>5d} {bg.moduli_count:>7d}  {abs(np.linalg.det(eta)):>10.0f}"
        )
    print("  moduli = d(d+1)/2 from G, d(d-1)/2 from B, and 16d Wilson lines = d(d+16),")
    print("  which is the dimension of O(16+d,d)/(O(16+d) x O(d)).")
    print("  Rank 16 + 2d: sixteen internal, plus one each from G and B per direction.")
    print()


def wilson_lines_are_rotations() -> None:
    print("=" * 72)
    print("A Wilson line is an O(16+d,d) rotation")
    print("=" * 72)
    rng = np.random.default_rng(4)
    for dim in (1, 2, 3):
        boost, eta = wilson_boost(rng.normal(size=(GAUGE_RANK, dim))), narain_form(dim)
        residual = np.max(np.abs(boost.T @ eta @ boost - eta))
        print(f"  d={dim}: max |Omega^T eta Omega - eta| = {residual:.2e}")
    print("  The shift pi -> pi + A w on its own would spoil pi^2 + 2 n.w; the")
    print("  compensating momentum shift is exactly what cancels the cross terms.")
    print()
    bg = HeteroticBackground(e8_squared(), CIRCLE, line([0.4, -0.9]))
    h, eta = bg.generalized_metric(), bg.narain_form()
    print(f"  H symmetric {np.allclose(h, h.T)}, positive definite "
          f"{bool(np.all(np.linalg.eigvalsh(h) > 0))}, in O(17,1): "
          f"{np.allclose(h @ np.linalg.inv(eta) @ h, eta, atol=1e-7)}")
    print()


def reductions() -> None:
    print("=" * 72)
    print("Two reductions the module has to satisfy")
    print("=" * 72)
    bare = HeteroticBackground(e8_squared())
    roots = bare.gauge_lattice.roots
    worst = max(
        max(abs(a - 2.0), abs(b))
        for a, b in (bare.momenta_squared(bare.charge_vector(r)) for r in roots)
    )
    print(f"  d = 0: all {len(roots)} roots give (p_L^2, p_R^2) = (2, 0) to {worst:.1e}")
    zero = bare.charge_vector(np.zeros(GAUGE_RANK))
    print(f"         alpha'M^2 at N_L = 0, 1: {bare.alpha_m2(zero):.1f}, "
          f"{bare.alpha_m2(zero, 1):.1f}   "
          f"(heterotic.spectrum gives {4 * float(left_mass(0, 0)):.1f}, "
          f"{4 * float(left_mass(1, 0)):.1f})")
    print()
    for dim in (1, 2, 3):
        metric = np.eye(dim) * 1.7 + 0.2 * (np.ones((dim, dim)) - np.eye(dim))
        torus = TorusBackground(metric, conventions=CONV)
        bg = HeteroticBackground(e8_squared(), torus)
        worst = 0.0
        for momentum in itertools.product((-1, 0, 2), repeat=dim):
            for winding in itertools.product((-2, 0, 1), repeat=dim):
                mine = bg.momenta_squared(bg.charge_vector(np.zeros(GAUGE_RANK), winding, momentum))
                left, right = narain_momenta(torus, momentum, winding)
                worst = max(
                    worst,
                    abs(mine[0] - float(left @ left)),
                    abs(mine[1] - float(right @ right)),
                )
        print(f"  d = {dim}, zero gauge charge: max difference from torus.py = {worst:.1e}")
    print("  A generalisation that cannot reproduce what it generalises is not one.")
    print()


def breaking() -> None:
    print("=" * 72)
    print("Wilson lines break the gauge group")
    print("=" * 72)
    print("  A gauge boson needs p_R = 0 and p_L^2 = 2.  With no winding that forces")
    print("  n_i = A_i . pi, and n is an integer, so a root survives only when")
    print("  A_i . pi is an integer.  That single condition is the whole mechanism.")
    print()
    cases = [
        ("A = 0", np.zeros((GAUGE_RANK, 1))),
        ("A = (1, 0^7 ; 0^8)", line([1.0])),
        ("A = (1/2, 1/2, 0^6 ; 0^8)", line([0.5, 0.5])),
        ("A = (1/3, 0^7 ; 0^8)", line([1 / 3])),
        ("A = (1/2^8 ; 0^8)", line([0.5] * 8)),
    ]
    for name, lattice in (("E8 x E8", e8_squared()), ("Spin(32)/Z2", d16_plus())):
        print(f"  --- {name} on a circle ---")
        for label, lines in cases:
            bg = HeteroticBackground(lattice, CIRCLE, lines)
            surviving = unbroken_roots(bg)
            print(f"    {label:<28s} {len(surviving):>4d} roots  ->  {gauge_algebra(bg)}")
        print()
    print("  A = (1/2^8 ; 0^8) leaves E8 x E8 alone: every E8 vector has even coordinate")
    print("  sum, so A . pi is an integer for all 480 roots.  The same Wilson line cuts")
    print("  Spin(32)/Z2 in half, because its roots do not share that property.")
    print()
    print("  Wilson lines are periodic: shifting A by a lattice vector changes nothing,")
    print("  since the lattice is integral and lambda . pi is then an integer.")
    base = line([0.5, 0.5])
    reference = gauge_algebra(HeteroticBackground(e8_squared(), CIRCLE, base))
    shifted = [
        gauge_algebra(
            HeteroticBackground(
                e8_squared(), CIRCLE, base + root.reshape(GAUGE_RANK, 1)
            )
        )
        for root in e8_squared().roots[:5]
    ]
    print(f"    A: {reference}")
    print(f"    A + lattice vectors: {set(shifted)}")
    print()


def winding_puts_it_back() -> None:
    print("=" * 72)
    print("Winding states, and a Wilson line that undoes itself")
    print("=" * 72)
    print("  unbroken_roots sees only w = 0.  The complete condition is")
    print("    |pi + A w|^2 = 2 - 2 w^T G w   and   E w + A^T pi + (1/2) A^T A w in Z,")
    print("  and it is finite: the left side cannot be negative, so w^T G w <= 1")
    print("  bounds the winding and each w leaves a ball of radius sqrt(2) for pi.")
    print("  On a circle the first equation *solves* for G, so the special radii")
    print("  are computed, not found by scanning the moduli space.")
    print()
    cases = [
        ("E8 x E8", e8_squared(), "A = 0", line([0.0])),
        ("E8 x E8", e8_squared(), "A = (1/2, 0^7 ; 0^8)", line([0.5])),
        ("E8 x E8", e8_squared(), "A = (1, 0^7 ; 0^8)", line([1.0])),
        ("Spin(32)/Z2", d16_plus(), "A = 0", line([0.0])),
        ("Spin(32)/Z2", d16_plus(), "A = (1, 0^15)", line([1.0])),
        ("Spin(32)/Z2", d16_plus(), "A = (1/4^16)", line([0.25] * 16)),
    ]
    for name, lattice, label, lines in cases:
        generic = HeteroticBackground(lattice, CIRCLE, lines)
        print(f"  {name}, {label}")
        print(f"    generic radius:      {len(massless_vectors(generic)):>4d} roots  ->  "
              f"{enhanced_algebra(generic)}")
        for metric in enhancement_radii(lattice, lines[:, 0], winding_max=3):
            bg = HeteroticBackground(lattice, TorusBackground(np.array([[metric]])), lines)
            charges = massless_vectors(bg)
            winding = int(np.sum(np.abs(charges[:, GAUGE_RANK]) > 1e-9))
            print(f"    G = {metric:<8.5f} {len(charges):>4d} roots  ->  {enhanced_algebra(bg)}"
                  f"   ({winding} carry winding)")
        print()
    print("  A = (1/2, 0^7; 0^8) breaks E8 x E8 to e8 + so(14) at a generic radius,")
    print("  and at G = 1/8 all 480 roots are back -- 156 of them with winding.")
    print("  A Wilson line is not gauge-invariant information on its own: that is")
    print("  the same point of moduli space as A = 0, reached by O(17,1;Z).")
    print()


def the_two_theories_meet() -> None:
    print("=" * 72)
    print("Nine dimensions: one theory, two names")
    print("=" * 72)
    pair = [
        ("E8 x E8", e8_squared(), line([1.0] + [0.0] * 7 + [1.0])),
        ("Spin(32)/Z2", d16_plus(), line([0.5] * 8)),
    ]
    for name, lattice, lines in pair:
        bg = HeteroticBackground(lattice, CIRCLE, lines)
        radii = enhancement_radii(lattice, lines[:, 0], winding_max=4)
        print(f"  {name:<12s} {len(massless_vectors(bg)):>4d} roots  ->  {enhanced_algebra(bg)}")
        print(f"               enhancement radii with G > 1/16: {radii if radii else 'none'}")
    print()
    print("  The same gauge content at every radius, and neither has a point where")
    print("  anything extra comes down.  The lattice statement behind it: both")
    print("  charge lattices are even, self-dual and of signature (17, 1), and such")
    print("  a lattice is unique up to isomorphism -- so this is one theory.")
    for name, lattice, lines in pair:
        bg = HeteroticBackground(lattice, CIRCLE, lines)
        gram = charge_lattice_gram(bg)
        eigenvalues = np.linalg.eigvalsh(gram)
        signature = (int(np.sum(eigenvalues > 0)), int(np.sum(eigenvalues < 0)))
        even = bool(np.all(np.abs(np.diag(gram) % 2) < 1e-9))
        print(f"    {name:<12s} Gamma_{{17,1}} Gram: even {even}, "
              f"|det| {abs(np.linalg.det(gram)):.0f}, signature {signature}")
    print()


def figures() -> None:
    unbroken = HeteroticBackground(e8_squared(), CIRCLE)
    broken = HeteroticBackground(e8_squared(), CIRCLE, line([0.5, 0.5]))
    plot_root_connectivity(
        [
            (f"$A = 0$: {len(unbroken_roots(unbroken))} roots, {gauge_algebra(unbroken)}",
             unbroken_roots(unbroken)),
            # the braces of \frac must be doubled, or the f-string eats them
            (f"$A = (\\frac{{1}}{{2}},\\frac{{1}}{{2}},0^6;0^8)$: "
             f"{len(unbroken_roots(broken))} roots, {gauge_algebra(broken)}",
             unbroken_roots(broken)),
        ],
        FIG / "wilson_breaking.png",
        title="A Wilson line deletes roots, and the surviving system falls apart",
    )
    print(f"  wrote {FIG / 'wilson_breaking.png'}")
    print("  The second panel has three blocks where the first had two: one E8 is")
    print("  untouched, the other has split into e7 and su(2).")
    print()

    lattice = e8_squared()
    points = []
    for value in np.linspace(0.0, 1.0, 81):
        lines = line([value])
        for metric in enhancement_radii(lattice, lines[:, 0], winding_max=2):
            bg = HeteroticBackground(lattice, TorusBackground(np.array([[metric]])), lines)
            points.append((value, metric, len(massless_vectors(bg))))
    plot_wilson_enhancement(
        points,
        FIG / "wilson_enhancement.png",
        generic_count=len(massless_vectors(HeteroticBackground(lattice, CIRCLE, line([0.5])))),
    )
    print(f"  wrote {FIG / 'wilson_enhancement.png'}")
    print("  Each arc is an exact locus, not a sampled one -- the radii are solved")
    print("  for.  The bright arc peaking at a = 1/2, G = 1/8 reaches 480: the")
    print("  broken group restored entirely by winding states.")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_lattice_grows()
    wilson_lines_are_rotations()
    reductions()
    breaking()
    winding_puts_it_back()
    the_two_theories_meet()
    figures()
    print("done.")
