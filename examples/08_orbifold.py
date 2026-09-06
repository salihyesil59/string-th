"""Orbifolds: the projection, the twisted sectors, and which ones exist at all.

Run:  python examples/08_orbifold.py

Finds the allowed orbifold orders by searching integer matrices, checks the
twisted-sector ground-state energy against a cut-off mode sum, shows the
Kaluza-Klein vectors being projected out of S^1/Z_2, and tabulates the twisted
spectra.  Writes ``figures/fixed_points_z3.png``, ``figures/fixed_points_z4.png``
and ``figures/orbifold_intercepts.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.compactification.orbifold import (  # noqa: E402
    Orbifold,
    crystallographic_orders,
    transverse_phases,
    twisted_spectrum,
    untwisted_degeneracy,
    untwisted_massless_content,
)
from stringsim.compactification.torus import TorusBackground  # noqa: E402
from stringsim.quantum.zeta import regularised_shifted_sum  # noqa: E402
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.plots import plot_fixed_points  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
CONV = Conventions(alpha_prime=1.0, dim=26)


def standard() -> dict[str, Orbifold]:
    return {
        "S^1/Z_2": Orbifold.inversion(TorusBackground.from_radii([1.7], CONV)),
        "T^2/Z_2": Orbifold.inversion(TorusBackground.from_radii([1.3, 2.1], CONV)),
        "T^4/Z_2": Orbifold.inversion(TorusBackground.from_radii([1.3, 2.1, 0.9, 1.5], CONV)),
        "T^2/Z_3": Orbifold.z3_hexagonal(CONV),
        "T^2/Z_4": Orbifold.z4_square(CONV),
        "T^2/Z_6": Orbifold.z6_hexagonal(CONV),
    }


def which_orbifolds_exist() -> None:
    print("=" * 72)
    print("Which orbifolds exist: the crystallographic restriction, by search")
    print("=" * 72)
    for dim in (1, 2):
        orders = sorted(crystallographic_orders(dim, bound=2))
        print(f"  d = {dim}: finite orders realised by integer matrices = {orders}")
    print("  Widening the search box changes nothing:",
          sorted(crystallographic_orders(2, bound=3)))
    print("  A finite-order integer matrix has roots of unity for eigenvalues, so its")
    print("  trace is both 2 cos(2 pi / N) and an integer.  That leaves N = 1, 2, 3, 4, 6,")
    print("  and it is why the list of T^2 orbifolds stops where it does.")
    print()


def geometry() -> None:
    print("=" * 72)
    print("Twist phases, ground-state energy and fixed points")
    print("=" * 72)
    print(f"  {'orbifold':<10s} {'N':>2s}  {'phases':<16s} {'a_1':>10s} {'fixed':>6s}")
    for name, orbifold in standard().items():
        phases = np.round(orbifold.twist_phases(1), 4)
        print(
            f"  {name:<10s} {orbifold.order:>2d}  {str(phases):<16s} "
            f"{orbifold.intercept(1):>10.6f} {orbifold.fixed_points(1):>6d}"
        )
    print()
    print("  Fixed-point positions in lattice coordinates:")
    for name in ("S^1/Z_2", "T^2/Z_3", "T^2/Z_4", "T^2/Z_6"):
        positions = standard()[name].fixed_point_positions(1)
        print(f"    {name:<10s} {np.round(positions, 4).tolist()}")
    print()


def ground_state_energy() -> None:
    print("=" * 72)
    print("The twisted intercept, two independent ways")
    print("=" * 72)
    print("  A boson with modes n + phi has zero-point energy (1/2) zeta(-1, phi),")
    print("  and zeta(-1, a) = -B_2(a)/2 comes out of a cut-off sum just as -1/12 did:")
    for shift in (1.0, 0.5, 1 / 3, 0.25):
        estimate = regularised_shifted_sum(shift)
        print(
            f"    zeta(-1, {shift:.4f}) = {estimate.value:+.10f}  "
            f"exact {estimate.exact:+.10f}  error {estimate.error:.1e}"
        )
    print()
    print("  Summing that over the 24 transverse bosons must reproduce the closed form")
    print("  a_k = 1 - (1/4) sum_j phi_j (1 - phi_j):")
    for name, orbifold in standard().items():
        for k in range(1, orbifold.order):
            phases = transverse_phases(orbifold, k)
            shifts = np.where(phases < 1e-12, 1.0, phases)
            zero_point = sum(0.5 * regularised_shifted_sum(float(s)).value for s in shifts)
            print(
                f"    {name:<10s} k={k}:  from mode sums {-zero_point:.8f}   "
                f"closed form {orbifold.intercept(k):.8f}   "
                f"difference {abs(-zero_point - orbifold.intercept(k)):.1e}"
            )
    print()


def projection() -> None:
    print("=" * 72)
    print("The untwisted projection: what the quotient deletes")
    print("=" * 72)
    for name in ("S^1/Z_2", "T^2/Z_2", "T^4/Z_2"):
        orbifold = standard()[name]
        content = untwisted_massless_content(orbifold)
        character = untwisted_degeneracy(orbifold, 1, 1)
        print(f"  {name}:  {content}")
        print(f"    {'':<9s} character projector gives {character}, index counting gives "
              f"{content.surviving} -- {'agree' if character == content.surviving else 'DISAGREE'}")
    print()
    print("  The deleted states are exactly the ones with one compact index: the")
    print("  Kaluza-Klein and winding gauge bosons g_{mu i} and B_{mu i}.  S^1/Z_2")
    print("  therefore has no massless vectors in its untwisted sector, while the")
    print("  graviton, the B field, the dilaton and the radius modulus all survive.")
    print()
    print("  For a rotation rather than a reflection only one compact pair survives:")
    for name in ("T^2/Z_3", "T^2/Z_4", "T^2/Z_6"):
        orbifold = standard()[name]
        print(f"    {name}: {untwisted_degeneracy(orbifold, 1, 1)} states "
              f"= 22^2 + 2 = {22**2 + 2}")
    print()


def twisted() -> None:
    print("=" * 72)
    print("Twisted sectors")
    print("=" * 72)
    for name, orbifold in standard().items():
        print(f"  {name}  (a_1 = {orbifold.intercept(1):.6f}):")
        for level in twisted_spectrum(orbifold, 1, n_levels=1)[:4]:
            print(f"    {level}")
    print()
    print("  The Z_4 orbifold's second sector is the Z_2 one, as it must be:")
    z4 = Orbifold.z4_square(CONV)
    for k in (1, 2, 3):
        print(f"    k={k}: phases {np.round(z4.twist_phases(k), 4)}, "
              f"a = {z4.intercept(k):.6f}, {z4.fixed_points(k)} fixed points")
    print()
    print("  An honest negative result: none of these has a massless twisted state.")
    print("  a_k never lands on the level lattice, so every twisted level is either")
    print("  tachyonic or massive.  The bosonic string keeps its instability.")
    print()


def figures() -> None:
    plot_fixed_points(Orbifold.z3_hexagonal(CONV), FIG / "fixed_points_z3.png")
    plot_fixed_points(Orbifold.z4_square(CONV), FIG / "fixed_points_z4.png", sectors=(1, 2))
    print(f"  wrote {FIG / 'fixed_points_z3.png'}")
    print(f"  wrote {FIG / 'fixed_points_z4.png'}")

    import matplotlib.pyplot as plt

    orbifolds = standard()
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=130)
    for name, orbifold in orbifolds.items():
        sectors = [k for k in range(1, orbifold.order) if orbifold.has_isolated_fixed_points(k)]
        ax.plot(
            [k / orbifold.order for k in sectors],
            [orbifold.intercept(k) for k in sectors],
            "o-",
            label=f"{name} ({orbifold.fixed_points(1)} fixed pts)",
        )
    ax.axhline(1.0, color="0.4", ls="--", lw=1.0, label="untwisted, $a = 1$")
    ax.set_xlabel("$k/N$ through the twisted sectors")
    ax.set_ylabel("$a_k$")
    ax.set_title("Twisting raises the ground-state energy, so $a_k$ falls below 1")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "orbifold_intercepts.png")
    plt.close(fig)
    print(f"  wrote {FIG / 'orbifold_intercepts.png'}")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    which_orbifolds_exist()
    geometry()
    ground_state_energy()
    projection()
    twisted()
    figures()
    print("done.")
