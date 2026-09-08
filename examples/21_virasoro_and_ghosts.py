"""The Virasoro algebra, built as matrices, and D = 26 from unitarity alone.

Run:  python examples/21_virasoro_and_ghosts.py

Everywhere else in the package the spectrum is *counted*.  Here the states are
built: an explicit Fock space of oscillators acting on ``|0; p>``, with the
indefinite Minkowski metric that makes timelike oscillators negative-norm.

Two things come out that were previously put in.  The central charge is read
off a commutator of matrices instead of quoted.  And the critical dimension --
already derived twice, from the normal-ordering constant and from the vanishing
total central charge, both anomaly arguments -- comes out a third time from
something unrelated: whether the physical states have positive norm.

Writes ``figures/central_charge.png``, ``figures/ghost_onset.png``,
``figures/no_ghost_region.png``, ``figures/no_ghost_region.gif`` and
``figures/physical_norms.gif``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.quantum.fock import (  # noqa: E402
    alpha_zero,
    basis_dimension,
    gram_diagonal,
    level_basis,
    momentum_for_level,
    state_norm,
)
from stringsim.quantum.states import sym_traceless_dim  # noqa: E402
from stringsim.quantum.virasoro import (  # noqa: E402
    algebra_residual,
    central_charge_from_algebra,
    constraint_residual,
    critical_dimension_from_norms,
    ghost_boundary,
    ghost_scan,
    inertia,
    lightcone_count,
    no_ghost_map,
    physical_states,
    virasoro_matrix,
)
from stringsim.quantum.zeta import central_charge, critical_dimension  # noqa: E402
from stringsim.viz.animate import (  # noqa: E402
    animate_no_ghost_region,
    animate_physical_norms,
)
from stringsim.viz.plots import (  # noqa: E402
    plot_central_charge,
    plot_ghost_onset,
    plot_no_ghost_region,
)

FIG = Path(__file__).resolve().parents[1] / "figures"
DIMS = list(range(18, 31))


def the_fock_space() -> None:
    """Explicit states, and the ghosts that are there from the beginning."""
    print("An explicit Fock space")
    print("-" * 64)
    print("  basis sizes in D = 26, levels 0..4:")
    print("   ", [basis_dimension(n, 26) for n in range(5)])
    print("  (the same numbers as prod (1-q^n)^-26, counted a completely")
    print("   different way -- by enumerating multisets)")
    print()
    print("  norms at level 2 in D = 4:")
    for state in level_basis(2, 4):
        label = " ".join(f"a(-{n})^{mu}" for n, mu in state)
        print(f"    {label:<22s} {state_norm(state, 4):+d}")
    print("  A single timelike oscillator has negative norm in every dimension.")
    print("  The critical dimension is not about that; it is about what")
    print("  survives the constraints.")
    print()


def the_algebra() -> None:
    """c from a commutator, not from a formula."""
    print("The algebra")
    print("-" * 64)
    alpha0 = alpha_zero(momentum_for_level(2, 6), 1.0)
    diagonal = np.diag(virasoro_matrix(2, 0, 6, alpha0))
    offdiag = virasoro_matrix(2, 0, 6, alpha0) - np.diag(diagonal)
    print(f"  L_0 at level 2, D = 6: diagonal {diagonal[0]:.12f}, "
          f"off-diagonal {np.abs(offdiag).max():.1e}")
    print("  = alpha' p^2 + N with alpha' p^2 = a - N, so exactly a = 1.")
    print()
    print("  c from [L_m, L_-m] = 2m L_0 + (c/12)(m^3 - m):")
    print("      D     m = 2     m = 3     zeta route")
    for dim in (4, 10, 26):
        two = central_charge_from_algebra(dim, m=2, level=0)
        three = central_charge_from_algebra(dim, m=3, level=0)
        print(f"    {dim:3d}  {two:8.4f}  {three:8.4f}   {central_charge(dim) + 26:8.4f}")
    print("  The central terms differ by a factor of four between m = 2 and")
    print("  m = 3; the c extracted from them does not.")
    print()
    print(f"  |[L_2,L_-2] - 4L_0 - (c/2)| at level 2, D = 6: "
          f"{algebra_residual(2, 2, -2, 6):.1e}")
    print(f"  |[L_1,L_2] + L_3|                             : "
          f"{algebra_residual(3, 1, 2, 6):.1e}")
    print()


def physical_states_and_gauge() -> None:
    """L_1 and L_2 generate everything, and level 1 sees nothing."""
    print("Physical states")
    print("-" * 64)
    basis = physical_states(2, 26)
    vector = basis @ np.random.default_rng(3).normal(size=basis.shape[1])
    residual = constraint_residual(vector, 2, 26, modes=(0, 1, 2, 3, 4))
    print("  a random physical combination at level 2, D = 26:")
    print("   ", {f"L_{m}": f"{r:.1e}" for m, r in residual.items()})
    print("  L_3 and L_4 were never imposed.  They follow from [L_1, L_2].")
    print()
    print("  level 1, in several dimensions:")
    for dim in (10, 26, 30):
        record = inertia(1, dim)
        print(f"    D = {dim:3d}:  {record.positive:3d} positive, "
              f"{record.zero} null, {record.negative} negative "
              f"(light-cone {lightcone_count(1, dim)})")
    print("  D - 2 states and one null, whatever D is.  Level 1 cannot see 26,")
    print("  which is why the argument has to go to level 2.")
    print()


def the_pincer() -> list:
    """Two bounds, opposite directions."""
    print("Level 2: two bounds that meet")
    print("-" * 64)
    records = ghost_scan(DIMS, level=2)
    print("      D    positive   null   ghosts   light-cone   SO(D-1) sym")
    for record in records:
        flag = "  <-- both agree" if record.dim == 26 else ""
        print(f"    {record.dim:3d}   {record.positive:7d}  {record.zero:5d}   "
              f"{record.negative:5d}   {lightcone_count(2, record.dim):10d}   "
              f"{sym_traceless_dim(record.dim - 1, 2):10d}{flag}")
    print()
    print("  Ghost-freedom fails above 26.  The covariant count exceeds the")
    print("  light-cone count below 26 -- one extra scalar, which at 26 turns")
    print("  null and drops out.  Neither bound alone gives 26; together they")
    print("  leave nothing else.")
    print()
    print(f"  critical dimension from norms      : {critical_dimension_from_norms(2)}")
    print(f"  critical dimension from zeta       : {critical_dimension('bosonic')}")
    print()
    return records


def the_intercept() -> None:
    """One level is a necessary condition, not the theorem."""
    print("The intercept, and why one level is not enough")
    print("-" * 64)
    print("      D    first ghost below a = 1")
    for dim in (24, 25, 26, 27, 28):
        boundary = ghost_boundary(dim, level=2)
        text = "none" if boundary is None else f"a = {boundary:+.9f}"
        print(f"    {dim:3d}    {text}")
    print("  Below 26 the whole line a < 1 is clean.  At 26 a crossing appears,")
    print("  at 3/8 to twelve digits; by 28 it has reached zero.")
    print()
    print("  So level 2 leaves a window at D = 26 for a < 3/8.  Level 3:")
    for value in (0.0, 1.0):
        record = inertia(3, 26, intercept=value)
        verdict = "ghost-free" if record.ghost_free else f"{record.negative} ghost(s)"
        print(f"    a = {value:.1f}:  {verdict}  (smallest norm {record.smallest:+.5f})")
    print("  The window closes.  At D = 26 the only surviving intercept is")
    print("  a = 1 -- which is what (D-2)/24 gives, by a route that never")
    print("  mentions norms.")
    print()
    print("  level 3 also reproduces the dimension boundary:")
    for dim in (26, 27):
        record = inertia(3, dim)
        print(f"    D = {dim}:  {record.positive} positive "
              f"(light-cone {lightcone_count(3, dim)}), {record.negative} ghosts")
    print()


def figures(records: list) -> None:
    print("Figures")
    print("-" * 64)

    measured = {
        m: [central_charge_from_algebra(d, m=m, level=0) for d in DIMS] for m in (2, 3)
    }
    plot_central_charge(DIMS, measured, FIG / "central_charge.png")
    print(f"  wrote {FIG / 'central_charge.png'}")

    plot_ghost_onset(
        records, [lightcone_count(2, r.dim) for r in records], FIG / "ghost_onset.png"
    )
    print(f"  wrote {FIG / 'ghost_onset.png'}")

    intercepts = np.linspace(-0.6, 1.4, 41)
    grid = no_ghost_map(DIMS, intercepts, level=2)
    boundary = [(d, ghost_boundary(d, level=2)) for d in DIMS]
    plot_no_ghost_region(DIMS, intercepts, grid, FIG / "no_ghost_region.png", boundary)
    print(f"  wrote {FIG / 'no_ghost_region.png'}")

    animate_no_ghost_region(DIMS, intercepts, grid, FIG / "no_ghost_region.gif")
    print(f"  wrote {FIG / 'no_ghost_region.gif'}")

    spectra, tracked = [], []
    for dim in DIMS:
        basis = physical_states(2, dim)
        metric = gram_diagonal(2, dim)
        values = np.linalg.eigvalsh(basis.T @ (metric[:, None] * basis))
        spectra.append((dim, values))
        # Set aside the D - 1 generic gauge nulls -- the descendants L_-1|chi>
        # of the level-1 physical states, present in every dimension -- and
        # take the smallest of what is left.  That is the one state whose norm
        # depends on D, and it reaches zero at 26.
        tracked.append(float(values[np.argsort(np.abs(values))[dim - 1 :]].min()))
    animate_physical_norms(spectra, tracked, FIG / "physical_norms.gif")
    print(f"  wrote {FIG / 'physical_norms.gif'}")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_fock_space()
    the_algebra()
    physical_states_and_gauge()
    scan = the_pincer()
    the_intercept()
    figures(scan)
    print("done.")
