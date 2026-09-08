"""Boundary states: a D-brane written as a closed string, and the eta it makes.

Run:  python examples/26_boundary_states.py

``examples/05`` describes a D-brane by what ends on it.  This describes the same
brane by what it emits: a coherent state of closed strings.  The state is an
ansatz until the gluing conditions are applied to it, and applying them is what
the explicit Fock space of ``examples/21`` was built for.

Two things come out.  The state really does satisfy the gluing conditions --
exactly, for every mode and every direction, and only because the worldsheet
metric is in the coefficients.  And its norm, summed state by state, is the
Dedekind eta that the cylinder's closed channel carries.

Writes ``figures/boundary_state.png`` and ``figures/boundary_touch.gif``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.amplitudes.oneloop import closed_channel_integrand  # noqa: E402
from stringsim.branes.boundary import (  # noqa: E402
    boundary_coefficients,
    boundary_residual,
    classical_boundary_state,
    eta_factor,
    eta_residual,
    gluing_residual,
    gluing_signs,
    neumann_count,
    oscillator_overlap,
    overlap_closed_form,
    overlap_truncation,
)
from stringsim.classical.modes import ClosedString  # noqa: E402
from stringsim.quantum.partition import dedekind_eta  # noqa: E402
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.animate import animate_boundary_state  # noqa: E402
from stringsim.viz.plots import plot_boundary_state  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"


def the_gluing() -> None:
    """The coherent state, checked rather than quoted."""
    print("The gluing conditions")
    print("-" * 70)
    print("  Along the brane the endpoint is free, across it the endpoint is")
    print("  nailed down.  On the closed-string Hilbert space both read")
    print("      (alpha_n - S alpha~_{-n}) |B> = 0,   S = -1 along, +1 across")
    print("  and the coherent state exp(sum (1/n) alpha_{-n} S alpha~_{-n})|0>")
    print("  is supposed to solve them.  Building it and applying the operator:")
    print()
    print("      D-brane   dim   worst residual over modes 1-3 and all directions")
    for p, dim in ((-1, 4), (0, 4), (1, 4), (2, 5)):
        signs = gluing_signs(p, dim)
        worst = max(
            gluing_residual(3, dim, signs, mode, index)
            for mode in (1, 2, 3)
            for index in range(dim)
        )
        print(f"    D{p:<7d} {dim:4d}   {worst:.1e}")
    print("  Exactly zero: the coefficients are rational and the cancellation")
    print("  is term by term.")
    print()
    print("  The worldsheet metric belongs in the coefficients.  [alpha_n^mu,")
    print("  alpha_{-n}^nu] = n eta^{mu nu}, so the exponent carries S_mu eta_mu")
    print("  and not S_mu.  Dropping it:")
    signs = gluing_signs(1, 4)
    with_metric = [gluing_residual(3, 4, signs, 1, i) for i in range(4)]
    without = [gluing_residual(3, 4, signs, 1, i, include_metric=False) for i in range(4)]
    print(f"      direction        {'  '.join(f'{i:8d}' for i in range(4))}")
    print(f"      with eta         {'  '.join(f'{v:8.1e}' for v in with_metric)}")
    print(f"      without eta      {'  '.join(f'{v:8.1e}' for v in without)}")
    print("  Only the timelike direction notices.  A check on one spatial")
    print("  direction would have passed a wrong state.")
    print()
    signs = gluing_signs(0, 3)
    print("  The coefficients themselves, for a D0-brane in three directions:")
    for level in (1, 2):
        for state, value in sorted(boundary_coefficients(level, 3, signs).items()):
            print(f"    level {level}  {str(state):<22s} {value:+.6f}")
    print()


def the_norm() -> list:
    """Summing over Fock states gives the eta function."""
    print("The norm")
    print("-" * 70)
    print("  Every diagonal state contributes 1 -- the coefficient cancels")
    print("  against the norm, whatever the signature -- so the overlap is a")
    print("  plain sum over the oscillator degeneracies.")
    print()
    print("      q     levels   sum over Fock states     the product        error")
    series = []
    for q in (0.1, 0.3, 0.5):
        cutoffs, errors = [], []
        exact = overlap_closed_form(q, 24)
        for cutoff in range(20, 421, 20):
            value = oscillator_overlap(q, 24, cutoff)
            cutoffs.append(cutoff)
            errors.append(abs(value / exact - 1.0))
        needed = next(c for c, e in zip(cutoffs, errors, strict=True) if e < 1e-13)
        print(f"    {q:5.2f}  {needed:6d}   {oscillator_overlap(q, 24, needed):20.8e}   "
              f"{exact:16.8e}   {overlap_truncation(q, 24, needed):.0e}")
        series.append((f"$q = {q}$", cutoffs, errors))
    print()
    print("  The cutoff climbs steeply, and that is the density of states")
    print("  rather than the code: the degeneracies grow like exp(4 pi sqrt N),")
    print("  so the terms rise before they fall.")
    print()
    return series


def the_eta() -> tuple:
    """With the ground-state energy back, it is the cylinder's oscillator factor."""
    print("What the norm turns out to be")
    print("-" * 70)
    print("      modulus   from the boundary state      from dedekind_eta    residual")
    moduli, from_state, from_eta = [], [], []
    for t in (0.4, 0.7, 1.0, 1.4, 1.9):
        mine = eta_factor(t, 24)
        theirs = abs(dedekind_eta(complex(0.0, t))) ** -24
        print(f"    {t:8.2f}   {mine:22.10e}   {theirs:18.10e}   {eta_residual(t):.1e}")
        moduli.append(t)
        from_state.append(mine)
        from_eta.append(theirs)
    print()
    print("  That factor is what closed_channel_integrand carries.  Its ratio to")
    print("  the full closed-channel cylinder is the zero-mode measure and the")
    print("  normalisation, neither of which is derived here:")
    for t in (0.6, 1.2):
        full = closed_channel_integrand(t, p=3, separation=1.0)
        print(f"    t = {t}:  full / oscillator factor = {full / eta_factor(t, 24):.6e}")
    print("  Getting that constant out of the boundary state is Polchinski's")
    print("  computation of T_p.  It is not attempted; dbrane.py supplies it.")
    print()
    return moduli, from_state, from_eta


def touching_the_brane() -> tuple:
    """The same conditions on a solution of the wave equation."""
    print("The same conditions, on a moving string")
    print("-" * 70)
    conv = Conventions(dim=4)
    rng = np.random.default_rng(7)
    p, brane = 1, 1.4
    position = np.zeros(4)
    position[p + 1 :] = brane
    modes = {
        1: rng.normal(size=4) + 1j * rng.normal(size=4),
        2: 0.6 * rng.normal(size=4),
    }
    glued = classical_boundary_state(modes, p, position=position, conventions=conv)
    signs = gluing_signs(p, 4)
    unglued = ClosedString(
        conventions=conv,
        x0=position,
        alphas=dict(modes),
        alphas_tilde={n: signs * a for n, a in modes.items()},
    )
    print(f"  A D{p}-brane at transverse position {brane}, "
          f"{neumann_count(signs)} Neumann directions.")
    print()
    print("      string                       |Xdot| along    |X - y| across")
    for label, string in (("glued (a boundary state)", glued), ("not glued", unglued)):
        neumann, dirichlet = boundary_residual(string, p, position=position)
        print(f"    {label:<28s} {neumann:12.2e}   {dirichlet:12.2e}")
    print()
    print("  At tau = 0 the glued string lies flat on the brane: Dirichlet puts")
    print("  every point of it at the brane's position, Neumann gives it no")
    print("  velocity along.  Then it peels off.  The unglued one has the same")
    print("  right-movers and never touches.")
    print()

    n_frames = 90
    taus = np.linspace(0.0, 1.6 * math.pi, n_frames)[:, None]
    sigmas = np.linspace(0.0, glued.sigma_max, 160)[None, :]
    panels = [
        ("a boundary state", glued.position(taus, sigmas)[:, :, 1:3]),
        ("not glued", unglued.position(taus, sigmas)[:, :, 1:3]),
    ]
    return panels, brane


def figures(series: list, eta_curve: tuple, panels: list, brane: float) -> None:
    print("Figures")
    print("-" * 70)
    plot_boundary_state(series, eta_curve, FIG / "boundary_state.png")
    print(f"  wrote {FIG / 'boundary_state.png'}")
    animate_boundary_state(panels, brane, FIG / "boundary_touch.gif")
    print(f"  wrote {FIG / 'boundary_touch.gif'}")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_gluing()
    series = the_norm()
    eta_curve = the_eta()
    panels, brane = touching_the_brane()
    figures(series, eta_curve, panels, brane)
    print("done.")
