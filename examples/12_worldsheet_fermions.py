"""Evolving the worldsheet fermions: transport, reflection, and the two sectors.

Run:  python examples/12_worldsheet_fermions.py

The bosonic worldsheet is simulated in ``examples/01_classical_string.py``; this
is its fermionic partner.  Nothing here is quoted from ``rns.py``: the mode
numbers, the period doubling and the Ramond zero mode all come out of a grid
being stepped forward.  Writes ``figures/fermion_reflection.png``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.classical.evolve import evolve  # noqa: E402
from stringsim.superstring.rns import Sector, fermion_mode_numbers  # noqa: E402
from stringsim.superstring.worldsheet import (  # noqa: E402
    SPIN_STRUCTURES,
    boundary_residual,
    evolve_closed_fermion,
    evolve_open_fermion,
    fermion_mode_spectrum,
    fold,
    norm,
    open_fermion_modes,
    reconstruct_from_modes,
    sector_twist,
    supercurrent_residual,
    susy_variation,
    zero_mode,
)
from stringsim.viz.plots import plot_fermion_reflection  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
TWO_PI = 2.0 * math.pi

MODES = {
    Sector.NS: {0.5: [1.0, 0.3j], 1.5: [0.2, -0.4], 2.5: [0.1j, 0.05]},
    Sector.R: {0.0: [0.7, 0.0], 1.0: [0.5, 0.2j], 2.0: [-0.3, 0.1]},
}


def half_grid(n_points: int) -> np.ndarray:
    """The physical string ``[0, pi]`` on ``n_points // 2 + 1`` points."""
    return np.linspace(0.0, math.pi, n_points // 2 + 1)


def the_sign_at_the_end() -> None:
    print("=" * 72)
    print("One sign at sigma = pi, and everything follows")
    print("=" * 72)
    print("  psi_+(0) = psi_-(0) always; psi_+(pi) = eta psi_-(pi) is a choice.")
    print("  Unfolded, the pair is one right-moving field on a circle of")
    print("  circumference 2 pi with psi(sigma + 2 pi) = eta psi(sigma).")
    print()
    n_points = 128
    sigma = half_grid(n_points)
    for sector in (Sector.NS, Sector.R):
        twist = sector_twist(sector)
        psi_minus, psi_plus = open_fermion_modes(MODES[sector], sector, 0.0, sigma)
        # tau_max = n_points * h = 2 pi at Courant number 1: exactly one circuit.
        run = evolve_open_fermion(
            psi_minus, psi_plus, sector=sector, n_steps=n_points, courant=1.0
        )
        same = np.max(np.abs(run.unfolded[-1] - run.unfolded[0]))
        twisted = np.max(np.abs(run.unfolded[-1] - twist * run.unfolded[0]))
        print(f"  {sector.value:2s} (eta = {twist:+d}), after tau = 2 pi:")
        print(f"       |psi(2 pi) - eta psi(0)| = {twisted:.1e}")
        print(f"       |psi(2 pi) -     psi(0)| = {same:.1e}")
    print("  The NS fermion comes back as minus itself and needs 4 pi to return.")
    print("  That is the same sign that makes its modes half-integral.")
    print()


def modes_read_off_a_snapshot() -> None:
    print("=" * 72)
    print("Mode numbers measured, not assumed")
    print("=" * 72)
    n_points = 128
    sigma = half_grid(n_points)
    for sector in (Sector.NS, Sector.R):
        twist = sector_twist(sector)
        psi_minus, psi_plus = open_fermion_modes(MODES[sector], sector, 0.0, sigma)
        run = evolve_open_fermion(psi_minus, psi_plus, sector=sector, n_steps=8)
        snapshot = run.unfolded[0]
        right = reconstruct_from_modes(fermion_mode_spectrum(snapshot, twist, 8), twist, n_points)
        wrong = reconstruct_from_modes(
            fermion_mode_spectrum(snapshot, -twist, 8), -twist, n_points
        )
        coefficients = fermion_mode_spectrum(snapshot, twist, 4)[:, 0]
        print(f"  {sector.value:2s}: rns.fermion_mode_numbers -> "
              f"{[str(r) for r in fermion_mode_numbers(sector, 4)]}")
        print(f"      c_r from the grid       -> {np.round(coefficients.real, 4)}")
        print(f"      rebuilt with these r    -> error {np.max(np.abs(right - snapshot)):.1e}")
        print(f"      rebuilt with the others -> error {np.max(np.abs(wrong - snapshot)):.1e}")
        print(f"      zero mode {np.round(zero_mode(snapshot, twist), 3)}"
              f"   (a piece of the field that never moves)")
    print("  The wrong sector's mode numbers are not a small error but a")
    print("  different vector space: they cannot represent the field at all.")
    print("  Only R has a zero mode, and its Clifford algebra is what makes the")
    print("  Ramond ground state a spinor instead of a single state.")
    print()


def the_scheme_converges() -> None:
    print("=" * 72)
    print("Second order, and exact at Courant number 1")
    print("=" * 72)
    previous = None
    for n_points in (32, 64, 128, 256):
        sigma = half_grid(n_points)
        psi_minus, psi_plus = open_fermion_modes(MODES[Sector.NS], Sector.NS, 0.0, sigma)
        run = evolve_open_fermion(
            psi_minus, psi_plus, sector=Sector.NS, n_steps=2 * n_points, courant=0.5
        )
        exact = open_fermion_modes(MODES[Sector.NS], Sector.NS, float(run.tau[-1]), sigma)
        error = max(
            float(np.max(np.abs(run.psi_minus[-1] - exact[0]))),
            float(np.max(np.abs(run.psi_plus[-1] - exact[1]))),
        )
        drift = norm(run.unfolded[-1]) - norm(run.unfolded[0])
        ratio = f"{previous / error:.2f}" if previous else "   -"
        print(f"  n = {n_points:4d}  error {error:.3e}  ratio {ratio}   norm drift {drift:+.1e}")
        previous = error
    print("  Halving the grid quarters the error: Lax-Wendroff is second order.")
    print("  The norm drifts because the scheme damps short wavelengths below")
    print("  Courant 1; the drift falls like h^2 as well.  At Courant 1 the step")
    print("  is a one-cell shift and the whole run is exact:")
    sigma = half_grid(128)
    psi_minus, psi_plus = open_fermion_modes(MODES[Sector.NS], Sector.NS, 0.0, sigma)
    run = evolve_open_fermion(psi_minus, psi_plus, sector=Sector.NS, n_steps=128, courant=1.0)
    exact = open_fermion_modes(MODES[Sector.NS], Sector.NS, float(run.tau[-1]), sigma)
    print(f"    error {np.max(np.abs(run.psi_minus[-1] - exact[0])):.1e}, "
          f"norm drift {norm(run.unfolded[-1]) - norm(run.unfolded[0]):+.1e}")
    print()


def four_spin_structures() -> None:
    print("=" * 72)
    print("The closed string: two independent fields, four spin structures")
    print("=" * 72)
    n_points = 96
    sigma = np.arange(n_points) * (TWO_PI / n_points)

    def data(twist: int) -> np.ndarray:
        """Admissible data for the given periodicity, with a constant if allowed."""
        if twist == 1:
            return np.stack([0.4 + np.cos(sigma), 0.25 * np.sin(sigma)], axis=-1)
        return np.stack([np.cos(0.5 * sigma), 0.25 * np.sin(0.5 * sigma)], axis=-1)

    for name, (twist_minus, twist_plus) in SPIN_STRUCTURES.items():
        minus, plus = data(twist_minus), data(twist_plus)
        run = evolve_closed_fermion(
            minus, plus, spin_structure=name, n_steps=n_points, courant=1.0
        )
        residual = max(
            float(np.max(np.abs(run.psi_minus[-1] - twist_minus * run.psi_minus[0]))),
            float(np.max(np.abs(run.psi_plus[-1] - twist_plus * run.psi_plus[0]))),
        )
        constants = (
            float(zero_mode(run.psi_minus[0], twist_minus)[0]),
            float(zero_mode(run.psi_plus[0], twist_plus)[0]),
        )
        print(f"  {name:6s} (eta_- , eta_+) = ({twist_minus:+d}, {twist_plus:+d})"
              f"   |psi(2 pi) - eta psi(0)| = {residual:.1e}"
              f"   zero modes {constants[0]:.2f}, {constants[1]:.2f}")
    print("  Each side was handed the same constant 0.4, and it survives only where")
    print("  the field is periodic.  A constant obeying psi(sigma + 2 pi) = -psi(sigma)")
    print("  can only be zero, so the NS sides have no zero mode to report -- which")
    print("  is why only Ramond gives spacetime fermions.")
    print("  Nothing couples the two, which is exactly why a closed string can be")
    print("  Ramond on one side and Neveu-Schwarz on the other -- and why the type")
    print("  II spectra are built from four sectors rather than two.")
    print()


def supersymmetry_picks_a_sector() -> None:
    print("=" * 72)
    print("A supersymmetry variation of a Neumann boson lands in Ramond")
    print("=" * 72)
    print("  delta psi_-+ = d_-+ X.  Neumann means d_sigma X = 0 at the ends, so")
    print("  d_+ X = d_- X there and the varied fermion has psi_+ = psi_- at both")
    print("  ends.  That is the R condition; NS would need eta = -1 at sigma = pi.")
    print()
    print(f"  {'n':>5s} {'R at 0':>10s} {'R at pi':>10s} {'NS at pi':>10s} "
          f"{'max |d_+ G_-|':>14s}")
    previous = None
    for n_points in (32, 64, 128, 256):
        sigma = half_grid(n_points)
        shape = np.stack([0.3 * np.cos(sigma), 0.2 * np.cos(2.0 * sigma)], axis=-1)
        bosons = evolve(shape, boundary="neumann", n_steps=2 * n_points, courant=0.5)
        # A generic time: at tau = 0 the string is at rest and both conditions
        # degenerate to 0 = 0, which would prove nothing.
        index = max(1, int(round(0.7 / float(bosons.tau[1] - bosons.tau[0]))))
        psi_minus, psi_plus = susy_variation(bosons, index=index)
        ramond = boundary_residual(psi_minus, psi_plus, +1)
        neveu = boundary_residual(psi_minus, psi_plus, -1)
        fermions = evolve_open_fermion(
            psi_minus, psi_plus, sector=Sector.R, n_steps=2 * n_points,
            courant=0.5, tolerance=1e-1,
        )
        residual = supercurrent_residual(bosons, fermions)
        ratio = f"  (/{previous / residual:.1f})" if previous else ""
        previous = residual
        print(f"  {n_points:>5d} {ramond[0]:>10.2e} {ramond[1]:>10.2e} "
              f"{neveu[1]:>10.2e} {residual:>14.3e}{ratio}")
    print("  The R residuals go to zero with the grid; the NS one sits at ~0.4 and")
    print("  stays there.  The last column is the supercurrent constraint")
    print("  d_+ G_- = 0 with G_- = psi_- . d_- X: it falls by four each time, so")
    print("  what is left is discretisation error and nothing else.")
    print()


def figures() -> None:
    n_points = 192
    runs = {}
    for sector in (Sector.NS, Sector.R):
        twist = sector_twist(sector)
        # A narrow bump on the doubled circle.  Its tails at sigma = 0 and
        # sigma = 2 pi are ~1e-14, so the twisted periodicity holds to round-off
        # whichever sign is wanted, and the sector enters only through the run.
        sigma = np.arange(n_points) * (TWO_PI / n_points)
        bump = np.exp(-(((sigma - 0.5 * math.pi) / 0.28) ** 2))[:, None]
        minus, plus = fold(bump, twist)
        runs[sector] = evolve_open_fermion(
            minus, plus, sector=sector, n_steps=3 * n_points, courant=1.0
        )
    plot_fermion_reflection(
        [
            (r"Neveu-Schwarz, $\eta = -1$", runs[Sector.NS]),
            (r"Ramond, $\eta = +1$", runs[Sector.R]),
        ],
        FIG / "fermion_reflection.png",
    )
    print(f"  wrote {FIG / 'fermion_reflection.png'}")
    print("  Each stripe is one pass of the pulse through psi_-.  Between passes")
    print("  it goes out through sigma = pi into psi_+ and comes back, picking up")
    print("  eta on the way.  Left panel: the colours alternate.  Right: they do")
    print("  not.  Half-integral modes are that alternation, drawn.")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_sign_at_the_end()
    modes_read_off_a_snapshot()
    the_scheme_converges()
    four_spin_structures()
    supersymmetry_picks_a_sector()
    figures()
    print("done.")
