"""Conventions, Lorentz contractions, and that the drawing code actually draws."""

from __future__ import annotations

import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from stringsim import Conventions, dot, minkowski  # noqa: E402
from stringsim.amplitudes.veneziano import veneziano, veneziano_pole_positions  # noqa: E402
from stringsim.branes.dbrane import stretched_spectrum  # noqa: E402
from stringsim.classical.evolve import evolve  # noqa: E402
from stringsim.classical.rotating import regge_trajectory, rigid_rotator  # noqa: E402
from stringsim.compactification.circle import self_dual_radius  # noqa: E402
from stringsim.compactification.torus import TorusBackground  # noqa: E402
from stringsim.quantum.partition import fit_hagedorn, oscillator_degeneracies  # noqa: E402
from stringsim.quantum.spectrum import open_bosonic_spectrum  # noqa: E402
from stringsim.viz.animate import animate_evolution, animate_modes, snapshot_grid  # noqa: E402
from stringsim.viz.plots import (  # noqa: E402
    plot_brane_separation,
    plot_degeneracy_growth,
    plot_mass_spectrum,
    plot_mode_spectrum,
    plot_regge_trajectory,
    plot_root_system,
    plot_tduality,
    plot_veneziano,
)

CONV = Conventions(alpha_prime=1.0, dim=26)


# -- conventions -------------------------------------------------------------


def test_derived_scales():
    conv = Conventions(alpha_prime=4.0, dim=26)
    assert conv.string_length == pytest.approx(2.0)
    assert conv.tension == pytest.approx(1 / (2 * math.pi * 4.0))
    assert conv.transverse_dim == 24


def test_conventions_validate_their_input():
    with pytest.raises(ValueError):
        Conventions(alpha_prime=0.0)
    with pytest.raises(ValueError):
        Conventions(dim=2)


def test_metric_signature_is_mostly_plus():
    eta = minkowski(5)
    assert eta[0] == -1.0
    assert np.all(eta[1:] == 1.0)


def test_dot_is_the_lorentz_product():
    a = np.array([2.0, 1.0, 0.0])
    assert dot(a, a) == pytest.approx(-3.0)
    stack = np.stack([a, a])
    assert np.allclose(dot(stack, stack), [-3.0, -3.0])
    with pytest.raises(ValueError):
        dot(np.zeros(3), np.zeros(4))


def test_a_massless_vector_has_zero_norm():
    k = np.array([1.0, 1.0, 0.0, 0.0])
    assert dot(k, k) == pytest.approx(0.0)


# -- figures -----------------------------------------------------------------


def test_static_plots_write_files(tmp_path):
    levels = open_bosonic_spectrum(4, CONV)
    assert plot_mass_spectrum(levels, tmp_path / "spectrum.png").exists()
    trajectory = regge_trajectory(conventions=CONV)
    assert plot_regge_trajectory(trajectory, tmp_path / "regge.png").exists()

    degen = oscillator_degeneracies(80, 24)
    fit = fit_hagedorn(n_max=80, n_fit=30)
    assert plot_degeneracy_growth(fit, degen, tmp_path / "hagedorn.png").exists()

    radii = np.geomspace(0.2, 5.0, 50)
    assert plot_tduality(
        radii, 1 / radii, radii, self_dual_radius(CONV), tmp_path / "tduality.png"
    ).exists()

    seps = np.linspace(0.0, 5.0, 20)
    assert plot_brane_separation(
        seps, [stretched_spectrum(d, 2, CONV) for d in seps], tmp_path / "branes.png"
    ).exists()

    s = np.linspace(-1.5, 3.5, 500)
    assert plot_veneziano(
        s, veneziano(s, -0.35), veneziano_pole_positions(4), tmp_path / "veneziano.png"
    ).exists()

    assert plot_mode_spectrum(np.arange(8.0), tmp_path / "modes.png").exists()


def test_animations_write_files(tmp_path):
    string = rigid_rotator(CONV, 1.5)
    assert animate_modes(string, tmp_path / "rot3d.gif", n_frames=6, n_sigma=24).exists()
    assert animate_modes(
        string, tmp_path / "rot2d.gif", projection=(1, 2), n_frames=6, n_sigma=24
    ).exists()
    assert snapshot_grid(string, tmp_path / "snaps.png", n_snapshots=3, n_sigma=24).exists()

    sigma = np.linspace(0.0, math.pi, 33)
    X0 = np.column_stack([np.cos(sigma), 0.3 * np.cos(2 * sigma)])
    ev = evolve(X0, boundary="neumann", n_steps=12)
    assert animate_evolution(ev, tmp_path / "plucked.gif", stride=3).exists()


def test_animation_rejects_a_bad_projection(tmp_path):
    string = rigid_rotator(CONV, 1.0)
    with pytest.raises(ValueError):
        animate_modes(string, tmp_path / "bad.gif", projection=(1,))
    with pytest.raises(ValueError):
        snapshot_grid(string, tmp_path / "bad.png", projection=(1, 2, 3))


def test_animate_evolution_needs_two_or_three_components(tmp_path):
    sigma = np.linspace(0.0, math.pi, 17)
    ev = evolve(np.cos(sigma)[:, None], boundary="neumann", n_steps=4)
    with pytest.raises(ValueError):
        animate_evolution(ev, tmp_path / "bad.gif")


def test_torus_plots_write_files(tmp_path):
    from stringsim.compactification.torus import TorusBackground, root_vectors
    from stringsim.viz.plots import plot_enhancement_map, plot_root_system

    left, _ = root_vectors(TorusBackground.su3_point(CONV))
    assert plot_root_system(left, tmp_path / "roots.png", "A2", "su(3)").exists()
    assert plot_root_system(np.zeros((0, 2)), tmp_path / "empty.png").exists()

    steps = np.linspace(-1.0, 1.0, 9)
    counts = np.zeros((9, 9), dtype=int)
    counts[4, 4] = 6
    assert plot_enhancement_map(steps, steps, counts, tmp_path / "map.png").exists()


def test_root_system_plot_needs_rank_two(tmp_path):
    with pytest.raises(ValueError):
        plot_root_system(np.zeros((3, 3)), tmp_path / "bad.png")


def test_fixed_point_plot(tmp_path):
    from stringsim.compactification.orbifold import Orbifold
    from stringsim.viz.plots import plot_fixed_points

    assert plot_fixed_points(Orbifold.z3_hexagonal(CONV), tmp_path / "z3.png").exists()
    assert plot_fixed_points(
        Orbifold.z4_square(CONV), tmp_path / "z4.png", sectors=(1, 2)
    ).exists()
    with pytest.raises(ValueError):
        plot_fixed_points(
            Orbifold.inversion(TorusBackground.self_dual(3, CONV)), tmp_path / "bad.png"
        )


def test_veneziano_plot_respects_the_clip(tmp_path):
    from stringsim.viz.plots import plot_veneziano

    s = np.linspace(-1.5, 3.5, 400)
    assert plot_veneziano(
        s, veneziano(s, -0.35), veneziano_pole_positions(4), tmp_path / "v.png", clip=8.0
    ).exists()


def test_supersymmetry_plot(tmp_path):
    from stringsim.quantum.partition import oscillator_degeneracies
    from stringsim.superstring.rns import Sector, open_superstring_levels, sector_degeneracies
    from stringsim.viz.plots import plot_supersymmetry

    assert plot_supersymmetry(
        open_superstring_levels(5),
        oscillator_degeneracies(30, 24),
        sector_degeneracies(Sector.NS, 30),
        tmp_path / "susy.png",
    ).exists()
