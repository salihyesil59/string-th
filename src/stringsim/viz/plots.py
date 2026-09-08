"""Static figures for each part of the package.

Every function takes an output path and returns it, so an example script reads
as a list of files produced.  Nothing here computes physics -- it only draws
what the other modules return, which keeps the figures honest.

The backend is left to matplotlib; scripts that run headless should select
``Agg`` before importing pyplot.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

__all__ = [
    "plot_mass_spectrum",
    "plot_regge_trajectory",
    "plot_degeneracy_growth",
    "plot_tduality",
    "plot_brane_separation",
    "plot_veneziano",
    "plot_mode_spectrum",
    "plot_root_system",
    "plot_enhancement_map",
    "plot_fixed_points",
    "plot_supersymmetry",
    "plot_root_connectivity",
    "plot_fermion_reflection",
    "plot_wilson_enhancement",
    "plot_twist_classification",
    "plot_dbi_field",
    "plot_bion_spike",
    "plot_fundamental_domain",
    "plot_one_loop_integrand",
    "plot_channel_duality",
    "plot_fuzzy_sphere",
    "plot_myers_landscape",
    "plot_hodge_diamond",
    "plot_fixed_loci",
    "plot_matrix_worldlines",
    "plot_lyapunov",
    "plot_shift_landscape",
    "plot_commutation",
    "plot_central_charge",
    "plot_ghost_onset",
    "plot_no_ghost_region",
    "plot_anomaly_conditions",
    "plot_anomaly_scan",
    "plot_orientifold_spectrum",
    "plot_worldsheet_surfaces",
    "plot_modular_invariance",
    "plot_narain_levels",
    "plot_koba_nielsen",
    "plot_five_point_moduli",
    "plot_boundary_state",
    "plot_black_hole_entropy",
    "plot_pq_strings",
    "plot_mirror_hodge",
    "plot_reflexive_duality",
]

FUNDAMENTAL_DOMAIN_FLOOR = math.sqrt(3.0) / 2.0

_STYLE = {
    "figure.figsize": (7.2, 4.6),
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}


def _fig(nrows: int = 1, ncols: int = 1, **kw):
    import matplotlib.pyplot as plt

    with plt.rc_context(_STYLE):
        return plt.subplots(nrows, ncols, **kw)


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def plot_mass_spectrum(levels, path, title: str = "Open bosonic string spectrum") -> Path:
    """Mass levels as a ladder, with the number of states at each rung.

    ``levels`` is a list of :class:`~stringsim.quantum.spectrum.Level`.
    """
    fig, (ax, ax2) = _fig(1, 2, figsize=(10.5, 4.6))
    m2 = [lv.alpha_m2 for lv in levels]
    deg = [lv.degeneracy for lv in levels]
    for lv in levels:
        colour = "tab:red" if lv.is_tachyonic else ("tab:green" if lv.is_massless else "tab:blue")
        ax.hlines(lv.alpha_m2, 0, 1, color=colour, lw=2.5)
        ax.text(1.04, lv.alpha_m2, f"N={lv.n}  x{lv.degeneracy:,}", va="center", fontsize=8)
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax.set_xlim(0, 2.3)
    ax.set_xticks([])
    ax.set_ylabel(r"$\alpha' M^2$")
    ax.set_title(title)

    ax2.semilogy(m2, deg, "o-", color="tab:blue")
    ax2.set_xlabel(r"$\alpha' M^2$")
    ax2.set_ylabel("number of states")
    ax2.set_title("degeneracy grows exponentially")
    return _save(fig, path)


def plot_regge_trajectory(points, path, title: str = "Leading Regge trajectory") -> Path:
    """``J`` against ``alpha' M^2`` for the classical rotating string."""
    fig, ax = _fig()
    x = np.array([p.alpha_m2 for p in points])
    y = np.array([p.spin for p in points])
    ax.plot(x, y, "o", ms=7, label="rotating string (measured)")
    line = np.linspace(0, max(x.max(), 1e-9) * 1.05, 50)
    ax.plot(line, line, "-", color="0.4", lw=1.2, label=r"$J = \alpha' M^2$")
    ax.set_xlabel(r"$\alpha' M^2$")
    ax.set_ylabel(r"$J$")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_degeneracy_growth(fit, degeneracies, path) -> Path:
    r"""The exponential growth of the density of states, and its slope.

    Left: ``log d_N`` against ``sqrt(N)`` with the *complete* fitted model
    ``beta_H M + b log N + c`` drawn over its fit window.  Plotting
    ``beta_H sqrt(N)`` on its own would float well above the data — the
    subleading terms are not small here, which is exactly why the fit includes
    them.

    Right: the local slope ``d(log d_N)/d(sqrt(N))``, which is the quantity
    that actually tends to ``beta_H``.  It approaches ``4 pi`` from below like
    ``1/sqrt(N)``, so the horizontal reference line is the asymptote, not a fit
    to the points.

    ``fit`` is a :class:`~stringsim.quantum.partition.HagedornFit` and
    ``degeneracies`` the full list it was computed from.
    """
    fig, (ax, ax2) = _fig(1, 2, figsize=(11.0, 4.6))
    n = np.arange(1, len(degeneracies))
    logd = np.array([math.log(d) for d in degeneracies[1:]])
    ax.plot(np.sqrt(n), logd, ".", ms=4, color="tab:blue", label=r"$\log d_N$")
    ax.plot(
        np.sqrt(fit.levels),
        fit.model(fit.levels),
        "-",
        color="tab:red",
        lw=2.0,
        label=rf"fit over $N \geq {int(fit.levels[0])}$: $\beta_H = {fit.beta_hagedorn:.3f}$",
    )
    ax.set_xlabel(r"$\sqrt{N} = \sqrt{\alpha'}\,M$")
    ax.set_ylabel(r"$\log d_N$")
    ax.set_title("Exponential density of states")
    ax.legend(loc="upper left")

    levels, slope = type(fit).local_slope(degeneracies, fit.alpha_prime)
    keep = levels >= 4  # the first few levels are not yet asymptotic
    ax2.plot(
        1.0 / np.sqrt(levels[keep]),
        slope[keep],
        ".",
        ms=4,
        color="tab:blue",
        label=r"$\Delta \log d_N / \Delta \sqrt{N}$",
    )
    ax2.axhline(
        fit.predicted_beta,
        color="0.4",
        ls="--",
        lw=1.3,
        label=rf"$4\pi = {fit.predicted_beta:.3f}$",
    )
    ax2.axhline(
        fit.beta_hagedorn,
        color="tab:red",
        ls=":",
        lw=1.3,
        label=rf"fitted {fit.beta_hagedorn:.3f}",
    )
    ax2.set_xlabel(r"$1/\sqrt{N}$   (asymptotic limit at the left edge)")
    ax2.set_ylabel(r"local slope")
    ax2.set_title(r"Slope converging on $\beta_H$")
    ax2.legend(loc="lower right")
    return _save(fig, path)


def plot_tduality(radii, kk, winding, self_dual: float, path) -> Path:
    """Kaluza-Klein and winding towers crossing at the self-dual radius."""
    fig, ax = _fig()
    ax.loglog(radii, kk, label=r"Kaluza-Klein $n/R$", color="tab:blue")
    ax.loglog(radii, winding, label=r"winding $wR/\alpha'$", color="tab:orange")
    ax.axvline(self_dual, color="0.4", ls="--", lw=1.2, label=r"$R = \sqrt{\alpha'}$")
    ax.set_xlabel(r"$R / \sqrt{\alpha'}$")
    ax.set_ylabel("mass")
    ax.set_title(r"T-duality: $R \to \alpha'/R$ swaps the two towers")
    ax.legend()
    return _save(fig, path)


def plot_brane_separation(separations, levels_by_sep, path) -> Path:
    """``M^2`` of stretched-string levels as two D-branes are pulled apart."""
    fig, ax = _fig()
    n_levels = len(levels_by_sep[0])
    for n in range(n_levels):
        ax.plot(
            separations,
            [lv[n].mass_squared for lv in levels_by_sep],
            label=f"N = {n}",
        )
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax.set_xlabel(r"brane separation $d / \sqrt{\alpha'}$")
    ax.set_ylabel(r"$M^2 \, \alpha'$")
    ax.set_title("Separating branes gives the gauge bosons mass")
    ax.legend()
    return _save(fig, path)


def plot_veneziano(s_values, amplitude, poles, path, clip: float = 25.0) -> Path:
    """The Veneziano amplitude along the real ``s`` axis, poles marked.

    Values beyond ``clip`` are masked rather than drawn, so each branch is a
    separate curve; without that, matplotlib joins ``+inf`` to ``-inf`` across
    every pole and the resulting vertical stroke hides the pole marker.
    """
    fig, ax = _fig()
    amplitude = np.asarray(amplitude, dtype=float)
    visible = np.where(np.abs(amplitude) > clip, np.nan, amplitude)
    ax.plot(s_values, visible, lw=1.4, color="tab:blue")
    for p in poles:
        ax.axvline(p, color="tab:red", ls=":", lw=1.0)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel(r"$s$  (units of $1/\alpha'$)")
    ax.set_ylabel(r"$A(s,t)$")
    ax.set_title(r"Veneziano amplitude: poles at $\alpha' s = N - 1$")
    ax.set_ylim(-clip, clip)
    return _save(fig, path)


def plot_mode_spectrum(coefficients, path, title: str = "Normal-mode content") -> Path:
    """Bar chart of a snapshot's Fourier coefficients."""
    fig, ax = _fig()
    amp = np.abs(np.asarray(coefficients))
    if amp.ndim > 1:
        amp = np.linalg.norm(amp, axis=1)
    ax.bar(np.arange(len(amp)), amp, color="tab:blue")
    ax.set_xlabel("mode number $n$")
    ax.set_ylabel(r"$|c_n|$")
    ax.set_title(title)
    return _save(fig, path)


def plot_root_system(roots, path, title: str = "Root system", label: str = "") -> Path:
    r"""Draw a two-dimensional root system as arrows from the origin.

    ``roots`` is an ``(n, 2)`` array of ``l_L`` (or ``l_R``) vectors from
    :func:`stringsim.compactification.torus.root_vectors`.  Every root has
    squared length 2, so they all end on the same circle; what distinguishes the
    algebras is the *angles*.  Four roots at right angles are ``su(2) + su(2)``;
    six at 60 degrees are ``su(3)``.
    """
    roots = np.asarray(roots, dtype=float)
    if roots.ndim != 2 or roots.shape[1] != 2:
        raise ValueError("plot_root_system draws rank-2 systems; give an (n, 2) array")

    fig, ax = _fig(figsize=(5.2, 5.2))
    limit = 1.6
    if len(roots):
        radius = float(np.linalg.norm(roots[0]))
        limit = 1.35 * radius
        for vector in roots:
            ax.annotate(
                "",
                xy=tuple(vector),
                xytext=(0.0, 0.0),
                arrowprops={"arrowstyle": "-|>", "color": "tab:blue", "lw": 1.6},
            )
        ax.plot(roots[:, 0], roots[:, 1], "o", ms=5, color="tab:blue")
        angle = np.linspace(0.0, 2.0 * np.pi, 240)
        ax.plot(radius * np.cos(angle), radius * np.sin(angle), "--", lw=0.9, color="0.7")
    ax.plot(0.0, 0.0, "+", ms=10, color="0.3")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\ell^1$")
    ax.set_ylabel(r"$\ell^2$")
    ax.set_title("\n".join([title, label]) if label else title)
    return _save(fig, path)


def plot_enhancement_map(g_values, b_values, root_counts, path) -> Path:
    r"""Where in the ``T^2`` moduli space the gauge symmetry grows.

    ``root_counts`` is a 2-D array indexed ``[g, b]`` over the off-diagonal
    metric modulus and the ``B``-field modulus.  Enhancement needs an
    integrality condition, so the picture is a set of *lines* -- rank-one
    enhancement -- meeting at isolated points where the symmetry becomes
    ``su(3)``.  It is not a smooth landscape, and that is the physics: the
    enhanced points form a measure-zero set.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    counts = np.asarray(root_counts)
    colours = ListedColormap(["#f4f4f4", "#c6dbef", "#6baed6", "#08519c"])
    edges = [0, 2, 4, 6, max(8, int(counts.max()) + 2)]

    fig, ax = _fig(figsize=(6.4, 5.2))
    mesh = ax.pcolormesh(
        np.asarray(g_values),
        np.asarray(b_values),
        counts.T,
        cmap=colours,
        norm=BoundaryNorm(edges, colours.N),
        shading="nearest",
    )
    bar = fig.colorbar(mesh, ax=ax, ticks=[1, 3, 5, 7])
    bar.ax.set_yticklabels(["0", "2", "4", "6+"])
    bar.set_label("left-moving roots")
    ax.set_xlabel(r"$G_{12}$")
    ax.set_ylabel(r"$B_{12}$")
    ax.set_title(r"Gauge enhancement on $T^2$ with $G_{11} = G_{22} = 1$")
    ax.set_aspect("equal", adjustable="box")
    return _save(fig, path)


def plot_fixed_points(orbifold, path, sectors=(1,), title: str | None = None) -> Path:
    r"""Fixed points of a two-dimensional orbifold, drawn in the torus cell.

    The parallelogram is the fundamental cell of the lattice, obtained from a
    Cholesky factor of ``G`` so that the drawn angles are the real ones -- the
    hexagonal lattice comes out at 60 degrees, the square one at 90.  Marked on
    it are the fixed points of each requested ``theta^k``, which is where the
    twisted strings live.  Their number is ``|det(1 - theta^k)|``: four for
    ``Z_2``, three for ``Z_3``, two for ``Z_4`` and one for ``Z_6``.
    """
    if orbifold.dim != 2:
        raise ValueError("plot_fixed_points draws two-dimensional orbifolds")

    basis = np.linalg.cholesky(np.asarray(orbifold.background.metric)).T
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]) @ basis

    fig, ax = _fig(figsize=(5.4, 5.4))
    ax.plot(corners[:, 0], corners[:, 1], "-", lw=1.2, color="0.55")
    markers = ["o", "s", "^", "D", "v"]
    for index, sector in enumerate(sectors):
        points = np.asarray(orbifold.fixed_point_positions(sector)) @ basis
        ax.plot(
            points[:, 0],
            points[:, 1],
            markers[index % len(markers)],
            ms=9 - 2 * index,
            label=rf"$\theta^{{{sector}}}$: {len(points)} points",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$X^1 / \sqrt{\alpha'}$")
    ax.set_ylabel(r"$X^2 / \sqrt{\alpha'}$")
    ax.set_title(title or f"Fixed points of $T^2/Z_{{{orbifold.order}}}$")
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, path)


def plot_supersymmetry(levels, bosonic_degeneracies, super_degeneracies, path) -> Path:
    r"""Boson and fermion counts level by level, and how fast each theory grows.

    Left: the GSO-projected superstring's bosons and fermions as paired bars,
    drawn side by side rather than stacked so that any inequality would show
    rather than average away.  They are equal at every level, which is what
    spacetime supersymmetry means for the spectrum.

    Right: the **local slope** ``d(log d)/d(sqrt(N))`` for both theories, which
    is the quantity that tends to ``beta_H``.  Plotting ``log d`` itself would
    mislead: the superstring has *more* states at low level (256 against 24 at
    ``N = 1``), so its curve sits higher even though it grows more slowly.  The
    slope separates the two claims, and the horizontal lines are the asymptotes
    ``4 pi`` and ``2 pi sqrt(2)``, not fits.

    ``levels`` is a short list of :class:`~stringsim.superstring.rns.SuperLevel`
    for the bars; the two degeneracy lists should run as far as convenient, since
    the slope approaches its limit only like ``1/sqrt(N)``.
    """
    fig, (ax, ax2) = _fig(1, 2, figsize=(11.0, 4.4))

    index = np.arange(len(levels))
    width = 0.38
    ax.bar(
        index - width / 2,
        [level.bosons for level in levels],
        width,
        label="bosons (NS)",
        color="tab:blue",
    )
    ax.bar(
        index + width / 2,
        [level.fermions for level in levels],
        width,
        label="fermions (R)",
        color="tab:orange",
    )
    ax.set_yscale("log")
    ax.set_xticks(index)
    ax.set_xticklabels([f"{level.alpha_m2:.0f}" for level in levels])
    ax.set_xlabel(r"$\alpha' M^2$")
    ax.set_ylabel("states")
    ax.set_title("Equal at every level: spacetime supersymmetry")
    ax.legend()

    def local_slope(degeneracies):
        n = np.arange(1, len(degeneracies))
        logd = np.array([math.log(d) for d in degeneracies[1:]])
        mass = np.sqrt(n)
        return n[:-1], np.diff(logd) / np.diff(mass)

    for degeneracies, colour, label, asymptote, name in (
        (bosonic_degeneracies, "tab:red", "bosonic", 4.0 * math.pi, r"$4\pi$"),
        (
            super_degeneracies,
            "tab:blue",
            "superstring",
            2.0 * math.pi * math.sqrt(2.0),
            r"$2\pi\sqrt{2}$",
        ),
    ):
        n, slope = local_slope(degeneracies)
        keep = n >= 4
        ax2.plot(1.0 / np.sqrt(n[keep]), slope[keep], ".", ms=4, color=colour, label=label)
        ax2.axhline(
            asymptote,
            color=colour,
            ls="--",
            lw=1.1,
            label=f"{name} = {asymptote:.3f}",
        )
    ax2.set_xlabel(r"$1/\sqrt{N}$   (asymptotic limit at the left edge)")
    ax2.set_ylabel(r"$\Delta \log d / \Delta \sqrt{N}$")
    ax2.set_title(r"The superstring's $\beta_H$ is smaller, so its $T_H$ is higher")
    ax2.legend(fontsize=8)
    return _save(fig, path)


def plot_root_connectivity(named_roots, path, title: str | None = None) -> Path:
    r"""Which roots are non-orthogonal to which -- the thing that separates the algebras.

    ``named_roots`` is a sequence of ``(label, roots)`` pairs.  Each panel shows
    the matrix that is 1 where two roots have non-zero inner product, which is
    precisely the graph
    :func:`stringsim.compactification.torus.decompose_roots` takes connected
    components of.  A root system that splits into orthogonal pieces shows
    **blank off-diagonal blocks**; a connected one does not.

    That is the honest picture for the two heterotic lattices: both have 480
    roots of squared length 2, so nothing about their *size* distinguishes them,
    and a projection to two dimensions shows only a shapeless cloud.  The block
    structure is the difference, and it is visible here.
    """
    entries = list(named_roots)
    if not entries:
        raise ValueError("give at least one (label, roots) pair")
    fig, axes = _fig(1, len(entries), figsize=(5.4 * len(entries), 5.2))
    axes = np.atleast_1d(axes)
    for ax, (label, roots) in zip(axes, entries, strict=True):
        roots = np.asarray(roots, dtype=float)
        gram = roots @ roots.T
        adjacency = (np.abs(gram) > 1e-8).astype(float)
        ax.imshow(adjacency, cmap="Blues", interpolation="nearest", vmin=0.0, vmax=1.4)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(
        title or "Non-orthogonal root pairs: blank blocks mean the algebra factorises",
        fontsize=11,
    )
    return _save(fig, path)


def plot_fermion_reflection(panels, path, title: str | None = None) -> Path:
    r"""Worldsheet history of ``psi_-``, where the sector is visible as a colour.

    ``panels`` is a sequence of ``(label, evolution)`` pairs of open-string
    :class:`stringsim.superstring.worldsheet.FermionEvolution` objects.  Each is
    drawn as ``psi_-^0(tau, sigma)`` over the strip ``[0, pi] x [0, tau_max]``,
    with ``sigma`` across and ``tau`` up.

    A pulse in ``psi_-`` runs from ``sigma = 0`` to ``sigma = pi``, leaves into
    ``psi_+`` picking up the factor ``eta``, comes back, and re-enters ``psi_-``
    unchanged at ``sigma = 0``.  So consecutive stripes differ by ``eta``: in
    Neveu-Schwarz they **alternate in sign** and the colours flip, in Ramond they
    do not.  That alternation is the whole reason NS modes are half-integral,
    and here it is something you can look at rather than derive.
    """
    entries = list(panels)
    if not entries:
        raise ValueError("give at least one (label, evolution) pair")
    fig, axes = _fig(1, len(entries), figsize=(4.6 * len(entries), 5.0))
    axes = np.atleast_1d(axes)
    scale = max(float(np.max(np.abs(ev.psi_minus[..., 0]))) for _, ev in entries)
    image = None
    for ax, (label, ev) in zip(axes, entries, strict=True):
        image = ax.imshow(
            np.asarray(ev.psi_minus[..., 0]),
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
            extent=(0.0, float(ev.sigma[-1]), 0.0, float(ev.tau[-1])),
        )
        ax.set_xlabel(r"$\sigma$")
        ax.set_title(label, fontsize=10)
        ax.grid(False)
    axes[0].set_ylabel(r"$\tau$")
    fig.colorbar(image, ax=axes[-1], label=r"$\psi_-^{0}$", fraction=0.06, pad=0.03)
    fig.suptitle(
        title or r"Each round trip multiplies the pulse by $\eta$", fontsize=11
    )
    return _save(fig, path)


def plot_wilson_enhancement(points, path, generic_count=None, title=None) -> Path:
    """Where in the (Wilson line, radius) plane the gauge group grows.

    ``points`` is a sequence of ``(a, G, n_roots)``: a one-parameter family of
    Wilson lines against the circle metric, with the number of massless vectors
    there.  Every point drawn is an enhancement -- away from these the count is
    ``generic_count``, which the colour bar is scaled against.

    The picture is the moduli space, not a scan of it: the radii come from
    solving for ``G`` rather than sampling it, so the arcs are exact loci.
    """
    points = np.asarray(list(points), dtype=float)
    if points.size == 0:
        raise ValueError("no enhancement points to draw")
    fig, ax = _fig(figsize=(7.4, 5.0))
    floor = float(points[:, 2].min() if generic_count is None else generic_count)
    scatter = ax.scatter(
        points[:, 0], points[:, 1], c=points[:, 2], s=16, cmap="viridis",
        vmin=floor, vmax=float(points[:, 2].max()), zorder=3,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Wilson line $a$")
    ax.set_ylabel(r"$G = (R/\sqrt{\alpha'})^2$")
    ax.set_title(title or "Enhancement loci in the moduli space of the circle")
    bar = fig.colorbar(scatter, ax=ax)
    bar.set_label("massless vectors")
    if generic_count is not None:
        bar.set_label(f"massless vectors  (generic radius: {generic_count})")
    return _save(fig, path)


def plot_twist_classification(rows, path, title: str | None = None) -> Path:
    """How the automorphisms of each background split, as fractions of the group.

    ``rows`` is a sequence of ``(label, geometric, asymmetric_failing,
    asymmetric_matched)``.  The totals run from 2 to several hundred, so the
    bars are drawn as *proportions* with ``|Aut|`` written at the end: a log
    axis would let the leftmost segment fill the bar whatever its share, which
    is precisely the comparison the picture is for.

    A background with no enhanced gauge symmetry is one solid colour: every
    symmetry of a generic torus is a motion of it.  Where the symmetry is
    enhanced the asymmetric twists take most of the group, and level matching
    then removes most of those again.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("give at least one row")
    labels = [row[0] for row in rows]
    counts = np.array([[row[1], row[2], row[3]] for row in rows], dtype=float)
    totals = counts.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("every background must have at least the identity")
    shares = counts / totals[:, None]

    fig, ax = _fig(figsize=(8.6, 4.8))
    positions = np.arange(len(rows))
    colours = ("tab:blue", "0.75", "tab:orange")
    names = ("geometric", "asymmetric, not level-matched", "asymmetric, level-matched")
    left = np.zeros(len(rows))
    for column, (colour, name) in enumerate(zip(colours, names, strict=True)):
        ax.barh(positions, shares[:, column], left=left, color=colour, label=name)
        for index, (share, start) in enumerate(zip(shares[:, column], left, strict=True)):
            if share > 0.06:
                ax.text(
                    start + share / 2.0,
                    index,
                    f"{int(counts[index, column])}",
                    va="center",
                    ha="center",
                    fontsize=9,
                    color="white" if colour != "0.75" else "0.15",
                )
        left = left + shares[:, column]
    for index, total in enumerate(totals):
        ax.text(1.02, index, f"|Aut| = {int(total)}", va="center", fontsize=9)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.30)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the automorphism group")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=9, frameon=False)
    fig.suptitle(title or "Every symmetry of a generic torus is a motion of it", fontsize=11)
    return _save(fig, path)


def plot_dbi_field(charges, dbi_energy, maxwell_energy, field_ratio, path, title=None) -> Path:
    r"""What Born-Infeld does that Maxwell does not, in the variable that shows it.

    ``charges`` is the dimensionless displacement ``D``; ``dbi_energy`` and
    ``maxwell_energy`` are the two energy densities in units of ``T_p``, and
    ``field_ratio`` is ``E/E_crit`` at that charge.

    Plotted against the *field* the Born-Infeld energy simply diverges, which
    says nothing interesting.  Against the **charge** the two theories separate
    properly: Maxwell's energy keeps going like ``D^2``, Born-Infeld's settles
    into a straight line, and the field it takes to hold that charge stops at
    ``E_crit = 1/(2 pi alpha')`` however much charge is piled on.  That ceiling
    is the fundamental string tension: pull on a string endpoint that hard and
    nothing is left holding it.
    """
    charges = np.asarray(charges, dtype=float)
    fig, (ax, ax2) = _fig(1, 2, figsize=(10.6, 4.4))
    ax.plot(charges, dbi_energy, lw=2.2, color="tab:blue", label="Born-Infeld")
    ax.plot(charges, maxwell_energy, lw=1.6, ls="--", color="tab:orange", label="Maxwell")
    ax.set_xlabel("charge $D$")
    ax.set_ylabel("energy density / $T_p$")
    ax.set_ylim(0.0, float(np.max(dbi_energy)) * 3.0)
    ax.set_title("energy: quadratic against linear")
    ax.legend(loc="upper left", fontsize=9)

    ax2.plot(charges, field_ratio, lw=2.2, color="tab:blue")
    ax2.axhline(1.0, color="tab:red", lw=1.2, ls=":", label=r"$E_{\rm crit} = 1/2\pi\alpha'$")
    ax2.set_xlabel("charge $D$")
    ax2.set_ylabel(r"$E / E_{\rm crit}$")
    ax2.set_ylim(0.0, 1.15)
    ax2.set_title(r"the field it takes never exceeds $E_{\rm crit}$")
    ax2.legend(loc="lower right", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    return _save(fig, path)


def plot_bion_spike(
    profiles, path, extent: float = 1.0, n_grid: int = 120,
    height: float | None = None, title=None,
) -> Path:
    r"""The brane's shape where strings end on it, one panel per string number.

    ``profiles`` is a sequence of ``(label, radial_function)``.  Each panel is a
    two-dimensional slice of the brane, and the spike is the string: far from
    the endpoint the brane is flat, and near it the transverse position runs
    away like the harmonic function ``q / r^(p-2)``.

    Every spike is infinitely tall -- ``q/r^(p-2)`` has no ceiling -- so height
    is not what distinguishes them.  The panels are therefore clipped at a
    common ``height`` and what varies is the **width of the funnel**: more
    strings make a wider mouth at the same depth, since ``q`` is proportional to
    their number.  What is finite, and proportional to ``n``, is the energy per
    unit height, which is the string tension.
    """
    import matplotlib.pyplot as plt

    entries = list(profiles)
    if not entries:
        raise ValueError("give at least one (label, profile) pair")
    axis = np.linspace(-extent, extent, n_grid)
    grid_x, grid_y = np.meshgrid(axis, axis)
    radius = np.maximum(np.hypot(grid_x, grid_y), extent / n_grid)
    surfaces = [np.asarray(fn(radius), dtype=float) for _, fn in entries]
    ceiling = (
        float(height)
        if height is not None
        else min(float(np.asarray(fn(np.array([extent / 6.0])))[0]) for _, fn in entries)
    )

    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(4.4 * len(entries), 4.2), dpi=130)
        for index, ((label, _), surface) in enumerate(zip(entries, surfaces, strict=True)):
            ax = fig.add_subplot(1, len(entries), index + 1, projection="3d")
            ax.plot_surface(
                grid_x,
                grid_y,
                np.minimum(surface, ceiling),
                cmap="viridis",
                linewidth=0,
                rstride=1,
                cstride=1,
            )
            ax.set_zlim(0.0, ceiling)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=10)
            ax.view_init(elev=24.0, azim=-58.0)
        fig.suptitle(title or "A string, seen from the brane it ends on", fontsize=11)
    return _save(fig, path)


def plot_fundamental_domain(images, path, points=(), links=(), title: str | None = None) -> Path:
    r"""The fundamental domain of ``SL(2,Z)`` and its images tiling the plane.

    ``images`` is a sequence of arrays of complex numbers, each the boundary of
    one image of the domain; ``points`` is a sequence of ``(label, tau)`` pairs
    to mark, and ``links`` a sequence of ``(start, end)`` pairs drawn as arrows
    from a point to its representative.

    The shaded region is where the one-loop integral is taken.  Everything below
    it -- the whole strip ``tau_2 -> 0`` that a field theory would integrate over
    and diverge on -- is *already counted*: it is a copy of the domain, reached
    by a modular transformation.  There is no ultraviolet region to regulate
    because there is no ultraviolet region.
    """
    entries = [np.asarray(image, dtype=complex) for image in images]
    if not entries:
        raise ValueError("give at least one image of the domain")
    fig, ax = _fig(figsize=(7.6, 5.4))
    for index, image in enumerate(entries):
        ax.fill(
            image.real,
            image.imag,
            facecolor="tab:blue" if index == 0 else "0.88",
            edgecolor="0.45",
            lw=0.7,
            alpha=0.55 if index == 0 else 0.9,
            zorder=2 if index == 0 else 1,
        )
    for start, end in links:
        start, end = complex(start), complex(end)
        ax.annotate(
            "",
            xy=(end.real, end.imag),
            xytext=(start.real, start.imag),
            arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 1.2, "alpha": 0.8},
            zorder=3,
        )
    for label, tau in points:
        tau = complex(tau)
        ax.plot([tau.real], [tau.imag], "o", ms=7, color="tab:red", zorder=4)
        ax.annotate(
            label,
            (tau.real, tau.imag),
            textcoords="offset points",
            xytext=(9, 5),
            fontsize=8,
            color="tab:red",
            zorder=5,
        )
    ax.axhline(FUNDAMENTAL_DOMAIN_FLOOR, color="tab:red", ls=":", lw=1.1)
    ax.text(
        -1.55,
        FUNDAMENTAL_DOMAIN_FLOOR + 0.04,
        r"$\tau_2 = \sqrt{3}/2$",
        fontsize=9,
        color="tab:red",
    )
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(0.0, 2.2)
    ax.set_xlabel(r"$\tau_1$")
    ax.set_ylabel(r"$\tau_2$")
    ax.set_title(title or "The integral is taken once, over the shaded region only")
    return _save(fig, path)


def plot_one_loop_integrand(heights, values, fitted, path, title: str | None = None) -> Path:
    r"""The torus integrand along the imaginary axis, and what its growth means.

    ``heights`` are values of ``tau_2``, ``values`` the integrand there and
    ``fitted`` the two-parameter model ``exp(-pi alpha' M^2 tau_2) tau_2^b``.

    The rise is not a failure of the calculation.  It is the tachyon: the
    lightest closed string has ``alpha' M^2 = -4``, so the trace grows like
    ``e^{4 pi tau_2}``, and reading that slope off the curve is a measurement of
    a mass from an amplitude.
    """
    heights = np.asarray(heights, dtype=float)
    fig, ax = _fig(figsize=(7.2, 4.6))
    ax.semilogy(heights, values, "o", ms=4, color="tab:blue", label="integrand")
    ax.semilogy(heights, fitted, "-", lw=1.8, color="tab:red", label="fitted growth")
    ax.set_xlabel(r"$\tau_2$")
    ax.set_ylabel("integrand")
    ax.set_title(title or r"Infrared growth: the tachyon, at $\alpha' M^2 = -4$")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, path)


def _decade_ticks(ax, values) -> None:
    """Readable ticks on a log axis that spans less than two decades.

    Matplotlib's minor labels collide badly over a short log range, which is
    exactly the range these figures use.
    """
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    values = np.asarray(values, dtype=float)
    ticks = [t for t in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0) if values.min() <= t <= values.max()]
    if not ticks:
        return
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_channel_duality(moduli, open_channel, closed_channel, path, title=None) -> Path:
    """One diagram in two languages: an open loop and a closed exchange.

    ``moduli`` are values of the open-string modulus ``t``; ``open_channel`` is
    the integrand there and ``closed_channel`` the same quantity computed from
    the closed-string formula at ``s = 1/t``.  The curves lie on top of each
    other because ``eta(i/t) = sqrt(t) eta(it)``, which is why a one-loop
    open-string diagram is a statement about gravity.
    """
    moduli = np.asarray(moduli, dtype=float)
    fig, (ax, ax2) = _fig(1, 2, figsize=(10.6, 4.4))
    ax.loglog(moduli, open_channel, lw=3.0, color="tab:blue", alpha=0.5, label="open loop, $t$")
    ax.loglog(
        moduli, closed_channel, lw=1.4, ls="--", color="tab:orange",
        label="closed exchange, $s = 1/t$",
    )
    ax.set_xlabel("$t$")
    ax.set_ylabel("integrand")
    ax.set_title("the same function, twice")
    ax.legend(loc="upper center", fontsize=9)
    _decade_ticks(ax, moduli)

    relative = np.abs(np.asarray(open_channel) / np.asarray(closed_channel) - 1.0)
    ax2.loglog(moduli, np.maximum(relative, 1e-18), lw=1.8, color="tab:blue")
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("relative difference")
    ax2.set_ylim(1e-18, 1.0)
    ax2.set_title("and it is the same to machine precision")
    _decade_ticks(ax2, moduli)
    if title:
        fig.suptitle(title, fontsize=11)
    return _save(fig, path)


def _sphere_scale(heights, radii) -> float:
    """The radius to normalise a fuzzy sphere by, so panels are comparable."""
    reach = max(float(np.max(np.abs(heights))), float(np.max(radii)))
    return reach if reach > 0.0 else 1.0


def plot_fuzzy_sphere(entries, path, title: str | None = None) -> Path:
    r"""A fuzzy sphere is a stack of circles, and only that.

    ``entries`` is a sequence of ``(label, heights, radii)``, one per panel:
    the eigenvalues of ``Phi_3`` and the circle radius at each, from
    :func:`stringsim.branes.myers.latitudes`.

    There is nothing between the circles.  ``N`` D0-branes in a flux have become
    a two-dimensional object made of ``N`` latitudes, and it only looks like a
    sphere once ``N`` is large -- which is the same statement as the commutators
    shrinking relative to the radius.

    Each panel is drawn at its own scale.  The physical radius grows like ``N``,
    so on a shared axis the small spheres shrink to dots and the thing the
    picture is about -- how many circles there are -- disappears.  Put the
    radius in the label.
    """
    import matplotlib.pyplot as plt

    entries = list(entries)
    if not entries:
        raise ValueError("give at least one (label, heights, radii) triple")
    # Each panel is scaled to its own radius: the sphere grows like N, and at a
    # shared absolute scale the small ones become invisible dots.  What the
    # picture is for is the *filling in*, so the size goes in the label instead.
    entries = [
        (label, np.asarray(z) / _sphere_scale(z, r), np.asarray(r) / _sphere_scale(z, r))
        for label, z, r in entries
    ]
    reach = 1.05
    angle = np.linspace(0.0, 2.0 * np.pi, 200)
    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(3.9 * len(entries), 4.0), dpi=130)
        for index, (label, heights, radii) in enumerate(entries):
            ax = fig.add_subplot(1, len(entries), index + 1, projection="3d")
            for height, radius in zip(np.asarray(heights), np.asarray(radii), strict=True):
                ax.plot(
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    np.full_like(angle, height),
                    lw=1.6,
                    color="tab:blue",
                    alpha=0.85,
                )
            ax.set_xlim(-reach, reach)
            ax.set_ylim(-reach, reach)
            ax.set_zlim(-reach, reach)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title(label, fontsize=10)
            ax.view_init(elev=18.0, azim=-62.0)
        fig.suptitle(title or "N D0-branes, drawn as the sphere they became", fontsize=11)
    return _save(fig, path)


def plot_myers_landscape(configurations, path, title: str | None = None) -> Path:
    """Every way of splitting N branes into blocks, and what each costs.

    ``configurations`` is a sequence of
    :class:`stringsim.branes.myers.Configuration`.  The single block sits at the
    bottom; ``(1, 1, ..., 1)`` -- commuting matrices, the configuration one
    would have called the vacuum -- sits at zero, at the top of the picture.

    The depth tracks ``sum N_a (N_a^2 - 1)``, which is why one big block beats
    every way of dividing the branes up.
    """
    entries = list(configurations)
    if not entries:
        raise ValueError("give at least one configuration")
    order = np.argsort([entry.energy for entry in entries])
    energies = np.array([entries[i].energy for i in order])
    labels = ["+".join(str(size) for size in entries[i].partition) for i in order]
    colours = ["tab:orange" if entries[i].is_irreducible else "tab:blue" for i in order]

    fig, ax = _fig(figsize=(max(6.4, 0.42 * len(entries) + 2.0), 4.4))
    positions = np.arange(len(entries))
    ax.bar(positions, energies, color=colours)
    ax.axhline(0.0, color="0.4", lw=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("block sizes")
    ax.set_ylabel("$V$")
    ax.set_title(title or "One block beats every way of dividing the branes up")
    ax.plot([], [], "s", color="tab:orange", label="a single fuzzy sphere")
    ax.plot([], [], "s", color="tab:blue", label="split into blocks")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, path)


def plot_hodge_diamond(entries, path, title: str | None = None) -> Path:
    r"""Hodge diamonds side by side, one per panel.

    ``entries`` is a sequence of ``(label, diamond)`` where ``diamond`` maps
    ``(p, q)`` to an integer, as
    :func:`stringsim.compactification.hodge.untwisted_hodge` returns and
    :func:`stringsim.compactification.hodge.hodge_numbers` completes.

    The diamond is drawn the usual way, ``h^{p,q}`` at the point
    ``(q - p, -(p + q))``, so the two numbers that are not forced by symmetry
    sit in the middle row.  Discrete torsion exchanges them: the same geometry,
    the same Euler characteristic up to a sign, and a mirror.
    """
    entries = list(entries)
    if not entries:
        raise ValueError("give at least one (label, diamond) pair")
    fig, axes = _fig(1, len(entries), figsize=(3.7 * len(entries) + 0.6, 4.3))
    axes = np.atleast_1d(axes)
    for ax, (label, diamond) in zip(axes, entries, strict=True):
        middle = {(1, 1), (2, 2), (1, 2), (2, 1)}
        for (p, q), value in diamond.items():
            x, y = q - p, -(p + q)
            highlight = (p, q) in middle
            ax.text(
                x,
                y,
                f"{value}",
                ha="center",
                va="center",
                fontsize=13 if highlight else 11,
                color="tab:blue" if highlight else "0.35",
                fontweight="bold" if highlight else "normal",
            )
        ax.set_xlim(-3.4, 3.4)
        ax.set_ylim(-6.7, 0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_frame_on(False)
        ax.set_title(label, fontsize=10)
    fig.suptitle(title or "The Hodge diamond, and what a phase does to it", fontsize=11)
    return _save(fig, path)


def plot_fixed_loci(entries, path, title: str | None = None) -> Path:
    """What each group element holds still, and how much of it there is.

    ``entries`` is a sequence of ``(label, [(element, components, dimension)])``.
    Bars are the number of components; the colour says whether the pieces are
    points or positive-dimensional, which is the whole difference between a
    contribution to the Euler characteristic and a zero.
    """
    entries = list(entries)
    if not entries:
        raise ValueError("give at least one orbifold")
    labels, counts, colours, dividers = [], [], [], []
    position = 0
    for name, rows in entries:
        for element, components, dimension in rows:
            labels.append(f"{name}\n{element}")
            counts.append(components)
            colours.append("tab:orange" if dimension == 0 else "tab:blue")
            position += 1
        dividers.append(position - 0.5)

    fig, ax = _fig(figsize=(max(6.4, 0.9 * len(counts) + 2.0), 4.2))
    ax.bar(np.arange(len(counts)), counts, color=colours)
    for index, value in enumerate(counts):
        ax.text(index, value + 0.6, str(value), ha="center", fontsize=9)
    for line in dividers[:-1]:
        ax.axvline(line, color="0.7", lw=0.9, ls="--")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("components of the fixed set")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.plot([], [], "s", color="tab:orange", label="isolated points (counts towards $\\chi$)")
    ax.plot([], [], "s", color="tab:blue", label="curves ($\\chi = 0$)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title or "What each element of the group holds still")
    return _save(fig, path)


def plot_matrix_worldlines(times, eigenvalues, path, title: str | None = None) -> Path:
    """Where the branes are, as the eigenvalues of one matrix through time.

    ``eigenvalues`` has shape ``(n_frames, n_branes)``, from
    :meth:`stringsim.branes.matrixmodel.Trajectory.eigenvalues`.

    When the lines are far apart the matrices nearly commute and the branes are
    genuinely separate objects moving freely.  Where the lines approach, the
    off-diagonal entries -- strings stretched between them -- become light, the
    potential wakes up, and the motion stops being predictable.  Everything
    interesting in the matrix model happens where the worldlines cross.
    """
    times = np.asarray(times, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if eigenvalues.ndim != 2:
        raise ValueError(f"expected (n_frames, n_branes), got {eigenvalues.shape}")
    fig, ax = _fig(figsize=(7.6, 4.6))
    for index in range(eigenvalues.shape[1]):
        ax.plot(times, eigenvalues[:, index], lw=1.3)
    ax.set_xlabel("$t$")
    ax.set_ylabel("eigenvalues of $X_1$")
    ax.set_title(title or "Brane worldlines: where they meet, the strings wake up")
    return _save(fig, path)


def plot_lyapunov(fit, energies, exponents, power, path, title: str | None = None) -> Path:
    r"""Exponential separation, and the power of the energy it scales with.

    ``fit`` is a :class:`stringsim.branes.matrixmodel.LyapunovFit`; ``energies``
    and ``exponents`` are the rescaled family, and ``power`` the fitted slope of
    ``log lambda`` against ``log E``.

    The left panel is the measurement: banked ``log`` growth against time, and a
    straight line through it.  The right is the prediction with nothing
    adjustable in it -- ``X -> sX``, ``t -> t/s`` is a symmetry, so ``E`` goes
    like ``s^4`` and ``lambda`` like ``s``, and the slope has to be ``1/4``.
    """
    energies = np.asarray(energies, dtype=float)
    exponents = np.asarray(exponents, dtype=float)
    fig, (ax, ax2) = _fig(1, 2, figsize=(10.6, 4.4))
    ax.plot(fit.times, fit.growth, ".", ms=4, color="tab:blue", label="banked $\\log$ growth")
    line = fit.exponent * fit.times + (fit.growth[-1] - fit.exponent * fit.times[-1])
    ax.plot(fit.times, line, "-", lw=1.8, color="tab:red",
            label=rf"$\lambda = {fit.exponent:.4f}$")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\sum \log(\mathrm{growth})$")
    ax.set_title("two nearby configurations, separating")
    ax.legend(loc="upper left", fontsize=9)

    ax2.loglog(energies, exponents, "o", ms=7, color="tab:blue", label="measured")
    grid = np.geomspace(energies.min(), energies.max(), 50)
    reference = exponents[0] * (grid / energies[0]) ** 0.25
    ax2.loglog(grid, reference, "-", lw=1.6, color="0.45", label=r"$E^{1/4}$")
    ax2.set_xlabel("$E$")
    ax2.set_ylabel(r"$\lambda$")
    ax2.set_title(rf"fitted power {power:.4f}, against $1/4$")
    ax2.legend(loc="upper left", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    return _save(fig, path)


def plot_shift_landscape(
    first, second, mismatch, marked, path, title: str | None = None
) -> Path:
    r"""The ground-state mismatch over the square of shifts, and what closes it.

    ``first`` and ``second`` are the two components of the shift on ``[0, 1)``,
    ``mismatch`` the array of ``E_L - E_R`` there, and ``marked`` a sequence of
    ``(w, n, level_matched)`` for the rational shifts that actually close into a
    finite-order element.

    Most of the square is unreachable: a shift has to be rational for the twist
    to have an order at all, so only the lattice points count.  Of those, only
    the ones where ``N (E_L - E_R)`` is an integer give a consistent orbifold --
    the filled markers.  The background is smooth and says nothing on its own;
    the arithmetic on top of it is the whole content.
    """
    marked = list(marked)
    fig, ax = _fig(figsize=(6.6, 5.2))
    image = ax.pcolormesh(first, second, mismatch, cmap="RdBu_r", shading="auto")
    fig.colorbar(image, ax=ax, label=r"$E_L - E_R$")
    for horizontal, vertical, matched in marked:
        ax.plot(
            [horizontal],
            [vertical],
            "o" if matched else "x",
            ms=10 if matched else 7,
            mew=2.0,
            color="k" if matched else "0.4",
            markerfacecolor="k" if matched else "none",
        )
    ax.plot([], [], "o", ms=8, color="k", label="closes and level-matches")
    ax.plot([], [], "x", ms=7, mew=2.0, color="0.4", label="closes but does not")
    ax.set_xlabel("shift in $w$")
    ax.set_ylabel("shift in $n$")
    ax.set_title(title or "Which shift repairs a twist that fails on its own")
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, path)


def plot_commutation(panels, path, title: str | None = None) -> Path:
    """Which pairs of a group commute, as a matrix of filled cells.

    ``panels`` is a sequence of ``(label, size, pairs)`` where ``pairs`` lists
    the commuting index pairs.

    An abelian group is a solid block: every pair commutes, and the Euler
    characteristic sums over all of ``G x G``.  A non-abelian one is a pattern,
    and the number of filled cells is ``|G|`` times the number of conjugacy
    classes -- an identity the picture makes visible rather than asserts.
    """
    entries = list(panels)
    if not entries:
        raise ValueError("give at least one (label, size, pairs) triple")
    fig, axes = _fig(1, len(entries), figsize=(4.2 * len(entries), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (label, size, pairs) in zip(axes, entries, strict=True):
        grid = np.zeros((size, size))
        for first, second in pairs:
            grid[first, second] = 1.0
        ax.imshow(grid, cmap="Blues", interpolation="nearest", vmin=0.0, vmax=1.4)
        ax.set_title(f"{label}\n{len(pairs)} of {size * size} pairs", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(title or "Commuting pairs: what the Euler characteristic sums over", fontsize=11)
    return _save(fig, path)


def plot_central_charge(
    dims, measured, path, title: str = "Central charge from the algebra"
) -> Path:
    r"""Measured ``c`` against the line ``c = D``.

    ``measured`` is a mapping ``{m: [c for each dim]}``, one series per
    commutator :math:`[L_m, L_{-m}]` used to extract it.  Different ``m`` give
    different central terms -- :math:`(m^3-m)/12` is 1/2 at ``m = 2`` and 2 at
    ``m = 3`` -- so the series landing on the same line is the check, not the
    line itself.

    A residual panel would be the usual way to show how close this is, but the
    residual is not close: it is zero to the last bit, at every point plotted.
    A log axis cannot draw that, so the figure states it instead.
    """
    dims = np.asarray(list(dims), dtype=float)
    fig, ax = _fig(figsize=(6.8, 4.6))
    ax.plot(dims, dims, color="0.4", lw=1.0, ls="--", label="c = D")
    markers = ["o", "s", "^", "D"]
    worst = 0.0
    for (m, values), marker in zip(sorted(measured.items()), markers, strict=False):
        values = np.asarray(values, dtype=float)
        worst = max(worst, float(np.max(np.abs(values - dims))))
        ax.plot(
            dims,
            values,
            marker,
            ms=7,
            mfc="none",
            label=f"from $[L_{{{m}}}, L_{{-{m}}}]$",
        )
    ax.set_xlabel("spacetime dimension $D$")
    ax.set_ylabel("central charge $c$")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    exact = "exactly, bit for bit" if worst == 0.0 else f"to {worst:.1e}"
    ax.text(
        0.97,
        0.06,
        f"$|c - D| = 0$ {exact}, for every point and both commutators",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="0.25",
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "0.75", "lw": 0.8},
    )
    return _save(fig, path)


def plot_ghost_onset(
    records, lightcone_counts, path, title: str = "The critical dimension from unitarity"
) -> Path:
    r"""Two bounds on ``D``, pointing opposite ways, meeting at 26.

    ``records`` is a sequence of
    :class:`~stringsim.quantum.virasoro.Inertia`, one per dimension, and
    ``lightcone_counts`` the matching light-cone degeneracies.

    Left: the smallest norm in the physical subspace.  It is non-negative up
    to 26 and turns over after it -- an upper bound.

    Right: how many states the covariant construction has that the light cone
    does not, and how many of them are ghosts.  The excess is 1 below 26 and 0
    from 26 on -- a lower bound.  Only at 26 are both zero.
    """
    records = list(records)
    dims = np.array([r.dim for r in records], dtype=float)
    smallest = np.array([r.smallest for r in records])
    negative = np.array([r.negative for r in records], dtype=float)
    excess = np.array([r.positive for r in records], dtype=float) - np.asarray(
        list(lightcone_counts), dtype=float
    )

    fig, axes = _fig(1, 2, figsize=(10.4, 4.3))
    left, right = axes

    left.axhline(0.0, color="0.3", lw=1.0)
    left.plot(dims, smallest, "o-", ms=4, color="#1f77b4")
    left.fill_between(dims, smallest, 0.0, where=smallest < 0, color="#d62728", alpha=0.25)
    left.set_xlabel("spacetime dimension $D$")
    left.set_ylabel("smallest physical norm")
    left.set_title("negative norms appear above 26")

    right.axhline(0.0, color="0.3", lw=1.0)
    right.plot(dims, excess, "o-", ms=4, color="#2ca02c", label="covariant $-$ light-cone")
    right.plot(dims, negative, "s-", ms=4, color="#d62728", label="ghosts")
    right.set_xlabel("spacetime dimension $D$")
    right.set_ylabel("state count")
    right.set_title("the two counts agree only at 26")
    right.legend(frameon=False, loc="upper left")

    for ax in axes:
        ax.axvline(26.0, color="0.55", lw=1.0, ls=":")
        ax.annotate(
            "$D = 26$",
            xy=(26.0, ax.get_ylim()[1]),
            xytext=(-4, -12),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color="0.35",
        )
    fig.suptitle(title, y=1.0)
    return _save(fig, path)


def plot_no_ghost_region(
    dims, intercepts, grid, path, boundary=None, title: str = "Where the ghosts are"
) -> Path:
    r"""The ``(D, a)`` plane, coloured by the smallest physical norm.

    ``grid`` is what :func:`~stringsim.quantum.virasoro.no_ghost_map` returns:
    rows indexed by dimension, columns by intercept, each entry normalised by
    the largest eigenvalue magnitude in its own cell so that different ``D``
    are comparable.  Blue is ghost-free, red is not.

    ``boundary`` may be a sequence of ``(dim, a)`` pairs from
    :func:`~stringsim.quantum.virasoro.ghost_boundary` -- the crossing that
    exists only for ``D >= 26``.

    The horizontal line at ``a = 1`` is where the normal-ordering constant
    actually sits in ``D = 26``; the picture shows that it is a single point on
    the boundary of the allowed region, not an interior choice.
    """
    from matplotlib.colors import TwoSlopeNorm

    dims = np.asarray(list(dims), dtype=float)
    intercepts = np.asarray(intercepts, dtype=float)
    grid = np.asarray(grid, dtype=float)
    limit = max(float(np.max(np.abs(grid))), 1e-12)

    fig, ax = _fig(figsize=(7.4, 5.0))
    mesh = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        extent=(
            float(intercepts[0]),
            float(intercepts[-1]),
            float(dims[0]) - 0.5,
            float(dims[-1]) + 0.5,
        ),
    )
    ax.contour(
        intercepts,
        dims,
        grid,
        levels=[0.0],
        colors="k",
        linewidths=1.2,
    )
    if boundary is not None:
        pairs = [(d, a) for d, a in boundary if a is not None]
        if pairs:
            ax.plot(
                [a for _, a in pairs],
                [d for d, _ in pairs],
                "wo",
                ms=5,
                mec="k",
                label="root of the smallest norm",
            )
            ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.axhline(26.0, color="0.2", lw=1.0, ls=":")
    ax.axvline(1.0, color="0.2", lw=1.0, ls=":")
    ax.plot([1.0], [26.0], "k*", ms=13)
    ax.annotate(
        "$a = 1$, $D = 26$",
        xy=(1.0, 26.0),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        fontsize=9,
    )
    ax.set_xlabel("intercept $a$")
    ax.set_ylabel("spacetime dimension $D$")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label="smallest physical norm (normalised)")
    return _save(fig, path)


def plot_anomaly_conditions(gravity, gauge, path, title: str = "What ten dimensions demand"):
    r"""The two conditions Green-Schwarz cancellation imposes, side by side.

    ``gravity`` is a sequence of ``(dim G, coefficient of tr R^6)`` and
    ``gauge`` a sequence of ``(N, coefficient of tr F^6, dim SO(N))``.  Both
    cross zero, and the crossings are not the same statement: the first fixes
    only how many gauge bosons there are, the second fixes which group they
    belong to.

    They meet because ``dim SO(32) = 496``, which is marked on the right-hand
    axis rather than asserted in the caption.
    """
    gravity = [(float(n), float(c)) for n, c in gravity]
    gauge = [(float(n), float(c), int(d)) for n, c, d in gauge]

    fig, (left, right) = _fig(1, 2, figsize=(10.6, 4.4))

    xs = [n for n, _ in gravity]
    ys = [c for _, c in gravity]
    left.axhline(0.0, color="0.3", lw=1.0)
    left.plot(xs, ys, "o-", ms=4, color="#1f77b4")
    left.axvline(496.0, color="#d62728", lw=1.0, ls=":")
    left.annotate(
        "496",
        xy=(496.0, 0.0),
        xytext=(6, 10),
        textcoords="offset points",
        fontsize=10,
        color="#d62728",
    )
    left.set_xlabel(r"gauge group dimension $\dim G$")
    left.set_ylabel(r"coefficient of ${\rm tr}\,R^6$")
    left.set_title("pure gravity: fixes the dimension")

    xs = [n for n, _, _ in gauge]
    ys = [c for _, c, _ in gauge]
    right.axhline(0.0, color="0.3", lw=1.0)
    right.plot(xs, ys, "s-", ms=4, color="#2ca02c")
    right.axvline(32.0, color="#d62728", lw=1.0, ls=":")
    marked = [(n, c, d) for n, c, d in gauge if c == 0.0]
    for n, c, d in marked:
        right.plot([n], [c], "*", ms=15, color="#d62728")
        right.annotate(
            f"SO({int(n)}), dim {d}",
            xy=(n, c),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=10,
            color="#d62728",
        )
    right.set_xlabel(r"$N$ in $SO(N)$")
    right.set_ylabel(r"coefficient of ${\rm tr}\,F^6$")
    right.set_title("pure gauge: fixes the group")

    fig.suptitle(title, y=1.0)
    return _save(fig, path)


def plot_anomaly_scan(points, survivors, path, title: str = "A family, and what survives it"):
    r"""Every candidate group against the two things that must vanish.

    ``points`` is a sequence of ``(dimension, sixth-order residue, label)`` and
    ``survivors`` a sequence of ``(dimension, label)``.  The horizontal axis is
    the gravitational condition -- everything off 496 is out -- and the
    vertical one the gauge condition, on a symmetric-log scale so that the
    exact zeros are visible rather than falling off the bottom.

    Only the origin of this picture is a string theory.
    """
    points = [(float(d), abs(float(r)), str(name)) for d, r, name in points]
    fig, ax = _fig(figsize=(7.6, 5.0))

    floor = min((r for _, r, _ in points if r > 0), default=1.0) / 3.0
    xs = [d for d, _, _ in points]
    ys = [max(r, 0.0) for _, r, _ in points]
    ax.set_yscale("symlog", linthresh=floor)
    ax.scatter(xs, ys, s=12, color="#8899aa", alpha=0.55, label="candidates")
    ax.axvline(496.0, color="#d62728", lw=1.0, ls=":")
    ax.axhline(0.0, color="#d62728", lw=1.0, ls=":")

    for slot, (dimension, name) in enumerate(survivors):
        ax.plot([float(dimension)], [0.0], "*", ms=18, color="#d62728", zorder=5)
        ax.annotate(
            name,
            xy=(float(dimension), 0.0),
            xytext=(14, 14 + 16 * slot),
            textcoords="offset points",
            fontsize=10,
            color="#d62728",
            arrowprops={"arrowstyle": "-", "color": "#d62728", "lw": 0.7},
        )
    ax.set_xlabel(r"$\dim G$ -- must be 496")
    ax.set_ylabel(r"largest surviving sixth-order trace -- must be 0")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    return _save(fig, path)


def plot_orientifold_spectrum(before, comparison, path, title: str = "Gauging worldsheet parity"):
    r"""What the projection removes, and what the result agrees with.

    ``before`` is a sequence of ``(sector, kept, removed)`` state counts for the
    massless level of type IIB.  ``comparison`` is a sequence of
    ``(label, supergravity, gauge)`` for the finished theories.

    Left: half of type IIB survives, and which half depends on the sector --
    the symmetric part of NS-NS, the antisymmetric part of R-R, and one
    diagonal copy of the two mixed sectors.

    Right: type I and the heterotic ``SO(32)`` string, built with nothing in
    common, at the same massless level and with the same split.
    """
    before = [(str(name), int(kept), int(removed)) for name, kept, removed in before]
    comparison = [(str(name), int(sugra), int(gauge)) for name, sugra, gauge in comparison]

    fig, (left, right) = _fig(1, 2, figsize=(10.8, 4.6))

    labels = [name for name, _, _ in before]
    kept = np.array([k for _, k, _ in before], dtype=float)
    gone = np.array([r for _, _, r in before], dtype=float)
    positions = np.arange(len(labels))
    left.bar(positions, kept, color="#1f77b4", label="survives")
    left.bar(positions, gone, bottom=kept, color="0.82", hatch="//", edgecolor="0.55",
             label="projected out")
    for x, (k, g) in enumerate(zip(kept, gone, strict=True)):
        left.text(x, k + g + 1.5, f"{int(k)}", ha="center", fontsize=9, color="#1f77b4")
    left.set_xticks(positions)
    left.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    left.set_ylabel("massless states")
    left.set_title(f"type IIB: {int(kept.sum() + gone.sum())} -> {int(kept.sum())}")
    left.legend(frameon=False, loc="upper left", fontsize=9)

    names = [name for name, _, _ in comparison]
    sugra = np.array([s for _, s, _ in comparison], dtype=float)
    gauge = np.array([g for _, _, g in comparison], dtype=float)
    positions = np.arange(len(names))
    right.bar(positions, sugra, color="#2ca02c", label="supergravity")
    right.bar(positions, gauge, bottom=sugra, color="#ff7f0e", label="gauge")
    for x, (s, g) in enumerate(zip(sugra, gauge, strict=True)):
        right.text(x, s + g + 120, f"{int(s + g)}", ha="center", fontsize=10)
        right.text(x, s + g / 2, f"{int(g)}", ha="center", va="center", fontsize=9, color="white")
        right.text(x, s / 2, f"{int(s)}", ha="center", va="center", fontsize=9, color="white")
    right.set_xticks(positions)
    right.set_xticklabels(names, fontsize=10)
    right.set_ylabel("massless states")
    right.set_ylim(0, max(sugra + gauge) * 1.32)
    right.set_title("two constructions, one spectrum")
    right.legend(frameon=False, loc="upper center", ncol=2, fontsize=9)

    fig.suptitle(title, y=1.0)
    return _save(fig, path)


_ARROWS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}


def _edge(ax, start, end, mark, colour):
    """Draw one side of the identification square."""
    (x0, y0), (x1, y1) = start, end
    if mark == "boundary":
        ax.plot([x0, x1], [y0, y1], color="#d62728", lw=3.5, solid_capstyle="round", zorder=3)
        return
    ax.plot([x0, x1], [y0, y1], color=colour, lw=1.6, zorder=3)
    dx, dy = _ARROWS[mark]
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ax.annotate(
        "",
        xy=(mx + 0.16 * dx, my + 0.16 * dy),
        xytext=(mx - 0.16 * dx, my - 0.16 * dy),
        arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.6},
        zorder=4,
    )


def plot_worldsheet_surfaces(panels, path, title: str = "The four one-loop worldsheets"):
    r"""Identification diagrams for the surfaces with ``chi = 0``.

    ``panels`` is a sequence of ``(name, euler, sides)`` where ``sides`` maps
    ``left, right, top, bottom`` to ``"boundary"`` or an arrow direction.  Two
    sides carrying arrows are glued; matching directions glue plainly, opposite
    ones glue with a flip and make the surface unoriented.  A red edge is a
    genuine boundary, where an open string ends.

    Nothing here is drawn from a picture book: the sides come from whichever
    ``(g, b, c)`` the caller enumerated, and the caption reports the Euler
    characteristic that put the surface at one loop.
    """
    from matplotlib.patches import Rectangle

    panels = list(panels)
    rows = (len(panels) + 1) // 2
    fig, axes = _fig(rows, 2, figsize=(7.8, 3.3 * rows))
    flat = np.atleast_1d(axes).ravel()

    for ax, (name, euler, sides) in zip(flat, panels, strict=False):
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(-0.45, 1.45)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#eef2f7", edgecolor="none", zorder=1))
        corners = {
            "bottom": ((0, 0), (1, 0)),
            "top": ((0, 1), (1, 1)),
            "left": ((0, 0), (0, 1)),
            "right": ((1, 0), (1, 1)),
        }
        colours = {"left": "#1f77b4", "right": "#1f77b4", "top": "#2ca02c", "bottom": "#2ca02c"}
        for side, (start, end) in corners.items():
            _edge(ax, start, end, sides[side], colours[side])
        ax.set_title(f"{name}   $\\chi = {euler}$", fontsize=11)

    for ax in flat[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, y=1.0)
    fig.text(
        0.5,
        0.005,
        "arrows mark glued sides -- opposite arrows glue with a flip; "
        "red edges are boundaries",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    return _save(fig, path)


def plot_modular_invariance(rows, weights, path, title: str = "What modular invariance needs"):
    r"""The two conditions on the lattice, failing separately.

    ``rows`` is a sequence of ``(label, T residual, S residual, truncation)``,
    one per lattice tried.  ``weights`` is a sequence of
    ``(|tau|, |Theta(-1/tau)| / |Theta(tau)|, |tau|^d)`` sampled along a path.

    Left: ``T`` invariance needs only that the lattice is even, so it survives
    every sublattice; ``S`` needs self-duality and does not.  The dashed line is
    the truncation of the lattice sum -- anything at that level is arithmetic,
    anything far above it is the lattice.

    Right: the theta series is *not* invariant on its own.  It carries weight,
    picking up exactly ``|tau|^d``, and the invariance of the full integrand is
    that factor cancelling against ``|eta|^{2d}``.
    """
    rows = [(str(name), float(t), float(s), float(gap)) for name, t, s, gap in rows]
    weights = [(float(a), float(b), float(c)) for a, b, c in weights]

    fig, (left, right) = _fig(1, 2, figsize=(10.8, 4.5))

    labels = [name for name, _, _, _ in rows]
    positions = np.arange(len(labels))
    floor = 1e-17
    t_values = np.maximum([t for _, t, _, _ in rows], floor)
    s_values = np.maximum([s for _, _, s, _ in rows], floor)
    width = 0.38
    left.bar(positions - width / 2, t_values, width, color="#1f77b4", label="$T$: $\\tau + 1$")
    left.bar(positions + width / 2, s_values, width, color="#d62728", label="$S$: $-1/\\tau$")
    gap = max(g for _, _, _, g in rows)
    left.axhline(max(gap, floor), color="0.35", lw=1.0, ls="--")
    left.annotate(
        "truncation of the lattice sum",
        xy=(-0.5, max(gap, floor)),
        xytext=(4, -12),
        textcoords="offset points",
        ha="left",
        fontsize=8,
        color="0.35",
    )
    left.set_yscale("log")
    left.set_xticks(positions)
    left.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    left.set_ylabel("relative change in the integrand")
    left.set_title("even is not enough")
    left.legend(frameon=False, fontsize=9)

    xs = [a for a, _, _ in weights]
    right.plot(xs, [b for _, b, _ in weights], "o", ms=6, mfc="none",
               color="#1f77b4", label=r"$|\Theta(-1/\tau)| / |\Theta(\tau)|$")
    right.plot(xs, [c for _, _, c in weights], "-", lw=1.4, color="#2ca02c",
               label=r"$|\tau|^{d}$")
    right.set_xlabel(r"$|\tau|$")
    right.set_ylabel("ratio")
    right.set_title("the theta series carries weight")
    right.legend(frameon=False, fontsize=9)

    fig.suptitle(title, y=1.0)
    return _save(fig, path)


def plot_narain_levels(panels, path, title: str = "Where the charges sit"):
    r"""The lattice's levels in the :math:`(\ell_L^2/2,\ \ell_R^2/2)` plane.

    ``panels`` is a sequence of ``(label, levels)`` where ``levels`` maps
    ``(h, h_bar)`` to a multiplicity.  Marker area is the multiplicity; the two
    axes carry the roots, so a point on an axis at 1 is a gauge boson.

    Each charge moves along a line of fixed ``h - h_bar = n.w`` as the moduli
    change, since that combination does not depend on them.  Sitting exactly on
    an axis is what a special point in moduli space *is*.
    """
    panels = list(panels)
    fig, axes = _fig(1, len(panels), figsize=(4.4 * len(panels), 4.3))
    axes = np.atleast_1d(axes)
    reach = max(
        (max(max(k) for k in levels) if levels else 1.0) for _, levels in panels
    )
    reach = min(float(reach), 4.0)

    for ax, (label, levels) in zip(axes, panels, strict=True):
        xs, ys, sizes = [], [], []
        roots, root_states = [], 0
        for (h, h_bar), count in levels.items():
            if h > reach + 1e-9 or h_bar > reach + 1e-9:
                continue
            xs.append(h)
            ys.append(h_bar)
            sizes.append(18 + 26 * (count - 1))
            if (abs(h - 1) < 1e-9 and h_bar < 1e-9) or (abs(h_bar - 1) < 1e-9 and h < 1e-9):
                roots.append((h, h_bar))
                # A root *level* can hold several charges -- (n, w) and its
                # negative at least -- and it is the states that are gauge
                # bosons, so the title counts multiplicities and not points.
                root_states += count
        ax.axhline(0.0, color="0.75", lw=1.0)
        ax.axvline(0.0, color="0.75", lw=1.0)
        ax.plot([0, reach], [0, reach], color="0.85", lw=1.0, ls="--")
        ax.scatter(xs, ys, s=sizes, color="#1f77b4", alpha=0.75, zorder=3)
        if roots:
            ax.scatter([x for x, _ in roots], [y for _, y in roots], s=90,
                       facecolors="none", edgecolors="#d62728", lw=1.6, zorder=4)
        ax.set_xlim(-0.25, reach + 0.25)
        ax.set_ylim(-0.25, reach + 0.25)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\ell_L^2 / 2$")
        ax.set_ylabel(r"$\ell_R^2 / 2$")
        ax.set_title(f"{label} -- {root_states} roots", fontsize=11)

    fig.suptitle(title, y=1.0)
    return _save(fig, path)


def plot_koba_nielsen(curves, comparison, gauges, path, title: str = "The amplitude, derived"):
    r"""Where the Veneziano amplitude comes from, in three pictures.

    ``curves`` is a sequence of ``(label, x, integrand)``; ``comparison`` a
    sequence of ``(t, from the integral, from the Beta function)``; ``gauges`` a
    sequence of ``(label, anchors, amplitudes)``.

    Left: the Koba-Nielsen integrand along the boundary.  As ``s`` climbs
    towards the first pole the exponent at ``x = 0`` reaches ``-1`` and the area
    under the curve stops being finite.  The pole is that endpoint.

    Middle: the integral against the closed form, over a range of ``t``.

    Right: the same amplitude computed in different ``SL(2,R)`` gauges.  Flat is
    the statement that three punctures can be put anywhere; the sloped lines are
    what happens when the external masses are nudged off shell, which is the
    condition the flatness rests on.
    """
    fig, axes = _fig(1, 3, figsize=(13.2, 4.2))
    left, middle, right = axes

    for label, xs, values in curves:
        left.plot(np.asarray(xs, dtype=float), np.asarray(values, dtype=float),
                  lw=1.6, label=str(label))
    left.set_yscale("log")
    left.set_xlabel("$x$, the free puncture")
    left.set_ylabel("Koba-Nielsen integrand")
    left.set_title("the pole is an endpoint")
    left.legend(frameon=False, fontsize=8)

    ts = np.array([row[0] for row in comparison], dtype=float)
    integral = np.array([row[1] for row in comparison], dtype=float)
    closed = np.array([row[2] for row in comparison], dtype=float)
    middle.plot(ts, closed, "-", lw=1.6, color="#2ca02c", label=r"$B(-\alpha(s), -\alpha(t))$")
    middle.plot(ts, integral, "o", ms=6, mfc="none", color="#1f77b4",
                label="worldsheet integral")
    middle.set_xlabel("$t$")
    middle.set_ylabel("$A(s, t)$")
    middle.set_title("integral against closed form")
    middle.legend(frameon=False, fontsize=9)

    styles = ["o-", "s--", "^:"]
    for (label, anchors, values), style in zip(gauges, styles, strict=False):
        anchors = np.asarray(anchors, dtype=float)
        values = np.asarray(values, dtype=float)
        right.plot(anchors, values / values[0], style, ms=5, mfc="none", label=str(label))
    right.set_xlabel(r"gauge: where the third puncture is fixed")
    right.set_ylabel("amplitude, relative to the first")
    right.set_title("on shell it does not matter")
    right.legend(frameon=False, fontsize=9)

    fig.suptitle(title, y=1.0)
    return _save(fig, path)


def plot_five_point_moduli(
    grid, extent, note: str = "", path=None, title: str = "Five punctures, two moduli"
):
    r"""The five-point integrand over the moduli space of the punctured disc.

    ``grid`` is the integrand on a square of ``(x, y)`` with ``extent`` the
    ``(x0, x1, y0, y1)`` it covers; only the triangle ``0 < x < y < 1`` is the
    integration region and the rest is masked.

    The three corners are where punctures collide -- ``x -> 0``, ``y -> 1``,
    ``x -> y`` -- and each is a channel of the amplitude.  There is no closed
    form to check this against; what says the construction is right is that the
    integral over it does not depend on where the other three punctures were
    fixed, and ``note`` carries that number.
    """
    from matplotlib.colors import LogNorm

    grid = np.asarray(grid, dtype=float)
    fig, ax = _fig(figsize=(6.4, 5.0))
    finite = grid[np.isfinite(grid) & (grid > 0)]
    image = ax.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        extent=tuple(float(v) for v in extent),
        aspect="equal",
        cmap="viridis",
        norm=LogNorm(vmin=max(finite.min(), finite.max() * 1e-4), vmax=finite.max()),
    )
    ax.plot([0, 1], [0, 1], color="white", lw=1.2, ls="--")
    ax.set_xlabel("$y_2$")
    ax.set_ylabel("$y_3$")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="integrand")
    if note:
        ax.text(
            0.03, 0.97, note, transform=ax.transAxes, va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.75", "lw": 0.8},
        )
    return _save(fig, path)


def plot_boundary_state(series, eta_curve, path, title: str = "A brane as a closed string"):
    r"""The coherent state's norm, and the function it turns out to be.

    ``series`` is a sequence of ``(label, cutoffs, relative errors)`` -- how far
    the truncated sum over Fock states is from the product, against where it was
    cut.  ``eta_curve`` is ``(moduli, from the boundary state, from eta)``.

    Left: the sum converges late, and later the larger ``q`` is.  That is the
    Hagedorn growth of the degeneracies seen from the inconvenient side: the
    terms rise before they fall.

    Right: with the ground-state energy put back, the norm is
    :math:`|\eta|^{-24}` -- the factor the closed channel of the cylinder
    carries.  The markers come from summing over states, the line from the eta
    product.
    """
    fig, (left, right) = _fig(1, 2, figsize=(10.6, 4.3))

    for label, cutoffs, errors in series:
        cutoffs = np.asarray(cutoffs, dtype=float)
        errors = np.maximum(np.asarray(errors, dtype=float), 1e-17)
        left.semilogy(cutoffs, errors, "o-", ms=4, mfc="none", label=str(label))
    left.axhline(1e-13, color="0.4", lw=1.0, ls="--")
    left.annotate(
        "machine precision", xy=(0.02, 1e-13), xycoords=("axes fraction", "data"),
        xytext=(0, 5), textcoords="offset points", fontsize=8, color="0.35",
    )
    left.set_xlabel("levels summed")
    left.set_ylabel("relative error against the product")
    left.set_title("the sum converges late")
    left.legend(frameon=False, fontsize=9)

    moduli, from_state, from_eta = (np.asarray(a, dtype=float) for a in eta_curve)
    right.semilogy(moduli, from_eta, "-", lw=1.6, color="#2ca02c",
                   label=r"$|\eta(i t)|^{-24}$")
    right.semilogy(moduli, from_state, "o", ms=6, mfc="none", color="#1f77b4",
                   label=r"$\langle B | q^{(N+\tilde N)/2} | B\rangle$")
    right.set_xlabel("modulus $t$")
    right.set_ylabel("oscillator factor")
    right.set_title("the norm is the eta function")
    right.legend(frameon=False, fontsize=9)

    fig.suptitle(title, y=1.0)
    return _save(fig, path)


def plot_black_hole_entropy(curves, scans, path, title: str = "An entropy, counted and measured"):
    r"""The microscopic count against Cardy, and the horizon against its moduli.

    ``curves`` is a sequence of ``(label, sqrt(N), log d_N, cardy line)``;
    ``scans`` a sequence of ``(label, couplings, entropies)``.

    Left: the exact logarithms climb towards the Cardy line and stay below it at
    every level that can be reached.  That is not a failure -- it is the
    subleading ``log N``, which is why the exponent has to be fitted rather than
    read off a ratio.

    Right: the Bekenstein-Hawking entropy against the string coupling.  Flat, for
    every charge, because an entropy counts states and cannot depend on a
    continuous modulus.  The dashed line is the same calculation with one power
    changed, and it is not flat.
    """
    fig, (left, right) = _fig(1, 2, figsize=(10.8, 4.4))

    colours = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    for (label, roots, counted, cardy), colour in zip(curves, colours, strict=False):
        roots = np.asarray(roots, dtype=float)
        left.plot(roots, np.asarray(cardy, dtype=float), "--", lw=1.2, color=colour)
        left.plot(roots, np.asarray(counted, dtype=float), "-", lw=1.8, color=colour,
                  label=str(label))
    left.set_xlabel(r"$\sqrt{N}$")
    left.set_ylabel(r"$\log d_N$")
    left.set_title("solid: the count.  dashed: $2\\pi\\sqrt{Q_1Q_5N}$")
    left.legend(frameon=False, fontsize=9, loc="upper left")

    styles = ["o-", "s-", "^--", "v--"]
    for (label, couplings, values), style, colour in zip(scans, styles, colours, strict=False):
        couplings = np.asarray(couplings, dtype=float)
        values = np.asarray(values, dtype=float)
        right.plot(couplings, values / values[0], style, ms=5, mfc="none", color=colour,
                   label=str(label))
    right.axhline(1.0, color="0.5", lw=0.9)
    right.set_xlabel("string coupling $g_s$")
    right.set_ylabel("entropy, relative to the first")
    right.set_title("the horizon forgets the moduli")
    right.legend(frameon=False, fontsize=9)

    fig.suptitle(title, y=1.0)
    return _save(fig, path)


def _draw_junction(ax, charges, tau, title, show_polygon=True):
    """Rays leaving a point, and the force vectors laid tip to tail."""
    vectors = [complex(p) + complex(q) * tau for p, q in charges]
    scale = max(abs(v) for v in vectors)
    colours = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    for (p, q), vector, colour in zip(charges, vectors, colours, strict=False):
        direction = vector / abs(vector)
        end = direction * 1.0
        ax.annotate(
            "",
            xy=(end.real, end.imag),
            xytext=(0.0, 0.0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": colour,
                "lw": 1.0 + 3.0 * abs(vector) / scale,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=3,
        )
        label = direction * 1.16
        ax.text(label.real, label.imag, f"$({p},{q})$", ha="center", va="center",
                fontsize=9, color=colour)

    if show_polygon:
        walk = [0j]
        for vector in vectors:
            walk.append(walk[-1] + vector / scale * 0.55)
        path = np.array([[z.real, z.imag] for z in walk]) + np.array([0.0, -1.75])
        ax.plot(path[:, 0], path[:, 1], "-o", ms=3, lw=1.2, color="0.45", zorder=2)
        ax.text(path[0, 0], path[0, 1] - 0.28, "force vectors, tip to tail",
                ha="center", fontsize=8, color="0.4")

    ax.plot([0], [0], "o", ms=6, color="black", zorder=4)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2.6, 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10)


def plot_pq_strings(curves, junctions, path, title: str = "The $SL(2,Z)$ multiplet"):
    r"""The tension lattice against the coupling, and a junction it holds together.

    ``curves`` is a sequence of ``(label, couplings, tensions)``; ``junctions``
    a sequence of ``(label, charges, tau)``.

    Left: :math:`T_{p,q} = |p + q\tau| / 2\pi\alpha'`.  The fundamental string
    is flat, the D1 falls like :math:`1/g_s`, and they cross at :math:`g_s = 1`
    -- which is the fixed point of ``S`` and the reason the two are the same
    kind of object.

    Right: a junction.  Each string leaves along the phase of :math:`p + q\tau`
    with a thickness set by its tension, and the same vectors laid tip to tail
    close into a polygon.  They close because the charges add to zero; the
    angles were never chosen.
    """
    panels = list(junctions)
    fig, axes = _fig(1, 1 + len(panels), figsize=(4.6 + 3.4 * len(panels), 4.4))
    axes = np.atleast_1d(axes)
    spectrum = axes[0]

    for label, couplings, values in curves:
        spectrum.loglog(
            np.asarray(couplings, dtype=float), np.asarray(values, dtype=float),
            lw=1.7, label=str(label),
        )
    spectrum.axvline(1.0, color="0.5", lw=1.0, ls=":")
    spectrum.annotate(
        "$g_s = 1$", xy=(1.0, spectrum.get_ylim()[0]), xytext=(4, 6),
        textcoords="offset points", fontsize=9, color="0.4",
    )
    spectrum.set_xlabel("string coupling $g_s$")
    spectrum.set_ylabel(r"$T_{p,q}$")
    spectrum.set_title("one lattice of strings", fontsize=11)
    spectrum.legend(frameon=False, fontsize=9)

    for ax, (label, charges, tau) in zip(axes[1:], panels, strict=True):
        _draw_junction(ax, charges, tau, label)

    fig.suptitle(title, y=1.0)
    return _save(fig, path)


def plot_mirror_hodge(points, highlights, path, title: str = "Mirror symmetry, as a reflection"):
    r"""The Hodge plot: every family and its mirror, reflected in the axis.

    ``points`` is a sequence of ``(h11, h21)``; ``highlights`` a sequence of
    ``(label, h11, h21)`` to name.  Each entry is drawn twice, once as itself
    and once with the two numbers swapped, because the dual polytope is a
    Calabi-Yau too.

    The horizontal axis is :math:`\chi = 2(h^{1,1} - h^{2,1})` and the vertical
    is :math:`h^{1,1} + h^{2,1}`.  The picture is symmetric about
    :math:`\chi = 0` -- not because it was made so, but because the two Hodge
    numbers are the same function of a polytope and of its dual.
    """
    points = [(int(a), int(b)) for a, b in points]
    fig, ax = _fig(figsize=(7.6, 5.2))

    forward = np.array([[2 * (a - b), a + b] for a, b in points], dtype=float)
    mirrored = np.array([[2 * (b - a), a + b] for a, b in points], dtype=float)
    ax.scatter(forward[:, 0], forward[:, 1], s=26, color="#1f77b4", alpha=0.8,
               label="from $\\Delta^*$")
    ax.scatter(mirrored[:, 0], mirrored[:, 1], s=26, facecolors="none",
               edgecolors="#d62728", linewidths=1.1, label="from $\\Delta$")
    ax.axvline(0.0, color="0.5", lw=1.0, ls=":")

    for label, a, b in highlights:
        for x, y, colour in (
            (2 * (a - b), a + b, "#1f77b4"),
            (2 * (b - a), a + b, "#d62728"),
        ):
            ax.annotate(
                str(label), xy=(x, y), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=8, color=colour,
            )
    ax.set_xlabel(r"$\chi = 2(h^{1,1} - h^{2,1})$")
    ax.set_ylabel(r"$h^{1,1} + h^{2,1}$")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def plot_reflexive_duality(panels, path, title: str = "A polytope and its dual"):
    r"""Two-dimensional reflexive polygons drawn beside their duals.

    ``panels`` is a sequence of ``(label, vertices, lattice points, dual
    vertices, dual lattice points)``, all integer arrays.

    Reflexive means the dual is a lattice polygon too, and that is the whole
    content of the picture: both sides have their vertices on lattice points and
    exactly one lattice point -- the origin -- in the interior.  In four
    dimensions the same condition is what makes a Calabi-Yau, and the pairing of
    the two polytopes is the mirror.
    """
    panels = list(panels)
    fig, axes = _fig(2, len(panels), figsize=(3.5 * len(panels), 6.8))
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:  # pragma: no cover - single row would be a bug
        axes = axes.T

    for column, (label, verts, pts, dual_verts, dual_pts) in enumerate(panels):
        for row, (vertices, points, colour, name) in enumerate(
            ((verts, pts, "#1f77b4", label), (dual_verts, dual_pts, "#d62728", "its dual"))
        ):
            ax = axes[row, column]
            vertices = np.asarray(vertices, dtype=float)
            points = np.asarray(points, dtype=float)
            order = np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))
            loop = np.vstack([vertices[order], vertices[order][:1]])
            ax.fill(loop[:, 0], loop[:, 1], color=colour, alpha=0.12, zorder=1)
            ax.plot(loop[:, 0], loop[:, 1], "-", lw=1.6, color=colour, zorder=2)
            reach = max(3.0, float(np.abs(vertices).max()) + 1.0)
            grid = np.arange(-int(reach), int(reach) + 1)
            mesh = np.array([[x, y] for x in grid for y in grid], dtype=float)
            ax.plot(mesh[:, 0], mesh[:, 1], ".", ms=1.6, color="0.8", zorder=0)
            ax.plot(points[:, 0], points[:, 1], "o", ms=4, color=colour, zorder=3)
            ax.plot([0], [0], "o", ms=7, mfc="none", mec="black", mew=1.3, zorder=4)
            ax.set_xlim(-reach - 0.4, reach + 0.4)
            ax.set_ylim(-reach - 0.4, reach + 0.4)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{name}   ({len(points)} points)", fontsize=10)

    fig.suptitle(title, y=1.0)
    return _save(fig, path)
