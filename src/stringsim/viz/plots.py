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
]

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
