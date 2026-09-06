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
