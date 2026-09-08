"""Animations: watching the string actually move.

Two entry points, matching the two ways the package produces a solution.

:func:`animate_modes`
    Takes an analytic :class:`~stringsim.classical.modes.OpenString` or
    :class:`~stringsim.classical.modes.ClosedString` and samples it on a
    ``(tau, sigma)`` grid.  Because the light-cone construction fixes the
    unphysical components from the transverse ones, what you see is a genuine
    solution of the constraints, not a cartoon of one.

:func:`animate_evolution`
    Takes the output of :func:`stringsim.classical.evolve.evolve`, i.e. a
    string that was plucked and released rather than assigned its Fourier
    coefficients.

Both write a GIF through matplotlib's Pillow writer, which needs no external
binary.  Pass ``writer="ffmpeg"`` and an ``.mp4`` path if ffmpeg is available
and a smaller file is wanted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "animate_modes",
    "animate_evolution",
    "snapshot_grid",
    "animate_fermion_reflection",
    "animate_twisted_string",
    "animate_bion_spike",
    "animate_modular_reduction",
    "animate_fuzzy_sphere",
    "animate_matrix_eigenvalues",
    "animate_no_ghost_region",
    "animate_physical_norms",
    "animate_anomaly_sweep",
    "animate_worldsheet_parity",
    "animate_narain_levels",
    "animate_pole_emergence",
    "animate_boundary_state",
    "animate_cardy_fit",
    "animate_pq_junction",
]


def _limits(frames: np.ndarray, pad: float = 0.15):
    lo = frames.reshape(-1, frames.shape[-1]).min(axis=0)
    hi = frames.reshape(-1, frames.shape[-1]).max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    return lo - pad * span, hi + pad * span


def _sample_modes(string, projection, n_frames: int, n_sigma: int, tau_max: float):
    tau = np.linspace(0.0, tau_max, n_frames)
    sigma = np.linspace(0.0, string.sigma_max, n_sigma)
    X = string.position(tau[:, None], sigma[None, :])
    return X[:, :, list(projection)]


def animate_modes(
    string,
    path,
    projection: tuple[int, ...] = (1, 2, 3),
    n_frames: int = 120,
    n_sigma: int = 200,
    tau_max: float = 2.0 * np.pi,
    fps: int = 24,
    title: str | None = None,
    trail: bool = True,
) -> Path:
    """Animate an analytic mode expansion.

    Parameters
    ----------
    string:
        An ``OpenString`` or ``ClosedString``.
    path:
        Output file; ``.gif`` is written with Pillow.
    projection:
        Which spacetime components to draw.  Three indices give a 3-D plot, two
        give a plane.  Index 0 is time, so the default ``(1, 2, 3)`` shows three
        spatial directions.
    trail:
        Draw the path swept by the endpoints (open string) or by one marked
        point (closed string).  For the rotating solution this is the circle
        traced at the speed of light.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if len(projection) not in (2, 3):
        raise ValueError("projection must name two or three components")
    frames = _sample_modes(string, projection, n_frames, n_sigma, tau_max)
    lo, hi = _limits(frames)

    fig = plt.figure(figsize=(5.6, 4.8), dpi=100)
    if len(projection) == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlim(lo[2], hi[2])
        ax.set_zlabel(f"$X^{projection[2]}$")
        (line,) = ax.plot([], [], [], lw=2.5, color="tab:blue")
        (ends,) = ax.plot([], [], [], "o", ms=6, color="tab:red")
        (path_a,) = ax.plot([], [], [], lw=0.8, color="0.6")
    else:
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        (line,) = ax.plot([], [], lw=2.5, color="tab:blue")
        (ends,) = ax.plot([], [], "o", ms=6, color="tab:red")
        (path_a,) = ax.plot([], [], lw=0.8, color="0.6")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel(f"$X^{projection[0]}$")
    ax.set_ylabel(f"$X^{projection[1]}$")
    ax.set_title(title or "String worldsheet")

    endpoint_track: list[np.ndarray] = []

    def update(k: int):
        pts = frames[k]
        endpoint_track.append(pts[0])
        track = np.array(endpoint_track) if trail else np.empty((0, len(projection)))
        if len(projection) == 3:
            line.set_data(pts[:, 0], pts[:, 1])
            line.set_3d_properties(pts[:, 2])
            ends.set_data(pts[[0, -1], 0], pts[[0, -1], 1])
            ends.set_3d_properties(pts[[0, -1], 2])
            if len(track):
                path_a.set_data(track[:, 0], track[:, 1])
                path_a.set_3d_properties(track[:, 2])
        else:
            line.set_data(pts[:, 0], pts[:, 1])
            ends.set_data(pts[[0, -1], 0], pts[[0, -1], 1])
            if len(track):
                path_a.set_data(track[:, 0], track[:, 1])
        return line, ends, path_a

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def animate_evolution(
    evolution,
    path,
    stride: int = 2,
    fps: int = 24,
    title: str | None = None,
) -> Path:
    """Animate the output of :func:`stringsim.classical.evolve.evolve`.

    The initial data had two or three components; those are the axes drawn.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    X = evolution.X[::stride]
    n_target = X.shape[-1]
    if n_target not in (2, 3):
        raise ValueError("evolution must carry two or three target-space components")
    lo, hi = _limits(X)

    fig = plt.figure(figsize=(5.6, 4.6), dpi=100)
    if n_target == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlim(lo[2], hi[2])
        (line,) = ax.plot([], [], [], lw=2.5, color="tab:blue")
    else:
        ax = fig.add_subplot(111)
        (line,) = ax.plot([], [], lw=2.5, color="tab:blue")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_title(title or f"plucked string, {evolution.boundary} ends")

    def update(k: int):
        pts = X[k]
        line.set_data(pts[:, 0], pts[:, 1])
        if n_target == 3:
            line.set_3d_properties(pts[:, 2])
        return (line,)

    anim = FuncAnimation(fig, update, frames=len(X), blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def snapshot_grid(
    string,
    path,
    projection: tuple[int, ...] = (1, 2),
    n_snapshots: int = 6,
    n_sigma: int = 200,
    tau_max: float = 2.0 * np.pi,
    title: str | None = None,
) -> Path:
    """A row of still frames -- the animation's content in one static figure."""
    import matplotlib.pyplot as plt

    if len(projection) != 2:
        raise ValueError("snapshot_grid draws a plane; give two components")
    frames = _sample_modes(string, projection, n_snapshots, n_sigma, tau_max)
    lo, hi = _limits(frames)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(2.1 * n_snapshots, 2.4), dpi=130)
    taus = np.linspace(0.0, tau_max, n_snapshots)
    for ax, pts, tau in zip(np.atleast_1d(axes), frames, taus, strict=True):
        ax.plot(pts[:, 0], pts[:, 1], lw=2.0, color="tab:blue")
        ax.plot(pts[[0, -1], 0], pts[[0, -1], 1], "o", ms=4, color="tab:red")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$\tau = {tau:.2f}$", fontsize=9)
    fig.suptitle(title or "String at successive worldsheet times", fontsize=11)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_animation(anim, fig, path, fps: int):
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def animate_fermion_reflection(panels, path, stride: int = 3, fps: int = 20) -> Path:
    r"""Watch a fermion pulse bounce, and watch the sign it comes back with.

    ``panels`` is a sequence of ``(label, evolution)`` of open-string
    :class:`stringsim.superstring.worldsheet.FermionEvolution` objects.  Each
    panel draws both components against ``sigma`` on ``[0, pi]``: ``psi_-``
    running to the right and ``psi_+`` to the left.

    The pulse leaves through ``sigma = pi`` into ``psi_+`` picking up ``eta``,
    returns, and re-enters ``psi_-`` unchanged at ``sigma = 0``.  In
    Neveu-Schwarz it therefore comes back **upside down** and needs a second
    round trip to be itself again; in Ramond it never flips.  That is the whole
    reason NS modes are half-integral, running in front of you rather than
    written down as a boundary condition.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    entries = list(panels)
    if not entries:
        raise ValueError("give at least one (label, evolution) pair")
    frames = min(len(evolution.tau) for _, evolution in entries) // stride

    fig, axes = plt.subplots(
        len(entries), 1, figsize=(6.4, 2.5 * len(entries)), dpi=110, sharex=True
    )
    axes = np.atleast_1d(axes)
    scale = 1.15 * max(
        float(np.max(np.abs(evolution.psi_minus[..., 0]))) for _, evolution in entries
    )
    lines = []
    for ax, (label, evolution) in zip(axes, entries, strict=True):
        (minus,) = ax.plot([], [], lw=2.2, color="tab:blue", label=r"$\psi_-$ (moves right)")
        (plus,) = ax.plot([], [], lw=2.2, color="tab:orange", label=r"$\psi_+$ (moves left)")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xlim(0.0, float(evolution.sigma[-1]))
        ax.set_ylim(-scale, scale)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
        lines.append((minus, plus, evolution))
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel(r"$\sigma$")
    clock = fig.suptitle("")

    def update(index: int):
        step = index * stride
        artists = []
        for minus, plus, evolution in lines:
            minus.set_data(evolution.sigma, evolution.psi_minus[step, :, 0])
            plus.set_data(evolution.sigma, evolution.psi_plus[step, :, 0])
            artists += [minus, plus]
        tau = lines[0][2].tau[step]
        clock.set_text(rf"$\tau = {tau:.2f}$   (one round trip every $2\pi$)")
        return artists

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    return _save_animation(anim, fig, path, fps)


def animate_twisted_string(
    phase: float,
    path,
    amplitudes=(0.55, 0.25),
    fixed_point=(0.0, 0.0),
    n_frames: int = 96,
    n_sigma: int = 400,
    fps: int = 20,
    title: str | None = None,
) -> Path:
    r"""A closed string that closes only up to a rotation.

    In a ``theta^k``-twisted sector the string obeys
    :math:`Z(\sigma + 2\pi) = e^{2\pi i \phi} Z(\sigma)` about a fixed point, so
    it is built from modes with *fractional* numbers: right-movers
    :math:`r \in \mathbb{Z} + \phi` and left-movers :math:`s \in \mathbb{Z} -
    \phi`.  Drawn in the plane it is an open curve whose two ends are related by
    the rotation -- which is what a twisted string is, and why it cannot leave
    the fixed point.

    ``phase`` is :math:`\phi = k/N`; ``amplitudes`` are the coefficients of the
    lowest right- and left-moving modes.  The dashed line joins the two ends
    through the fixed point, so the angle between them is the twist.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if not 0.0 < phase < 1.0:
        raise ValueError(f"phase must lie strictly in (0, 1), got {phase}")
    right, left = (float(value) for value in amplitudes)
    centre = np.asarray(fixed_point, dtype=float).reshape(2)

    sigma = np.linspace(0.0, 2.0 * np.pi, n_sigma)
    tau = np.linspace(0.0, 2.0 * np.pi / phase, n_frames, endpoint=False)

    def profile(time: float) -> np.ndarray:
        z = right * np.exp(-1j * phase * (time - sigma)) + left * np.exp(
            -1j * (1.0 - phase) * (time + sigma)
        )
        return np.stack([centre[0] + z.real, centre[1] + z.imag], axis=-1)

    frames = np.array([profile(t) for t in tau])
    span = 1.25 * float(np.max(np.abs(frames - centre)))

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    (curve,) = ax.plot([], [], lw=2.4, color="tab:blue")
    (ends,) = ax.plot([], [], "o", ms=7, color="tab:red")
    (chord,) = ax.plot([], [], "--", lw=1.0, color="0.55")
    ax.plot(centre[0], centre[1], "x", ms=10, mew=2.0, color="k")
    ax.text(centre[0], centre[1] - 0.14 * span, "fixed point", ha="center", fontsize=9)
    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[1] - span, centre[1] + span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        title or rf"Twisted string: the ends differ by a rotation of $2\pi\cdot{phase:.3g}$",
        fontsize=10,
    )

    def update(index: int):
        points = frames[index]
        curve.set_data(points[:, 0], points[:, 1])
        ends.set_data(points[[0, -1], 0], points[[0, -1], 1])
        chord.set_data(
            [points[0, 0], centre[0], points[-1, 0]], [points[0, 1], centre[1], points[-1, 1]]
        )
        return curve, ends, chord

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    return _save_animation(anim, fig, path, fps)


def animate_bion_spike(
    profiles,
    path,
    n_grid: int = 90,
    extent: float = 1.0,
    hold: int = 8,
    fps: int = 12,
    height: float | None = None,
    title: str | None = None,
) -> Path:
    r"""A D-brane stretching into a spike as fundamental strings are attached.

    ``profiles`` is a sequence of ``(label, radial_function)``: each is called
    with an array of radii and returns the transverse position ``X(r)``.  What
    is drawn is a two-dimensional slice of the brane through the string's
    endpoint, so the spike *is* the string -- the same object seen from the
    brane's side rather than from the string's.

    Every frame is a genuine BPS solution rather than an interpolation: the
    flux is quantised, so the family is discrete, and each held frame solves the
    equations exactly.  All are clipped at the same ``height``, because all of
    them are infinitely tall; what grows with the number of strings is the width
    of the funnel.  The camera turns while it grows.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    entries = list(profiles)
    if not entries:
        raise ValueError("give at least one (label, profile) pair")

    axis = np.linspace(-extent, extent, n_grid)
    grid_x, grid_y = np.meshgrid(axis, axis)
    radius = np.hypot(grid_x, grid_y)
    inner = extent / n_grid
    surfaces = [np.asarray(fn(np.maximum(radius, inner)), dtype=float) for _, fn in entries]
    ceiling = (
        float(height)
        if height is not None
        else min(float(np.asarray(fn(np.array([extent / 6.0])))[0]) for _, fn in entries)
    )

    fig = plt.figure(figsize=(5.6, 5.0), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    frames = len(entries) * hold

    def update(index: int):
        which = index // hold
        ax.clear()
        ax.plot_surface(
            grid_x,
            grid_y,
            np.minimum(surfaces[which], ceiling),
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1,
        )
        ax.set_zlim(0.0, ceiling)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zlabel("transverse $X$")
        ax.view_init(elev=26.0, azim=-60.0 + 360.0 * index / max(frames - 1, 1))
        ax.set_title(f"{title or 'BIon spike'}\n{entries[which][0]}", fontsize=10)
        return ()

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    return _save_animation(anim, fig, path, fps)


def animate_modular_reduction(
    tau,
    path,
    domain_boundary=None,
    hold: int = 6,
    fps: int = 6,
    title: str | None = None,
) -> Path:
    r"""Watch a point walk into the fundamental domain.

    ``tau`` is the starting point in the upper half plane and
    ``domain_boundary`` a complex array outlining the fundamental domain (drawn
    shaded, if given).  Each frame applies one generator -- an integer shift
    ``T`` or the inversion ``S`` -- and the trail shows where the point has
    been.

    Start deep in what a field theory would call the ultraviolet, at
    ``tau_2 = 0.011``, and the walk carries it up to ``tau_2 ~ 1``.  That is the
    whole reason a string one-loop amplitude has no ultraviolet divergence:
    small ``tau_2`` is not a region of the moduli space that has been cut off,
    it is a copy of a region already counted.
    """
    import math

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    point = complex(tau)
    if point.imag <= 0:
        raise ValueError("tau must lie in the upper half plane")

    trail = [point]
    labels = ["start"]
    for _ in range(200):
        shift = math.floor(trail[-1].real + 0.5)
        if shift:
            trail.append(trail[-1] - shift)
            labels.append(rf"$T^{{{-shift}}}$")
        if abs(trail[-1]) < 1.0 - 1e-9:
            trail.append(-1.0 / trail[-1])
            labels.append("$S$")
        else:
            break
    else:  # pragma: no cover - the walk always terminates
        raise RuntimeError("no representative found")

    span = max(2.0, 1.3 * max(abs(z.real) for z in trail))
    top = max(2.2, 1.25 * max(z.imag for z in trail))

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=110)
    if domain_boundary is not None:
        outline = np.asarray(domain_boundary, dtype=complex)
        ax.fill(outline.real, outline.imag, facecolor="tab:blue", alpha=0.25, edgecolor="0.4")
    ax.axhline(math.sqrt(3.0) / 2.0, color="tab:red", ls=":", lw=1.0)
    (path_line,) = ax.plot([], [], "-o", ms=4, lw=1.0, color="0.55")
    (current,) = ax.plot([], [], "o", ms=10, color="tab:red")
    caption = ax.set_title("")
    ax.set_xlim(-span, span)
    ax.set_ylim(0.0, top)
    ax.set_xlabel(r"$\tau_1$")
    ax.set_ylabel(r"$\tau_2$")
    ax.grid(alpha=0.25)

    frames = len(trail) * hold

    def update(index: int):
        which = min(index // hold, len(trail) - 1)
        visible = trail[: which + 1]
        path_line.set_data([z.real for z in visible], [z.imag for z in visible])
        current.set_data([visible[-1].real], [visible[-1].imag])
        caption.set_text(
            f"{title or 'walking into the fundamental domain'}\n"
            rf"{labels[which]}:  $\tau_2 = {visible[-1].imag:.4f}$"
        )
        return path_line, current

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    return _save_animation(anim, fig, path, fps)


def _sphere_scale(heights, radii) -> float:
    """The radius to normalise a fuzzy sphere by, so panels are comparable."""
    reach = max(float(np.max(np.abs(heights))), float(np.max(radii)))
    return reach if reach > 0.0 else 1.0


def animate_fuzzy_sphere(frames, path, hold: int = 7, fps: int = 10, title=None) -> Path:
    r"""Watch a sphere assemble itself out of D0-branes as ``N`` grows.

    ``frames`` is a sequence of ``(label, heights, radii)`` from
    :func:`stringsim.branes.myers.latitudes`, one per value of ``N``.  Each is a
    genuine solution of the matrix equations of motion, not an interpolation:
    the flux quantum is an integer, so the family is discrete.

    Two things happen at once, and they are the same thing.  The circles get
    more numerous, so the surface fills in; and the radius grows like ``N``
    while the commutators grow like ``N``, so their *relative* size falls like
    ``1/N`` and the noncommutative sphere turns into an ordinary one.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    entries = list(frames)
    if not entries:
        raise ValueError("give at least one (label, heights, radii) triple")
    entries = [
        (label, np.asarray(z) / _sphere_scale(z, r), np.asarray(r) / _sphere_scale(z, r))
        for label, z, r in entries
    ]
    reach = 1.05
    angle = np.linspace(0.0, 2.0 * np.pi, 200)

    fig = plt.figure(figsize=(5.2, 5.0), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    total = len(entries) * hold

    def update(index: int):
        label, heights, radii = entries[min(index // hold, len(entries) - 1)]
        ax.clear()
        for height, radius in zip(np.asarray(heights), np.asarray(radii), strict=True):
            ax.plot(
                radius * np.cos(angle),
                radius * np.sin(angle),
                np.full_like(angle, height),
                lw=1.5,
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
        ax.view_init(elev=18.0, azim=-62.0 + 300.0 * index / max(total - 1, 1))
        ax.set_title(f"{title or 'the fuzzy sphere fills in'}\n{label}", fontsize=10)
        return ()

    anim = FuncAnimation(fig, update, frames=total, blit=False)
    return _save_animation(anim, fig, path, fps)


def animate_matrix_eigenvalues(
    times,
    eigenvalues,
    kinetic,
    potential,
    path,
    stride: int = 2,
    fps: int = 24,
    title: str | None = None,
) -> Path:
    r"""D0-branes scattering, and the string energy that appears when they meet.

    ``eigenvalues`` has shape ``(n_frames, n_branes)`` -- the positions along one
    direction -- and ``kinetic`` and ``potential`` are the two halves of the
    energy at the same times.

    Watch the two panels together.  While the branes are apart the potential is
    flat on the floor: commuting matrices, free motion, nothing stretched
    between them.  When they come close the off-diagonal entries become light,
    the potential takes energy from the motion, and the branes leave in
    directions no one could have predicted from where they came in.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    times = np.asarray(times, dtype=float)[::stride]
    eigenvalues = np.asarray(eigenvalues, dtype=float)[::stride]
    kinetic = np.asarray(kinetic, dtype=float)[::stride]
    potential = np.asarray(potential, dtype=float)[::stride]
    if eigenvalues.ndim != 2:
        raise ValueError(f"expected (n_frames, n_branes), got {eigenvalues.shape}")

    reach = 1.15 * float(np.max(np.abs(eigenvalues)))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.4), dpi=110,
                                  gridspec_kw={"height_ratios": [1.0, 1.2]})

    (points,) = ax.plot([], [], "o", ms=13, color="tab:blue")
    trails = [
        ax.plot([], [], "-", lw=0.9, color="0.75", zorder=0)[0]
        for _ in range(eigenvalues.shape[1])
    ]
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("position along $X_1$ (eigenvalues)")
    ax.grid(alpha=0.2)

    ax2.plot(times, kinetic, lw=1.4, color="tab:orange", label="kinetic")
    ax2.plot(times, potential, lw=1.4, color="tab:blue", label="potential (stretched strings)")
    marker = ax2.axvline(times[0], color="0.35", lw=1.2)
    ax2.set_xlabel("$t$")
    ax2.set_ylabel("energy")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.25)
    caption = fig.suptitle("")
    fig.subplots_adjust(hspace=0.55, top=0.90, bottom=0.10)

    trail_length = max(2, len(times) // 12)

    def update(index: int):
        row = eigenvalues[index]
        points.set_data(row, np.zeros_like(row))
        start = max(0, index - trail_length)
        for which, line in enumerate(trails):
            piece = eigenvalues[start : index + 1, which]
            line.set_data(piece, np.full(len(piece), -0.45))
        marker.set_xdata([times[index], times[index]])
        caption.set_text(
            f"{title or 'D0-branes, scattering'}   "
            f"$t = {times[index]:.2f}$,  $V = {potential[index]:.3f}$"
        )
        return (points, marker, *trails)

    anim = FuncAnimation(fig, update, frames=len(times), blit=False)
    return _save_animation(anim, fig, path, fps)


def animate_no_ghost_region(dims, intercepts, grid, path, fps: int = 8) -> Path:
    r"""Sweep the intercept and watch the ghost-free region close.

    ``grid`` is a precomputed :func:`~stringsim.quantum.virasoro.no_ghost_map`
    -- rows indexed by dimension, columns by intercept.  Recomputing it per
    frame would be the same work several dozen times over, so the animation
    only draws.

    Left, the smallest physical norm across dimensions at the current ``a``,
    with everything below zero shaded: that band is the set of dimensions the
    theory has already lost.  Right, the ``(D, a)`` map filling in column by
    column.

    At ``a`` well below 1 no dimension in the window is excluded.  As ``a``
    rises the band eats down from large ``D``, reaching 26 exactly as ``a``
    reaches 1 -- and one step past ``a = 1`` the band covers every dimension,
    which is why the intercept cannot be treated as a free parameter.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import TwoSlopeNorm

    dims = np.asarray(list(dims), dtype=float)
    intercepts = np.asarray(intercepts, dtype=float)
    grid = np.asarray(grid, dtype=float)
    limit = max(float(np.max(np.abs(grid))), 1e-12)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    extent = (
        float(intercepts[0]),
        float(intercepts[-1]),
        float(dims[0]) - 0.5,
        float(dims[-1]) + 0.5,
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.4), dpi=130)
    fig.subplots_adjust(wspace=0.28)

    (curve,) = left.plot([], [], "o-", ms=4, color="#1f77b4")
    shade = [None]
    left.axhline(0.0, color="0.3", lw=1.0)
    left.axvline(26.0, color="0.55", lw=1.0, ls=":")
    left.set_xlim(dims[0] - 0.5, dims[-1] + 0.5)
    left.set_ylim(-limit * 1.1, limit * 1.1)
    left.set_xlabel("spacetime dimension $D$")
    left.set_ylabel("smallest physical norm")
    left.grid(alpha=0.25)

    masked = np.full_like(grid, np.nan)
    image = right.imshow(
        masked, origin="lower", aspect="auto", cmap="RdBu", norm=norm, extent=extent
    )
    marker = right.axvline(intercepts[0], color="k", lw=1.0)
    right.axhline(26.0, color="0.2", lw=1.0, ls=":")
    right.set_xlabel("intercept $a$")
    right.set_ylabel("spacetime dimension $D$")
    right.set_title("the $(D, a)$ plane")
    fig.colorbar(image, ax=right, label="smallest norm (normalised)")

    def frame(j: int):
        column = grid[:, j]
        curve.set_data(dims, column)
        if shade[0] is not None:
            shade[0].remove()
        shade[0] = left.fill_between(
            dims, column, 0.0, where=column < 0, color="#d62728", alpha=0.3
        )
        drawn = np.full_like(grid, np.nan)
        drawn[:, : j + 1] = grid[:, : j + 1]
        image.set_data(drawn)
        marker.set_xdata([intercepts[j], intercepts[j]])
        lost = int(np.sum(column < 0))
        left.set_title(f"$a = {intercepts[j]:+.2f}$ -- {lost} dimensions excluded")
        return curve, image, marker

    anim = FuncAnimation(fig, frame, frames=len(intercepts), blit=False)
    return _save_animation(anim, fig, path, fps)


def animate_physical_norms(spectra, trace, path, fps: int = 2) -> Path:
    r"""The physical Gram spectrum, dimension by dimension, and one point crossing.

    ``spectra`` is a sequence of ``(dim, eigenvalues)``.  ``trace`` is the value
    to follow underneath, one number per entry -- see the example script, which
    sets aside the generic gauge nulls and takes the smallest of what is left.
    It has to be supplied rather than derived here: the plain minimum sits on
    the null pile until ``D`` passes 26 and would show a flat line and then a
    cliff, hiding the very thing that moves.

    Top: every physical norm as a strip.  Most sit in tight clusters -- the
    null pile at the origin, the bulk further out -- and one lone point drifts
    left as ``D`` rises.  Bottom: that point, drawn in as the frames advance.

    Nothing else moves.  The critical dimension is the frame where the lone
    point reaches zero.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    spectra = [(int(d), np.asarray(v, dtype=float)) for d, v in spectra]
    dims = np.array([d for d, _ in spectra], dtype=float)
    trace = np.asarray(list(trace), dtype=float)
    if trace.size != dims.size:
        raise ValueError(f"trace has {trace.size} values for {dims.size} spectra")
    span = max(float(np.max(np.abs(v))) for _, v in spectra)
    reach = max(float(np.max(np.abs(trace))), 1e-3) * 1.3

    fig, (strip, below) = plt.subplots(
        2, 1, figsize=(7.8, 5.4), dpi=130, gridspec_kw={"height_ratios": [1.0, 1.4]}
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.10, hspace=0.55)

    scatter = strip.scatter([], [], s=14, color="#1f77b4", alpha=0.55)
    (marked,) = strip.plot([], [], "v", ms=11, color="#d62728", zorder=5)
    strip.axvspan(-0.4 * span, 0.0, color="#d62728", alpha=0.08)
    strip.axvline(0.0, color="0.3", lw=1.0)
    strip.set_xlim(-0.35 * span, 1.08 * span)
    strip.set_ylim(-1.0, 1.0)
    strip.set_yticks([])
    strip.set_xlabel("physical Gram eigenvalue")
    strip.grid(alpha=0.25, axis="x")

    below.axhline(0.0, color="0.3", lw=1.0)
    below.axvline(26.0, color="0.55", lw=1.0, ls=":")
    below.axhspan(-reach, 0.0, color="#d62728", alpha=0.08)
    (history,) = below.plot([], [], "o-", ms=4, color="#1f77b4")
    (head,) = below.plot([], [], "o", ms=9, color="#d62728")
    below.set_xlim(dims[0] - 0.5, dims[-1] + 0.5)
    below.set_ylim(-reach, reach)
    below.set_xlabel("spacetime dimension $D$")
    below.set_ylabel("the state that moves")
    below.grid(alpha=0.25)

    def frame(i: int):
        dim, values = spectra[i]
        jitter = np.linspace(-0.75, 0.75, values.size)
        scatter.set_offsets(np.column_stack([values, jitter]))
        marked.set_data([trace[i]], [0.0])
        history.set_data(dims[: i + 1], trace[: i + 1])
        head.set_data([dims[i]], [trace[i]])
        negative = int(np.sum(values < -1e-8 * span))
        word = "negative norm" if negative == 1 else "negative norms"
        fig.suptitle(
            f"$D = {dim}$ -- {negative} {word}",
            color="#d62728" if negative else "#2ca02c",
        )
        return scatter, marked, history, head

    anim = FuncAnimation(fig, frame, frames=len(spectra), blit=False)
    return _save_animation(anim, fig, path, fps)


def animate_anomaly_sweep(frames, path, fps: int = 3):
    r"""Sweep ``N`` through ``SO(N)`` and watch the fatal terms shrink to zero.

    ``frames`` is a sequence of ``(N, labels, values, fatal, verdict)``: the
    twelve-form's coefficients in a fixed monomial order, a boolean per bar
    saying whether that monomial is one no ``X_4 X_8`` product can reach, and
    a short verdict string.

    The two red bars are ``tr R^6`` and ``tr F^6``.  Nothing can absorb them,
    so both have to vanish on their own -- and they do so at different values
    of ``N``, for different reasons, meeting only at 32.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    frames = [
        (int(n), list(labels), [float(v) for v in values], list(fatal), str(verdict))
        for n, labels, values, fatal, verdict in frames
    ]
    labels = frames[0][1]
    reach = max(max(abs(v) for v in values) for _, _, values, _, _ in frames)

    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=130)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.30)
    positions = np.arange(len(labels))
    bars = ax.bar(positions, np.zeros(len(labels)), color="#1f77b4")
    ax.axhline(0.0, color="0.3", lw=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(-1.15 * reach, 1.15 * reach)
    ax.set_ylabel("twelve-form coefficient")
    ax.grid(alpha=0.25, axis="y")

    def draw(i: int):
        n, _, values, fatal, verdict = frames[i]
        for bar, value, bad in zip(bars, values, fatal, strict=True):
            bar.set_height(value)
            bar.set_color("#d62728" if bad and value != 0.0 else "#1f77b4")
        clean = all(v == 0.0 for v, bad in zip(values, fatal, strict=True) if bad)
        ax.set_title(
            f"$SO({n})$ -- {verdict}", color="#2ca02c" if clean else "#d62728"
        )
        return bars

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    return _save_animation(anim, fig, path, fps)


def animate_worldsheet_parity(panels, path, beads=(0.0, 0.25, 0.5, 0.75), fps: int = 20):
    r"""What gauging :math:`\Omega` does to a string that is actually moving.

    ``panels`` is a sequence of ``(label, frames)`` where ``frames`` has shape
    ``(n_tau, n_sigma, 2)`` -- a closed string sampled in two transverse
    directions over time.  ``beads`` are fractions of the way round the string;
    each is drawn as a coloured marker.

    **The shape alone cannot show the projection.**  :math:`X(\tau, -\sigma)`
    traces exactly the same curve as :math:`X(\tau, \sigma)`, backwards, so a
    string and its image are the same picture.  What differs is *where each
    point of the string sits on it*, which is what the beads are for: they run
    round one way in the first panel and the other way in the second.

    In the third the beads collide.  An :math:`\Omega`-even string satisfies
    :math:`X(\sigma) = X(-\sigma) = X(2\pi - \sigma)`, so it is **folded**:
    the two halves lie on top of each other and the curve is traced twice.  That
    is what "half the states survive" looks like on a moving string.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    panels = [(str(label), np.asarray(frames, dtype=float)) for label, frames in panels]
    lengths = {frames.shape[0] for _, frames in panels}
    if len(lengths) != 1:
        raise ValueError(f"panels disagree on the number of frames: {sorted(lengths)}")
    n_frames = lengths.pop()
    stacked = np.concatenate([frames.reshape(-1, 2) for _, frames in panels])
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    pad = 0.2 * np.maximum(hi - lo, 1e-9)
    colours = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 4.1), dpi=130)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.06, wspace=0.16)

    curves, markers = [], []
    for ax, (label, frames) in zip(axes, panels, strict=True):
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=11)
        (curve,) = ax.plot([], [], color="#1f77b4", lw=2.2, zorder=2)
        curves.append(curve)
        row = []
        for slot in range(len(beads)):
            (dot,) = ax.plot([], [], "o", ms=8, color=colours[slot % len(colours)], zorder=3)
            row.append(dot)
        markers.append(row)
        del frames

    def draw(i: int):
        artists = []
        for curve, row, (_, frames) in zip(curves, markers, panels, strict=True):
            shape = frames[i]
            closed = np.vstack([shape, shape[:1]])
            curve.set_data(closed[:, 0], closed[:, 1])
            artists.append(curve)
            for dot, fraction in zip(row, beads, strict=True):
                index = int(round(fraction * shape.shape[0])) % shape.shape[0]
                dot.set_data([shape[index, 0]], [shape[index, 1]])
                artists.append(dot)
        return artists

    anim = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    return _save_animation(anim, fig, path, fps)


def animate_narain_levels(frames, path, fps: int = 12):
    r"""Sweep a radius and watch the charge levels slide past the self-dual point.

    ``frames`` is a sequence of ``(radius, levels)`` where ``levels`` maps
    ``(l_L^2/2, l_R^2/2)`` to a multiplicity.

    Every charge moves along a line of constant ``h - h_bar = n.w``, because
    that combination is moduli-independent.  What changes is where along the
    line it sits.  At the self-dual radius four of them land exactly on the
    axes -- squared length 2 on one side, zero on the other -- and those are the
    gauge bosons of the enhanced symmetry.

    Sweeping past it, the picture reflects in the diagonal: ``R -> 1/R``
    exchanges the two axes, which is T-duality drawn rather than argued.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    frames = [(float(radius), dict(levels)) for radius, levels in frames]
    reach = 3.0

    fig, (plane, trace) = plt.subplots(
        1, 2, figsize=(9.4, 4.4), dpi=130, gridspec_kw={"width_ratios": [1.0, 1.0]}
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.13, wspace=0.3)

    plane.axhline(0.0, color="0.75", lw=1.0)
    plane.axvline(0.0, color="0.75", lw=1.0)
    plane.plot([0, reach], [0, reach], color="0.85", lw=1.0, ls="--")
    scatter = plane.scatter([], [], s=[], color="#1f77b4", alpha=0.75, zorder=3)
    highlight = plane.scatter([], [], s=110, facecolors="none", edgecolors="#d62728",
                              lw=1.8, zorder=4)
    plane.set_xlim(-0.3, reach + 0.3)
    plane.set_ylim(-0.3, reach + 0.3)
    plane.set_aspect("equal")
    plane.set_xlabel(r"$\ell_L^2 / 2$")
    plane.set_ylabel(r"$\ell_R^2 / 2$")
    plane.grid(alpha=0.25)

    radii = np.array([r for r, _ in frames])
    counts = np.array([
        sum(m for (a, b), m in levels.items()
            if (abs(a - 1) < 1e-9 and b < 1e-9) or (abs(b - 1) < 1e-9 and a < 1e-9))
        for _, levels in frames
    ], dtype=float)
    trace.plot(radii, counts, "-", lw=1.4, color="#2ca02c")
    (head,) = trace.plot([], [], "o", ms=9, color="#d62728")
    trace.axvline(1.0, color="0.55", lw=1.0, ls=":")
    trace.set_xscale("log")
    trace.set_xlabel(r"radius $R / \sqrt{\alpha'}$")
    trace.set_ylabel("massless gauge bosons")
    trace.set_ylim(-0.4, max(counts.max(), 1.0) + 0.6)
    trace.grid(alpha=0.25)
    trace.set_title("only at the self-dual radius", fontsize=11)

    def draw(i: int):
        radius, levels = frames[i]
        xs, ys, sizes, marked = [], [], [], []
        for (a, b), count in levels.items():
            if a > reach + 1e-9 or b > reach + 1e-9:
                continue
            xs.append(a)
            ys.append(b)
            sizes.append(20 + 30 * (count - 1))
            if (abs(a - 1) < 1e-9 and b < 1e-9) or (abs(b - 1) < 1e-9 and a < 1e-9):
                marked.append((a, b))
        scatter.set_offsets(np.column_stack([xs, ys]) if xs else np.zeros((0, 2)))
        scatter.set_sizes(sizes)
        highlight.set_offsets(np.array(marked) if marked else np.zeros((0, 2)))
        head.set_data([radius], [counts[i]])
        plane.set_title(f"$R = {radius:.3f}$ -- {len(marked)} roots", fontsize=11)
        return scatter, highlight, head

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    return _save_animation(anim, fig, path, fps)


def animate_pole_emergence(frames, path, fps: int = 10):
    r"""Watch a pole grow out of the end of an integral.

    ``frames`` is a sequence of ``(alpha_s, x, integrand, residue estimate)``.

    As :math:`\alpha(s)` climbs towards zero the exponent of the leftmost gap
    reaches :math:`-1`, and the integrand's tail at ``x = 0`` stops being
    integrable.  Nothing else on the worldsheet changes.  So the tachyon pole
    of the amplitude is not a feature of a Beta function -- it is the region
    where two vertex operators sit on top of each other, and the right-hand
    panel watches the residue it leaves behind settle on ``-1``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    frames = [
        (float(a), np.asarray(x, dtype=float), np.asarray(y, dtype=float), float(r))
        for a, x, y, r in frames
    ]
    ceiling = max(float(np.max(y)) for _, _, y, _ in frames)
    floor = min(float(np.min(y[y > 0])) for _, _, y, _ in frames)
    alphas = np.array([a for a, _, _, _ in frames])
    residues = np.array([r for _, _, _, r in frames])

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=130)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.14, wspace=0.3)

    (curve,) = left.plot([], [], lw=2.0, color="#1f77b4")
    left.set_yscale("log")
    left.set_xlim(0.0, 1.0)
    left.set_ylim(floor * 0.7, ceiling * 1.5)
    left.set_xlabel("$x$, the free puncture")
    left.set_ylabel("integrand")
    left.grid(alpha=0.25)

    right.plot(alphas, residues, "-", lw=1.4, color="#2ca02c")
    (head,) = right.plot([], [], "o", ms=9, color="#d62728")
    right.axhline(-1.0, color="0.4", lw=1.0, ls="--")
    right.annotate(
        "residue $-1$", xy=(alphas[0], -1.0), xytext=(4, 6),
        textcoords="offset points", fontsize=9, color="0.35",
    )
    right.set_xlabel(r"$\alpha(s)$, approaching the pole")
    right.set_ylabel(r"$\alpha(s)\, A(s,t)$")
    right.set_ylim(min(residues.min(), -1.05) - 0.05, max(residues.max(), -0.95) + 0.05)
    right.grid(alpha=0.25)

    def draw(i: int):
        alpha_s, xs, values, residue = frames[i]
        curve.set_data(xs, values)
        head.set_data([alpha_s], [residue])
        left.set_title(rf"$\alpha(s) = {alpha_s:+.4f}$", fontsize=11)
        return curve, head

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    return _save_animation(anim, fig, path, fps)


def animate_boundary_state(panels, brane, path, fps: int = 20):
    r"""A closed string that touches the brane, beside one that does not.

    ``panels`` is a sequence of ``(label, frames)`` with ``frames`` of shape
    ``(n_tau, n_sigma, 2)``; ``brane`` is the transverse position of the brane,
    drawn as a line.

    At :math:`\tau = 0` the glued string lies *flat on the brane*: the
    Dirichlet condition puts every point of it at the brane's transverse
    position, and the Neumann one gives it no velocity along the brane.  Then it
    peels off.  That is what the coherent state means -- a boundary state is the
    closed string the brane can emit and reabsorb, and the moment it touches is
    the moment the boundary conditions hold.

    The second panel is a closed string with unglued modes.  It never lies on
    the brane at all, which is what makes the first panel a statement rather
    than a picture.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    panels = [(str(label), np.asarray(frames, dtype=float)) for label, frames in panels]
    lengths = {frames.shape[0] for _, frames in panels}
    if len(lengths) != 1:
        raise ValueError(f"panels disagree on the number of frames: {sorted(lengths)}")
    n_frames = lengths.pop()
    stacked = np.concatenate([frames.reshape(-1, 2) for _, frames in panels])
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    pad = 0.18 * np.maximum(hi - lo, 1e-9)

    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.3), dpi=130)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.12, wspace=0.2)

    curves, touches = [], []
    for ax, (label, _) in zip(axes, panels, strict=True):
        ax.axhline(float(brane), color="#d62728", lw=3.0, alpha=0.85, zorder=1)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xlabel("along the brane")
        ax.set_ylabel("across it")
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.25)
        (curve,) = ax.plot([], [], lw=2.2, color="#1f77b4", zorder=3)
        curves.append(curve)
        touches.append(ax)

    def draw(i: int):
        for curve, ax, (label, frames) in zip(curves, touches, panels, strict=True):
            shape = frames[i]
            closed = np.vstack([shape, shape[:1]])
            curve.set_data(closed[:, 0], closed[:, 1])
            gap = float(np.max(np.abs(shape[:, 1] - brane)))
            curve.set_color("#d62728" if gap < 1e-6 else "#1f77b4")
            ax.set_title(f"{label}   (gap {gap:.2f})", fontsize=11)
        return curves

    anim = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    return _save_animation(anim, fig, path, fps)


def animate_cardy_fit(frames, target, path, fps: int = 6):
    r"""Watch a famous number get measured out of a list of integers.

    ``frames`` is a sequence of ``(n_max, sqrt(N), log d_N, fitted curve,
    slope)``; ``target`` the value the slope should approach.

    Left, the exact logarithms with the three-term fit laid over them.  Right,
    the fitted coefficient of :math:`\sqrt N` against how many levels went into
    the fit.  It climbs towards :math:`2\pi\sqrt{Q_1Q_5}` and does not get there
    quickly: the subleading term is comparable to the leading one at every level
    that can be reached, which is why the number has to be extracted rather than
    read off.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    frames = [
        (int(n), np.asarray(x, dtype=float), np.asarray(y, dtype=float),
         np.asarray(f, dtype=float), float(s))
        for n, x, y, f, s in frames
    ]
    target = float(target)
    cutoffs = np.array([n for n, _, _, _, _ in frames], dtype=float)
    slopes = np.array([s for _, _, _, _, s in frames])
    top_x = max(float(x.max()) for _, x, _, _, _ in frames)
    top_y = max(float(y.max()) for _, _, y, _, _ in frames)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.8, 4.2), dpi=130)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.14, wspace=0.3)

    (points,) = left.plot([], [], "o", ms=3, color="#1f77b4", alpha=0.7)
    (model,) = left.plot([], [], "-", lw=1.8, color="#d62728")
    left.set_xlim(0.0, top_x * 1.05)
    left.set_ylim(0.0, top_y * 1.08)
    left.set_xlabel(r"$\sqrt{N}$")
    left.set_ylabel(r"$\log d_N$")
    left.grid(alpha=0.25)

    right.axhline(target, color="0.4", lw=1.0, ls="--")
    right.annotate(
        rf"$2\pi\sqrt{{Q_1Q_5}} = {target:.4f}$",
        xy=(cutoffs[0], target), xytext=(6, -14), textcoords="offset points",
        fontsize=9, color="0.35",
    )
    right.plot(cutoffs, slopes, "-", lw=1.4, color="#2ca02c")
    (head,) = right.plot([], [], "o", ms=9, color="#d62728")
    right.set_xscale("log")
    right.set_xlabel("levels used")
    right.set_ylabel(r"fitted coefficient of $\sqrt{N}$")
    right.set_ylim(min(slopes.min(), target) - 0.02, max(slopes.max(), target) + 0.02)
    right.grid(alpha=0.25)

    def draw(i: int):
        n_max, xs, ys, fitted, slope = frames[i]
        points.set_data(xs, ys)
        model.set_data(xs, fitted)
        head.set_data([cutoffs[i]], [slope])
        left.set_title(f"levels up to {n_max}", fontsize=11)
        right.set_title(f"slope {slope:.5f}   ({abs(slope / target - 1):.1e} off)", fontsize=11)
        return points, model, head

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    return _save_animation(anim, fig, path, fps)


def animate_pq_junction(frames, path, fps: int = 14):
    r"""Move the coupling and watch a junction change shape.

    ``frames`` is a sequence of ``(tau, charges, residual)``.

    A BPS :math:`(p,q)` string is not free to point where it likes: its
    direction is the phase of :math:`p + q\tau`.  So a junction is a rigid
    object whose shape the coupling sets, and moving :math:`\tau` deforms it
    while it stays in equilibrium -- the force vectors close into a polygon at
    every frame, because the charges have not changed.

    The right-hand panel is those vectors laid tip to tail.  It closes, always,
    and the number beside it is how far from closing.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    frames = [
        (complex(tau), tuple((int(p), int(q)) for p, q in charges), float(residual))
        for tau, charges, residual in frames
    ]
    colours = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    count = len(frames[0][1])

    fig, (rays, polygon) = plt.subplots(1, 2, figsize=(9.0, 4.4), dpi=130)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.86, bottom=0.06, wspace=0.15)

    arrows, labels = [], []
    for slot in range(count):
        colour = colours[slot % len(colours)]
        (line,) = rays.plot([], [], lw=2.2, color=colour, solid_capstyle="round")
        arrows.append(line)
        labels.append(rays.text(0, 0, "", ha="center", va="center", fontsize=9, color=colour))
    rays.plot([0], [0], "o", ms=7, color="black", zorder=4)
    for ax, title in ((rays, "the junction"), (polygon, "force vectors, tip to tail")):
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=11)
    (walk,) = polygon.plot([], [], "-o", ms=4, lw=1.6, color="0.35")

    def draw(i: int):
        tau, charges, residual = frames[i]
        vectors = [complex(p) + complex(q) * tau for p, q in charges]
        scale = max(abs(v) for v in vectors)
        for line, label, (p, q), vector in zip(arrows, labels, charges, vectors, strict=True):
            direction = vector / abs(vector)
            line.set_data([0.0, direction.real], [0.0, direction.imag])
            line.set_linewidth(1.0 + 3.5 * abs(vector) / scale)
            spot = direction * 1.2
            label.set_position((spot.real, spot.imag))
            label.set_text(f"({p},{q})")
        trail = [0j]
        for vector in vectors:
            trail.append(trail[-1] + vector / scale)
        centre = sum(trail) / len(trail)
        points = np.array([[z.real - centre.real, z.imag - centre.imag] for z in trail])
        walk.set_data(points[:, 0], points[:, 1])
        fig.suptitle(
            rf"$\tau = {tau.real:+.2f} {tau.imag:+.2f}i$   "
            rf"($g_s = {1.0 / tau.imag:.2f}$)   net force {residual:.1e}",
            fontsize=11,
        )
        return [*arrows, walk]

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    return _save_animation(anim, fig, path, fps)
