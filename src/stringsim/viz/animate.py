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
