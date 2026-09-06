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

__all__ = ["animate_modes", "animate_evolution", "snapshot_grid"]


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
