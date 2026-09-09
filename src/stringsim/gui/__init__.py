"""A window for the package, built on what is already in the standard library.

``python -m stringsim.gui``, or ``stringsim --gui``.

The package's figures are files, and a viewer for files would add nothing.  What
a window can do that a script cannot is let a parameter *move* -- and, while it
moves, recompute the independent checks and show them changing beside the
result.  Drag the radius past the self-dual point and the two towers swap while
the whole enumerated spectrum stays equal to its dual's, to the last digit, on
screen.  That is the package's habit made visible rather than merely tested.

Tkinter and matplotlib are the whole of it: the first is in the standard
library and the second was already required, so the window costs no new
dependency.  The panels themselves are free of both -- see
:mod:`stringsim.gui.panel` for why that is worth the small effort it takes.
"""

from __future__ import annotations

from .panel import Choice, Control, Integer, Line, Panel, Slider
from .panels import REGISTRY
from .runner import Runner

__all__ = [
    "launch",
    "REGISTRY",
    "Runner",
    "Panel",
    "Line",
    "Slider",
    "Integer",
    "Choice",
    "Control",
]

_NO_TK = """\
stringsim --gui needs tkinter, which is part of the standard library but is
packaged separately by some distributions.  On Debian and Ubuntu that is
"apt install python3-tk"; on Fedora "dnf install python3-tkinter".  Everything
else in the package works without it."""

_NO_DISPLAY = """\
tkinter is installed but could not open a window: {reason}

There is no display here.  Over SSH try "ssh -X", and headless machines can
still produce every figure with the example scripts in examples/."""


def launch(panels: tuple[Panel, ...] = REGISTRY, *, title: str = "stringsim") -> int:
    """Open the window and run until it is closed.  Returns a process exit code.

    The two ways this fails are ordinary and neither deserves a traceback: no
    tkinter installed, and no display to open.  Both are reported as sentences.
    """
    try:
        import tkinter as tk
    except ImportError:
        print(_NO_TK)
        return 1

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(_NO_DISPLAY.format(reason=exc))
        return 1

    from .app import App

    root.title(title)
    # As much of the screen as the figures want, without ever running off it:
    # two subplots side by side need height a wide short strip does not give
    # them, and a laptop should not open a window taller than its display.
    width = min(1180, root.winfo_screenwidth() - 80)
    height = min(1000, root.winfo_screenheight() - 120)
    root.geometry(f"{max(760, width)}x{max(560, height)}")
    root.minsize(760, 560)
    _use_native_theme(root)

    app = App(root, panels)

    def on_close() -> None:
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0


def _use_native_theme(root: object) -> None:
    """Prefer the platform's own look, and ``clam`` where there is none."""
    from tkinter import ttk

    style = ttk.Style(root)
    available = style.theme_names()
    for wanted in ("vista", "aqua", "clam"):
        if wanted in available:
            style.theme_use(wanted)
            return
