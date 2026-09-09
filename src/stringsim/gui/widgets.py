"""Turning control *specifications* into ttk widgets.

This is the only place besides :mod:`~stringsim.gui.app` that imports tkinter.
A panel declares what it wants adjusted -- a range, a default, a label -- and
this module decides what that looks like.  A slider becomes a ``Scale`` with
its current value beside it, a truncation becomes a ``Spinbox`` because a
truncation is read as much as it is dragged, and a choice becomes a read-only
``Combobox``.

A logarithmic slider is dragged in the exponent, which matters more than it
sounds: a panel about ``R -> alpha'/R`` should put the self-dual radius in the
middle of its track and the two dual radii symmetrically about it, and a linear
slider puts ``R = 1`` a fifth of the way along a range that reaches 4.  The
widget holds the logarithm; the panel is handed the radius.

Every change is reported through one callback.  Deciding *when* to act on it is
the app's problem, not this module's: a slider fires continuously while it is
dragged, and the app debounces.
"""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import Any

from .panel import Choice, Control, Integer, Slider

__all__ = ["ControlBar"]


class ControlBar:
    """The row of controls for one panel, and the values they currently hold."""

    def __init__(
        self,
        master: tk.Misc,
        controls: tuple[Control, ...],
        on_change: Callable[[], None],
    ) -> None:
        self.frame = ttk.Frame(master, padding=(8, 6))
        self._controls = controls
        self._on_change = on_change
        self._vars: dict[str, tk.Variable] = {}
        self._read: dict[str, Callable[[], Any]] = {}
        # The controls keep a readable width and the slack goes to the right, so
        # a slider's value sits next to the slider rather than adrift at the far
        # edge of a wide window.
        self.frame.columnconfigure(3, weight=1)

        for row, control in enumerate(controls):
            ttk.Label(self.frame, text=control.label).grid(
                row=row, column=0, sticky="w", padx=(0, 10), pady=2
            )
            builder = {Slider: self._slider, Integer: self._integer, Choice: self._choice}[
                type(control)
            ]
            builder(control, row)

    # -- the three kinds -----------------------------------------------

    def _slider(self, control: Slider, row: int) -> None:
        forward, back = _scales(control)
        var = tk.DoubleVar(value=forward(control.default))
        self._vars[control.name] = var
        self._read[control.name] = lambda: _snap(back(var.get()), control)
        shown = ttk.Label(self.frame, width=7, anchor="w", text=f"{control.default:g}")

        def moved(_value: str) -> None:
            shown.configure(text=f"{self._read[control.name]():g}")
            self._on_change()

        scale = ttk.Scale(
            self.frame,
            from_=forward(control.low),
            to=forward(control.high),
            variable=var,
            command=moved,
            orient="horizontal",
            length=320,
        )
        scale.grid(row=row, column=1, sticky="w", pady=2)
        shown.grid(row=row, column=2, sticky="w", padx=(10, 0))

    def _integer(self, control: Integer, row: int) -> None:
        var = tk.IntVar(value=control.default)
        self._vars[control.name] = var
        self._read[control.name] = var.get
        spin = ttk.Spinbox(
            self.frame,
            from_=control.low,
            to=control.high,
            textvariable=var,
            width=6,
            command=self._on_change,
        )
        spin.grid(row=row, column=1, sticky="w", pady=2)
        # Typing into the box counts too, but only once the field parses.
        spin.bind("<Return>", lambda _event: self._on_change())
        spin.bind("<FocusOut>", lambda _event: self._on_change())

    def _choice(self, control: Choice, row: int) -> None:
        var = tk.StringVar(value=control.default)
        self._vars[control.name] = var
        self._read[control.name] = var.get
        box = ttk.Combobox(
            self.frame,
            values=list(control.options),
            textvariable=var,
            state="readonly",
            width=max(len(option) for option in control.options) + 2,
        )
        box.grid(row=row, column=1, sticky="w", pady=2)
        box.bind("<<ComboboxSelected>>", lambda _event: self._on_change())

    # -- reading them --------------------------------------------------

    def values(self) -> dict[str, Any]:
        """What to pass to ``compute``, snapped and clamped to each range.

        A ``Spinbox`` can be typed into and can therefore hold nonsense, so
        every value goes through its control's own ``clamp`` before it reaches
        any physics.
        """
        out: dict[str, Any] = {}
        for control in self._controls:
            try:
                raw = self._read[control.name]()
            except tk.TclError:  # half-typed text in a spinbox
                raw = control.default
            out[control.name] = control.clamp(raw)
        return out

    def set(self, name: str, value: Any) -> None:
        """Move a control from code, in the units the panel speaks."""
        control = next(c for c in self._controls if c.name == name)
        forward = _scales(control)[0] if isinstance(control, Slider) else (lambda v: v)
        self._vars[name].set(forward(control.clamp(value)))

    def destroy(self) -> None:
        self.frame.destroy()


def _scales(control: Slider) -> tuple[Callable[[float], float], Callable[[float], float]]:
    """The map between what the widget holds and what the panel is given."""
    if control.log:
        return math.log10, lambda t: 10.0**t
    return (lambda v: v), (lambda t: t)


def _snap(value: float, control: Slider) -> float:
    """Round to the slider's step, so the readout does not jitter in its tail."""
    if control.step <= 0:
        return float(value)
    return round(round(float(value) / control.step) * control.step, 10)
