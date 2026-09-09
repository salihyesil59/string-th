"""The window: a list of panels on the left, and one panel's world on the right.

Three rules shape this module.

*The physics never runs here.*  A control moves, the app collects the values,
and :class:`~stringsim.gui.runner.Runner` computes on another thread.  What
comes back is a plain result object.

*Drawing never runs there.*  matplotlib is not thread-safe, so ``draw`` is
called on the main thread once the result has arrived.  That is the whole
reason a panel has ``compute`` and ``draw`` as separate methods rather than one
method that returns a figure.

*Nothing crosses threads except through a queue.*  The worker puts finished
jobs in a :class:`queue.Queue` and the main thread drains it on a timer.  Tk
objects are touched by the main thread only.
"""

from __future__ import annotations

import queue
import tkinter as tk
import traceback
from tkinter import ttk
from typing import Any

from .panel import (
    Line,
    Panel,
    background_of,
    check_controls,
    notes_of,
    suggestions_of,
)
from .panels import REGISTRY
from .runner import Runner
from .widgets import ControlBar

__all__ = ["App"]

DEBOUNCE_MS = 120
"""How long a control must sit still before the job is submitted."""

POLL_MS = 40
"""How often the main thread looks for a finished job."""


class App:
    """The application, over a Tk root that somebody else created."""

    def __init__(self, root: tk.Misc, panels: tuple[Panel, ...] = REGISTRY) -> None:
        if not panels:
            raise ValueError("there are no panels to show")
        for panel in panels:
            check_controls(panel.controls)

        self.root = root
        self.panels = panels
        self.panel: Panel = panels[0]

        self._results: queue.Queue[tuple[int, bool, Any]] = queue.Queue()
        self.runner = Runner(lambda job_id, ok, payload: self._results.put((job_id, ok, payload)))
        self._pending_after: str | None = None
        self._canvas = None
        self._figure = None
        self._controls: ControlBar | None = None

        self._build()
        self._select(0)
        self.root.after(POLL_MS, self._drain)

    # -- layout --------------------------------------------------------

    def _build(self) -> None:
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        sidebar = ttk.Frame(outer, padding=(8, 8))
        sidebar.pack(side="left", fill="y")
        ttk.Label(sidebar, text="panels", font=_bold()).pack(anchor="w", pady=(0, 4))
        self._list = tk.Listbox(
            sidebar,
            height=len(self.panels),
            width=26,
            exportselection=False,
            activestyle="none",
        )
        for panel in self.panels:
            self._list.insert("end", f"  {panel.title}")
        self._list.pack(fill="y", expand=True)
        self._list.bind("<<ListboxSelect>>", self._on_select)

        right = ttk.Frame(outer, padding=(4, 8, 8, 8))
        right.pack(side="left", fill="both", expand=True)

        self._title = ttk.Label(right, text="", font=_bold(12))
        self._title.pack(anchor="w")
        self._blurb = ttk.Label(right, text="", justify="left", wraplength=700)
        self._blurb.pack(anchor="w", pady=(2, 8), fill="x")
        right.bind(
            "<Configure>",
            lambda event: self._blurb.configure(wraplength=max(320, event.width - 30)),
        )

        # A draggable split, because how much room a figure needs is a property
        # of the figure: one pair of axes is legible in a wide short strip and
        # two side by side are not, and no fixed division suits both.  The
        # panes below are packed from the bottom upwards so the readout keeps
        # its height rather than being squeezed off the window.
        split = ttk.PanedWindow(right, orient="vertical")
        split.pack(fill="both", expand=True)

        self._plot = ttk.Frame(split)
        lower = ttk.Frame(split)
        # All spare height goes to the figure and none to the controls, which
        # need exactly as much as they need.  With any weight at all on the
        # lower pane the surplus lands under the readout as blank grey.
        split.add(self._plot, weight=1)
        split.add(lower, weight=0)

        # Two views of the same result, side by side in the same space: the
        # numbers, and what they mean here.  Tabs rather than a second column
        # because the readout's lines are long and a figure wants the width.
        tabs = ttk.Notebook(lower)
        tabs.pack(side="bottom", fill="x")

        readout_box = ttk.Frame(tabs)
        self._readout = tk.Text(readout_box, height=9, wrap="none", font=_fixed(), relief="flat")
        scroll = ttk.Scrollbar(readout_box, orient="horizontal", command=self._readout.xview)
        self._readout.configure(xscrollcommand=scroll.set)
        scroll.pack(side="bottom", fill="x")
        self._readout.pack(side="bottom", fill="both", expand=True)
        tabs.add(readout_box, text="  numbers  ")

        notes_box = ttk.Frame(tabs)
        self._notes = tk.Text(
            notes_box, height=9, wrap="word", relief="flat", padx=8, pady=6, spacing3=4
        )
        notes_scroll = ttk.Scrollbar(notes_box, orient="vertical", command=self._notes.yview)
        self._notes.configure(yscrollcommand=notes_scroll.set)
        notes_scroll.pack(side="right", fill="y")
        self._notes.pack(side="left", fill="both", expand=True)
        tabs.add(notes_box, text="  what this means  ")

        self._notes.tag_configure("heading", foreground="#444444", spacing1=8, spacing3=3)
        self._notes.tag_configure("now", foreground="#1a4d7f")
        self._notes.configure(state="disabled")

        self._status = ttk.Label(lower, text="", anchor="w")
        self._status.pack(side="bottom", fill="x", pady=(4, 2))

        self._control_host = ttk.Frame(lower)
        self._control_host.pack(side="bottom", fill="x")

        self._readout.tag_configure("ok", foreground="#1a7f37")
        self._readout.tag_configure("bad", foreground="#b3261e")
        self._readout.tag_configure("label", foreground="#555555")
        self._readout.configure(state="disabled")

    # -- switching panels ----------------------------------------------

    def _on_select(self, _event: object) -> None:
        picked = self._list.curselection()
        if picked and self.panels[picked[0]] is not self.panel:
            self._select(picked[0])

    def _select(self, index: int) -> None:
        self.panel = self.panels[index]
        self._list.selection_clear(0, "end")
        self._list.selection_set(index)
        self._title.configure(text=self.panel.title)
        self._blurb.configure(text=self.panel.blurb)
        if self._controls is not None:
            self._controls.destroy()
        self._controls = ControlBar(self._control_host, self.panel.controls, self._changed)
        self._controls.frame.pack(fill="x")
        self._request()

    # -- the loop ------------------------------------------------------

    def _changed(self) -> None:
        """A control moved.  Wait for it to settle before doing any work."""
        if self._pending_after is not None:
            self.root.after_cancel(self._pending_after)
        self._pending_after = self.root.after(DEBOUNCE_MS, self._request)

    def _request(self) -> None:
        self._pending_after = None
        assert self._controls is not None
        panel, values = self.panel, self._controls.values()
        self._status.configure(text="computing ...")
        self.runner.submit(lambda: (panel, panel.compute(**values)))

    def _drain(self) -> None:
        """Main thread: take whatever the worker finished and put it on screen."""
        try:
            while True:
                _job_id, ok, payload = self._results.get_nowait()
                if not ok:
                    self._show_failure(payload)
                    continue
                panel, result = payload
                if panel is self.panel:  # a slower panel may finish after a switch
                    self._show(panel, result)
        except queue.Empty:
            pass
        finally:
            self.root.after(POLL_MS, self._drain)

    # -- putting it on screen -------------------------------------------

    def _show(self, panel: Panel, result: Any) -> None:
        try:
            figure = panel.draw(result)
            lines = panel.readout(result)
        except Exception as exc:  # drawing can fail on its own
            self._show_failure(exc)
            return
        self._show_figure(figure)
        self._write(lines)
        self._write_notes(panel, result)
        failed = sum(1 for line in lines if line.ok is False)
        checks = sum(1 for line in lines if line.is_check)
        self._status.configure(
            text=f"{checks} independent check(s), {failed} failing"
            if failed
            else f"{checks} independent check(s), all agreeing"
        )

    def _show_figure(self, figure: Any) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
        # ``_save`` laid the figure out once, at its own size.  A canvas resizes
        # it, and without a layout engine the title is then cropped -- so hand
        # the figure one, which re-runs the layout on every draw.
        figure.set_layout_engine("tight")
        self._figure = figure  # keep it alive; the canvas only holds a weak claim
        self._canvas = FigureCanvasTkAgg(figure, master=self._plot)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    def _write(self, lines: list[Line]) -> None:
        width = max((len(line.label) for line in lines), default=0)
        self._readout.configure(state="normal")
        self._readout.delete("1.0", "end")
        for line in lines:
            self._readout.insert("end", f"{line.label:<{width}}  ", "label")
            self._readout.insert("end", line.value)
            if line.check:
                self._readout.insert("end", f"   [{line.check}]")
            if line.is_check:
                self._readout.insert("end", f"  {line.verdict()}", "ok" if line.ok else "bad")
            self._readout.insert("end", "\n")
        self._readout.configure(state="disabled")

    def _write_notes(self, panel: Panel, result: Any) -> None:
        """The prose: what is true here, where it comes from, what to try next.

        The first section is recomputed with the result and the other two are
        not, which is the whole distinction -- a sentence that holds only at the
        self-dual radius belongs above one that holds always.
        """
        sections = (
            ("what is happening here", notes_of(panel, result), "now"),
            ("background", list(background_of(panel)), ""),
            ("things to try", list(suggestions_of(panel)), ""),
        )
        self._notes.configure(state="normal")
        self._notes.delete("1.0", "end")
        for heading, paragraphs, tag in sections:
            if not paragraphs:
                continue
            self._notes.insert("end", f"{heading.upper()}\n", "heading")
            for paragraph in paragraphs:
                self._notes.insert("end", f"{paragraph}\n\n", tag)
        if self._notes.get("1.0", "end").strip() == "":
            self._notes.insert("end", "This panel carries no commentary yet.")
        self._notes.configure(state="disabled")

    def _show_failure(self, exc: BaseException) -> None:
        self._status.configure(text=f"{type(exc).__name__}: {exc}")
        self._readout.configure(state="normal")
        self._readout.delete("1.0", "end")
        self._readout.insert(
            "end", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), "bad"
        )
        self._readout.configure(state="disabled")

    # -- shutdown ------------------------------------------------------

    def close(self) -> None:
        """Stop the worker.  Called on window close, and by the tests."""
        self.runner.close()


def _bold(size: int | None = None) -> tuple[str, int, str]:
    from tkinter import font

    base = font.nametofont("TkDefaultFont")
    return (base.cget("family"), size or base.cget("size"), "bold")


def _fixed() -> tuple[str, int]:
    from tkinter import font

    base = font.nametofont("TkFixedFont")
    return (base.cget("family"), base.cget("size"))
