"""What a panel is: four small things, and no widget among them.

A panel names itself, declares its controls as *data*, computes a result from
them, draws that result, and reports the numbers.  Nothing in this module
imports ``tkinter`` and nothing in it opens a window, so a panel's
:meth:`compute` and :meth:`readout` are ordinary functions over ordinary values
-- tested the way the rest of the package is tested, with no display anywhere.

That separation is not tidiness.  The physics in a panel is the part worth
checking, and it stays checkable exactly as long as it never touches a widget.
It also leaves the door open: a second front end over the same panels would be
another ``app.py``, not a rewrite.

The readout is where the package's habit shows up on screen.  A
:class:`Line` carries a label and a value, and where the package has a *second,
independent* route to that number it carries the second value too, together
with the residual between them -- recomputed every time a control moves, rather
than asserted once in a test and then forgotten.

Beside the numbers a panel may carry prose, in three kinds.  ``background`` is
static and says where the physics comes from.  ``suggestions`` says what is
worth doing to the controls.  :meth:`Panel.notes` is the one that earns the
window: it describes *this* result, and a sentence that is only true at the
self-dual radius is a sentence a static page cannot write.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

__all__ = [
    "Slider",
    "Integer",
    "Choice",
    "Control",
    "Line",
    "Panel",
    "defaults",
    "check_controls",
    "background_of",
    "suggestions_of",
    "notes_of",
]


# --------------------------------------------------------------------------
# controls, as data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Slider:
    """A continuous parameter."""

    name: str
    label: str
    low: float
    high: float
    default: float
    step: float = 0.0
    """Resolution of the widget.  Zero means as fine as the widget allows."""
    log: bool = False
    """Drag the exponent rather than the value.

    Right whenever the parameter's *ratios* are what matter -- a radius under a
    duality that sends ``R`` to ``alpha'/R`` belongs in the middle of a
    logarithmic track, with the two dual radii symmetric about it.
    """

    def clamp(self, value: float) -> float:
        return min(max(float(value), self.low), self.high)


@dataclass(frozen=True)
class Integer:
    """A whole-numbered parameter, usually a truncation."""

    name: str
    label: str
    low: int
    high: int
    default: int

    def clamp(self, value: int) -> int:
        return min(max(int(value), self.low), self.high)


@dataclass(frozen=True)
class Choice:
    """One of a fixed set of options."""

    name: str
    label: str
    options: tuple[str, ...]
    default: str

    def clamp(self, value: str) -> str:
        return value if value in self.options else self.default


Control = Slider | Integer | Choice


def defaults(controls: tuple[Control, ...]) -> dict[str, Any]:
    """The parameter dictionary a panel starts from."""
    return {control.name: control.default for control in controls}


def check_controls(controls: tuple[Control, ...]) -> None:
    """Raise if a control specification could not build a working widget.

    Cheap to run and worth running over every panel in the registry: a
    misspelled name or a default outside its own range is the kind of mistake
    that otherwise shows up as an empty window.
    """
    seen: set[str] = set()
    for control in controls:
        if not control.name or not control.name.isidentifier():
            raise ValueError(f"control name {control.name!r} is not an identifier")
        if control.name in seen:
            raise ValueError(f"two controls are both named {control.name!r}")
        seen.add(control.name)
        if not control.label:
            raise ValueError(f"control {control.name!r} has no label")
        if isinstance(control, Choice):
            if not control.options:
                raise ValueError(f"control {control.name!r} offers nothing to choose")
            if control.default not in control.options:
                raise ValueError(
                    f"control {control.name!r} defaults to {control.default!r}, "
                    "which is not among its options"
                )
            continue
        if control.low >= control.high:
            raise ValueError(f"control {control.name!r} has an empty range")
        if isinstance(control, Slider) and control.log and control.low <= 0:
            raise ValueError(f"control {control.name!r} is logarithmic but reaches zero")
        if not control.low <= control.default <= control.high:
            raise ValueError(
                f"control {control.name!r} defaults to {control.default}, "
                f"outside [{control.low}, {control.high}]"
            )


# --------------------------------------------------------------------------
# the readout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Line:
    """One row of a panel's readout.

    ``value`` is the number the panel computed.  ``check`` is the same quantity
    reached another way, when there is another way, and ``ok`` says whether the
    two agree -- so a reader watches the agreement survive a parameter moving,
    which is a stronger thing to see than a passing test.
    """

    label: str
    value: str
    check: str = ""
    ok: bool | None = None

    @property
    def is_check(self) -> bool:
        return self.ok is not None

    def verdict(self) -> str:
        """``''``, ``'ok'`` or ``'FAILS'`` -- what the app puts in the margin."""
        if self.ok is None:
            return ""
        return "ok" if self.ok else "FAILS"

    def agreeing(self, ok: bool) -> Line:
        return replace(self, ok=ok)

    def __str__(self) -> str:
        tail = f"   [{self.check}]" if self.check else ""
        mark = f" {self.verdict()}" if self.is_check else ""
        return f"{self.label}: {self.value}{tail}{mark}"


# --------------------------------------------------------------------------
# the protocol
# --------------------------------------------------------------------------


class Panel(Protocol):
    """One screen: a title, a paragraph, some controls, and three methods."""

    title: str
    blurb: str
    controls: tuple[Control, ...]

    def compute(self, **params: Any) -> Any:
        """Do the physics.  Pure, and free of matplotlib and of tkinter."""

    def draw(self, result: Any) -> Figure:
        """Return a figure for ``result``, detached from pyplot.

        In practice this calls one of the package's existing ``plot_*``
        functions with ``path=None``, so the window and the example scripts
        draw the same picture from the same code.
        """

    def readout(self, result: Any) -> list[Line]:
        """The numbers, and the independent checks on them."""

    # Everything below is optional.  A panel that defines none of it still
    # works; a panel that defines all of it can be read rather than merely
    # operated, which for most of this package's subjects is the difference
    # between a control surface and an explanation.

    background: tuple[str, ...]
    """Paragraphs that do not change: where the physics comes from."""

    suggestions: tuple[str, ...]
    """Things worth doing to the controls, and what to watch when you do."""

    def notes(self, result: Any) -> list[str]:
        """Paragraphs about *this* result -- what is true at these settings.

        The one part of the commentary that a static page cannot carry.  A
        radius that happens to be self-dual, an intercept that has walked the
        poles off the spectrum, a coupling past the point where the D1 became
        the lighter object: each is a sentence that is worth writing only while
        it is true.
        """


# The three optional pieces, read off a panel that may not define them.  A
# panel is a plain class rather than a subclass of anything, so this is where
# the defaults live.


def background_of(panel: Any) -> tuple[str, ...]:
    return tuple(getattr(panel, "background", ()))


def suggestions_of(panel: Any) -> tuple[str, ...]:
    return tuple(getattr(panel, "suggestions", ()))


def notes_of(panel: Any, result: Any) -> list[str]:
    method = getattr(panel, "notes", None)
    return list(method(result)) if callable(method) else []
