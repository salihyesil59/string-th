r"""Every independent comparison in the package, run at once, on one screen.

The other panels each make a claim and check it two ways.  This one runs all of
them and shows the verdicts together, which is the only place the package's
argument appears whole: not "here is a number" but "here are forty numbers, each
reached twice by code that shares nothing, and here is whether they agree".

It is built out of the other panels rather than out of a list.  Adding a panel
adds its checks here with no edit, and a check that is quietly removed
disappears from the count -- which is the property a hand-written summary would
not have.

The control is a corner of every panel's parameter space.  At the defaults
everything agrees.  At the low or high end of every slider and spinbox at once,
two panels part company on purpose: the critical-dimension panel has an
intercept control whose whole point is that ``a = 1`` and ``D = 26`` stand or
fall together, and the Veneziano panel has one that walks the amplitude's poles
off the string's mass levels.  The rest hold, and holding at the corners of
their ranges is a stronger statement than holding at the settings they were
written with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...viz.plots import plot_check_summary
from ..panel import Choice, Line, defaults

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from matplotlib.figure import Figure

__all__ = ["ChecksPanel", "Checks", "WHERE", "settings_for"]

WHERE = ("their defaults", "the low end of every control", "the high end")


def settings_for(panel: Any, where: str) -> dict[str, Any]:
    """Parameters at one corner of a panel's control space.

    Generic on purpose: it reads the controls a panel already declares rather
    than asking each panel for a special stress setting, so a corner cannot be
    chosen to be one the checks survive.
    """
    if where == WHERE[0]:
        return defaults(panel.controls)
    low = where == WHERE[1]
    out: dict[str, Any] = {}
    for control in panel.controls:
        if isinstance(control, Choice):
            out[control.name] = control.options[0 if low else -1]
        else:
            out[control.name] = control.low if low else control.high
    return out


@dataclass(frozen=True)
class Checks:
    """What every panel said when it was asked."""

    where: str
    rows: tuple[tuple[str, tuple[tuple[str, bool | None], ...]], ...]
    failures: tuple[tuple[str, str], ...]
    """``(panel, check)`` for each disagreement."""
    errors: tuple[tuple[str, str], ...]
    """``(panel, message)`` for each panel that raised instead of answering."""
    seconds: float

    @property
    def panels(self) -> int:
        return len(self.rows)

    @property
    def made(self) -> int:
        """Comparisons that were actually made at these settings."""
        return sum(1 for _, checks in self.rows for _, v in checks if v is not None)

    @property
    def skipped(self) -> int:
        return sum(1 for _, checks in self.rows for _, v in checks if v is None)

    @property
    def agreed(self) -> int:
        return self.made - len(self.failures)

    @property
    def all_agree(self) -> bool:
        return not self.failures and not self.errors


class ChecksPanel:
    """The package's argument, as one screen."""

    title = "Every check at once"
    blurb = (
        "One dot per independent comparison, one row per subject.  Green is agreement, "
        "red is disagreement, and a hollow ring is a comparison that does not apply at "
        "these settings -- which is a different statement from a failure and is drawn "
        "differently for that reason.  The rows are built from the other panels, so a "
        "check added anywhere appears here without this panel being touched."
    )
    background = (
        "Every number in this package that can be reached two ways is reached two ways, "
        "and the two routes are required to share no code.  A partition count against "
        "Cardy's formula; a Gram matrix's signature against a conformal anomaly; a "
        "membrane wrapping a torus cycle against a ten-dimensional tension formula; a "
        "numerical Legendre transform against a gradient.  Separately these are lines in "
        "a test file.  Together they are the reason to believe any of it.",
        "That is what this screen is for.  A test suite reports pass or fail and then "
        "vanishes; here the comparisons are recomputed on demand and shown side by side, "
        "with the same verdicts the individual panels give.  Nothing is cached and "
        "nothing is quoted.",
        "The rows come from the registry rather than from a list written by hand.  If a "
        "panel is added its checks appear; if a check is deleted the count drops.  A "
        "summary maintained separately would drift from what is actually being "
        "computed, and would drift silently.",
        "The corner control is the interesting one.  Running everything at the settings "
        "each panel was written with proves less than running it at the ends of every "
        "range at once, because the defaults are where a mistake is least likely to "
        "show.  Two panels are expected to part at the corners -- both have an intercept "
        "control whose purpose is that moving it breaks an agreement -- and the notes "
        "name them rather than leaving a red dot unexplained.",
    )
    suggestions = (
        "Start at the defaults: everything agrees, and the title says how many "
        "comparisons that is.",
        "Switch to the low or the high end.  Two rows go red, and they are the two "
        "panels whose sliders are built to break an agreement -- the critical dimension "
        "at a intercept away from 1, and the Veneziano amplitude with its poles walked "
        "off the spectrum.  Everything else survives its own extremes.",
        "Count the hollow rings.  Those are comparisons that do not apply where the "
        "controls now are: an axion away from zero, a scan window too narrow to hold "
        "SO(32), a momentum transfer sitting on a resonance.  None of them is a failure "
        "and none is drawn as one.",
        "Open any red row's own panel to read what happened.  This screen says which "
        "comparisons disagree; the panel says why, and whether that is the point.",
    )
    controls = (Choice("where", "run every panel at", WHERE, WHERE[0]),)

    def compute(self, where: str = WHERE[0]) -> Checks:
        import time

        from . import REGISTRY

        started = time.perf_counter()
        rows: list[tuple[str, tuple[tuple[str, bool | None], ...]]] = []
        failures: list[tuple[str, str]] = []
        errors: list[tuple[str, str]] = []

        for panel in REGISTRY:
            if isinstance(panel, ChecksPanel):
                continue  # this panel is not one of its own rows
            try:
                result = panel.compute(**settings_for(panel, where))
                lines = panel.readout(result)
            except Exception as exc:  # a panel that cannot answer is not a pass
                errors.append((panel.title, f"{type(exc).__name__}: {exc}"))
                rows.append((panel.title, ()))
                continue
            # Both states are collected: a verdict, and a comparison that
            # declined to give one.  Dropping the second would make the screen
            # claim more comparisons were made than were.
            checks = tuple(
                (line.label.strip(), line.ok)
                for line in lines
                if line.is_comparison
            )
            rows.append((panel.title, checks))
            failures.extend(
                (panel.title, label) for label, ok in checks if ok is False
            )

        return Checks(
            where=where,
            rows=tuple(rows),
            failures=tuple(failures),
            errors=tuple(errors),
            seconds=time.perf_counter() - started,
        )

    def draw(self, result: Checks) -> Figure:
        return plot_check_summary(
            [(name, list(checks)) for name, checks in result.rows],
            path=None,
            title=f"Every check, run at {result.where}",
        )

    def readout(self, result: Checks) -> list[Line]:
        lines = [
            Line(
                "comparisons made",
                f"{result.made} across {result.panels} panels",
                f"recomputed in {result.seconds:.2f} s, nothing cached",
            ),
            Line(
                "agreeing",
                f"{result.agreed} of {result.made}",
                "each reached twice by code with nothing in common",
                ok=result.all_agree,
            ),
        ]
        if result.skipped:
            lines.append(
                Line(
                    "not applicable here",
                    f"{result.skipped}",
                    "drawn as a ring, not as a failure",
                )
            )
        for panel, label in result.failures:
            lines.append(Line(f"  disagreeing: {panel}", label))
        for panel, message in result.errors:
            lines.append(Line(f"  raised: {panel}", message))
        for name, checks in result.rows:
            agreed = sum(1 for _, ok in checks if ok is True)
            made = sum(1 for _, ok in checks if ok is not None)
            lines.append(
                Line(
                    f"  {name}",
                    f"{agreed}/{made}" if made else "no comparison here",
                    ", ".join(label for label, _ in checks)[:70],
                )
            )
        return lines

    def notes(self, result: Checks) -> list[str]:
        """What agreed, what did not, and whether the difference was designed."""
        out = [
            f"{result.made} independent comparisons across {result.panels} panels, "
            f"recomputed in {result.seconds:.1f} seconds with nothing cached and nothing "
            "quoted.  Each of them is a number reached twice by code that shares no "
            "path: a partition count against Cardy's formula, a Gram matrix's signature "
            "against a conformal anomaly, an M2-brane wrapping a cycle against a "
            "ten-dimensional tension formula, a numerical Legendre transform against a "
            "gradient.  Separately they are lines in a test file.  Together they are the "
            "reason to believe any of it."
        ]
        if result.all_agree:
            out.append(
                f"All {result.agreed} of them agree at {result.where}.  That is the "
                "whole claim the package makes, and this is the only screen on which it "
                "appears at once."
            )
        else:
            named = ", ".join(sorted({panel for panel, _ in result.failures}))
            expected = {"D = 26, from two sides", "The Veneziano amplitude"}
            designed = set(panel for panel, _ in result.failures) <= expected
            out.append(
                f"{len(result.failures)} comparison(s) disagree here, in: {named}.  "
                + (
                    "Both of those panels have an intercept control whose purpose is "
                    "that moving it breaks an agreement -- a = 1 and D = 26 are one "
                    "statement, and the Veneziano amplitude's poles sit on the string's "
                    "mass levels only at the string's own intercept.  So this is the "
                    "panels working, and the rest of the package surviving the ends of "
                    "its own ranges."
                    if designed
                    else "That is not a set this panel expects, and it is worth opening "
                    "the panels named to find out why rather than assuming it is by "
                    "design."
                )
            )
        if result.skipped:
            out.append(
                f"{result.skipped} comparison(s) are not made at these settings and are "
                "drawn as hollow rings.  A comparison that does not apply is not a "
                "comparison that failed: an axion away from zero, a scan window too "
                "narrow to contain SO(32), a momentum transfer sitting on a resonance.  "
                "Colouring those red would be the easiest way to make this screen lie."
            )
        if result.errors:
            out.append(
                f"{len(result.errors)} panel(s) raised instead of answering, and they "
                "are listed in the readout with the message.  A panel that cannot "
                "compute is not a panel that passed, so its row is empty rather than "
                "absent."
            )
        return out
