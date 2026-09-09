"""The window, tested mostly without one.

A panel's ``compute`` and ``readout`` are ordinary functions, so the physics in
them is tested here the way it is tested everywhere else in the package -- no
display, no widgets, no waiting.  The scheduler is the same: it knows nothing
about tkinter, so its cancellation behaviour is checked with plain threads.

What genuinely needs a window gets one smoke test, skipped where there is no
display: build the app, walk every panel in the registry, and let each one
compute, draw and report.  That catches import errors, malformed control
specifications and drawing crashes across the whole registry at once.  Nothing
here asserts anything about how the window *looks*.
"""

from __future__ import annotations

import importlib
import math
import re
import sys
import threading
import time

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from stringsim.gui import REGISTRY, Runner  # noqa: E402
from stringsim.gui.panel import (  # noqa: E402
    Choice,
    Integer,
    Line,
    Slider,
    background_of,
    check_controls,
    defaults,
    notes_of,
    suggestions_of,
)
from stringsim.gui.panels.anomaly import AnomalyPanel  # noqa: E402
from stringsim.gui.panels.bion import BIonPanel  # noqa: E402
from stringsim.gui.panels.branes import BranePanel  # noqa: E402
from stringsim.gui.panels.critical import CriticalDimensionPanel  # noqa: E402
from stringsim.gui.panels.hagedorn import HagedornPanel  # noqa: E402
from stringsim.gui.panels.mirror import FAMILIES, MirrorPanel  # noqa: E402
from stringsim.gui.panels.myers import MyersPanel  # noqa: E402
from stringsim.gui.panels.orbifold import ORBIFOLDS, OrbifoldPanel  # noqa: E402
from stringsim.gui.panels.pq import PQPanel  # noqa: E402
from stringsim.gui.panels.tduality import TDualityPanel  # noqa: E402
from stringsim.gui.panels.veneziano import VenezianoPanel  # noqa: E402
from stringsim.quantum.virasoro import lightcone_count  # noqa: E402

TDUALITY = TDualityPanel()
TDUALITY_IN_REGISTRY = next(p for p in REGISTRY if isinstance(p, TDualityPanel))


# --------------------------------------------------------------------------
# controls, as data
# --------------------------------------------------------------------------


def test_every_panel_in_the_registry_declares_workable_controls() -> None:
    """The cheapest guard there is against a panel that opens empty."""
    assert REGISTRY
    for panel in REGISTRY:
        check_controls(panel.controls)
        assert panel.title and panel.blurb
        assert set(defaults(panel.controls)) == {c.name for c in panel.controls}


@pytest.mark.parametrize(
    ("controls", "message"),
    [
        ((Slider("r", "R", 1.0, 2.0, 5.0),), "outside"),
        ((Slider("r", "R", 2.0, 1.0, 1.5),), "empty range"),
        ((Slider("r", "R", 0.0, 1.0, 0.5), Integer("r", "again", 0, 2, 1)), "both named"),
        ((Integer("n max", "N", 0, 3, 1),), "not an identifier"),
        ((Slider("r", "", 0.0, 1.0, 0.5),), "no label"),
        ((Choice("k", "kind", (), "a"),), "nothing to choose"),
        ((Choice("k", "kind", ("a", "b"), "c"),), "not among its options"),
        ((Slider("r", "R", 0.0, 4.0, 1.0, log=True),), "logarithmic"),
    ],
)
def test_a_broken_control_is_refused(controls, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_controls(controls)


def test_controls_clamp_what_a_widget_might_hand_them() -> None:
    """A spinbox can be typed into, so nothing reaches the physics unchecked."""
    assert Slider("r", "R", 0.5, 2.0, 1.0).clamp(9.0) == 2.0
    assert Slider("r", "R", 0.5, 2.0, 1.0).clamp(0.0) == 0.5
    assert Integer("n", "N", 1, 4, 2).clamp(99) == 4
    assert Choice("k", "kind", ("a", "b"), "a").clamp("nonsense") == "a"


def test_a_readout_line_knows_whether_it_is_a_check() -> None:
    plain = Line("radius", "1.0000")
    assert not plain.is_check and plain.verdict() == ""
    checked = Line("masses", "0.25", "0.25", ok=True)
    assert checked.is_check and checked.verdict() == "ok"
    assert checked.agreeing(False).verdict() == "FAILS"
    assert "0.25" in str(checked) and "ok" in str(checked)


def test_the_panels_import_without_tkinter(monkeypatch) -> None:
    """The physics side must not drag a widget in behind it.

    This is what makes the panels testable at all, and it is the reason a
    second front end would be another ``app.py`` rather than a rewrite -- so it
    is worth a test rather than a comment.
    """
    for name in [m for m in sys.modules if m.startswith("stringsim.gui")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "tkinter", None)

    with pytest.raises(ImportError):
        importlib.import_module("tkinter")
    module = importlib.import_module("stringsim.gui.panels")
    assert module.REGISTRY


# --------------------------------------------------------------------------
# the scheduler
# --------------------------------------------------------------------------


def test_a_job_runs_and_its_result_is_delivered() -> None:
    seen: list[tuple[int, bool, object]] = []
    runner = Runner(lambda *args: seen.append(args))
    try:
        job_id = runner.submit(lambda: 6 * 7)
        assert runner.wait_idle(5.0)
    finally:
        runner.close()
    assert seen == [(job_id, True, 42)]


def test_a_failing_job_arrives_as_a_message_not_a_crash() -> None:
    """A panel that raises should be readable on screen, not fatal."""
    seen: list[tuple[int, bool, object]] = []
    runner = Runner(lambda *args: seen.append(args))
    try:
        runner.submit(lambda: 1 / 0)
        assert runner.wait_idle(5.0)
    finally:
        runner.close()
    (_, ok, payload) = seen[0]
    assert not ok and isinstance(payload, ZeroDivisionError)


def test_a_newer_job_cancels_the_ones_it_overtook() -> None:
    """Dragging a slider forty times must not compute forty spectra.

    The job already running cannot be interrupted, so it finishes -- but its
    answer is to a question nobody is holding any more, and it is dropped.  The
    one waiting behind it is simply replaced and never runs at all.
    """
    delivered: list[object] = []
    started, release = threading.Event(), threading.Event()

    def slow() -> str:
        started.set()
        release.wait(5.0)
        return "slow"

    runner = Runner(lambda _id, _ok, payload: delivered.append(payload))
    try:
        runner.submit(slow)
        assert started.wait(5.0)
        runner.submit(lambda: "displaced")
        runner.submit(lambda: "latest")
        release.set()
        assert runner.wait_idle(5.0)
    finally:
        runner.close()

    assert delivered == ["latest"]


def test_jobs_submitted_in_a_row_all_arrive_when_none_overtakes() -> None:
    delivered: list[object] = []
    runner = Runner(lambda _id, _ok, payload: delivered.append(payload))
    try:
        for n in range(5):
            runner.submit(lambda n=n: n)
            assert runner.wait_idle(5.0)
    finally:
        runner.close()
    assert delivered == [0, 1, 2, 3, 4]


def test_closing_is_idempotent_and_shuts_the_door() -> None:
    runner = Runner(lambda *_: None)
    runner.close()
    runner.close()
    assert not runner.busy
    with pytest.raises(RuntimeError, match="closed"):
        runner.submit(lambda: None)


def test_closing_while_a_job_runs_does_not_hang() -> None:
    release = threading.Event()
    runner = Runner(lambda *_: None)
    runner.submit(lambda: release.wait(5.0))
    time.sleep(0.05)
    release.set()
    runner.close()
    assert not runner.busy


# --------------------------------------------------------------------------
# the T-duality panel: the physics, with no window anywhere
# --------------------------------------------------------------------------


def test_the_spectrum_at_the_self_dual_radius() -> None:
    r"""Eight extra massless states, four of them the ``SU(2)`` gauge bosons."""
    result = TDUALITY.compute(radius=1.0)
    assert result.dual_radius == pytest.approx(1.0)
    assert len(result.extra_here) == 8
    assert result.oscillator_extras == 4
    charges = sorted((s.n, s.w) for s in result.extra_here)
    assert charges == [(-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 2), (1, -1), (1, 1), (2, 0)]


@pytest.mark.parametrize("radius", [0.25, 0.5, 0.83, 1.0, 1.7, 2.0, 3.6])
def test_the_two_spectra_agree_at_every_radius(radius: float) -> None:
    """The panel's own residual, and the package's predicate, from two routes.

    :func:`~stringsim.compactification.circle.spectrum_is_t_dual` decides the
    question itself; the panel decides it again by sorting both enumerations
    and subtracting.  Neither uses the other.
    """
    result = TDUALITY.compute(radius=radius)
    assert result.residual < 1e-12
    assert result.is_dual
    assert result.lightest_here == pytest.approx(result.lightest_there)
    assert len(result.extra_here) == len(result.extra_there)


def test_the_extra_massless_states_at_half_and_at_two_are_each_other_s_image() -> None:
    r"""Away from the self-dual radius the tachyon tower still crosses zero.

    At ``R = 1/2`` it does so through momentum, ``(n, w) = (\pm 1, 0)``; at
    ``R = 2`` through winding, ``(0, \pm 1)``.  T-duality exchanges the two, and
    the panel finds both by enumeration rather than being told where to look --
    which is why the readout reports the count rather than asserting a rule
    about the self-dual radius that would have been wrong here.
    """
    half = TDUALITY.compute(radius=0.5)
    twice = TDUALITY.compute(radius=2.0)

    assert sorted((s.n, s.w) for s in half.extra_here) == [(-1, 0), (1, 0)]
    assert sorted((s.n, s.w) for s in twice.extra_here) == [(0, -1), (0, 1)]
    assert half.oscillator_extras == twice.oscillator_extras == 0
    assert sorted((s.n, s.w) for s in half.extra_there) == sorted(
        (s.n, s.w) for s in twice.extra_here
    )


def test_a_wider_truncation_finds_more_states_and_changes_no_verdict() -> None:
    narrow = TDUALITY.compute(radius=1.0, n_max=1, w_max=1, level_max=1)
    wide = TDUALITY.compute(radius=1.0, n_max=4, w_max=4, level_max=3)
    assert wide.states > narrow.states
    assert narrow.is_dual and wide.is_dual
    assert narrow.residual < 1e-12 and wide.residual < 1e-12


def test_the_readout_carries_checks_and_they_all_agree() -> None:
    lines = TDUALITY.readout(TDUALITY.compute(radius=1.3))
    checks = [line for line in lines if line.is_check]
    assert len(checks) >= 3
    assert all(line.ok for line in checks)
    assert any("spectrum_is_t_dual" in line.check for line in lines)


def test_the_panel_draws_the_package_s_own_figure_with_the_radii_marked() -> None:
    """Three tower lines from ``plot_tduality``, plus ``R`` and ``alpha'/R``."""
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    figure = TDUALITY.draw(TDUALITY.compute(radius=1.6))
    assert isinstance(figure, Figure)
    assert len(figure.axes[0].lines) == 5
    assert plt.get_fignums() == before, "the panel must hand back a detached figure"


# --------------------------------------------------------------------------
# the window itself
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def interpreter():
    """One Tcl interpreter for the whole file.

    Starting and stopping a ``Tk()`` per test fails intermittently on Windows
    with a "usable init.tcl" error -- which the fixture below would report as
    though there were no display, turning a real test into one that sometimes
    silently does not run.  Starting the interpreter once removes that.
    """
    tk = pytest.importorskip("tkinter")
    try:
        window = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"tkinter could not start a window: {exc}")
    window.withdraw()
    yield window
    window.destroy()


@pytest.fixture
def root(interpreter):
    """A fresh, empty window per test, over that one interpreter."""
    import tkinter as tk

    window = tk.Toplevel(interpreter)
    window.withdraw()
    yield window
    window.destroy()


def test_the_window_builds_and_every_panel_computes_draws_and_reports(root) -> None:
    """One walk over the registry, which is where a new panel gets caught.

    ``_drain`` is called directly rather than waited for: the app polls on a
    timer, and a test should not have to sleep through it.
    """
    from stringsim.gui.app import App

    app = App(root)
    try:
        for index, panel in enumerate(app.panels):
            app._select(index)
            root.update_idletasks()
            assert app.runner.wait_idle(60.0), f"{panel.title} never finished"
            app._drain()

            assert app._title.cget("text") == panel.title
            assert app._canvas is not None
            assert set(app._controls.values()) == {c.name for c in panel.controls}
            assert "check" in app._status.cget("text")
            assert app._readout.get("1.0", "end").strip()
    finally:
        app.close()


def test_moving_a_control_redraws_with_the_new_value(root) -> None:
    """Selected by title rather than by position, since the order is editorial."""
    from stringsim.gui.app import App

    app = App(root)
    try:
        index = next(
            i for i, panel in enumerate(app.panels) if panel is TDUALITY_IN_REGISTRY
        )
        app._select(index)
        assert app.runner.wait_idle(60.0)
        app._drain()
        first = app._readout.get("1.0", "end")

        app._controls.set("radius", 2.5)
        app._request()  # the debounce timer is what the widget would trigger
        assert app.runner.wait_idle(60.0)
        app._drain()

        assert app._readout.get("1.0", "end") != first
        assert "2.5000" in app._readout.get("1.0", "end")
    finally:
        app.close()


def test_a_panel_that_raises_is_reported_rather_than_fatal(root) -> None:
    from stringsim.gui.app import App

    class Broken:
        title = "broken"
        blurb = "a panel that does not work"
        controls = ()

        def compute(self):
            raise ValueError("deliberately broken")

        def draw(self, result):  # pragma: no cover - never reached
            raise AssertionError

        def readout(self, result):  # pragma: no cover - never reached
            raise AssertionError

    app = App(root, (Broken(),))
    try:
        assert app.runner.wait_idle(10.0)
        app._drain()
        assert "deliberately broken" in app._status.cget("text")
        assert "ValueError" in app._readout.get("1.0", "end")
    finally:
        app.close()


def test_an_empty_registry_is_refused(root) -> None:
    from stringsim.gui.app import App

    with pytest.raises(ValueError, match="no panels"):
        App(root, ())


def test_a_logarithmic_slider_is_symmetric_about_the_self_dual_radius(root) -> None:
    """Which is the whole reason the panel asks for one.

    Under ``R -> alpha'/R`` a radius and its dual are reflections of each other,
    and on a linear track they are not: over ``[0.25, 4]`` the fixed point sits
    a fifth of the way along.  In the exponent it sits in the middle, and the
    two dual radii are equidistant from it.
    """
    from stringsim.gui.widgets import ControlBar

    control = Slider("radius", "R", 0.25, 4.0, 1.0, step=0.01, log=True)
    bar = ControlBar(root, (control,), lambda: None)

    positions = {}
    for radius in (0.5, 1.0, 2.0):
        bar.set("radius", radius)
        positions[radius] = bar._vars["radius"].get()
        assert bar.values()["radius"] == pytest.approx(radius)

    assert positions[1.0] == pytest.approx((positions[0.5] + positions[2.0]) / 2)
    bar.destroy()


# --------------------------------------------------------------------------
# D-branes pulled apart
# --------------------------------------------------------------------------

BRANES = BranePanel()


def test_coincident_branes_carry_u_of_n() -> None:
    """``N^2`` massless vectors, counted by grouping and by walking pairs."""
    result = BRANES.compute(separation=0.0, branes=4)
    assert result.group == "U(4)"
    assert result.massless_grouped == result.massless_paired == 16
    assert result.stretched == 0


def test_pulling_one_away_breaks_the_group() -> None:
    r""":math:`U(4) \to U(3) \times U(1)`, and ten vectors survive."""
    result = BRANES.compute(separation=3.0, branes=4)
    assert result.group == "U(3) x U(1)"
    assert result.massless_grouped == result.massless_paired == 3**2 + 1**2
    assert result.stretched == 16 - 10


@pytest.mark.parametrize("branes", [2, 3, 5])
@pytest.mark.parametrize("separation", [0.0, 1.5, 9.0])
def test_the_two_counts_of_massless_vectors_always_agree(branes, separation) -> None:
    """One sums squares over stacks; the other asks every ordered pair."""
    result = BRANES.compute(separation=separation, branes=branes)
    assert result.massless_grouped == result.massless_paired


def test_the_tachyon_survives_until_two_pi_root_alpha_prime() -> None:
    r"""The closed form against a bisection that knows no closed form.

    ``stretched_spectrum`` returns numbers.  Bisecting its level-0 mass finds
    the crossing without ever being told where it is, and
    :func:`~stringsim.branes.dbrane.tachyon_free_separation` derives it by hand.
    """
    result = BRANES.compute(separation=1.0)
    assert result.threshold_closed == pytest.approx(2 * math.pi)
    assert result.threshold_found == pytest.approx(result.threshold_closed, abs=1e-9)
    assert result.tachyonic
    assert not BRANES.compute(separation=7.0).tachyonic


def test_the_brane_readout_agrees_with_itself() -> None:
    lines = BRANES.readout(BRANES.compute(separation=2.5, branes=3))
    checks = [line for line in lines if line.is_check]
    assert len(checks) >= 2
    assert all(line.ok for line in checks)


# --------------------------------------------------------------------------
# (p,q) strings
# --------------------------------------------------------------------------

PQ = PQPanel()


@pytest.mark.parametrize(("p", "q"), [(1, 0), (0, 1), (1, 1), (2, -3), (-1, 2)])
@pytest.mark.parametrize("coupling", [0.05, 0.5, 1.0, 3.7])
def test_the_tension_from_ten_dimensions_and_from_eleven(p, q, coupling) -> None:
    """One evaluates ``|p + q tau| / 2 pi alpha'``; the other wraps an M2-brane."""
    result = PQ.compute(coupling=coupling, p=p, q=q)
    assert result.membrane_residual < 1e-12
    assert result.tension == pytest.approx(result.membrane)


@pytest.mark.parametrize(("p", "q"), [(1, 0), (0, 1), (3, 2)])
def test_sl2z_moves_the_charges_and_leaves_the_tension(p, q) -> None:
    result = PQ.compute(coupling=0.7, axion=0.3, p=p, q=q)
    assert result.s_residual < 1e-12
    assert result.t_residual < 1e-12


def test_s_exchanges_the_fundamental_string_and_the_d1() -> None:
    assert PQ.compute(p=1, q=0).s_charges == (0, 1)
    assert PQ.compute(p=0, q=1).s_charges == (-1, 0)


@pytest.mark.parametrize("coupling", [0.2, 1.0, 4.0])
def test_the_junction_balances_because_the_charges_do(coupling) -> None:
    """No angle is imposed anywhere; the net force is what comes out."""
    result = PQ.compute(coupling=coupling, axion=0.25)
    assert result.junction_charge == (0, 0)
    assert result.junction_force < 1e-12


def test_the_d1_matches_dp_brane_tension_only_without_an_axion() -> None:
    r"""``|tau| = 1/g_s`` when ``C_0 = 0`` and not otherwise, and the panel says so."""
    zero = PQ.compute(coupling=0.4, axion=0.0)
    assert zero.axion_is_zero
    assert zero.d1_residual < 1e-12
    assert zero.d1_tension == pytest.approx(zero.d1_from_dbrane)

    tilted = PQ.compute(coupling=0.4, axion=0.5)
    assert not tilted.axion_is_zero
    assert tilted.d1_residual > 1e-3
    # The line for it carries no verdict rather than a failing one, because the
    # equality was never claimed away from C_0 = 0.
    line = PQ.readout(tilted)[-1]
    assert not line.is_check and "only at C_0 = 0" in line.value


def test_a_charge_with_a_common_factor_sits_at_threshold() -> None:
    """``(2,2)`` is two ``(1,1)`` strings, so nothing is gained by binding."""
    single = PQ.compute(p=1, q=1)
    doubled = PQ.compute(p=2, q=2)
    assert single.primitive and not doubled.primitive
    assert single.binding > 0.0
    assert doubled.binding == pytest.approx(0.0, abs=1e-12)


def test_a_string_needs_a_charge() -> None:
    with pytest.raises(ValueError, match="not a string"):
        PQ.compute(p=0, q=0)


# --------------------------------------------------------------------------
# the Veneziano amplitude
# --------------------------------------------------------------------------

VENEZIANO = VenezianoPanel()


def test_the_poles_sit_on_the_mass_levels_at_the_string_s_intercept() -> None:
    r"""Two modules, no shared code, and the same list of numbers.

    The amplitude's poles come from where a gamma function's argument reaches a
    non-positive integer; the levels come from ``alpha' M^2 = N - (D-2)/24``.
    """
    result = VENEZIANO.compute(intercept=1.0)
    assert result.string_intercept == pytest.approx(1.0)
    assert result.pole_offset < 1e-12
    assert list(result.poles) == pytest.approx(list(result.levels))


@pytest.mark.parametrize("intercept", [0.55, 0.8, 1.2, 1.45])
def test_moving_the_intercept_parts_them_by_exactly_the_shift(intercept) -> None:
    """And the amplitude itself is untouched -- which is the point of the panel."""
    result = VENEZIANO.compute(intercept=intercept)
    assert result.pole_offset == pytest.approx(abs(1.0 - intercept))
    assert result.crossing < 1e-12
    check = next(
        line for line in VENEZIANO.readout(result) if "poles against" in line.label
    )
    assert check.ok is False


@pytest.mark.parametrize("intercept", [0.55, 1.0, 1.2])
@pytest.mark.parametrize("mandelstam_t", [-0.35, -1.4, -2.6])
def test_the_residues_from_a_limit_and_from_the_closed_form(intercept, mandelstam_t) -> None:
    r"""A two-sided numerical limit against ``-prod (alpha_t + k) / n!``."""
    result = VENEZIANO.compute(intercept=intercept, mandelstam_t=mandelstam_t)
    if result.t_on_shell:  # pragma: no cover - only for special (a, t) pairs
        pytest.skip("alpha(t) is an integer here, so A is singular for every s")
    assert result.residues
    assert result.residue_error < 1e-6


def test_a_momentum_transfer_on_a_resonance_is_named_rather_than_failed() -> None:
    r"""``alpha(t) = 1`` at ``a = 1.35``, ``alpha' t = -0.35``: reachable from the sliders.

    ``A`` is then singular at every ``s`` and the residue limit has nothing to
    converge to.  That is a fact about the configuration rather than a failure,
    so the line carries no verdict and says which fact it is.
    """
    result = VENEZIANO.compute(intercept=1.35, mandelstam_t=-0.35)
    assert result.t_on_shell
    assert result.residues == ()
    line = next(line for line in VENEZIANO.readout(result) if line.label == "residues")
    assert not line.is_check
    assert "non-negative integer" in line.check


def test_the_veneziano_figure_carries_both_sets_of_marks() -> None:
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    figure = VENEZIANO.draw(VENEZIANO.compute(intercept=0.8))
    assert isinstance(figure, Figure)
    assert "off the spectrum" in figure.axes[0].get_title()
    assert plt.get_fignums() == before


# --------------------------------------------------------------------------
# the commentary
# --------------------------------------------------------------------------


def test_every_panel_carries_prose_of_all_three_kinds() -> None:
    """Static background, things to try, and notes about the current result."""
    for panel in REGISTRY:
        assert background_of(panel), panel.title
        assert suggestions_of(panel), panel.title
        assert notes_of(panel, panel.compute()), panel.title


def test_the_prose_is_paragraphs_rather_than_fragments() -> None:
    """Each piece is a sentence or more, so the notes read rather than list."""
    for panel in REGISTRY:
        for paragraph in (
            *background_of(panel),
            *suggestions_of(panel),
            *notes_of(panel, panel.compute()),
        ):
            # No rule about the first character: a paragraph may open with a
            # charge, "(1, 1) has coprime charges", or with a symbol,
            # "alpha(t) = 1 is a non-negative integer here".
            assert len(paragraph) > 60, (panel.title, paragraph)
            assert paragraph.rstrip().endswith((".", "!", "?")), paragraph


def test_the_prose_carries_no_markup_the_window_cannot_render() -> None:
    """A Tk text widget shows ``*emphasis*`` and backticks literally.

    A lone star is not markup: ``Delta*`` is what the dual of a lattice
    polytope is called, and forbidding the character outright would ban the
    notation along with the markdown.  What is looked for is a *pair* around a
    word, which is the only form that would read as an artifact on screen.
    """
    emphasis = re.compile(r"\*[A-Za-z][A-Za-z ]{0,30}\*")
    for panel in REGISTRY:
        for paragraph in (
            *background_of(panel),
            *suggestions_of(panel),
            *notes_of(panel, panel.compute()),
        ):
            assert "`" not in paragraph, (panel.title, paragraph)
            assert not emphasis.search(paragraph), (panel.title, paragraph)


def test_a_panel_without_commentary_is_still_a_panel() -> None:
    """All three pieces are optional, and their absence is not an error."""

    class Bare:
        title = "bare"
        blurb = "no commentary at all"
        controls = ()

    assert background_of(Bare()) == ()
    assert suggestions_of(Bare()) == ()
    assert notes_of(Bare(), None) == []


def test_the_notes_change_with_the_radius() -> None:
    """The one part of the commentary a static page could not have written."""
    at_self_dual = " ".join(TDUALITY.notes(TDUALITY.compute(radius=1.0)))
    at_half = " ".join(TDUALITY.notes(TDUALITY.compute(radius=0.5)))
    generic = " ".join(TDUALITY.notes(TDUALITY.compute(radius=1.7)))

    assert "self-dual radius" in at_self_dual and "SU(2)" in at_self_dual
    assert "not the self-dual radius" in at_half and "tachyon tower" in at_half
    assert "generic radius" in generic
    assert at_self_dual != at_half != generic


def test_the_notes_name_which_tower_supplies_the_first_excitation() -> None:
    r"""A momentum mode above :math:`\sqrt{\alpha'}` and a winding mode below."""
    assert "momentum mode" in " ".join(TDUALITY.notes(TDUALITY.compute(radius=3.0)))
    assert "winding mode" in " ".join(TDUALITY.notes(TDUALITY.compute(radius=0.4)))


def test_the_veneziano_notes_say_what_did_not_break() -> None:
    """Moving the intercept is meant to be read, not merely seen to go red."""
    moved = " ".join(VENEZIANO.notes(VENEZIANO.compute(intercept=0.7)))
    assert "did not break" in moved
    assert "crossing" in moved

    on_shell = " ".join(
        VENEZIANO.notes(VENEZIANO.compute(intercept=1.35, mandelstam_t=-0.35))
    )
    assert "not a failure" in on_shell


def test_the_brane_notes_follow_the_stack_apart() -> None:
    together = " ".join(BRANES.notes(BRANES.compute(separation=0.0, branes=4)))
    apart = " ".join(BRANES.notes(BRANES.compute(separation=8.0, branes=4)))
    assert "All 4 branes are together" in together
    assert "still tachyonic" in " ".join(BRANES.notes(BRANES.compute(separation=1.0)))
    assert "stable one" in apart


def test_the_pq_notes_follow_the_coupling_across_one() -> None:
    weak = " ".join(PQ.notes(PQ.compute(coupling=0.2)))
    strong = " ".join(PQ.notes(PQ.compute(coupling=4.0)))
    fixed = " ".join(PQ.notes(PQ.compute(coupling=1.0)))
    assert "weakly coupled" in weak
    assert "lighter object" in strong
    assert "fixed point of S" in fixed


# --------------------------------------------------------------------------
# D = 26, from two sides
# --------------------------------------------------------------------------

CRITICAL = CriticalDimensionPanel()


def test_three_routes_to_twenty_six() -> None:
    """Norms, counting, and a conformal anomaly, with no code in common.

    The first reads the signature of a Gram matrix, the second compares a state
    count against the light cone's, and the third is ``c = D - 26``.  None of
    them mentions the others.
    """
    result = CRITICAL.compute()
    assert result.from_norms == 26
    assert result.from_counting == 26
    assert result.from_anomaly == 26
    assert result.bounds_meet


def test_the_intercept_comes_from_zeta_rather_than_from_a_formula() -> None:
    r"""``a = -(D-2)/2 * zeta(-1)`` against ``(D-2)/24``, at ``D = 26``."""
    result = CRITICAL.compute()
    assert result.zeta_intercept == pytest.approx(1.0, abs=1e-6)
    assert result.string_intercept == pytest.approx(1.0)
    assert result.zeta_intercept == pytest.approx(result.string_intercept, abs=1e-6)


def test_the_central_charge_from_two_different_commutators() -> None:
    r""":math:`[L_2, L_{-2}]` and :math:`[L_3, L_{-3}]` both give ``c = D``."""
    result = CRITICAL.compute()
    assert result.measured_c == pytest.approx(26.0)
    assert result.measured_c_other == pytest.approx(26.0)


def test_at_twenty_six_the_survivors_are_the_light_cone_s() -> None:
    """324 positive, 26 null, none negative -- and 324 is what the light cone has."""
    result = CRITICAL.compute()
    here = result.at(26)
    assert (here.positive, here.zero, here.negative) == (324, 26, 0)
    assert here.positive == 324 == lightcone_count(2, 26)
    assert abs(here.smallest) < 1e-9


def test_level_one_cannot_see_the_critical_dimension() -> None:
    """So its two bound lines carry no verdict rather than a failing one.

    Level 1's physical states are the ``D - 1`` polarisations with one null
    direction, in every dimension.  Nothing there disagrees with 26; nothing
    there has an opinion about it, and a red mark would claim otherwise.
    """
    result = CRITICAL.compute(level=1)
    assert result.from_norms == 30  # the top of the scan, not a bound
    assert result.from_counting == 4  # the bottom of it
    assert not result.bounds_meet

    bounds = [
        line for line in CRITICAL.readout(result)
        if line.label in ("largest ghost-free D", "smallest D matching the light cone")
    ]
    assert len(bounds) == 2
    assert all(not line.is_check for line in bounds)
    assert "level 1" in " ".join(line.check for line in bounds)
    assert "no upper bound to find" in " ".join(CRITICAL.notes(result))


@pytest.mark.parametrize("intercept", [0.9, 1.1])
def test_moving_the_intercept_parts_the_two_bounds(intercept: float) -> None:
    r"""``a = 1`` and ``D = 26`` are one statement, so neither survives alone."""
    result = CRITICAL.compute(intercept=intercept)
    assert not result.bounds_meet
    assert result.from_norms != 26 or result.from_counting != 26

    bound = next(
        line for line in CRITICAL.readout(result) if line.label == "largest ghost-free D"
    )
    assert bound.ok is False


def test_the_notes_do_not_claim_an_equality_that_has_stopped_holding() -> None:
    """The prose is conditional where the numbers are.

    At ``a = 0.9`` the surviving count at ``D = 26`` is 349 and the light cone
    still has 324, and the smallest norm is ``-1.5e-2`` rather than zero.  A
    paragraph written for the good case would assert both equalities anyway.
    """
    good = " ".join(CRITICAL.notes(CRITICAL.compute()))
    bad = " ".join(CRITICAL.notes(CRITICAL.compute(intercept=0.9)))

    assert "is exactly the light cone's" in good
    assert "zero to machine precision" in good

    assert "is not the light cone's" in bad
    assert "not zero and not small" in bad
    assert "is exactly the light cone's" not in bad


def test_setting_a_control_a_panel_does_not_have_says_so(root) -> None:
    """A bare StopIteration from a generator is not a usable error message."""
    from stringsim.gui.widgets import ControlBar

    bar = ControlBar(root, TDUALITY.controls, lambda: None)
    with pytest.raises(KeyError, match="no control named 'coupling'"):
        bar.set("coupling", 1.0)
    bar.destroy()


# --------------------------------------------------------------------------
# the Hagedorn temperature
# --------------------------------------------------------------------------

HAGEDORN = HagedornPanel()


def test_the_fitted_slope_approaches_cardy_s_closed_form() -> None:
    r"""Counted partitions against :math:`2\pi\sqrt{c/6}`, which is ``4 pi`` at 24."""
    result = HAGEDORN.compute()
    assert result.predicted == pytest.approx(4 * math.pi)
    assert result.gap < 0.01
    assert result.fit.beta_hagedorn < result.predicted  # it approaches from below


def test_the_gap_closes_when_more_levels_are_counted() -> None:
    """The check that means something, since the two are not meant to be equal.

    A fit over a finite window cannot reach the limit -- there is a subleading
    ``log N`` -- so the panel fits twice and asks whether the distance shrank.
    A wrong closed form would not be approached.
    """
    for n_max in (120, 240, 400):
        result = HAGEDORN.compute(n_max=n_max, n_fit=min(100, n_max // 2))
        assert result.converging
        assert result.gap < result.coarse_gap


def test_more_levels_gets_closer() -> None:
    small = HAGEDORN.compute(n_max=120, n_fit=60)
    large = HAGEDORN.compute(n_max=400, n_fit=60)
    assert large.gap < small.gap


def test_the_agreement_is_not_about_the_number_twenty_four() -> None:
    """Both routes move together when the number of species changes."""
    for species in (8, 12, 16, 24):
        result = HAGEDORN.compute(n_species=species)
        assert result.predicted == pytest.approx(2 * math.pi * math.sqrt(species / 6))
        assert result.gap < 0.01


def test_the_fitted_slope_line_carries_no_verdict() -> None:
    """It is not supposed to equal the closed form at a finite truncation.

    Marking it pass or fail against a tolerance would report how many levels
    were counted, dressed up as a statement about the physics.
    """
    lines = HAGEDORN.readout(HAGEDORN.compute(n_max=60, n_fit=50))
    slope = next(line for line in lines if line.label == "beta_H")
    assert not slope.is_check

    checks = [line for line in lines if line.is_check]
    assert len(checks) == 1
    assert "the gap closes" in checks[0].check
    assert checks[0].ok


def test_a_coarse_truncation_says_it_is_coarse() -> None:
    assert "long way from converged" in " ".join(
        HAGEDORN.notes(HAGEDORN.compute(n_max=60, n_fit=50))
    )
    assert "long way from converged" not in " ".join(
        HAGEDORN.notes(HAGEDORN.compute(n_max=400, n_fit=150))
    )


# --------------------------------------------------------------------------
# the Myers effect
# --------------------------------------------------------------------------

MYERS = MyersPanel()


@pytest.mark.parametrize("total", [2, 4, 6, 9])
def test_the_single_block_is_the_cheapest_configuration(total: int) -> None:
    """Every partition of N is evaluated; one big sphere beats all of them."""
    result = MYERS.compute(total=total)
    assert result.sphere_wins
    assert result.winner.partition == (total,)
    assert result.commuting.energy == pytest.approx(0.0)
    assert result.depth > 0.0


@pytest.mark.parametrize("total", [2, 5, 7])
def test_a_shrunk_d2_with_n_flux_quanta_weighs_n_d0_branes(total: int) -> None:
    r""":math:`4\pi^2\alpha' T_2 = T_0`, with nothing fitted to make it so.

    One side is a Born-Infeld energy at zero radius, the other is a D0-brane
    tension times ``N``.  They agree because the tension formula has no freedom
    in it, which is why a matrix model and a wrapped brane describe one object.
    """
    result = MYERS.compute(total=total)
    assert result.shrunk_energy == pytest.approx(result.d0_energy)
    assert result.shrunk_energy == pytest.approx(float(total))


@pytest.mark.parametrize("total", [2, 3, 6, 9])
def test_the_discretisation_error_is_exact(total: int) -> None:
    r""":math:`\mathrm{Tr}\,J^2 / (N^3/4) = 1 - 1/N^2`, not approximately."""
    result = MYERS.compute(total=total)
    assert result.ratio == pytest.approx(1.0 - 1.0 / total**2, abs=1e-15)
    assert result.ratio == pytest.approx(result.predicted_ratio, abs=1e-15)


@pytest.mark.parametrize("flux", [0.1, 1.0, 3.0])
def test_the_fuzzy_sphere_is_a_critical_point_of_the_potential(flux: float) -> None:
    """Checked rather than declared: the gradient there, and the algebra closing."""
    result = MYERS.compute(flux=flux)
    assert result.gradient < 1e-9
    assert result.algebra < 1e-12


def test_every_myers_check_agrees_across_the_controls() -> None:
    for total in (2, 5, 8):
        for flux in (0.1, 1.0, 2.5):
            lines = MYERS.readout(MYERS.compute(total=total, flux=flux))
            assert all(line.ok for line in lines if line.is_check), (total, flux)


# --------------------------------------------------------------------------
# mirror symmetry
# --------------------------------------------------------------------------

MIRROR = MirrorPanel()


def test_the_quintic_and_its_mirror() -> None:
    """``(1, 101)`` one way and ``(101, 1)`` the other, with chi opposite."""
    here = MIRROR.compute(label="P(1,1,1,1,1)[5]", side="the polytope")
    there = MIRROR.compute(label="P(1,1,1,1,1)[5]", side="its dual")

    assert (here.numbers.h11, here.numbers.h21) == (1, 101)
    assert (there.numbers.h11, there.numbers.h21) == (101, 1)
    assert here.swapped and there.swapped
    assert here.numbers.euler == -there.numbers.euler == -200


def test_the_quintic_euler_from_a_second_route() -> None:
    r"""``int c_3`` over the hypersurface, with no polytope in the computation."""
    result = MIRROR.compute(label="P(1,1,1,1,1)[5]")
    assert result.has_chern_check
    assert result.chern_euler == -200
    assert result.euler_agrees


@pytest.mark.parametrize("label", [name for name in FAMILIES if name != "P(1,1,1,1,1)[5]"])
def test_the_chern_route_is_refused_where_it_does_not_apply(label: str) -> None:
    """It is a formula for a smooth hypersurface in *ordinary* projective space.

    Calabi-Yau there means degree = n + 1, which only the quintic satisfies
    among these.  The rest sit in singular weighted spaces and the manifold is a
    resolution.  Running the formula anyway compares ``-204`` against ``-516``
    and blames Batyrev's count for the mismatch, so the panel does not run it.
    """
    result = MIRROR.compute(label=label)
    assert not result.has_chern_check
    assert result.chern_euler is None

    line = next(
        item for item in MIRROR.readout(result) if item.label == "Euler characteristic"
    )
    assert not line.is_check
    assert "the ambient space is weighted" in line.check
    assert "does not manufacture one" in " ".join(MIRROR.notes(result))


@pytest.mark.parametrize(
    ("label", "h11", "h21"),
    [
        ("P(1,1,1,1,1)[5]", 1, 101),
        ("P(1,1,1,1,2)[6]", 1, 103),
        ("P(1,1,1,1,4)[8]", 1, 149),
        ("P(1,1,1,2,5)[10]", 1, 145),
        ("P(1,1,1,6,9)[18]", 2, 272),
    ],
)
def test_the_famous_families(label: str, h11: int, h21: int) -> None:
    """Nothing in the computation knows these numbers; it counts lattice points."""
    result = MIRROR.compute(label=label)
    assert (result.numbers.h11, result.numbers.h21) == (h11, h21)
    assert (result.mirror.h11, result.mirror.h21) == (h21, h11)


def test_the_greene_plesser_scalings_are_enumerated() -> None:
    r"""The smallest exponent is right for the quintic and wrong for ``P(1,1,1,2,5)``."""
    quintic = MIRROR.compute(label="P(1,1,1,1,1)[5]")
    assert (quintic.phases, quintic.scalings, quintic.group_order) == (5**4, 5, 125)

    other = MIRROR.compute(label="P(1,1,1,2,5)[10]")
    assert other.scalings == 10  # not min(10, 10, 10, 5, 2) = 2
    assert other.group_order == 100


def test_the_cloud_holds_each_family_once() -> None:
    """The plot draws a point and its mirror itself, so passing both doubles it.

    Every marker was being drawn four times before this was noticed: twice by
    the scan and twice again by ``plot_mirror_hodge``, which reflects whatever
    it is given.
    """
    cloud = MIRROR.compute().cloud
    assert cloud
    assert len(set(cloud)) == len(cloud)
    # h11 is small and h21 large for these families, so the unswapped
    # orientation is the one present.
    assert all(a <= b for a, b in cloud)
    assert (1, 101) in cloud and (101, 1) not in cloud


# --------------------------------------------------------------------------
# orbifold fixed points
# --------------------------------------------------------------------------

ORBIFOLD = OrbifoldPanel()


@pytest.mark.parametrize(
    ("label", "sector", "points"),
    [
        ("T^2/Z_2", 1, 4),
        ("T^2/Z_3 hexagonal", 1, 3),
        ("T^2/Z_3 hexagonal", 2, 3),
        ("T^2/Z_4 square", 1, 2),
        ("T^2/Z_4 square", 2, 4),
        ("T^2/Z_6 hexagonal", 1, 1),
        ("T^2/Z_6 hexagonal", 2, 3),
        ("T^2/Z_6 hexagonal", 3, 4),
    ],
)
def test_the_fixed_points_are_counted_and_then_found(label, sector, points) -> None:
    r"""``|det(1 - theta^k)|`` against an enumeration of lattice positions.

    ``theta^2`` on the square lattice is the inversion, which is why ``Z_4``
    has two fixed points at ``k = 1`` and four at ``k = 2``.
    """
    result = ORBIFOLD.compute(label=label, sector=sector)
    assert result.fixed_from_determinant == points
    assert result.fixed_from_enumeration == points
    assert result.fixed_points_agree


@pytest.mark.parametrize("label", list(ORBIFOLDS))
def test_the_twisted_intercept_from_a_closed_form_and_from_a_mode_sum(label) -> None:
    r"""``1 - (1/4) sum phi(1-phi)`` against a measured :math:`\zeta(-1,\phi)`.

    The second route fits the small-``eps`` expansion of
    ``sum (n+phi) exp(-eps(n+phi))`` and reads the constant term off it.  It
    evaluates no closed form for the intercept anywhere.
    """
    result = ORBIFOLD.compute(label=label)
    assert result.intercept_gap < 1e-6
    assert result.intercept == pytest.approx(result.intercept_from_zeta, abs=1e-6)
    # Loose enough to be the numerical route's precision and no looser.
    assert result.intercept_gap > 0.0


def test_the_z2_intercept_is_seven_eighths() -> None:
    r"""Two directions at ``phi = 1/2``: ``a = 1 - 2/16 = 7/8``."""
    result = ORBIFOLD.compute(label="T^2/Z_2")
    assert result.phases == pytest.approx((0.5, 0.5))
    assert result.intercept == pytest.approx(0.875)


def test_a_sector_the_orbifold_does_not_have_is_named_rather_than_refused() -> None:
    """Two independent controls can reach ``Z_2`` with ``k = 3``; one of them wins.

    Refusing would put a traceback on screen for a combination a reader can
    produce by moving one spinbox, so the panel uses the nearest sector that
    exists and says in both the readout and the notes that it did.
    """
    result = ORBIFOLD.compute(label="T^2/Z_2", sector=4)
    assert result.requested_sector == 4
    assert result.sector == 1
    assert result.sector_was_clamped
    assert "does not exist here" in ORBIFOLD.readout(result)[0].check
    assert "There is no sector k = 4" in " ".join(ORBIFOLD.notes(result))

    plain = ORBIFOLD.compute(label="T^2/Z_6 hexagonal", sector=4)
    assert not plain.sector_was_clamped
    assert "There is no sector" not in " ".join(ORBIFOLD.notes(plain))


@pytest.mark.parametrize("label", list(ORBIFOLDS))
def test_the_projection_leaves_a_whole_number_of_states(label) -> None:
    """Not automatic: a sign or conjugation error breaks exactly this.

    ``untwisted_degeneracy`` raises rather than rounding if the character sum
    comes out non-integral, so reaching a number at all is the check.
    """
    result = ORBIFOLD.compute(label=label)
    assert isinstance(result.untwisted_massless, int)
    assert 0 < result.untwisted_massless < result.torus_massless
    assert result.torus_massless == 24**2


@pytest.mark.parametrize("label", list(ORBIFOLDS))
def test_the_twisted_ground_state_is_tachyonic_in_every_one(label) -> None:
    """Which is the bosonic string's own tachyon, not one the orbifold added."""
    result = ORBIFOLD.compute(label=label)
    ground = result.ground_state
    assert ground is not None
    assert ground.is_tachyonic
    assert ground.alpha_m2 == pytest.approx(-4.0 * result.intercept)
    assert "this is a zero-point energy" in " ".join(ORBIFOLD.notes(result))


def test_every_orbifold_draws_its_cell() -> None:
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    for label in ORBIFOLDS:
        figure = ORBIFOLD.draw(ORBIFOLD.compute(label=label))
        assert isinstance(figure, Figure)
    assert plt.get_fignums() == before


# --------------------------------------------------------------------------
# a string ending on a brane
# --------------------------------------------------------------------------

BION = BIonPanel()


@pytest.mark.parametrize("p", [3, 4, 5, 6])
@pytest.mark.parametrize("n_strings", [1, 3, 6])
def test_the_flux_counts_the_strings_at_every_radius(p: int, n_strings: int) -> None:
    r"""It must not depend on the surface, because :math:`X` is harmonic.

    Measured through spheres of three different radii, so a formula that
    happened to give the right answer at one of them would be caught.
    """
    result = BION.compute(n_strings=n_strings, p=p)
    assert result.flux_error < 1e-9
    assert result.flux_spread < 1e-12
    assert all(value == pytest.approx(n_strings) for value in result.flux)


@pytest.mark.parametrize("p", [3, 4, 5, 6])
@pytest.mark.parametrize("n_strings", [1, 2, 5])
def test_the_spike_weighs_that_many_fundamental_strings(p: int, n_strings: int) -> None:
    r"""``T_p int |grad X|^2`` divided by the height, against ``n / 2 pi alpha'``.

    One side is a numerical integral over the brane; the other is a string
    tension.  The sphere area, the brane tension and the harmonic exponent all
    change with ``p`` and the ratio does not.
    """
    result = BION.compute(n_strings=n_strings, p=p)
    assert result.tension_ratio == pytest.approx(1.0, abs=1e-6)
    assert result.string_tension == pytest.approx(n_strings / (2 * math.pi))


@pytest.mark.parametrize("p", [3, 4, 5, 6])
@pytest.mark.parametrize("slope", [0.05, 0.3, 1.2, 2.0])
def test_the_legendre_transform_lands_on_the_gradient(p: int, slope: float) -> None:
    r"""``D = dL/dE`` by central differences, not by writing ``grad X`` twice.

    At ``2 pi alpha' E = grad X`` the two coincide and the Bogomolny gap
    vanishes, which is what makes the spike BPS.  Differentiating the
    Lagrangian numerically is the whole point: the bound and the configuration
    are then computed from different code.
    """
    result = BION.compute(p=p, slope=slope)
    assert result.displacement == pytest.approx(result.gradient, abs=1e-7)
    assert result.bps
    assert abs(result.gap) < 1e-12
    assert result.energy == pytest.approx(result.bound, abs=1e-12)


def test_the_determinant_two_ways() -> None:
    """The matrix's determinant against the closed form the literature writes."""
    result = BION.compute()
    assert result.determinant_gap < 1e-12
    assert result.determinant_matrix == pytest.approx(result.determinant_closed)


def test_the_probe_field_is_below_critical() -> None:
    """So the determinant shown is one the Lagrangian is real at.

    An earlier version probed at a field past ``E_crit``.  Both routes still
    agreed there -- on a number describing a configuration that does not exist,
    since the square root is imaginary beyond it.
    """
    result = BION.compute()
    assert result.probe_fraction == (0.4, 0.2)
    assert sum(f**2 for f in result.probe_fraction) < 1.0
    assert result.determinant_matrix > 0.0


def test_the_critical_field_is_the_string_tension() -> None:
    r"""``E_crit = 1 / 2 pi alpha'``, which is what a fundamental string weighs."""
    result = BION.compute()
    assert result.critical == pytest.approx(1.0 / (2 * math.pi))
    assert result.critical == pytest.approx(result.string_tension / result.n_strings)


def test_more_strings_widen_the_mouth_rather_than_deepening_it() -> None:
    """``q`` is proportional to ``n``, and every spike is infinitely tall."""
    result = BION.compute(n_strings=4)
    charges = [charge for _, charge in result.profiles]
    assert len(charges) == 4
    for n, charge in enumerate(charges, start=1):
        assert charge == pytest.approx(n * charges[0])


def test_every_bion_check_agrees_across_the_controls() -> None:
    for p in (3, 5):
        for n_strings in (1, 4):
            for slope in (0.1, 1.5):
                lines = BION.readout(
                    BION.compute(n_strings=n_strings, p=p, slope=slope)
                )
                assert all(line.ok for line in lines if line.is_check), (p, n_strings, slope)


# --------------------------------------------------------------------------
# anomaly cancellation
# --------------------------------------------------------------------------

ANOMALY = AnomalyPanel()


def test_only_two_groups_survive_the_scan() -> None:
    """819 candidates, three conditions, and the two heterotic strings left.

    They were not looked for: the scan builds a family and reports what comes
    through it.
    """
    result = ANOMALY.compute()
    assert result.total == 819
    assert set(result.survivors) == {"SO(32)", "E8 x E8"}
    assert result.found_both


def test_four_hundred_and_ninety_six_from_two_unrelated_places() -> None:
    r"""A twelve-form's ``tr R^6`` coefficient, and a count of short lattice vectors.

    One solves for the gauge dimension that kills a pure-gravity term; the
    other counts the 480 roots of an even self-dual rank-16 lattice and adds
    the 16 Cartan directions.
    """
    result = ANOMALY.compute()
    assert result.dimension_from_anomaly == 496
    assert result.lattice_dimension == 496
    assert result.dimensions_agree
    assert {roots for _, roots, _ in result.lattice_dimensions} == {480}
    assert len(result.lattice_dimensions) == 2


def test_passing_the_gravitational_condition_is_not_passing() -> None:
    r"""``SO(26) x SO(19)`` has dimension 496 and still leaves a ``tr F^6``.

    More candidates reach 496 than survive, which is the whole point of the
    picture: the two conditions are separate.
    """
    result = ANOMALY.compute()
    assert result.right_dimension > len(result.survivors)
    assert result.right_dimension == 3


def test_the_factorisation_coefficient_comes_out_rather_than_going_in() -> None:
    r"""``X_4 = tr R^2 + b Tr F^2`` with ``b = 1/30``, for every surviving factor."""
    result = ANOMALY.compute()
    assert result.coefficients
    assert {value for _, value in result.coefficients} == {"1/30"}
    assert len(result.coefficients) == 3  # one for SO(32), two for E8 x E8


@pytest.mark.parametrize(
    ("max_so", "max_factors", "expected", "unreachable"),
    [
        (30, 2, {"E8 x E8"}, ("SO(32)",)),
        (40, 1, {"SO(32)"}, ("E8 x E8",)),
        (30, 1, set(), ("SO(32)", "E8 x E8")),
    ],
)
def test_a_window_that_cannot_hold_them_makes_no_claim(
    max_so, max_factors, expected, unreachable
) -> None:
    """Narrowing the search is not the scan disagreeing with anything.

    ``SO(32)`` needs the scan to reach ``N = 32`` and ``E_8 x E_8`` needs two
    factors.  Outside that the survivors line drops its verdict and names what
    is out of reach, rather than reporting a failure of ten dimensions.
    """
    result = ANOMALY.compute(max_so=max_so, max_factors=max_factors)
    assert set(result.survivors) == expected
    assert not result.window_holds_both
    assert result.out_of_reach == unreachable

    line = next(item for item in ANOMALY.readout(result) if item.label == "survivors")
    assert not line.is_check
    assert "cannot be reached by this window" in line.check
    assert "told to look" in " ".join(ANOMALY.notes(result))


def test_a_window_that_can_hold_them_does_make_a_claim() -> None:
    result = ANOMALY.compute(max_so=32, max_factors=2)
    assert result.window_holds_both
    assert result.out_of_reach == ()
    line = next(item for item in ANOMALY.readout(result) if item.label == "survivors")
    assert line.ok is True


def test_widening_the_scan_finds_nothing_further() -> None:
    """More candidates, the same two survivors -- which is the interesting part."""
    narrow = ANOMALY.compute(max_so=32)
    wide = ANOMALY.compute(max_so=48)
    assert wide.total > narrow.total
    assert set(wide.survivors) == set(narrow.survivors) == {"SO(32)", "E8 x E8"}


def test_the_panel_says_it_is_not_a_uniqueness_proof() -> None:
    """A finite search over one family, and the readout does not let it pass for more."""
    result = ANOMALY.compute()
    line = next(
        item for item in ANOMALY.readout(result) if item.label == "what this is not"
    )
    assert "uniqueness proof" in line.value
    assert "SU(N)" in line.check
    assert "uniqueness proof" in " ".join(
        background_of(ANOMALY)
    )
