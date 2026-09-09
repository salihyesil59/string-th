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
    check_controls,
    defaults,
)
from stringsim.gui.panels.branes import BranePanel  # noqa: E402
from stringsim.gui.panels.pq import PQPanel  # noqa: E402
from stringsim.gui.panels.tduality import TDualityPanel  # noqa: E402
from stringsim.gui.panels.veneziano import VenezianoPanel  # noqa: E402

TDUALITY = TDualityPanel()


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
    from stringsim.gui.app import App

    app = App(root)
    try:
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
