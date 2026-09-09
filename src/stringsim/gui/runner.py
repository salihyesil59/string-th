"""One worker thread, and a new job cancels the one still waiting.

Most of what the package computes is fast enough to run between two frames --
a circle spectrum is microseconds, the quintic's Hodge numbers eleven
milliseconds.  A few things are not: the physical states at level three take
close to three seconds, and a scan over gauge groups takes half of one.  A
window that computed on the calling thread would freeze for those, and one that
computed only the fast panels inline would need two code paths for no reason.

So everything goes through here.  ``submit`` hands a callable to a single
worker; a second ``submit`` before the first is picked up simply *replaces* it,
which is what makes dragging a slider cheap -- forty moves queue one job, not
forty.  A job already running cannot be interrupted, so it finishes, but its
result is dropped if something newer has since been asked for.  What arrives at
the caller is therefore always the answer to the most recent question.

This module knows nothing about tkinter.  It calls ``deliver`` *from the worker
thread*, and it is the caller's business to get from there to wherever the
result is wanted -- for Tk that means ``after``, which is the only thread-safe
way in.  Keeping that knowledge out of here is what lets the whole scheduler be
tested without a display.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

__all__ = ["Runner", "Delivery"]

Delivery = Callable[[int, bool, Any], None]
"""``deliver(job_id, ok, payload)``, called from the worker thread.

``payload`` is the return value when ``ok``, and the exception when not.  A
panel that raises should show its message, not take the window down with it.
"""


class Runner:
    """A single background worker with room for one pending job."""

    def __init__(self, deliver: Delivery, *, name: str = "stringsim-runner") -> None:
        self._deliver = deliver
        self._pending: tuple[int, Callable[[], Any]] | None = None
        self._latest = 0
        self._running: int | None = None
        self._closed = False
        self._idle = threading.Event()
        self._idle.set()
        self._cv = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    # -- public --------------------------------------------------------

    def submit(self, job: Callable[[], Any]) -> int:
        """Queue ``job``, displacing any job not yet started.  Returns its id."""
        with self._cv:
            if self._closed:
                raise RuntimeError("runner is closed")
            self._latest += 1
            self._pending = (self._latest, job)
            self._idle.clear()
            self._cv.notify()
            return self._latest

    @property
    def busy(self) -> bool:
        """True while a job is running or waiting -- what the status line reads."""
        with self._cv:
            return self._pending is not None or self._running is not None

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until nothing is pending or running.  For tests, not for Tk."""
        return self._idle.wait(timeout)

    def close(self, timeout: float = 2.0) -> None:
        """Stop the worker.  Safe to call twice, and safe to call while busy."""
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._pending = None
            self._cv.notify_all()
        self._thread.join(timeout)

    # -- the worker ----------------------------------------------------

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._closed:
                    self._cv.wait()
                if self._closed:
                    self._idle.set()
                    return
                job_id, job = self._pending
                self._pending = None
                self._running = job_id
            try:
                ok, payload = True, job()
            except Exception as exc:  # a panel's failure is a message, not a crash
                ok, payload = False, exc
            with self._cv:
                self._running = None
                # Something newer has been asked for, so this answer is to a
                # question nobody is holding any more.
                stale = job_id != self._latest or self._closed
            if not stale:
                self._deliver(job_id, ok, payload)
            with self._cv:
                # Set only after delivering, so a test that waits for idle has
                # the result in hand rather than racing it.
                if self._pending is None and not self._closed:
                    self._idle.set()
