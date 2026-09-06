r"""Light-cone gauge: transverse oscillators that solve the constraints exactly.

Light-cone coordinates are :math:`X^\pm = (X^0 \pm X^{D-1})/\sqrt2`, so that
:math:`A \cdot B = -A^+B^- - A^-B^+ + A^iB^i` with ``i`` running over the
``D - 2`` transverse directions.  The residual conformal freedom is used to set

.. math::  X^+ = 2\alpha' p^+ \tau ,

i.e. :math:`\alpha_n^+ = 0` for :math:`n \neq 0` and
:math:`\alpha_0^+ = \sqrt{2\alpha'}\, p^+`.  The Virasoro conditions
:math:`L_n = \tfrac12 \sum_m \alpha_{n-m} \cdot \alpha_m = 0` then *solve* for
the minus components,

.. math::
   \alpha_n^- = \frac{1}{\sqrt{2\alpha'}\, p^+} \,
                \frac{1}{2} \sum_{m \in \mathbb{Z}} \alpha_{n-m}^i \alpha_m^i .

Nothing is left over: the transverse amplitudes are free data and the string is
physical by construction.  The ``n = 0`` component of that same equation is the
mass-shell condition, and it gives -- with no input beyond the amplitudes --

.. math::  \alpha' M^2 = \sum_{m \geq 1} \alpha_{-m}^i \alpha_m^i \equiv N .

Quantising replaces ``N`` by an integer operator and adds the normal-ordering
constant, ``alpha' M^2 = N - 1`` in ``D = 26``; see
:mod:`stringsim.quantum.zeta`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from ..units import Conventions
from .modes import OpenString

__all__ = ["LightconeOpenString", "oscillator_amplitudes"]


def oscillator_amplitudes(
    excitations: Mapping[tuple[int, int], int],
    transverse_dim: int,
    phases: Mapping[tuple[int, int], float] | None = None,
) -> dict[int, np.ndarray]:
    r"""Classical amplitudes matching a quantum oscillator state.

    In the quantum theory :math:`\alpha_{-n}^i = \sqrt{n}\, a_{n}^{i\dagger}`,
    so a state with ``k`` quanta in mode ``n``, direction ``i``, contributes
    ``n * k`` to the level.  This helper returns amplitudes with
    :math:`|\alpha_n^i|^2 = n k`, so that the classical level equals the quantum
    level ``N = sum n k``.

    Parameters
    ----------
    excitations:
        ``{(n, i): k}`` with ``n >= 1``, ``0 <= i < transverse_dim`` and ``k >= 1``.
    transverse_dim:
        ``D - 2``.
    phases:
        Optional ``{(n, i): phi}``; the amplitude is multiplied by
        ``exp(i phi)``.  Phases do not change the mass, only the shape traced
        out in spacetime -- a relative phase of ``pi/2`` between two directions
        turns a standing wave into a rotating one.

    Returns
    -------
    dict
        ``{n: alpha_n}`` with complex arrays of length ``transverse_dim``,
        ready for :class:`LightconeOpenString`.
    """
    phases = dict(phases or {})
    out: dict[int, np.ndarray] = {}
    for (n, i), k in excitations.items():
        n, i, k = int(n), int(i), int(k)
        if n < 1:
            raise ValueError(f"mode number must be >= 1, got {n}")
        if not 0 <= i < transverse_dim:
            raise ValueError(f"transverse index {i} outside [0, {transverse_dim})")
        if k < 1:
            raise ValueError(f"occupation must be >= 1, got {k}")
        vec = out.setdefault(n, np.zeros(transverse_dim, dtype=complex))
        vec[i] += math.sqrt(n * k) * np.exp(1j * phases.get((n, i), 0.0))
    return out


@dataclass
class LightconeOpenString:
    """An open string built from free transverse data, constraints already solved.

    Parameters
    ----------
    conventions:
        Sets ``alpha'`` and ``D``.
    p_plus:
        The light-cone momentum ``p^+``; must be non-zero (it is the gauge
        choice's denominator).  Only the overall boost is affected by its
        value, not the mass.
    p_transverse:
        ``p^i``, shape ``(D - 2,)``.  Defaults to zero, i.e. the rest frame in
        the transverse directions.
    transverse_modes:
        ``{n: alpha_n^i}`` for ``n >= 1``, complex arrays of shape ``(D - 2,)``.
        Build them by hand or with :func:`oscillator_amplitudes`.
    """

    conventions: Conventions = field(default_factory=Conventions)
    p_plus: float = 1.0
    p_transverse: np.ndarray | None = None
    transverse_modes: Mapping[int, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.p_plus == 0.0:
            raise ValueError("p_plus must be non-zero in light-cone gauge")
        dt = self.conventions.transverse_dim
        if self.p_transverse is None:
            self.p_transverse = np.zeros(dt)
        else:
            self.p_transverse = np.asarray(self.p_transverse, float).reshape(dt)
        modes: dict[int, np.ndarray] = {}
        for n, a in (self.transverse_modes or {}).items():
            n = int(n)
            if n < 1:
                raise ValueError(f"mode numbers must be >= 1, got {n}")
            modes[n] = np.asarray(a, dtype=complex).reshape(dt)
        self.transverse_modes = modes

    # -- transverse data ----------------------------------------------------

    @property
    def n_max(self) -> int:
        """Highest excited mode number (0 if the string is unexcited)."""
        return max(self.transverse_modes, default=0)

    def alpha_transverse(self, n: int) -> np.ndarray:
        r"""``alpha_n^i`` for any integer ``n``, using ``alpha_{-n} = conj(alpha_n)``.

        ``n = 0`` returns :math:`\sqrt{2\alpha'}\, p^i`.
        """
        n = int(n)
        if n == 0:
            return math.sqrt(2.0 * self.conventions.alpha_prime) * self.p_transverse.astype(complex)
        if n > 0:
            got = self.transverse_modes.get(n)
            return np.zeros(self.conventions.transverse_dim, dtype=complex) if got is None else got
        got = self.transverse_modes.get(-n)
        if got is None:
            return np.zeros(self.conventions.transverse_dim, dtype=complex)
        return np.conjugate(got)

    # -- derived quantities -------------------------------------------------

    def level(self) -> float:
        r"""``N = sum_{m >= 1} |alpha_m^i|^2``, the classical level."""
        return float(sum(float(np.sum(np.abs(a) ** 2)) for a in self.transverse_modes.values()))

    def alpha_minus(self, n: int) -> complex:
        r"""``alpha_n^-`` from the Virasoro constraint, for any integer ``n``."""
        n = int(n)
        pref = 1.0 / (math.sqrt(2.0 * self.conventions.alpha_prime) * self.p_plus)
        nm = self.n_max
        # alpha_{n-m} and alpha_m both vanish outside [-nm, nm] (except m = 0,
        # which carries the momentum), so the sum is finite.
        lo, hi = min(-nm, n - nm), max(nm, n + nm)
        total = 0.0 + 0.0j
        for m in range(lo, hi + 1):
            total += np.dot(self.alpha_transverse(n - m), self.alpha_transverse(m))
        return complex(pref * 0.5 * total)

    def p_minus(self) -> float:
        r"""``p^-``, fixed by the mass-shell condition ``alpha_0^- = sqrt(2 alpha') p^-``."""
        return float(np.real(self.alpha_minus(0)) / math.sqrt(2.0 * self.conventions.alpha_prime))

    def mass_squared(self) -> float:
        r"""Classical ``M^2 = N / alpha'``, read off from ``2 p^+ p^- - p^i p^i``."""
        return float(
            2.0 * self.p_plus * self.p_minus() - float(np.dot(self.p_transverse, self.p_transverse))
        )

    def mass_squared_quantum(self, normal_ordering_constant: float = 1.0) -> float:
        r"""``M^2 = (N - a)/alpha'``, with ``a = 1`` the critical bosonic value.

        The level is *not* rounded: build the amplitudes with
        :func:`oscillator_amplitudes` if an integer level is wanted.
        """
        return (self.level() - normal_ordering_constant) / self.conventions.alpha_prime

    def momentum(self) -> np.ndarray:
        """The full ``p^mu`` in Cartesian components ``(X^0, X^1, ..., X^{D-1})``."""
        pm = self.p_minus()
        d = self.conventions.dim
        p = np.zeros(d)
        p[0] = (self.p_plus + pm) / math.sqrt(2.0)
        p[1 : d - 1] = self.p_transverse
        p[d - 1] = (self.p_plus - pm) / math.sqrt(2.0)
        return p

    # -- export -------------------------------------------------------------

    def to_open_string(self, modes: Iterable[int] | None = None) -> OpenString:
        """Assemble the full ``D``-dimensional mode expansion.

        The returned :class:`~stringsim.classical.modes.OpenString` satisfies the
        Virasoro constraints to machine precision, which
        :func:`stringsim.classical.constraints.virasoro_residual` will confirm.

        Parameters
        ----------
        modes:
            Which ``n >= 1`` to include.  By default every ``n`` for which some
            component is non-zero: the transverse modes plus the ``alpha_n^-``
            they generate, which reach up to ``2 * n_max``.
        """
        d = self.conventions.dim
        nm = self.n_max
        if modes is None:
            modes = range(1, max(1, 2 * nm) + 1)
        alphas: dict[int, np.ndarray] = {}
        for n in modes:
            n = int(n)
            if n < 1:
                continue
            vec = np.zeros(d, dtype=complex)
            am = self.alpha_minus(n)
            # alpha^+ = 0 for n != 0, so X^0 and X^{D-1} carry alpha^- alone.
            vec[0] = am / math.sqrt(2.0)
            vec[1 : d - 1] = self.alpha_transverse(n)
            vec[d - 1] = -am / math.sqrt(2.0)
            if np.any(vec != 0):
                alphas[n] = vec
        return OpenString(conventions=self.conventions, p=self.momentum(), alphas=alphas)
