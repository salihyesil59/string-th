r"""Boundary states: a D-brane as a closed string, and the eta function it makes.

A D-brane has two descriptions.  In the open channel it is where strings end,
and :mod:`stringsim.branes.dbrane` counts the states stretched between branes.
In the closed channel it is a *state* -- a coherent superposition of closed
strings that the brane emits -- and this module builds it.

**The gluing conditions.**  Along the brane the endpoint is free, so
:math:`\partial_\tau X^\mu = 0`; transverse to it the endpoint is nailed down,
so :math:`X^i` is fixed.  On the closed-string Hilbert space both read

.. math::  (\alpha_n^\mu - S^\mu{}_\nu\, \tilde\alpha_{-n}^\nu)\,|B\rangle = 0 ,

with :math:`S = -1` along the brane (Neumann) and :math:`+1` across it
(Dirichlet).  A coherent state solves them:

.. math::
   |B\rangle = \exp\Bigl(\sum_{n>0} \tfrac{1}{n}\,
   \alpha_{-n}\!\cdot S\cdot\tilde\alpha_{-n}\Bigr)|0\rangle .

That is an *ansatz* until it is checked.  :func:`gluing_residual` builds the
state level by level in the explicit Fock space of
:mod:`stringsim.quantum.fock` and applies the operator to it, which is the
whole reason that Fock space exists.

**Its norm is the Dedekind eta.**  Expanding the exponential, the component on
:math:`|s\rangle \otimes |s\rangle` has exactly the weight that makes every
diagonal state contribute 1, so

.. math::
   \langle B | q^{(N + \tilde N)/2} | B \rangle
   = \sum_{N} d_N\, q^{N} = \prod_{n\geq1}(1 - q^n)^{-(D-2)} ,

with :math:`d_N` the transverse oscillator degeneracies of
:mod:`stringsim.quantum.partition`.  Multiplying by the ground-state energy
turns it into :math:`\eta^{-(D-2)}` -- which is the factor
:func:`stringsim.amplitudes.oneloop.closed_channel_integrand` carries.  So the
cylinder's closed-channel oscillator content is the norm of a coherent state,
and :func:`oscillator_overlap` computes it by summing over Fock states one at a
time rather than by quoting the product.

**The same conditions on a moving string.**  Classically the gluing is
:math:`\alpha_n = S\tilde\alpha_{-n}`, and a closed string whose modes obey it
has :math:`\dot X^\mu = 0` along the brane and :math:`X^i` independent of
:math:`\sigma` across it: at :math:`\tau = 0` it is a string lying still on the
brane, at a point in the transverse directions.  That is what a boundary state
*is*, and :func:`classical_boundary_state` builds one out of the
:class:`~stringsim.classical.modes.ClosedString` of section 1.

**What is not derived here.**  The zero-mode measure and the overall
normalisation.  The power of the modulus that counts the transverse momentum
integral, and the constant that is the brane tension, are taken from
:mod:`stringsim.amplitudes.oneloop` and :mod:`stringsim.branes.dbrane`
respectively; getting them out of the boundary state is Polchinski's
computation of :math:`T_p`, and it is not attempted.  What this module supplies
is the oscillator content, and it supplies it from the state rather than from a
product formula.

Reference: J. Polchinski, *String Theory* Vol. I, section 8.7 and Vol. II,
section 13.4.
"""

from __future__ import annotations

import math

import numpy as np

from ..classical.modes import ClosedString
from ..quantum.fock import BasisState, apply_alpha, level_basis, signature
from ..quantum.partition import dedekind_eta, oscillator_degeneracies
from ..units import Conventions

__all__ = [
    "gluing_signs",
    "neumann_count",
    "boundary_coefficients",
    "gluing_residual",
    "oscillator_overlap",
    "overlap_truncation",
    "overlap_closed_form",
    "eta_factor",
    "classical_boundary_state",
    "boundary_residual",
]


def gluing_signs(p: int, dim: int) -> np.ndarray:
    r"""``S``: ``-1`` along the brane, ``+1`` across it.

    ``dim`` counts the directions the oscillators run over.  Called with the
    transverse count ``D - 2`` it describes the light-cone content, which is
    what the partition function needs; called with ``D`` it is the covariant
    statement.  The brane occupies the first ``p + 1`` of them.
    """
    if dim < 1:
        raise ValueError("dim must be positive")
    if not -1 <= p < dim:
        raise ValueError(f"a D{p}-brane does not fit in {dim} directions")
    signs = np.ones(dim)
    signs[: p + 1] = -1.0
    return signs


def neumann_count(signs) -> int:
    """How many directions are Neumann -- the brane's worldvolume dimension."""
    return int(np.sum(np.asarray(signs) < 0))


def _multiplicities(state: BasisState) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for item in state:
        counts[item] = counts.get(item, 0) + 1
    return counts


def boundary_coefficients(
    level: int, dim: int, signs, include_metric: bool = True
) -> dict[BasisState, float]:
    r"""The component of :math:`|B\rangle` on :math:`|s\rangle\otimes|s\rangle`.

    Expanding the exponential and rewriting the number states in the basis of
    :mod:`stringsim.quantum.fock` -- whose elements are products of
    :math:`\alpha_{-n}`, not normalised -- gives

    .. math::
       \beta_s = \prod_{(n,\mu)} \frac{(S_\mu \eta_\mu)^{\,m_{n\mu}}}
       {n^{m_{n\mu}}\, m_{n\mu}!} ,

    the product running over the distinct oscillators in ``s`` with their
    multiplicities.  Only diagonal pairs appear: the exponential creates the
    two sides together.

    **The metric belongs in there.**  From
    :math:`[\alpha_n^\mu, \alpha_{-n}^\nu] = n\,\eta^{\mu\nu}`, acting with
    :math:`\alpha_n` on the exponential brings down :math:`\lambda_\mu\eta_\mu`,
    so the gluing condition fixes :math:`\lambda_\mu = S_\mu \eta_\mu` rather
    than :math:`S_\mu`.  Leaving the :math:`\eta` out gives a state that fails
    the timelike gluing condition by a factor of two and satisfies every
    spacelike one, which is exactly the kind of error that survives a careless
    check.

    The factors cancel against
    :math:`\langle s|s\rangle = \prod (n\eta_\mu)^{m} m!`, so every diagonal
    state contributes 1 to the norm whatever its signature.  That cancellation
    is why the overlap is a plain sum over degeneracies -- and why the
    light-cone count of ``D - 2`` directions is the right one to use in it.
    """
    signs = np.asarray(signs, dtype=float)
    eta = signature(len(signs)).astype(float) if include_metric else np.ones(len(signs))
    out: dict[BasisState, float] = {}
    for state in level_basis(level, len(signs)):
        value = 1.0
        for (mode, index), multiplicity in _multiplicities(state).items():
            value *= (signs[index] * eta[index]) ** multiplicity / (
                mode**multiplicity * math.factorial(multiplicity)
            )
        out[state] = value
    return out


def _apply_left(vector, mode, index, dim, alpha0):
    out: dict = {}
    for (left, right), coeff in vector.items():
        for target, value in apply_alpha(left, mode, index, dim, alpha0).items():
            key = (target, right)
            out[key] = out.get(key, 0.0) + coeff * value
    return out


def _apply_right(vector, mode, index, dim, alpha0):
    out: dict = {}
    for (left, right), coeff in vector.items():
        for target, value in apply_alpha(right, mode, index, dim, alpha0).items():
            key = (left, target)
            out[key] = out.get(key, 0.0) + coeff * value
    return out


def gluing_residual(
    level_max: int,
    dim: int,
    signs,
    mode: int = 1,
    index: int = 0,
    include_metric: bool = True,
) -> float:
    r"""How badly the coherent state fails :math:`(\alpha_n - S\tilde\alpha_{-n})|B\rangle = 0`.

    The state is built out to ``level_max`` on each side, the operator applied,
    and the result inspected.  Only the bi-levels the truncation determines are
    looked at: :math:`\alpha_n` lowers the left level by ``n`` and
    :math:`\tilde\alpha_{-n}` raises the right one by ``n``, so both terms land
    on :math:`(M, M+n)` and both are complete for :math:`M + n \leq`
    ``level_max``.

    Zero to round-off.  It is a check of the coefficient formula, and it is the
    kind of thing that is easy to get wrong by a factor of ``n`` and impossible
    to notice without doing it.

    ``include_metric`` is there to show that it is not decoration.  Setting it
    to ``False`` drops the :math:`\eta_\mu` of
    :func:`boundary_coefficients`, and the state then satisfies every spacelike
    gluing condition and fails the timelike one -- which is exactly the sort of
    error a check on one direction would miss.
    """
    signs = np.asarray(signs, dtype=float)
    if mode < 1:
        raise ValueError("the mode number must be positive")
    if not 0 <= index < len(signs):
        raise ValueError(f"index {index} out of range for {len(signs)} directions")
    alpha0 = np.zeros(len(signs))

    state: dict = {}
    for level in range(level_max + 1):
        coefficients = boundary_coefficients(level, len(signs), signs, include_metric)
        for basis, value in coefficients.items():
            state[(basis, basis)] = value

    lowered = _apply_left(state, mode, index, len(signs), alpha0)
    raised = _apply_right(state, -mode, index, len(signs), alpha0)
    difference: dict = dict(lowered)
    for key, value in raised.items():
        difference[key] = difference.get(key, 0.0) - signs[index] * value

    worst = 0.0
    for (_left, right), value in difference.items():
        if sum(n for n, _ in right) <= level_max:
            worst = max(worst, abs(value))
    return worst


def oscillator_overlap(q: float, dim: int, level_max: int = 90) -> float:
    r""":math:`\langle B|q^{(N+\tilde N)/2}|B\rangle`, summed over Fock states.

    Every diagonal state contributes 1 -- the coefficient squared times the
    norm squared -- so the sum is :math:`\sum_N d_N q^N` with ``d_N`` the
    oscillator degeneracies.  ``dim`` is the number of directions the
    oscillators run over, ``D - 2`` for the light-cone content.

    Truncated at ``level_max``, and the truncation is not gentle.  The
    degeneracies grow like :math:`e^{4\pi\sqrt N}`, so the terms rise before
    they fall and the cutoff needed for a given precision climbs steeply with
    ``q``: about 60 levels at ``q = 0.1``, 120 at 0.3, 320 at 0.5 and 500 at
    0.6.  The default suits the first two.  :func:`overlap_truncation` measures
    what raising it changes, and no number from here should be quoted without
    it.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between 0 and 1 for the sum to converge")
    degeneracies = oscillator_degeneracies(level_max, dim)
    return float(sum(d * q**n for n, d in enumerate(degeneracies)))


def overlap_truncation(
    q: float, dim: int, level_max: int = 90, extra: int = 60
) -> float:
    """Relative change in :func:`oscillator_overlap` when the cutoff is raised.

    The honest error bar on the sum, and the same device the Narain theta
    series uses: a residual near this is arithmetic, one far above it is
    something else.
    """
    tight = oscillator_overlap(q, dim, level_max)
    wide = oscillator_overlap(q, dim, level_max + extra)
    return abs(wide / tight - 1.0)


def overlap_closed_form(q: float, dim: int, n_terms: int = 400) -> float:
    r""":math:`\prod_{n\geq1}(1-q^n)^{-\dim}`, the same thing as a product."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between 0 and 1")
    total = 0.0
    for n in range(1, n_terms + 1):
        total -= dim * math.log1p(-(q**n))
    return math.exp(total)


def eta_factor(modulus: float, dim: int, n_terms: int = 400) -> float:
    r""":math:`|\eta(i\,\text{modulus})|^{-\dim}` built from the boundary state.

    The overlap supplies :math:`\prod(1-q^n)^{-\dim}` with
    :math:`q = e^{-2\pi\,\text{modulus}}`; multiplying by the ground-state
    factor :math:`q^{-\dim/24}` gives the eta function.  That is the oscillator
    content of the closed channel of the cylinder, and
    :func:`stringsim.amplitudes.oneloop.closed_channel_integrand` carries it at
    ``dim = 24``.

    The zero modes and the normalisation are *not* here -- see the module
    docstring.
    """
    if modulus <= 0:
        raise ValueError("the modulus must be positive")
    q = math.exp(-2.0 * math.pi * modulus)
    return q ** (-dim / 24.0) * overlap_closed_form(q, dim, n_terms)


def classical_boundary_state(
    modes, p: int, position=None, conventions: Conventions | None = None
) -> ClosedString:
    r"""A classical closed string obeying the gluing conditions.

    ``modes`` maps a mode number to a coefficient vector, taken as the
    right-movers :math:`\alpha_n`.  The gluing is
    :math:`\alpha_n = S\,\tilde\alpha_{-n}`, and a real :math:`X` has
    :math:`\alpha_{-n} = \alpha_n^{*}`, so the left-movers must be
    :math:`\tilde\alpha_n = S\,\alpha_n^{*}`.  The conjugate matters as soon as
    a mode carries a phase; with real coefficients it is invisible, which makes
    it easy to leave out and hard to notice.

    ``position`` places the brane in the transverse directions; the string sits
    on it.
    """
    conv = conventions or Conventions()
    signs = gluing_signs(p, conv.dim)
    right = {n: np.asarray(a) for n, a in modes.items()}
    tilde = {n: signs * np.conj(a) for n, a in right.items()}
    origin = np.zeros(conv.dim) if position is None else np.asarray(position, dtype=float)
    return ClosedString(conventions=conv, x0=origin, alphas=right, alphas_tilde=tilde)


def boundary_residual(
    string: ClosedString, p: int, position=None, n_sigma: int = 200
) -> tuple[float, float]:
    r"""What the gluing conditions look like on the string itself.

    Returns ``(Neumann residual, Dirichlet residual)`` at :math:`\tau = 0`:
    the largest :math:`|\dot X^\mu|` along the brane, and the largest
    :math:`|X^i - y^i|` across it, both normalised by the string's own extent
    so the numbers mean something.

    A string built by :func:`classical_boundary_state` gives round-off for
    both.  It is lying still on the brane, which is the classical shadow of the
    coherent state: the brane is where a closed string can end up at rest.
    """
    signs = gluing_signs(p, string.conventions.dim)
    sigma = np.linspace(0.0, string.sigma_max, n_sigma)
    tau = np.zeros_like(sigma)
    x = string.position(tau, sigma)
    v = string.velocity(tau, sigma)
    scale = max(float(np.max(np.abs(x - x.mean(axis=0)))), 1e-12)

    along = signs < 0
    across = ~along
    neumann = float(np.max(np.abs(v[:, along]))) / scale if along.any() else 0.0
    if across.any():
        target = (
            np.zeros(int(across.sum()))
            if position is None
            else np.asarray(position, dtype=float)[across]
        )
        dirichlet = float(np.max(np.abs(x[:, across] - target))) / scale
    else:  # pragma: no cover - a space-filling brane has nothing across it
        dirichlet = 0.0
    return neumann, dirichlet


def eta_residual(modulus: float, dim: int = 24, n_terms: int = 400) -> float:
    r"""The boundary state's oscillator factor against :func:`dedekind_eta`.

    Two implementations of the same function that share no code: one sums the
    coherent state's components and multiplies in the ground-state energy, the
    other evaluates the eta product in
    :mod:`stringsim.quantum.partition`.  Returned as a relative difference.
    """
    reference = abs(dedekind_eta(complex(0.0, modulus), n_terms)) ** (-dim)
    return abs(eta_factor(modulus, dim, n_terms) / reference - 1.0)


__all__ += ["eta_residual"]
