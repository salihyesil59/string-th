r"""An explicit Fock space for one chiral tower of string oscillators.

Everything else in :mod:`stringsim.quantum` *counts* states.  This module
*builds* them.  A basis element is a multiset of creation operators

.. math::
   \alpha_{-n_1}^{\mu_1} \alpha_{-n_2}^{\mu_2} \cdots |0; p\rangle ,
   \qquad \sum_i n_i = N ,

recorded as a sorted tuple of ``(n, mu)`` pairs.  The creation operators all
commute -- :math:`[\alpha_{-m}^\mu, \alpha_{-n}^\nu] = -m\,\eta^{\mu\nu}
\delta_{m+n,0}` vanishes when both indices are negative -- so a multiset is a
faithful label and no ordering convention is needed.

The algebra is

.. math::
   [\alpha_m^\mu, \alpha_n^\nu] = m \, \eta^{\mu\nu} \delta_{m+n,0} ,
   \qquad \alpha_0^\mu = \sqrt{2\alpha'}\, p^\mu

for the open string, and :math:`(\alpha_{-n}^\mu)^\dagger = \alpha_n^\mu`.

**The metric is indefinite and that is the whole point.**  Because
:math:`\eta^{00} = -1`, a single timelike oscillator gives a state of negative
norm:

.. math::
   \langle 0 | \alpha_1^0 \alpha_{-1}^0 | 0 \rangle = \eta^{00} = -1 .

The full Fock space therefore has ghosts in every dimension.  Whether the
*physical* subspace does is a different question, and it is the question
:mod:`stringsim.quantum.virasoro` answers.

Since :math:`\eta` is diagonal, distinct multisets are orthogonal and the Gram
matrix of this basis is diagonal with exactly computable integer entries --
see :func:`state_norm`.  That makes the inner product cheap and exact, which
matters: the no-ghost analysis is a statement about *signs* of eigenvalues, and
a Gram matrix accumulated in floating point from non-orthogonal states would
put those signs at the mercy of round-off.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np

__all__ = [
    "BasisState",
    "level_basis",
    "basis_dimension",
    "basis_index",
    "state_level",
    "state_norm",
    "gram_diagonal",
    "signature",
    "alpha_zero",
    "momentum_for_level",
    "apply_alpha",
    "apply_alpha_dict",
    "alpha_matrix",
]

#: A basis element: a sorted tuple of ``(mode, index)`` pairs, repeats allowed.
#: ``((1, 0), (1, 3), (2, 1))`` is
#: ``alpha_{-1}^0 alpha_{-1}^3 alpha_{-2}^1 |0; p>`` at level 4.
BasisState = tuple[tuple[int, int], ...]


def signature(dim: int) -> np.ndarray:
    """The mostly-plus metric diagonal as integers: ``[-1, +1, ..., +1]``.

    Same content as :func:`stringsim.units.minkowski`, but integer-typed so
    that norms stay exact.
    """
    if dim < 2:
        raise ValueError("dim must be at least 2")
    eta = np.ones(dim, dtype=np.int64)
    eta[0] = -1
    return eta


def _generate(level: int, dim: int, floor: tuple[int, int]) -> list[BasisState]:
    """Multisets of ``(n, mu)`` summing to ``level``, items ``>= floor``."""
    if level == 0:
        return [()]
    out: list[BasisState] = []
    for n in range(1, level + 1):
        for mu in range(dim):
            item = (n, mu)
            if item < floor:
                continue
            for rest in _generate(level - n, dim, item):
                out.append((item,) + rest)
    return out


@cache
def level_basis(level: int, dim: int) -> tuple[BasisState, ...]:
    """Every oscillator state at the given level, as sorted multisets.

    The count is the coefficient of ``q^level`` in
    :math:`\\prod_{n\\geq1}(1-q^n)^{-D}` -- the same partition function as
    :func:`stringsim.quantum.partition.oscillator_degeneracies`, but with ``D``
    colours rather than ``D - 2``, because nothing has been projected yet.

    Cached: the basis is regenerated often and the recursion is not cheap.
    """
    if level < 0:
        raise ValueError("level must be non-negative")
    if dim < 2:
        raise ValueError("dim must be at least 2")
    return tuple(_generate(level, dim, (1, 0)))


def basis_dimension(level: int, dim: int) -> int:
    """Number of oscillator states at ``level`` in ``dim`` dimensions."""
    return len(level_basis(level, dim))


@cache
def basis_index(level: int, dim: int) -> dict[BasisState, int]:
    """Map from basis state to its position in :func:`level_basis`."""
    return {state: i for i, state in enumerate(level_basis(level, dim))}


def state_level(state: BasisState) -> int:
    """``N`` for a basis state: the sum of its mode numbers."""
    return sum(n for n, _ in state)


def state_norm(state: BasisState, dim: int) -> int:
    r"""The exact norm :math:`\langle s | s \rangle`, an integer.

    For a single mode repeated ``k`` times,
    :math:`\langle 0 | (\alpha_n^\mu)^k (\alpha_{-n}^\mu)^k | 0 \rangle
    = (n\,\eta^{\mu\mu})^k \, k!`, and distinct ``(n, mu)`` factorise.  An odd
    number of timelike oscillators makes this negative -- the ghosts are here
    from the start.
    """
    eta = signature(dim)
    counts: dict[tuple[int, int], int] = {}
    for item in state:
        counts[item] = counts.get(item, 0) + 1
    norm = 1
    for (n, mu), k in counts.items():
        if not 0 <= mu < dim:
            raise ValueError(f"index {mu} out of range for dim {dim}")
        norm *= (n * int(eta[mu])) ** k * math.factorial(k)
    return norm


def gram_diagonal(level: int, dim: int) -> np.ndarray:
    """Norms of every basis state at ``level``, in basis order.

    The Gram matrix of the Fock basis is ``diag(gram_diagonal(...))``: the
    basis is orthogonal because the metric is.
    """
    return np.array([state_norm(s, dim) for s in level_basis(level, dim)], dtype=float)


def alpha_zero(momentum: np.ndarray, alpha_prime: float = 1.0) -> np.ndarray:
    r"""``alpha_0^mu = sqrt(2 alpha') p^mu``, the open-string zero mode."""
    return math.sqrt(2.0 * alpha_prime) * np.asarray(momentum, dtype=float)


def momentum_for_level(
    level: int, dim: int, intercept: float = 1.0, alpha_prime: float = 1.0
) -> np.ndarray:
    r"""A momentum obeying the mass shell :math:`\alpha' p^2 = a - N`.

    The physical-state condition :math:`(L_0 - a)|\psi\rangle = 0` with
    :math:`L_0 = \alpha' p^2 + N` fixes :math:`p^2` and nothing else, so any
    representative will do; Lorentz invariance of the answer is a check, not an
    assumption.  Three cases:

    * ``a < N`` -- massive, :math:`p^2 < 0`: the rest frame ``(M, 0, ..., 0)``;
    * ``a == N`` -- massless: a lightlike ``(E, 0, ..., 0, E)``;
    * ``a > N`` -- spacelike momentum, which happens at level 0 for ``a > 0``
      (the tachyon): ``(0, ..., 0, k)``.
    """
    if dim < 2:
        raise ValueError("dim must be at least 2")
    p_squared = (intercept - level) / alpha_prime
    p = np.zeros(dim)
    if p_squared < 0.0:
        p[0] = math.sqrt(-p_squared)
    elif p_squared > 0.0:
        p[-1] = math.sqrt(p_squared)
    else:
        p[0] = 1.0
        p[-1] = 1.0
    return p


def apply_alpha(
    state: BasisState, mode: int, index: int, dim: int, alpha0: np.ndarray
) -> dict[BasisState, float]:
    r"""Act with :math:`\alpha_{\text{mode}}^{\text{index}}` on one basis state.

    Returns the result as ``{state: coefficient}``; an empty dict means the
    operator annihilated it.

    * ``mode < 0`` -- creation: append ``(-mode, index)``, coefficient 1.
    * ``mode > 0`` -- annihilation: :math:`[\alpha_n^\mu, \alpha_{-n}^\nu] =
      n \eta^{\mu\nu}`, so with ``k`` copies of ``(mode, index)`` present the
      coefficient is :math:`k \, n \, \eta^{\mu\mu}` and one copy is removed.
      Off-diagonal ``eta`` would mix indices; the diagonal metric means only
      the matching index contributes.
    * ``mode == 0`` -- diagonal, coefficient ``alpha0[index]``.
    """
    if not 0 <= index < dim:
        raise ValueError(f"index {index} out of range for dim {dim}")
    if mode == 0:
        value = float(alpha0[index])
        return {state: value} if value != 0.0 else {}
    if mode < 0:
        item = (-mode, index)
        return {tuple(sorted(state + (item,))): 1.0}
    item = (mode, index)
    multiplicity = state.count(item)
    if multiplicity == 0:
        return {}
    rest = list(state)
    rest.remove(item)
    eta = -1.0 if index == 0 else 1.0
    return {tuple(rest): multiplicity * mode * eta}


def apply_alpha_dict(
    vector: dict[BasisState, float], mode: int, index: int, dim: int, alpha0: np.ndarray
) -> dict[BasisState, float]:
    """Act with one oscillator on a whole linear combination."""
    out: dict[BasisState, float] = {}
    for state, coeff in vector.items():
        for target, value in apply_alpha(state, mode, index, dim, alpha0).items():
            out[target] = out.get(target, 0.0) + coeff * value
    return {k: v for k, v in out.items() if v != 0.0}


def alpha_matrix(
    level: int, dim: int, mode: int, alpha0: np.ndarray, alpha_prime: float = 1.0
) -> np.ndarray:
    r"""Matrix of :math:`\alpha_{\text{mode}}^{\mu}` stacked over ``mu``.

    Returns shape ``(dim, rows, cols)``, where ``cols`` is the level-``level``
    basis size and ``rows`` is the size of level ``level - mode``.  Useful for
    checking hermiticity: :math:`\alpha_{-n}` and :math:`\alpha_n` must be
    adjoint *with respect to the indefinite Gram matrix*, not the Euclidean
    one.
    """
    del alpha_prime  # alpha' enters only through alpha0
    target = level - mode
    if target < 0:
        return np.zeros((dim, 0, basis_dimension(level, dim)))
    source_states = level_basis(level, dim)
    target_index = basis_index(target, dim)
    out = np.zeros((dim, len(target_index), len(source_states)))
    for col, state in enumerate(source_states):
        for mu in range(dim):
            for produced, coeff in apply_alpha(state, mode, mu, dim, alpha0).items():
                out[mu, target_index[produced], col] += coeff
    return out
