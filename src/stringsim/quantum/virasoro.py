r"""The Virasoro algebra as matrices, and the critical dimension from unitarity.

The generators are built from the oscillators of :mod:`stringsim.quantum.fock`:

.. math::
   L_m = \tfrac12 \sum_{n \in \mathbb{Z}} :\!\alpha_{m-n} \cdot \alpha_n\!: ,
   \qquad
   L_0 = \alpha' p^2 + \sum_{n \geq 1} \alpha_{-n} \cdot \alpha_n .

For ``m != 0`` the two factors commute -- :math:`[\alpha_a, \alpha_b]` is
proportional to :math:`\delta_{a+b,0}` and ``a + b = m`` -- so no ordering
prescription is needed and the sum is finite on any level: only ``2N + O(|m|)``
terms can act at all.  The bound is derived in :func:`_summation_range` and
checked by widening it.

**Nothing here is asserted.**  The central charge is not written down; it is
read off the commutator

.. math::
   [L_m, L_{-m}] = 2m L_0 + \frac{c}{12}(m^3 - m) ,

which comes out ``c = D`` for whatever ``D`` the Fock space was built in.  That
is an independent route to :func:`stringsim.quantum.zeta.central_charge`, which
gets ``c`` from zeta regularisation of the ghost and matter contributions.

**The critical dimension, a third way.**  The package already derives ``D = 26``
twice: from the normal-ordering constant :math:`a = (D-2)/24 = 1`, and from the
vanishing of the total central charge.  Both are statements about *anomalies*.
This module gets the same number out of *unitarity* instead.  The physical
subspace at level ``N`` is

.. math::
   L_1|\psi\rangle = L_2|\psi\rangle = 0 , \qquad (L_0 - a)|\psi\rangle = 0 ,

and its Gram matrix -- inherited from the indefinite Fock metric -- is what
decides the matter.  Two independent counts pin ``D`` from opposite sides:

* **no negative norms** holds for ``D <= 26`` and fails at 27, where one state
  at level 2 (and ``D - 1`` of them at level 3) goes negative;
* **the positive-norm count equals the light-cone count** for ``D >= 26``;
  below 26 the covariant spectrum carries one extra scalar that the light cone
  does not have.

Neither bound alone singles out 26.  Together they leave exactly one
dimension, and that is the derivation.  At ``D = 26`` the extra scalar is
null -- it has zero norm and drops out of every inner product -- which is the
same statement seen from the third side.

**One level is not enough.**  Scanning the intercept at level 2 leaves a
spurious ghost-free window at ``D = 26`` for ``a < 3/8`` (see
:func:`ghost_boundary`, which locates that ``3/8`` to twelve digits).  Level 3
closes it: there the only ghost-free intercept at ``D = 26`` is ``a = 1``, the
value zeta regularisation independently demands.  What level 2 gives is a
necessary condition; the example script shows the window shutting.

Only the *signs* of the Gram eigenvalues carry meaning: a different basis of
the same null space rescales them.  Sylvester's law of inertia says the counts
of positive, zero and negative eigenvalues do not move, and those counts are
what this module reports.

Reference: Polchinski, *String Theory* Vol. I, sections 2.7 and 4.1; Green,
Schwarz and Witten, *Superstring Theory* Vol. I, section 2.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fock import (
    BasisState,
    alpha_zero,
    apply_alpha,
    basis_dimension,
    basis_index,
    gram_diagonal,
    level_basis,
    momentum_for_level,
    state_level,
)
from .partition import oscillator_degeneracies

__all__ = [
    "virasoro_action",
    "virasoro_matrix",
    "l_zero_eigenvalues",
    "commutator",
    "algebra_residual",
    "central_charge_from_algebra",
    "physical_states",
    "Inertia",
    "inertia",
    "lightcone_count",
    "ghost_scan",
    "ghost_boundary",
    "constraint_residual",
    "critical_dimension_from_norms",
    "no_ghost_map",
]


def _summation_range(m: int, level: int, pad: int = 2) -> range:
    r"""Which ``a`` in :math:`\sum_a \alpha_a \cdot \alpha_{m-a}` can contribute.

    Acting on a level-``N`` state with ``m != 0``, write the term as
    :math:`\alpha_a \alpha_b` with ``b = m - a`` applied first.

    * ``b > 0`` annihilates a mode of the state, so ``b <= N`` and
      ``a >= m - N``;
    * ``b <= 0`` creates mode ``|b|``; a following ``a > 0`` must match a mode
      that is present, which is either one of the original ones (``a <= N``) or
      ``|b|`` itself -- and ``a = -b`` would force ``m = 0``;
    * ``a <= 0`` with ``b <= 0`` raises the level by ``-m``, possible only for
      ``m < 0``, and then ``a >= m``.

    So ``m - N <= a <= max(N, m - 1)``.  ``pad`` widens that; the test suite
    checks the result does not change when it does.
    """
    return range(m - level - pad, max(level, m - 1) + pad + 1)


def virasoro_action(
    state: BasisState, m: int, dim: int, alpha0: np.ndarray, pad: int = 2
) -> dict[BasisState, float]:
    r"""Act with :math:`L_m` on one basis state, returning ``{state: coeff}``.

    ``m == 0`` uses the normal-ordered form
    :math:`\tfrac12 \alpha_0\!\cdot\!\alpha_0 + \sum_{n\geq1}
    \alpha_{-n}\!\cdot\!\alpha_n`, evaluated as operator products rather than
    replaced by ``alpha' p^2 + N``; that identity is then something the test
    suite can check instead of something the code assumes.
    """
    level = state_level(state)
    if m == 0:
        pairs = [(0, 0)] + [(-n, n) for n in range(1, level + 1)]
        weights = [0.5] + [1.0] * level
    else:
        pairs = [(a, m - a) for a in _summation_range(m, level, pad)]
        weights = [0.5] * len(pairs)

    out: dict[BasisState, float] = {}
    for (a, b), weight in zip(pairs, weights, strict=True):
        for mu in range(dim):
            eta = -1.0 if mu == 0 else 1.0
            inner = apply_alpha(state, b, mu, dim, alpha0)
            if not inner:
                continue
            for middle, first in inner.items():
                for target, second in apply_alpha(middle, a, mu, dim, alpha0).items():
                    out[target] = out.get(target, 0.0) + weight * eta * first * second
    return {k: v for k, v in out.items() if v != 0.0}


def virasoro_matrix(
    level: int, m: int, dim: int, alpha0: np.ndarray, pad: int = 2
) -> np.ndarray:
    r""":math:`L_m` as a matrix from level ``level`` to level ``level - m``.

    Shape ``(rows, cols)`` with ``cols`` the source basis size.  A target level
    below zero gives a ``(0, cols)`` matrix -- the operator annihilates
    everything, which is the correct statement, not an error.
    """
    source = level_basis(level, dim)
    target_level = level - m
    if target_level < 0:
        return np.zeros((0, len(source)))
    index = basis_index(target_level, dim)
    out = np.zeros((len(index), len(source)))
    for col, state in enumerate(source):
        for produced, coeff in virasoro_action(state, m, dim, alpha0, pad).items():
            out[index[produced], col] += coeff
    return out


def l_zero_eigenvalues(level: int, dim: int, alpha0: np.ndarray, alpha_prime: float = 1.0) -> float:
    r"""The value :math:`L_0` *should* take: :math:`\alpha' p^2 + N`.

    Computed from ``alpha0`` by inverting :math:`\alpha_0 = \sqrt{2\alpha'}p`,
    so that comparing it against ``diag(virasoro_matrix(level, 0, ...))`` is a
    real check of the operator construction.
    """
    eta = np.ones(dim)
    eta[0] = -1.0
    p_squared = float(np.sum(eta * np.asarray(alpha0) ** 2)) / (2.0 * alpha_prime)
    return alpha_prime * p_squared + level


def commutator(level: int, m: int, n: int, dim: int, alpha0: np.ndarray) -> np.ndarray:
    r""":math:`[L_m, L_n]` as a matrix from ``level`` to ``level - m - n``.

    Assembled as ``L_m L_n - L_n L_m`` with each factor built at the level it
    actually acts on; the intermediate levels differ between the two orderings,
    so this needs the Fock space at ``level - n`` *and* ``level - m``.  Either
    ordering may run off the bottom of the tower -- ``L_2 L_{-2}`` on the
    ground state does, since ``L_2`` annihilates it -- and that term is then
    genuinely zero rather than an error.
    """
    target = level - m - n
    cols = basis_dimension(level, dim)
    if target < 0:
        return np.zeros((0, cols))
    out = np.zeros((basis_dimension(target, dim), cols))
    if level - n >= 0:
        out += virasoro_matrix(level - n, m, dim, alpha0) @ virasoro_matrix(level, n, dim, alpha0)
    if level - m >= 0:
        out -= virasoro_matrix(level - m, n, dim, alpha0) @ virasoro_matrix(level, m, dim, alpha0)
    return out


def algebra_residual(level: int, m: int, n: int, dim: int, alpha_prime: float = 1.0) -> float:
    r"""How far :math:`[L_m,L_n] - (m-n)L_{m+n}` is from its central term.

    For ``m + n != 0`` there is no central term and this is the whole content
    of the algebra, so the residual should be zero to round-off.  For
    ``m + n == 0`` the leftover must be a multiple of the identity, and this
    returns the deviation from *the best such multiple* -- the multiple itself
    is the central charge, extracted by
    :func:`central_charge_from_algebra`.
    """
    momentum = momentum_for_level(level, dim, intercept=1.0, alpha_prime=alpha_prime)
    alpha0 = alpha_zero(momentum, alpha_prime)
    comm = commutator(level, m, n, dim, alpha0)
    rest = virasoro_matrix(level, m + n, dim, alpha0)
    if rest.shape[0] == 0:
        rest = np.zeros_like(comm)
    residual = comm - (m - n) * rest
    if m + n != 0:
        return float(np.max(np.abs(residual))) if residual.size else 0.0
    scale = float(np.mean(np.diag(residual)))
    return float(np.max(np.abs(residual - scale * np.eye(residual.shape[0]))))


def central_charge_from_algebra(
    dim: int, m: int = 2, level: int = 0, alpha_prime: float = 1.0
) -> float:
    r"""Read ``c`` off :math:`[L_m, L_{-m}] = 2m L_0 + \frac{c}{12}(m^3-m)`.

    Parameters
    ----------
    dim:
        Spacetime dimension the Fock space is built in.  The answer should be
        ``c = dim``: one free scalar per direction, ghosts not included.
    m:
        Which commutator to use.  ``m = 1`` has no central term at all
        (:math:`m^3 - m = 0`) and so cannot measure ``c``; use it as a control.
    level:
        Which level to evaluate on.  ``0`` is cheapest -- the space is
        one-dimensional and only levels ``0`` and ``m`` are ever built -- but
        also the weakest check, since it compares a single number.  Level
        ``ell`` needs the basis up to ``ell + m``, which grows quickly in large
        ``dim``.

    Raises
    ------
    ValueError
        If ``m == 1``, where the central term vanishes identically and ``c``
        is not determined.
    """
    if abs(m) < 2:
        raise ValueError("m^3 - m vanishes for |m| < 2; that commutator cannot see c")
    momentum = momentum_for_level(level, dim, intercept=1.0, alpha_prime=alpha_prime)
    alpha0 = alpha_zero(momentum, alpha_prime)
    comm = commutator(level, m, -m, dim, alpha0)
    l_zero = virasoro_matrix(level, 0, dim, alpha0)
    leftover = comm - 2 * m * l_zero
    scale = float(np.mean(np.diag(leftover)))
    return 12.0 * scale / (m**3 - m)


def physical_states(
    level: int,
    dim: int,
    intercept: float = 1.0,
    alpha_prime: float = 1.0,
    tol: float = 1e-9,
    momentum: np.ndarray | None = None,
) -> np.ndarray:
    r"""An orthonormal basis of the physical subspace, as columns.

    Solves :math:`L_1|\psi\rangle = L_2|\psi\rangle = 0` at fixed level, with
    the momentum already on the mass shell :math:`\alpha' p^2 = a - N` so that
    :math:`(L_0 - a)|\psi\rangle = 0` holds identically.  ``L_1`` and ``L_2``
    generate the rest of the positive modes through the algebra, so imposing
    those two is imposing all of them -- something the test suite verifies by
    applying ``L_3`` to the result.

    ``momentum`` overrides the representative chosen by
    :func:`~stringsim.quantum.fock.momentum_for_level`.  It must sit on the
    same mass shell; anything else is a different problem, so it is rejected
    rather than silently used.  The individual states depend on the frame --
    they are polarisation tensors -- but the signature of their Gram matrix
    does not, and running :func:`inertia` in a boosted frame is how the test
    suite checks that nothing has quietly become frame-dependent.

    The returned columns are orthonormal in the *Euclidean* sense, which is
    just a convenient basis choice; the physics is in :func:`inertia`.
    """
    if momentum is None:
        momentum = momentum_for_level(level, dim, intercept, alpha_prime)
    else:
        momentum = np.asarray(momentum, dtype=float)
        if momentum.shape != (dim,):
            raise ValueError(f"momentum must have shape ({dim},), got {momentum.shape}")
        eta = np.ones(dim)
        eta[0] = -1.0
        shell = alpha_prime * float(np.sum(eta * momentum**2)) - (intercept - level)
        if abs(shell) > 1e-9 * max(1.0, abs(intercept - level)):
            raise ValueError(
                f"momentum is off the mass shell by {shell:.3e}; "
                f"alpha' p^2 must equal a - N = {intercept - level:g}"
            )
    alpha0 = alpha_zero(momentum, alpha_prime)
    blocks = [virasoro_matrix(level, m, dim, alpha0) for m in (1, 2)]
    blocks = [b for b in blocks if b.shape[0] > 0]
    width = basis_dimension(level, dim)
    if not blocks:
        return np.eye(width)
    stack = np.vstack(blocks)
    _, singular, right = np.linalg.svd(stack)
    padded = np.zeros(right.shape[0])
    padded[: singular.size] = singular
    cutoff = tol * max(1.0, float(padded.max()))
    return right[padded <= cutoff].T


@dataclass(frozen=True)
class Inertia:
    """Signature of the Gram matrix on the physical subspace.

    ``positive`` counts propagating states, ``zero`` counts null states -- pure
    gauge, decoupled from every amplitude -- and ``negative`` counts ghosts.
    Only ``negative == 0`` makes the theory unitary.
    """

    dim: int
    level: int
    intercept: float
    positive: int
    zero: int
    negative: int
    smallest: float

    @property
    def total(self) -> int:
        """Dimension of the physical subspace, nulls included."""
        return self.positive + self.zero + self.negative

    @property
    def ghost_free(self) -> bool:
        """``True`` when no negative-norm physical state exists."""
        return self.negative == 0

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"D = {self.dim}, N = {self.level}, a = {self.intercept:g}: "
            f"{self.positive} pos / {self.zero} null / {self.negative} neg, "
            f"(smallest eigenvalue {self.smallest:+.4g})"
        )


def inertia(
    level: int,
    dim: int,
    intercept: float = 1.0,
    alpha_prime: float = 1.0,
    tol: float = 1e-8,
    momentum: np.ndarray | None = None,
) -> Inertia:
    r"""Count positive, null and negative-norm physical states.

    The Gram matrix is :math:`V^{T} \eta V` with ``V`` the physical basis and
    :math:`\eta` the diagonal Fock metric of
    :func:`stringsim.quantum.fock.gram_diagonal`.  Its eigenvalue *magnitudes*
    depend on the basis; the signs do not.

    ``tol`` is relative to the largest eigenvalue magnitude.  At the critical
    dimension the null eigenvalues come out at ``1e-15`` of the largest while
    the smallest non-null one is ``O(1)``, so the classification is not
    delicate -- the test suite checks that gap directly.
    """
    basis = physical_states(level, dim, intercept, alpha_prime, momentum=momentum)
    metric = gram_diagonal(level, dim)
    gram = basis.T @ (metric[:, None] * basis)
    values = np.linalg.eigvalsh(gram) if gram.size else np.zeros(0)
    cutoff = tol * max(1.0, float(np.max(np.abs(values))) if values.size else 1.0)
    return Inertia(
        dim=dim,
        level=level,
        intercept=intercept,
        positive=int(np.sum(values > cutoff)),
        zero=int(np.sum(np.abs(values) <= cutoff)),
        negative=int(np.sum(values < -cutoff)),
        smallest=float(values.min()) if values.size else 0.0,
    )


def lightcone_count(level: int, dim: int) -> int:
    """Number of light-cone states at ``level``: ``D - 2`` transverse towers.

    This is the count the covariant construction has to reproduce once the
    null states are removed, and it does so only at the critical dimension.
    """
    return oscillator_degeneracies(level, dim - 2)[level]


def ghost_scan(
    dims: list[int] | range,
    level: int = 2,
    intercept: float = 1.0,
    alpha_prime: float = 1.0,
) -> list[Inertia]:
    """:func:`inertia` across a range of spacetime dimensions."""
    return [inertia(level, int(d), intercept, alpha_prime) for d in dims]


def critical_dimension_from_norms(
    level: int = 2, search: range | None = None, intercept: float = 1.0
) -> int:
    """The largest ``D`` whose physical subspace still has no negative norms.

    Returns 26 for ``level >= 2``.  Level 1 cannot see it: there the physical
    states are the ``D - 1`` transverse-plus-longitudinal polarisations with a
    single null direction, in every dimension.  Raising ``level`` past 2 does
    not change the answer either, which is the point -- 26 is not a level-2
    accident.
    """
    window = search if search is not None else range(4, 31)
    ghost_free = [rec.dim for rec in ghost_scan(window, level, intercept) if rec.ghost_free]
    if not ghost_free:
        raise RuntimeError("no ghost-free dimension in the search window")
    return max(ghost_free)


def no_ghost_map(
    dims: list[int] | range,
    intercepts: np.ndarray,
    level: int = 2,
    alpha_prime: float = 1.0,
) -> np.ndarray:
    r"""Smallest Gram eigenvalue over a ``(D, a)`` grid.

    Rows are dimensions, columns are intercepts.  The sign of each entry marks
    the boundary of the no-ghost region, which the literature states as
    ``D <= 26`` with ``a <= 1``.  Normalised by the largest magnitude in each
    cell so that entries at different ``D`` are comparable at all.
    """
    out = np.zeros((len(list(dims)), len(intercepts)))
    for i, d in enumerate(dims):
        for j, a in enumerate(intercepts):
            basis = physical_states(level, int(d), float(a), alpha_prime)
            if basis.shape[1] == 0:
                out[i, j] = 0.0
                continue
            metric = gram_diagonal(level, int(d))
            gram = basis.T @ (metric[:, None] * basis)
            values = np.linalg.eigvalsh(gram)
            out[i, j] = float(values.min()) / max(1.0, float(np.max(np.abs(values))))
    return out


def ghost_boundary(
    dim: int,
    level: int = 2,
    window: tuple[float, float] = (-3.0, 0.999),
    alpha_prime: float = 1.0,
) -> float | None:
    r"""The intercept below ``a = 1`` at which the first ghost appears, or ``None``.

    Scans :attr:`Inertia.smallest` as a function of ``a`` and brackets its sign
    change.  The upper end of the default window stops just short of ``a = 1``,
    where ``D - 1`` eigenvalues hit zero together and the root finder would be
    bracketing a degeneracy rather than a crossing.

    At level 2 no such crossing exists for ``dim <= 25``: the whole line
    ``a < 1`` is free of negative norms, and the function returns ``None``.  It
    appears first at ``dim = 26``, where the root is ``0.375`` to twelve
    digits, and moves down through zero at ``dim = 28``.

    A ``None`` here is a statement about level 2 only.  The window it leaves
    open at ``dim = 26`` is closed by level 3 -- see the module docstring --
    which is why this returns the boundary rather than a verdict.
    """
    from scipy.optimize import brentq

    def smallest(a: float) -> float:
        return inertia(level, dim, float(a), alpha_prime).smallest

    lo, hi = window
    if smallest(hi) > 0.0:
        return None
    if smallest(lo) < 0.0:
        raise ValueError(f"already ghostly at a = {lo}; widen the window downwards")
    return float(brentq(smallest, lo, hi, xtol=1e-13, rtol=1e-14))


def constraint_residual(
    vector: np.ndarray,
    level: int,
    dim: int,
    intercept: float = 1.0,
    alpha_prime: float = 1.0,
    modes: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[int, float]:
    r"""How badly a vector fails :math:`L_m|\psi\rangle = 0`, mode by mode.

    Mode ``0`` is measured against the mass shell,
    :math:`\|(L_0 - a)|\psi\rangle\|`; the rest against zero.

    The point of allowing ``m > 2`` is that :func:`physical_states` imposes
    only ``L_1`` and ``L_2``.  Every higher generator follows from those
    through :math:`[L_1, L_2] = -L_3` and so on, so a state that satisfies the
    two and fails a third would mean the algebra had been built wrong.  The
    test suite checks ``L_3`` and ``L_4`` on states that never saw them.
    """
    momentum = momentum_for_level(level, dim, intercept, alpha_prime)
    alpha0 = alpha_zero(momentum, alpha_prime)
    vector = np.asarray(vector, dtype=float)
    out: dict[int, float] = {}
    for m in modes:
        matrix = virasoro_matrix(level, m, dim, alpha0)
        if matrix.shape[0] == 0:
            out[m] = 0.0
            continue
        image = matrix @ vector
        if m == 0:
            image = image - intercept * vector
        out[m] = float(np.max(np.abs(image)))
    return out
