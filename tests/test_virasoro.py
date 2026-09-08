"""The Virasoro algebra as matrices, and the critical dimension from unitarity.

The package derives ``D = 26`` twice already -- from the normal-ordering
constant and from the vanishing central charge, both anomaly statements.  These
tests check the third route, which uses neither: build the physical states and
look at the sign of their norms.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.quantum.fock import (
    alpha_zero,
    basis_dimension,
    gram_diagonal,
    level_basis,
    momentum_for_level,
)
from stringsim.quantum.partition import oscillator_degeneracies
from stringsim.quantum.states import sym_traceless_dim
from stringsim.quantum.virasoro import (
    algebra_residual,
    central_charge_from_algebra,
    constraint_residual,
    critical_dimension_from_norms,
    ghost_boundary,
    ghost_scan,
    inertia,
    l_zero_eigenvalues,
    lightcone_count,
    no_ghost_map,
    physical_states,
    virasoro_action,
    virasoro_matrix,
)
from stringsim.quantum.zeta import central_charge, critical_dimension

# --------------------------------------------------------------------------
# the operators themselves
# --------------------------------------------------------------------------


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("intercept", [0.0, 1.0])
def test_l_zero_is_diagonal_with_the_expected_eigenvalue(level: int, intercept: float) -> None:
    r"""``L_0`` built from operator products equals ``alpha' p^2 + N``.

    The code never substitutes that formula; it multiplies out
    ``(1/2) alpha_0.alpha_0 + sum alpha_{-n}.alpha_n`` on every basis state.
    Getting a diagonal matrix at all is already a check, since off-diagonal
    entries would mean the normal ordering was wrong.
    """
    dim, alpha_prime = 5, 1.3
    alpha0 = alpha_zero(momentum_for_level(level, dim, intercept, alpha_prime), alpha_prime)
    matrix = virasoro_matrix(level, 0, dim, alpha0)
    diagonal = np.diag(matrix)
    assert np.allclose(matrix - np.diag(diagonal), 0.0)
    assert np.allclose(diagonal, l_zero_eigenvalues(level, dim, alpha0, alpha_prime))
    assert np.allclose(diagonal, intercept)


@pytest.mark.parametrize("dim", [3, 4, 6, 10, 26])
@pytest.mark.parametrize("m", [2, 3])
def test_central_charge_is_the_dimension(dim: int, m: int) -> None:
    r"""``c`` read off ``[L_m, L_{-m}] = 2m L_0 + (c/12)(m^3-m)``.

    Nothing in :mod:`~stringsim.quantum.virasoro` contains the number; it comes
    out of the commutator of matrices built from the oscillator algebra.  The
    comparison is against :func:`stringsim.quantum.zeta.central_charge`, which
    reaches ``c`` by zeta-regularising the mode sums instead -- and which
    counts the ghosts too, hence the ``+ 26``.
    """
    measured = central_charge_from_algebra(dim, m=m, level=0)
    assert math.isclose(measured, float(dim), abs_tol=1e-9)
    assert math.isclose(measured, central_charge(dim, "bosonic") + 26.0, abs_tol=1e-9)


@pytest.mark.parametrize("dim", [3, 4, 6])
def test_central_charge_from_the_full_matrix(dim: int) -> None:
    """The same ``c`` at level 2, where the commutator is a matrix not a number.

    Restricted to small ``dim``: this needs the basis up to level 4, which in
    ``D = 26`` has 33930 states.
    """
    assert math.isclose(central_charge_from_algebra(dim, m=2, level=2), float(dim), abs_tol=1e-9)
    assert algebra_residual(2, 2, -2, dim) < 1e-12


@pytest.mark.parametrize(("m", "n"), [(1, -1), (1, 2), (2, -1), (3, -1), (2, 1)])
def test_commutators_without_a_central_term(m: int, n: int) -> None:
    r"""``[L_m, L_n] = (m-n) L_{m+n}`` exactly whenever ``m + n != 0``.

    ``(1, -1)`` is included even though ``m + n = 0``: there ``m^3 - m``
    vanishes, so the central term is absent and the identity is just as sharp.
    """
    for dim in (4, 6):
        assert algebra_residual(3, m, n, dim) < 1e-12


def test_m_equals_one_cannot_measure_the_central_charge() -> None:
    """``m^3 - m = 0`` at ``m = 1``: that commutator carries no information."""
    with pytest.raises(ValueError, match="cannot see c"):
        central_charge_from_algebra(26, m=1)


def test_summation_range_is_wide_enough() -> None:
    r"""Widening the bound on :math:`\sum_a \alpha_a \cdot \alpha_{m-a}` changes nothing.

    The range is derived, not guessed, so padding it by six more terms in each
    direction must produce identical matrices.  If the derivation were wrong
    this is where it would show.
    """
    dim = 5
    alpha0 = alpha_zero(momentum_for_level(3, dim), 1.0)
    for m in (-2, -1, 1, 2, 3):
        tight = virasoro_matrix(3, m, dim, alpha0, pad=2)
        wide = virasoro_matrix(3, m, dim, alpha0, pad=8)
        assert np.array_equal(tight, wide)


def test_virasoro_action_off_the_bottom_is_empty() -> None:
    """``L_3`` on a level-2 state gives nothing, and the matrix has no rows."""
    alpha0 = alpha_zero(momentum_for_level(2, 4), 1.0)
    assert virasoro_action(((2, 1),), 5, 4, alpha0) == {}
    assert virasoro_matrix(2, 5, 4, alpha0).shape == (0, basis_dimension(2, 4))


# --------------------------------------------------------------------------
# physical states
# --------------------------------------------------------------------------


@pytest.mark.parametrize("level", [1, 2, 3])
def test_physical_states_satisfy_generators_never_imposed(level: int) -> None:
    r"""Only ``L_1`` and ``L_2`` are solved for; ``L_3`` and ``L_4`` follow.

    They follow through :math:`[L_1, L_2] = -L_3`, so a state passing the first
    two and failing the third would mean the algebra had been built wrong.
    """
    dim = 8
    basis = physical_states(level, dim)
    assert basis.shape[1] > 0
    rng = np.random.default_rng(7)
    vector = basis @ rng.normal(size=basis.shape[1])
    residual = constraint_residual(vector, level, dim, modes=(0, 1, 2, 3, 4))
    assert max(residual.values()) < 1e-10


def test_level_one_cannot_see_the_critical_dimension() -> None:
    r"""Level 1 is ``D - 2`` transverse states and one null, in every dimension.

    The famous ``zeta . p = 0`` with ``zeta ~ p`` pure gauge.  Nothing here
    depends on ``D``, which is why the no-ghost analysis has to go to level 2.
    """
    for dim in (6, 10, 26, 27, 40):
        record = inertia(1, dim)
        assert (record.positive, record.zero, record.negative) == (dim - 2, 1, 0)
        assert record.positive == lightcone_count(1, dim)


@pytest.mark.parametrize("dim", [10, 20, 24, 25, 26])
def test_no_ghosts_at_or_below_twenty_six(dim: int) -> None:
    """Level 2 with ``a = 1``: the physical norms stay non-negative."""
    assert inertia(2, dim).ghost_free


@pytest.mark.parametrize("dim", [27, 28, 30])
def test_ghosts_above_twenty_six(dim: int) -> None:
    """One negative norm appears at level 2 the moment ``D`` passes 26."""
    record = inertia(2, dim)
    assert record.negative == 1
    assert record.smallest < -1e-3


def test_the_dimension_boundary_is_sharp() -> None:
    """26 in, 27 out -- checked as a single scan rather than two assertions."""
    scan = {r.dim: r.ghost_free for r in ghost_scan(range(4, 31), level=2)}
    assert all(scan[d] for d in range(4, 27))
    assert not any(scan[d] for d in range(27, 31))


def test_covariant_count_meets_the_light_cone_only_at_twenty_six() -> None:
    r"""The second half of the pincer, and it points the other way.

    Ghost-freedom bounds ``D`` from above.  This bounds it from below: for
    ``D < 26`` the covariant level-2 spectrum carries one state the light cone
    does not have -- a scalar on top of the symmetric traceless tensor of
    ``SO(D-1)``.  At 26 that scalar goes null and the two counts agree.

    The light-cone number comes from
    :func:`~stringsim.quantum.partition.oscillator_degeneracies`, which knows
    nothing about Virasoro generators; the ``SO(D-1)`` number comes from
    :func:`~stringsim.quantum.states.sym_traceless_dim`, which knows nothing
    about either.  All three agree at 26.
    """
    for dim in (22, 23, 24, 25):
        record = inertia(2, dim)
        assert record.positive == lightcone_count(2, dim) + 1
        assert record.positive == sym_traceless_dim(dim - 1, 2) + 1
    for dim in (26, 27, 28):
        record = inertia(2, dim)
        assert record.positive == lightcone_count(2, dim)
        assert record.positive == sym_traceless_dim(dim - 1, 2)


def test_critical_dimension_agrees_with_zeta_regularisation() -> None:
    """Unitarity and the anomaly give the same 26, by unrelated routes."""
    assert critical_dimension_from_norms(level=2) == 26
    assert critical_dimension_from_norms(level=2) == critical_dimension("bosonic")


def test_null_states_appear_exactly_at_the_critical_dimension() -> None:
    r"""``D - 1`` null states below 26, one more at 26, back to ``D - 1`` above.

    The extra null is the scalar of the previous test.  Counting it here and
    counting the positive-norm deficit there are two views of one state.
    """
    assert inertia(2, 25).zero == 24
    assert inertia(2, 26).zero == 26
    assert inertia(2, 27).zero == 26


def test_the_null_gap_is_not_delicate() -> None:
    """Null eigenvalues sit 1e-15 below the rest, so the tolerance is not a knob.

    A classification that depended on where ``tol`` was set would be worthless.
    Here the two clusters are fifteen orders of magnitude apart.
    """
    for dim, level in ((26, 2), (20, 2), (26, 1)):
        basis = physical_states(level, dim)
        metric = gram_diagonal(level, dim)
        values = np.linalg.eigvalsh(basis.T @ (metric[:, None] * basis))
        magnitude = np.abs(values)
        null = magnitude <= 1e-8 * magnitude.max()
        assert magnitude[null].max() / magnitude[~null].min() < 1e-10


def test_a_ghost_is_a_genuine_physical_state() -> None:
    r"""At ``D = 27`` the negative-norm direction really satisfies every constraint.

    Otherwise the "ghost" would just be a state that failed to be physical.
    ``L_3`` and ``L_4`` annihilate it although they were never imposed, and
    ``L_0`` returns the intercept.
    """
    dim, level = 27, 2
    basis = physical_states(level, dim)
    metric = gram_diagonal(level, dim)
    values, vectors = np.linalg.eigh(basis.T @ (metric[:, None] * basis))
    ghost = basis @ vectors[:, 0]
    assert float(ghost @ (metric * ghost)) < -1e-3
    assert max(constraint_residual(ghost, level, dim).values()) < 1e-10


def test_inertia_does_not_depend_on_the_frame() -> None:
    r"""Boosting the momentum moves the states, not the signature.

    The physical states are polarisation tensors and every one of them changes
    under a boost; the counts of positive, null and negative norms cannot, and
    a frame-dependent bug would be invisible in the rest frame alone.
    """
    dim, level = 26, 2
    rest = inertia(level, dim)
    for rapidity in (0.4, 1.1):
        boosted_p = np.zeros(dim)
        boosted_p[0] = math.cosh(rapidity)
        boosted_p[3] = math.sinh(rapidity)
        moving = inertia(level, dim, momentum=boosted_p)
        assert (moving.positive, moving.zero, moving.negative) == (
            rest.positive,
            rest.zero,
            rest.negative,
        )


def test_off_shell_momentum_is_rejected() -> None:
    """A momentum that does not satisfy ``alpha' p^2 = a - N`` is a different problem."""
    with pytest.raises(ValueError, match="off the mass shell"):
        physical_states(2, 6, momentum=np.ones(6))
    with pytest.raises(ValueError, match="shape"):
        physical_states(2, 6, momentum=np.ones(5))


# --------------------------------------------------------------------------
# the intercept, and what one level can and cannot decide
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [10, 20, 26])
def test_intercept_above_one_is_always_ghostly(dim: int) -> None:
    """``a > 1`` puts a whole vector's worth of ghosts in, in any dimension.

    This is the level-2 reason ``a`` is not a free parameter, independent of
    the zeta-function argument that fixes it to ``(D-2)/24``.
    """
    record = inertia(2, dim, intercept=1.2)
    assert record.negative == dim - 1


@pytest.mark.parametrize("dim", [10, 20, 25])
def test_no_boundary_below_twenty_six(dim: int) -> None:
    """For ``D <= 25`` the whole line ``a < 1`` is free of negative norms."""
    assert ghost_boundary(dim, level=2) is None


def test_the_boundary_at_twenty_six_is_three_eighths() -> None:
    """It appears first at ``D = 26``, at an exact rational, and moves down.

    ``3/8`` and the zero at ``D = 28`` come out to twelve digits from a root
    find on the smallest eigenvalue -- nothing is fitted to them.
    """
    assert ghost_boundary(26, level=2) == pytest.approx(0.375, abs=1e-10)
    assert ghost_boundary(28, level=2) == pytest.approx(0.0, abs=1e-10)
    assert ghost_boundary(27, level=2) < 0.375
    assert ghost_boundary(30, level=2) < ghost_boundary(29, level=2)


def test_level_three_closes_the_window_level_two_leaves() -> None:
    r"""Level 2 is a necessary condition, not the theorem.

    At ``D = 26`` the level-2 analysis finds no ghost for ``a < 3/8``, which
    would leave a whole interval of intercepts looking consistent.  Level 3
    finds one, and the only intercept that survives is ``a = 1`` -- the value
    zeta regularisation demands on entirely separate grounds.

    This is the slowest check in the file (the level-3 basis has 3978 states);
    it is here because the claim it tests is the one the README makes.
    """
    assert inertia(2, 26, intercept=0.0).ghost_free
    assert inertia(3, 26, intercept=0.0).negative == 1
    assert inertia(3, 26, intercept=1.0).ghost_free


def test_level_three_reproduces_the_level_two_boundary() -> None:
    """26 stays the edge one level up, so it is not a level-2 accident.

    At level 3 the ghosts arrive in force: ``D - 1`` of them at ``D = 27``
    rather than one.
    """
    assert inertia(3, 26).ghost_free
    assert inertia(3, 26).positive == lightcone_count(3, 26)
    assert inertia(3, 27).negative == 26


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def test_lightcone_count_is_the_transverse_partition_function() -> None:
    for dim in (10, 26):
        for level in range(4):
            assert lightcone_count(level, dim) == oscillator_degeneracies(level, dim - 2)[level]


def test_no_ghost_map_marks_the_region() -> None:
    """The map is negative exactly where :func:`inertia` reports a ghost."""
    dims = [24, 26, 28]
    intercepts = np.array([0.0, 0.6, 1.2])
    grid = no_ghost_map(dims, intercepts, level=2)
    assert grid.shape == (3, 3)
    for i, dim in enumerate(dims):
        for j, a in enumerate(intercepts):
            expected = inertia(2, dim, float(a)).negative > 0
            assert (grid[i, j] < -1e-8) == expected


def test_inertia_totals_the_physical_subspace() -> None:
    record = inertia(2, 26)
    assert record.total == physical_states(2, 26).shape[1]
    assert record.total == len(level_basis(2, 26)) - 26 - 1
    assert "pos" in str(record)
