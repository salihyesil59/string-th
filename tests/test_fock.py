"""The explicit oscillator Fock space: basis, norms, and the indefinite metric."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.quantum.fock import (
    alpha_matrix,
    alpha_zero,
    apply_alpha,
    apply_alpha_dict,
    basis_dimension,
    basis_index,
    gram_diagonal,
    level_basis,
    momentum_for_level,
    signature,
    state_level,
    state_norm,
)
from stringsim.quantum.partition import oscillator_degeneracies
from stringsim.units import minkowski


@pytest.mark.parametrize("dim", [3, 4, 6, 10, 26])
def test_basis_size_is_the_partition_function(dim: int) -> None:
    """Counting the basis must reproduce the generating function it came from.

    ``level_basis`` enumerates multisets by recursion; ``oscillator_degeneracies``
    expands ``prod (1-q^n)^{-D}`` in exact integers.  Neither knows about the
    other.
    """
    counted = [basis_dimension(n, dim) for n in range(5)]
    expanded = oscillator_degeneracies(4, dim)
    assert counted == expanded


def test_basis_states_are_well_formed() -> None:
    """Sorted, distinct, and all at the level they were asked for."""
    for level in range(5):
        states = level_basis(level, 4)
        assert len(set(states)) == len(states)
        for state in states:
            assert list(state) == sorted(state)
            assert state_level(state) == level
            assert all(n >= 1 for n, _ in state)


def test_basis_index_inverts_the_basis() -> None:
    states = level_basis(3, 5)
    index = basis_index(3, 5)
    assert all(states[index[s]] == s for s in states)


def test_signature_matches_the_package_metric() -> None:
    """Same metric as everything else, just integer-typed."""
    assert np.array_equal(signature(26).astype(float), minkowski(26))


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ((), 1),
        (((1, 1),), 1),
        (((1, 0),), -1),
        (((2, 0),), -2),
        (((3, 0),), -3),
        (((1, 0), (1, 0)), 2),
        (((1, 0), (1, 0), (1, 0)), -6),
        (((1, 0), (1, 1)), -1),
        (((1, 1), (2, 2)), 2),
        (((2, 1), (2, 1)), 8),
    ],
)
def test_state_norm_by_hand(state: tuple, expected: int) -> None:
    r"""``(n eta)^k k!`` per distinct mode, worked out on paper.

    ``((2,1),(2,1))`` is ``(2 * (+1))^2 * 2! = 8``; ``((1,0),(1,0),(1,0))`` is
    ``(1 * (-1))^3 * 3! = -6``.
    """
    assert state_norm(state, 4) == expected


def test_norm_matches_direct_commutator_evaluation() -> None:
    r"""``<s|s>`` from the closed form against moving annihilators through.

    The bra is built by acting with the adjoints one at a time and reading off
    the coefficient of the vacuum.  Each ``alpha_n`` already carries its own
    ``eta`` from the commutator, so nothing else is inserted by hand -- and
    that is why the timelike states come out negative on their own.
    """
    dim = 5
    for state in level_basis(4, dim):
        vector: dict = {state: 1.0}
        for mode, index in reversed(state):
            vector = apply_alpha_dict(vector, mode, index, dim, np.zeros(dim))
        assert math.isclose(vector.get((), 0.0), float(state_norm(state, dim)))


@pytest.mark.parametrize("dim", [3, 6, 26])
def test_the_full_fock_space_always_has_ghosts(dim: int) -> None:
    """One timelike oscillator is a negative-norm state in every dimension.

    The critical dimension is not about *this*; it is about what survives the
    Virasoro constraints.
    """
    norms = gram_diagonal(2, dim)
    assert np.any(norms < 0)
    assert np.any(norms > 0)
    assert state_norm(((2, 0),), dim) == -2


def test_annihilation_and_creation_are_adjoint_in_the_indefinite_metric() -> None:
    r"""``G_{N+n} alpha_{-n} = alpha_n^T G_N``, the statement that
    :math:`(\alpha_{-n})^\dagger = \alpha_n`.

    Adjointness holds with respect to the *indefinite* Gram matrix.  Checking
    it against the Euclidean one would fail, and that failure is the physics.
    """
    dim, level = 4, 2
    alpha0 = alpha_zero(momentum_for_level(level, dim), 1.0)
    for n in (1, 2):
        up = alpha_matrix(level, dim, -n, alpha0)
        down = alpha_matrix(level + n, dim, n, alpha0)
        g_source = gram_diagonal(level, dim)
        g_target = gram_diagonal(level + n, dim)
        for mu in range(dim):
            left = g_target[:, None] * up[mu]
            right = down[mu].T * g_source[None, :]
            assert np.allclose(left, right)


def test_alpha_matrix_off_the_bottom_is_empty() -> None:
    """Lowering past the vacuum annihilates: an empty matrix, not an error."""
    alpha0 = alpha_zero(momentum_for_level(1, 4), 1.0)
    assert alpha_matrix(1, 4, 3, alpha0).shape == (4, 0, 4)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("intercept", [0.0, 1.0, 2.0])
def test_momentum_sits_on_the_mass_shell(level: int, intercept: float) -> None:
    r"""``alpha' p^2 = a - N`` in all three sign cases."""
    alpha_prime = 0.7
    p = momentum_for_level(level, 6, intercept, alpha_prime)
    eta = minkowski(6)
    assert math.isclose(alpha_prime * float(np.sum(eta * p * p)), intercept - level, abs_tol=1e-12)


def test_zero_mode_scales_with_alpha_prime() -> None:
    r"""``alpha_0 = sqrt(2 alpha') p``."""
    p = np.array([1.0, 0.0, 0.0, 2.0])
    assert np.allclose(alpha_zero(p, 2.0), math.sqrt(4.0) * p)


def test_apply_alpha_creation_and_annihilation() -> None:
    """Creation appends; annihilation pulls out ``k n eta`` and drops one copy."""
    dim = 4
    alpha0 = np.zeros(dim)
    assert apply_alpha(((1, 2),), -3, 1, dim, alpha0) == {((1, 2), (3, 1)): 1.0}
    assert apply_alpha(((2, 3),), 2, 3, dim, alpha0) == {(): 2.0}
    assert apply_alpha(((2, 0),), 2, 0, dim, alpha0) == {(): -2.0}
    assert apply_alpha(((2, 3), (2, 3)), 2, 3, dim, alpha0) == {((2, 3),): 4.0}
    assert apply_alpha(((1, 1),), 2, 1, dim, alpha0) == {}


def test_apply_alpha_zero_mode_is_diagonal() -> None:
    alpha0 = np.array([0.0, 5.0, 0.0, 0.0])
    assert apply_alpha(((1, 1),), 0, 1, 4, alpha0) == {((1, 1),): 5.0}
    assert apply_alpha(((1, 1),), 0, 2, 4, alpha0) == {}


def test_bad_index_and_level_are_rejected() -> None:
    with pytest.raises(ValueError):
        apply_alpha((), 1, 9, 4, np.zeros(4))
    with pytest.raises(ValueError):
        level_basis(-1, 4)
    with pytest.raises(ValueError):
        signature(1)
    with pytest.raises(ValueError):
        momentum_for_level(1, 1)
