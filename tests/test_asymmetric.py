"""Asymmetric orbifolds: twists of the charge lattice that are not motions of space.

The automorphism search is the load-bearing part, so it is checked as a group
(closed, inverses present, identity present) and against counts that have an
independent derivation: at the fully self-dual ``T^d`` the answer must be
``2^(2d) d!`` with a geometric subgroup of ``2^d d!``, and on the hexagonal
``T^2`` -- which has no enhanced symmetry -- it must be the order-12
automorphism group of the lattice and nothing asymmetric at all.

The physics claims are then re-derived rather than trusted: a geometric twist
reproduces the intercept ``orbifold.py`` computes from the Hurwitz zeta
function, and level matching is checked to coincide with "the two sides turn by
the same angles" across every background, which is where that statement in the
module docstring came from.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.compactification.asymmetric import (
    AsymmetricTwist,
    classify_automorphisms,
    is_geometric,
    momentum_action,
    momentum_frame,
    narain_automorphisms,
    phase_intercept,
    twist_phases,
)
from stringsim.compactification.orbifold import Orbifold
from stringsim.compactification.torus import (
    TorusBackground,
    basis_change,
    factorized_duality,
    gauge_algebra,
    is_odd_integer,
    odd_metric,
)

SELF_DUAL_1 = TorusBackground(np.array([[1.0]]))
GENERIC_1 = TorusBackground(np.array([[1.7]]))
SELF_DUAL_2 = TorusBackground(np.eye(2))
HEXAGONAL = TorusBackground(np.array([[2.0, -1.0], [-1.0, 2.0]]))
HALF_DUAL = TorusBackground(np.diag([1.0, 2.3]))
SELF_DUAL_3 = TorusBackground(np.eye(3))

ALL = [SELF_DUAL_1, GENERIC_1, SELF_DUAL_2, HEXAGONAL, HALF_DUAL, SELF_DUAL_3]


# --------------------------------------------------------------------------
# the momentum frame
# --------------------------------------------------------------------------


@pytest.mark.parametrize("background", ALL)
def test_the_frame_diagonalises_both_forms(background):
    """In the (l_L, l_R) frame H is the identity and eta is diag(I, -I)."""
    dim = background.dim
    frame = momentum_frame(background)
    inverse = np.linalg.inv(frame)
    signature = np.diag([1.0] * dim + [-1.0] * dim)
    assert np.allclose(inverse.T @ background.generalized_metric() @ inverse, np.eye(2 * dim))
    assert np.allclose(inverse.T @ odd_metric(dim) @ inverse, signature)


def test_an_element_that_moves_the_moduli_is_refused():
    """factorized_duality is a symmetry only at the self-dual radius."""
    with pytest.raises(ValueError, match="mixes left and right"):
        momentum_action(GENERIC_1, factorized_duality(1, 0))


# --------------------------------------------------------------------------
# the automorphism search
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("background", "expected"),
    [
        (SELF_DUAL_1, 4),
        (GENERIC_1, 2),
        (SELF_DUAL_2, 32),
        (HEXAGONAL, 12),
        (HALF_DUAL, 8),
        (SELF_DUAL_3, 384),
    ],
)
def test_automorphism_counts(background, expected):
    assert len(narain_automorphisms(background)) == expected


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_the_self_dual_torus_gives_two_to_the_2d_times_d_factorial(dim):
    """And the geometric part is the signed permutations, Aut of Z^d itself."""
    background = TorusBackground(np.eye(dim))
    twists = classify_automorphisms(background)
    geometric = [twist for twist in twists if twist.is_geometric]
    assert len(twists) == 2 ** (2 * dim) * math.factorial(dim)
    assert len(geometric) == 2**dim * math.factorial(dim)


@pytest.mark.parametrize("background", ALL)
def test_the_result_is_a_group(background):
    found = narain_automorphisms(background)
    rows = {tuple(np.rint(omega).astype(int).ravel()) for omega in found}
    assert tuple(np.eye(2 * background.dim).astype(int).ravel()) in rows
    for first in found:
        assert tuple(np.rint(np.linalg.inv(first)).astype(int).ravel()) in rows
        for second in found:
            assert tuple(np.rint(first @ second).astype(int).ravel()) in rows


@pytest.mark.parametrize("background", ALL)
def test_every_automorphism_preserves_both_forms(background):
    metric = background.generalized_metric()
    for omega in narain_automorphisms(background):
        assert is_odd_integer(omega)
        assert np.allclose(omega.T @ metric @ omega, metric, atol=1e-8)


@pytest.mark.parametrize("background", ALL)
def test_asymmetric_twists_appear_exactly_where_the_symmetry_is_enhanced(background):
    """No roots, no asymmetry: on a generic torus every symmetry is geometric."""
    twists = classify_automorphisms(background)
    asymmetric = [twist for twist in twists if twist.is_asymmetric]
    assert bool(asymmetric) == gauge_algebra(background).is_enhanced


@pytest.mark.parametrize("background", ALL)
def test_the_asymmetric_part_is_indexed_by_one_weyl_group(background):
    """|Aut| / |Aut_geometric| is the order of one side's Weyl group."""
    twists = classify_automorphisms(background)
    geometric = sum(twist.is_geometric for twist in twists)
    components = gauge_algebra(background).components_left
    # every root here has squared length 2 and the components are su(2)^k or
    # larger simply-laced pieces; for these backgrounds the left algebra is a
    # sum of su(2)s, so the Weyl order is 2 per component.
    weyl = 2 ** len(components)
    assert len(twists) == geometric * weyl


# --------------------------------------------------------------------------
# geometric or not
# --------------------------------------------------------------------------


def test_t_duality_is_not_geometric_and_the_reflection_is():
    assert not is_geometric(factorized_duality(1, 0))
    assert is_geometric(basis_change(np.array([[-1.0]])))
    left, right = momentum_action(SELF_DUAL_1, factorized_duality(1, 0))
    assert left.ravel()[0] == pytest.approx(1.0)
    assert right.ravel()[0] == pytest.approx(-1.0)


@pytest.mark.parametrize("background", ALL)
def test_a_basis_change_is_always_geometric(background):
    for omega in narain_automorphisms(background):
        if is_geometric(omega):
            twist = AsymmetricTwist(background, omega)
            assert twist.has_equal_phases
            assert twist.is_level_matched


# --------------------------------------------------------------------------
# level matching
# --------------------------------------------------------------------------


@pytest.mark.parametrize("background", ALL)
def test_level_matching_is_exactly_equal_phase_multisets(background):
    """The module docstring claims this; here it is re-derived every run."""
    for twist in classify_automorphisms(background):
        assert twist.is_level_matched == twist.has_equal_phases


@pytest.mark.parametrize("background", ALL)
def test_level_matched_twists_have_square_fixed_point_counts(background):
    for twist in classify_automorphisms(background):
        if not twist.is_level_matched or twist.fixed_points < 1e-9:
            continue
        degeneracy = twist.twisted_degeneracy
        assert degeneracy is not None
        assert degeneracy**2 == round(twist.fixed_points)


def test_the_self_dual_circle_has_asymmetric_twists_and_none_of_them_work():
    """T-duality alone is not a consistent orbifold; a shift would be needed."""
    twists = [t for t in classify_automorphisms(SELF_DUAL_1) if t.is_asymmetric]
    assert len(twists) == 2
    assert not any(twist.is_level_matched for twist in twists)
    duality = AsymmetricTwist(SELF_DUAL_1, factorized_duality(1, 0))
    assert duality.left_intercept == pytest.approx(1.0)
    assert duality.right_intercept == pytest.approx(1.0 - 1.0 / 16.0)
    assert duality.level_matching_defect == pytest.approx(1.0 / 8.0)


@pytest.mark.parametrize(
    ("background", "asymmetric", "matched"),
    [(SELF_DUAL_2, 24, 6), (SELF_DUAL_3, 336, 84), (HALF_DUAL, 4, 0)],
)
def test_how_many_survive(background, asymmetric, matched):
    twists = [t for t in classify_automorphisms(background) if t.is_asymmetric]
    assert len(twists) == asymmetric
    assert sum(twist.is_level_matched for twist in twists) == matched


def test_the_flagship_example():
    """An order-4 asymmetric twist of the self-dual T^2, with degeneracy 2."""
    found = [
        twist
        for twist in classify_automorphisms(SELF_DUAL_2)
        if twist.is_asymmetric and twist.is_level_matched and twist.fixed_points > 1e-9
    ]
    assert len(found) == 2
    for twist in found:
        assert twist.order == 4
        assert np.allclose(twist.left_phases, [0.25, 0.75])
        assert np.allclose(twist.right_phases, [0.25, 0.75])
        assert round(twist.fixed_points) == 4
        assert twist.twisted_degeneracy == 2


# --------------------------------------------------------------------------
# agreement with the symmetric machinery
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "orbifold",
    [Orbifold.z3_hexagonal(), Orbifold.z4_square(), Orbifold.z6_hexagonal()],
)
def test_a_geometric_twist_reproduces_the_orbifold_intercept(orbifold):
    """orbifold.py gets a_k from the Hurwitz zeta function; this must agree."""
    twist = AsymmetricTwist(orbifold.background, basis_change(orbifold.rotation))
    assert twist.is_geometric
    assert twist.order == orbifold.order
    assert twist.left_intercept == pytest.approx(orbifold.intercept(1), abs=1e-12)
    assert twist.right_intercept == pytest.approx(orbifold.intercept(1), abs=1e-12)
    assert twist.level_matching_defect == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(twist.left_phases, np.sort(orbifold.twist_phases(1)))


def test_phase_intercept_matches_the_untwisted_value():
    assert phase_intercept([0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert phase_intercept([0.5]) == pytest.approx(1.0 - 1.0 / 16.0)


def test_twist_phases_of_a_rotation():
    quarter = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert np.allclose(twist_phases(quarter), [0.25, 0.75])
    assert np.allclose(twist_phases(np.eye(2)), [0.0, 0.0])


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------


def test_bad_twists_are_refused():
    with pytest.raises(ValueError, match="must be 2x2"):
        AsymmetricTwist(SELF_DUAL_1, np.eye(4))
    with pytest.raises(ValueError, match="integer matrix"):
        AsymmetricTwist(SELF_DUAL_1, np.array([[1.0, 0.5], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="finite order"):
        AsymmetricTwist(SELF_DUAL_1, np.array([[1.0, 1.0], [0.0, 1.0]]))


def test_a_continuous_fixed_set_reports_no_degeneracy():
    """|det(1 - Omega)| = 0 means a fixed direction, not a fixed point count."""
    identity = AsymmetricTwist(SELF_DUAL_2, np.eye(4))
    assert identity.fixed_points == pytest.approx(0.0)
    assert identity.twisted_degeneracy is None
