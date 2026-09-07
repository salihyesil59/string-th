"""Shifts on the Narain lattice, and the twists they repair.

``test_asymmetric.py`` establishes that the self-dual circle's T-duality twist
misses level matching by exactly ``1/8``. These tests find what closes it, and
the value is that nothing about the answer was put in: the search enumerates
rational shifts, computes the order each one forces, and keeps the ones whose
defect vanishes. That halves and thirds do nothing and quarters work is a
result, not a setting.

The three things a shift can change are checked separately -- the order, the
level-matching condition through ``<v,v>``, and whether any fixed points remain
-- because a shift that repairs the first two while leaving the orbifold with
fixed points is a different object from one that acts freely.
"""

from __future__ import annotations

import numpy as np
import pytest

from stringsim.compactification.asymmetric import (
    AsymmetricTwist,
    classify_automorphisms,
    shifts_that_close,
)
from stringsim.compactification.torus import (
    TorusBackground,
    basis_change,
    factorized_duality,
    odd_metric,
)

CIRCLE = TorusBackground(np.array([[1.0]]))
HALF_DUAL = TorusBackground(np.diag([1.0, 2.3]))
DUALITY = factorized_duality(1, 0)


# --------------------------------------------------------------------------
# what a shift does to the order
# --------------------------------------------------------------------------


def test_no_shift_leaves_everything_as_it_was():
    plain = AsymmetricTwist(CIRCLE, DUALITY)
    zero = AsymmetricTwist(CIRCLE, DUALITY, np.zeros(2))
    assert plain.order == zero.order == 2
    assert plain.rotation_order == 2
    assert not plain.has_shift
    assert plain.shift_norm == 0.0
    assert plain.level_matching_defect == pytest.approx(zero.level_matching_defect)
    assert plain.level_matching_defect == pytest.approx(0.125)


def test_a_shift_can_raise_the_order():
    """(1 + Omega) v has to land on the lattice before the element closes."""
    quarter = AsymmetricTwist(CIRCLE, DUALITY, np.array([0.25, 0.25]))
    assert quarter.rotation_order == 2
    assert quarter.order == 4
    assert quarter.has_shift
    half = AsymmetricTwist(CIRCLE, DUALITY, np.array([0.25, 0.75]))
    assert half.order == 2  # (1 + Omega) v = (1, 1) already integral


def test_an_irrational_shift_never_closes():
    with pytest.raises(ValueError, match="never closes"):
        AsymmetricTwist(CIRCLE, DUALITY, np.array([np.sqrt(2.0) / 3.0, 0.0]))


@pytest.mark.parametrize("denominator", [2, 3, 4, 5])
def test_a_pure_shift_has_the_order_of_its_denominator(denominator):
    shift = np.array([0.0, 1.0 / denominator])
    twist = AsymmetricTwist(CIRCLE, np.eye(2), shift)
    assert twist.order == denominator
    assert twist.rotation_order == 1


# --------------------------------------------------------------------------
# the level-matching condition
# --------------------------------------------------------------------------


def test_the_shift_enters_only_through_its_norm():
    """<v, v> = v^T eta v, and on a circle that is 2 n w."""
    form = odd_metric(1)
    for entries in ([1, 1], [1, 3], [2, 2], [3, 1]):
        shift = np.array(entries) / 4.0
        twist = AsymmetricTwist(CIRCLE, DUALITY, shift)
        assert twist.shift_norm == pytest.approx(shift @ form @ shift)
        assert twist.shift_norm == pytest.approx(2.0 * shift[0] * shift[1])


@pytest.mark.parametrize("denominator", [2, 3, 4, 5, 7])
def test_pure_momentum_and_winding_shifts_are_always_consistent(denominator):
    """Their norm vanishes, so they cannot break level matching at any order."""
    for entries in ([0, 1], [1, 0]):
        shift = np.array(entries) / denominator
        twist = AsymmetricTwist(CIRCLE, np.eye(2), shift)
        assert twist.shift_norm == pytest.approx(0.0)
        assert twist.is_level_matched


def test_a_mixed_shift_usually_is_not():
    twist = AsymmetricTwist(CIRCLE, np.eye(2), np.array([1 / 3, 1 / 3]))
    assert twist.shift_norm == pytest.approx(2.0 / 9.0)
    assert not twist.is_level_matched


def test_the_twisted_energies_include_the_shift():
    """E = -a + v^2/2 on each side, and their difference is what must quantise."""
    twist = AsymmetricTwist(CIRCLE, DUALITY, np.array([0.25, 0.25]))
    left, right = twist.twisted_energies
    left_norm, right_norm = twist.shift_momenta()
    assert left == pytest.approx(-twist.left_intercept + left_norm / 2.0)
    assert right == pytest.approx(-twist.right_intercept + right_norm / 2.0)
    assert left_norm - right_norm == pytest.approx(twist.shift_norm)
    assert left - right == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------


@pytest.mark.parametrize("denominator", [2, 3, 5, 6])
def test_nothing_but_quarters_repairs_t_duality(denominator):
    assert shifts_that_close(CIRCLE, DUALITY, denominator) == []


def test_quarters_repair_it_and_the_answer_is_forced():
    found = shifts_that_close(CIRCLE, DUALITY, 4)
    assert len(found) == 2
    shifts = sorted(tuple(np.round(twist.shift, 6)) for twist in found)
    assert shifts == [(0.25, 0.25), (0.75, 0.75)]
    for twist in found:
        assert twist.is_level_matched
        assert twist.order == 4
        assert twist.is_freely_acting


def test_the_repaired_twist_really_does_close():
    """N (E_L - E_R) has to be an integer, and it is checked directly."""
    for twist in shifts_that_close(CIRCLE, DUALITY, 4):
        left, right = twist.twisted_energies
        value = twist.order * (left - right)
        assert value == pytest.approx(round(value), abs=1e-12)


def test_the_search_validates_its_denominator():
    with pytest.raises(ValueError, match="denominator must be"):
        shifts_that_close(CIRCLE, DUALITY, 0)


def test_a_twist_that_already_works_stays_working_with_no_shift():
    """The zero shift is always in the search's output for a matched twist."""
    geometric = AsymmetricTwist(CIRCLE, basis_change(np.array([[-1.0]])))
    assert geometric.is_level_matched
    found = shifts_that_close(CIRCLE, geometric.omega, 2)
    assert any(np.max(np.abs(twist.shift)) == 0.0 for twist in found)


# --------------------------------------------------------------------------
# fixed points
# --------------------------------------------------------------------------


def test_a_rotation_alone_is_never_freely_acting():
    for background in (CIRCLE, HALF_DUAL):
        for twist in classify_automorphisms(background):
            assert not twist.is_freely_acting


def test_the_repairing_shift_removes_the_fixed_points():
    r"""(1 - Omega) x = v has no solution: w + n = 1/2 is not an integer."""
    twist = AsymmetricTwist(CIRCLE, DUALITY, np.array([0.25, 0.25]))
    assert twist.is_freely_acting
    kept = AsymmetricTwist(CIRCLE, DUALITY, np.array([0.25, 0.75]))
    assert not kept.is_freely_acting  # w + n = 1 is


def test_a_shift_in_the_image_leaves_the_fixed_points_alone():
    """Shifting by something (1 - Omega) can produce just moves them."""
    matrix = np.eye(2) - DUALITY
    for column in range(2):
        shift = matrix[:, column] / 2.0
        twist = AsymmetricTwist(CIRCLE, DUALITY, shift)
        assert not twist.is_freely_acting


# --------------------------------------------------------------------------
# across the backgrounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("background", [CIRCLE, HALF_DUAL])
def test_every_repair_is_a_genuine_repair(background):
    """Whatever the search returns must satisfy the condition it was searching for."""
    for twist in classify_automorphisms(background):
        if twist.is_level_matched:
            continue
        for repaired in shifts_that_close(background, twist.omega, 4):
            assert repaired.is_level_matched
            assert repaired.level_matching_defect < 1e-9
            assert np.array_equal(repaired.omega, twist.omega)
            assert repaired.order % twist.rotation_order == 0


def test_the_repairs_do_not_all_raise_the_order():
    """Most do, because the shift has to work its way back onto the lattice."""
    raised = kept = 0
    for twist in classify_automorphisms(HALF_DUAL):
        if twist.is_level_matched:
            continue
        for repaired in shifts_that_close(HALF_DUAL, twist.omega, 4):
            if repaired.order > twist.order:
                raised += 1
            else:
                kept += 1
    assert raised > 0
    assert kept > 0
    assert raised > kept
