"""Hodge numbers of toroidal orbifolds, and the sign discrete torsion puts on them.

The value of these tests is that the two halves of the answer are computed from
things with nothing in common. ``h11 - h21`` comes from a gcd of integer minors
-- how many lattice points a pair of rotations holds still. ``h11 + h21`` comes
from averaging a character over the group, plus one modulus per singular locus.
Nothing links them, so both coming out as non-negative integers is a check, and
that they reproduce three standard orbifolds is another.

``(51, 3)`` and ``(3, 51)`` for ``T^6/(Z_2 x Z_2)`` are the point: the two
differ only by the phase from ``torsion.py``, which changes nothing about the
geometry and flips the sign of every term in the Euler characteristic.
"""

from __future__ import annotations

import numpy as np
import pytest

from stringsim.compactification.hodge import (
    FixedSet,
    OrbifoldAction,
    blowup_moduli,
    euler_characteristic,
    fixed_locus,
    fixed_set,
    hodge_numbers,
    joint_euler,
    lattice_rotation,
    minor_gcd,
    untwisted_hodge,
    z2_z2_orbifold,
    z3_orbifold,
    z4_orbifold,
)
from stringsim.compactification.torsion import TorsionGroup

MAKERS = {"z3": z3_orbifold, "z4": z4_orbifold, "z2z2": z2_z2_orbifold}


# --------------------------------------------------------------------------
# fixed sets
# --------------------------------------------------------------------------


def test_minor_gcd_on_things_with_known_answers():
    assert minor_gcd(np.eye(3, dtype=int), 3) == 1
    assert minor_gcd(2 * np.eye(3, dtype=int), 3) == 8
    assert minor_gcd(np.diag([2, 3, 0]), 2) == 6
    assert minor_gcd(np.zeros((3, 3), dtype=int), 1) == 0
    assert minor_gcd(np.eye(3, dtype=int), 0) == 1
    with pytest.raises(ValueError, match="size must lie"):
        minor_gcd(np.eye(3, dtype=int), 4)


def test_fixed_set_of_a_reflection_is_the_half_periods():
    """1 - (-1) = 2, so 2^d classes: the well-known 2, 4, 16, 64."""
    for dim in (1, 2, 4, 6):
        found = fixed_set(2 * np.eye(dim, dtype=int))
        assert found == FixedSet(components=2**dim, dimension=0)
        assert found.euler_characteristic == 2**dim


def test_fixed_set_of_the_identity_is_the_whole_torus():
    found = fixed_set(np.zeros((6, 6), dtype=int))
    assert found.dimension == 6
    assert found.euler_characteristic == 0


def test_a_mixed_fixed_set_is_curves():
    """Two directions untouched, four reflected: 16 copies of T^2."""
    matrix = np.diag([0, 0, 2, 2, 2, 2])
    found = fixed_set(matrix)
    assert found == FixedSet(components=16, dimension=2)
    assert found.euler_characteristic == 0
    assert "T^2" in str(found)


def test_fixed_set_rejects_a_non_matrix():
    with pytest.raises(ValueError, match="expected a matrix"):
        fixed_set(np.zeros(4, dtype=int))


# --------------------------------------------------------------------------
# the action and its validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_the_standard_actions_are_representations(name):
    action = MAKERS[name]()
    group = action.group
    assert action.dim == 6
    for first in group.elements():
        for second in group.elements():
            product = action.rotations[first] @ action.rotations[second]
            assert np.array_equal(product, action.rotations[group.add(first, second)])
            assert np.allclose(
                np.array(action.phases[first]) * np.array(action.phases[second]),
                np.array(action.phases[group.add(first, second)]),
            )


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_every_action_is_calabi_yau(name):
    action = MAKERS[name]()
    for element in action.group.elements():
        assert np.prod(action.phases[element]) == pytest.approx(1.0)
    assert untwisted_hodge(action)[(3, 0)] == 1


def test_an_action_outside_su3_is_refused():
    """Phases that do not multiply to 1 are not a Calabi-Yau quotient."""
    group = TorsionGroup((2,))
    unit = np.eye(2, dtype=int)
    flip = np.array([[-1, 0], [0, -1]])
    # one torus reflected: the phases multiply to -1, so the holonomy is U(3)
    rotations = {
        (0,): lattice_rotation(unit, unit, unit),
        (1,): lattice_rotation(flip, unit, unit),
    }
    phases = {(0,): (1, 1, 1), (1,): (-1, 1, 1)}
    with pytest.raises(ValueError, match="not Calabi-Yau"):
        OrbifoldAction(group, rotations, phases)


def test_phases_that_are_not_the_eigenvalues_are_refused():
    group = TorsionGroup((2,))
    unit = np.eye(2, dtype=int)
    flip = np.array([[-1, 0], [0, -1]])
    rotations = {
        (0,): lattice_rotation(unit, unit, unit),
        (1,): lattice_rotation(flip, flip, unit),
    }
    phases = {(0,): (1, 1, 1), (1,): (1j, -1j, 1)}  # product 1, wrong eigenvalues
    with pytest.raises(ValueError, match="not the eigenvalues"):
        OrbifoldAction(group, rotations, phases)


def test_a_missing_element_is_refused():
    group = TorsionGroup((2,))
    with pytest.raises(ValueError, match="every group element"):
        OrbifoldAction(group, {(0,): np.eye(6, dtype=int)}, {(0,): (1, 1, 1)})


def test_lattice_rotation_builds_blocks():
    built = lattice_rotation(np.eye(2, dtype=int), 2 * np.eye(2, dtype=int))
    assert built.shape == (4, 4)
    assert np.array_equal(built, np.diag([1, 1, 2, 2]))


# --------------------------------------------------------------------------
# the fixed loci of the standard orbifolds
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("z3", {(1,): (27, 0), (2,): (27, 0)}),
        ("z4", {(1,): (16, 0), (2,): (16, 2), (3,): (16, 0)}),
        ("z2z2", {(1, 0): (16, 2), (0, 1): (16, 2), (1, 1): (16, 2)}),
    ],
)
def test_the_fixed_loci_are_the_known_ones(name, expected):
    action = MAKERS[name]()
    for element, (components, dimension) in expected.items():
        locus = fixed_locus(action, element)
        assert locus.components == components
        assert locus.dimension == dimension


def test_an_element_and_its_inverse_hold_the_same_set():
    for name in MAKERS:
        action = MAKERS[name]()
        for element in action.group.elements():
            assert fixed_locus(action, element) == fixed_locus(
                action, action.group.inverse(element)
            )


def test_z2_z2_has_no_isolated_points_from_a_single_element():
    """Its 64 points come only from pairs, which is why chi is a sum over pairs."""
    action = z2_z2_orbifold()
    for element in action.group.elements():
        assert fixed_locus(action, element).euler_characteristic == 0
    pairs = [
        (first, second)
        for first in action.group.elements()
        for second in action.group.elements()
        if joint_euler(action, first, second)
    ]
    assert len(pairs) == 6
    assert all(joint_euler(action, *pair) == 64 for pair in pairs)
    assert all(first != second for first, second in pairs)
    assert all(first != action.identity and second != action.identity for first, second in pairs)


# --------------------------------------------------------------------------
# the Euler characteristic
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expected"), [("z3", 72), ("z4", 48), ("z2z2", 96)])
def test_the_euler_characteristic_without_torsion(name, expected):
    action = MAKERS[name]()
    assert euler_characteristic(action) == pytest.approx(expected)
    assert euler_characteristic(action, action.group.pairings()[0]) == pytest.approx(expected)


def test_discrete_torsion_flips_the_sign():
    action = z2_z2_orbifold()
    trivial, torsion = action.group.pairings()
    assert euler_characteristic(action, trivial) == pytest.approx(96.0)
    assert euler_characteristic(action, torsion) == pytest.approx(-96.0)


@pytest.mark.parametrize("name", ["z3", "z4"])
def test_a_cyclic_orbifold_has_only_one_answer(name):
    """No pairs to weight, so no choice to make."""
    action = MAKERS[name]()
    assert len(action.group.pairings()) == 1


# --------------------------------------------------------------------------
# the untwisted forms
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"), [("z3", (9, 0)), ("z4", (5, 1)), ("z2z2", (3, 3))]
)
def test_the_untwisted_hodge_numbers(name, expected):
    diamond = untwisted_hodge(MAKERS[name]())
    assert (diamond[(1, 1)], diamond[(2, 1)]) == expected


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_the_diamond_has_the_symmetries_a_diamond_has(name):
    diamond = untwisted_hodge(MAKERS[name]())
    for p in range(4):
        for q in range(4):
            assert diamond[(p, q)] == diamond[(q, p)]  # complex conjugation
            assert diamond[(p, q)] == diamond[(3 - p, 3 - q)]  # Serre duality
    assert diamond[(0, 0)] == 1
    assert diamond[(1, 0)] == 0  # no continuous isometries left


def test_the_untwisted_forms_of_the_torus_itself():
    """With the trivial group nothing is projected out: T^6 has h11 = 9."""
    group = TorsionGroup((1,))
    action = OrbifoldAction(
        group, {(0,): np.eye(6, dtype=int)}, {(0,): (1, 1, 1)}
    )
    diamond = untwisted_hodge(action)
    assert diamond[(1, 1)] == 9
    assert diamond[(2, 1)] == 9
    assert euler_characteristic(action) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# putting it together
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expected"), [("z3", 27), ("z4", 32), ("z2z2", 48)])
def test_the_blow_up_count(name, expected):
    """Loci counted once per {g, g^-1}, since inverses hold the same set."""
    assert blowup_moduli(MAKERS[name]()) == expected


@pytest.mark.parametrize(
    ("name", "expected"), [("z3", (36, 0)), ("z4", (31, 7)), ("z2z2", (51, 3))]
)
def test_the_hodge_numbers_of_the_standard_orbifolds(name, expected):
    result = hodge_numbers(MAKERS[name]())
    assert (result.h11, result.h21) == expected
    assert result.h11 - result.h21 == result.euler // 2
    assert result.total == result.untwisted[0] + result.untwisted[1] + result.twisted


def test_discrete_torsion_produces_a_mirror():
    action = z2_z2_orbifold()
    trivial, torsion = action.group.pairings()
    plain, twisted = hodge_numbers(action, trivial), hodge_numbers(action, torsion)
    assert (plain.h11, plain.h21) == (51, 3)
    assert (twisted.h11, twisted.h21) == (3, 51)
    assert plain.euler == -twisted.euler
    assert plain.total == twisted.total == 54
    assert plain.untwisted == twisted.untwisted
    assert plain.twisted == twisted.twisted


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_the_two_routes_have_to_agree_on_parity(name):
    """The consistency check the module relies on, stated as a test."""
    action = MAKERS[name]()
    for pairing in action.group.pairings():
        chi = euler_characteristic(action, pairing)
        diamond = untwisted_hodge(action)
        total = diamond[(1, 1)] + diamond[(2, 1)] + blowup_moduli(action)
        assert (total + round(chi) // 2) % 2 == 0
        result = hodge_numbers(action, pairing)
        assert result.h11 >= 0 and result.h21 >= 0
