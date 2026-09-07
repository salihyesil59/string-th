"""Orbifold groups that need not be abelian.

Two of the three ingredients in ``test_hodge.py`` never needed commutativity --
the Euler characteristic sums over commuting pairs, which for an abelian group
happens to be all of them, and the untwisted Hodge numbers average a character.
The check that the general code is right is that re-presenting the three
verified abelian orbifolds through it gives exactly the same numbers back.

The group theory carries its own check: the number of commuting pairs must equal
``|G|`` times the number of conjugacy classes, and the two are computed by
unrelated enumerations. A closure that missed an element, or a conjugation with
a wrong inverse, would break it.

The blow-up count does *not* generalise, and that is asserted too: for a
non-abelian group the twisted sectors are labelled by conjugacy classes and
projected by centralizers, so ``hodge_numbers`` stays where it was verified.
"""

from __future__ import annotations

import cmath

import numpy as np
import pytest

from stringsim.compactification.hodge import (
    GroupAction,
    block_permutation,
    close_group,
    delta27_orbifold,
    elementary_symmetric,
    euler_characteristic,
    from_abelian,
    lattice_rotation,
    untwisted_hodge,
    z2_z2_orbifold,
    z3_orbifold,
    z4_orbifold,
)

MAKERS = {"z3": z3_orbifold, "z4": z4_orbifold, "z2z2": z2_z2_orbifold}
OMEGA = cmath.exp(2j * cmath.pi / 3.0)
ROTATION = np.array([[0, -1], [1, -1]], dtype=np.int64)
UNIT = np.eye(2, dtype=np.int64)


def z3_squared() -> GroupAction:
    """The abelian ``Z_3 x Z_3`` inside Delta(27)."""
    return close_group(
        [
            (
                lattice_rotation(UNIT, ROTATION, ROTATION @ ROTATION),
                np.diag([1.0 + 0j, OMEGA, OMEGA**2]),
            ),
            (
                lattice_rotation(ROTATION @ ROTATION, UNIT, ROTATION),
                np.diag([OMEGA**2, 1.0 + 0j, OMEGA]),
            ),
        ]
    )


# --------------------------------------------------------------------------
# the character, without diagonalising
# --------------------------------------------------------------------------


def test_elementary_symmetric_on_things_with_known_answers():
    identity = np.eye(3, dtype=complex)
    assert elementary_symmetric(0, identity) == pytest.approx(1.0)
    assert elementary_symmetric(1, identity) == pytest.approx(3.0)
    assert elementary_symmetric(2, identity) == pytest.approx(3.0)
    assert elementary_symmetric(3, identity) == pytest.approx(1.0)
    assert elementary_symmetric(4, identity) == pytest.approx(0.0)


def test_it_matches_the_eigenvalues_for_a_permutation():
    """The point of using minors: a permutation is never diagonalised."""
    cyclic = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
    eigenvalues = np.linalg.eigvals(cyclic)
    assert elementary_symmetric(1, cyclic) == pytest.approx(np.sum(eigenvalues), abs=1e-9)
    assert elementary_symmetric(3, cyclic) == pytest.approx(np.prod(eigenvalues), abs=1e-9)
    assert elementary_symmetric(1, cyclic) == pytest.approx(0.0, abs=1e-12)
    assert elementary_symmetric(3, cyclic) == pytest.approx(1.0, abs=1e-12)


def test_it_agrees_with_the_diagonal_case():
    rng = np.random.default_rng(3)
    for _ in range(10):
        values = rng.normal(size=3) + 1j * rng.normal(size=3)
        matrix = np.diag(values)
        assert elementary_symmetric(1, matrix) == pytest.approx(np.sum(values))
        assert elementary_symmetric(3, matrix) == pytest.approx(np.prod(values))


# --------------------------------------------------------------------------
# closing a group
# --------------------------------------------------------------------------


def test_block_permutation_is_what_it_says():
    matrix = block_permutation((1, 2, 0))
    assert matrix.shape == (6, 6)
    assert np.trace(matrix) == 0
    assert np.array_equal(np.linalg.matrix_power(matrix, 3), np.eye(6, dtype=np.int64))
    assert np.array_equal(block_permutation((0, 1, 2)), np.eye(6, dtype=np.int64))
    with pytest.raises(ValueError, match="must be a permutation"):
        block_permutation((0, 0, 1))


def test_delta27_is_the_group_it_claims_to_be():
    group = delta27_orbifold()
    assert group.size == 27
    assert not group.is_abelian
    classes = group.conjugacy_classes()
    assert len(classes) == 11
    assert sorted(len(entry) for entry in classes) == [1, 1, 1] + [3] * 8
    assert sum(len(entry) for entry in classes) == 27


def test_the_commuting_pair_identity():
    """|G| x (number of classes), from two unrelated enumerations."""
    for group in (z3_squared(), delta27_orbifold(), from_abelian(z4_orbifold())):
        pairs = group.commuting_pairs()
        assert len(pairs) == group.size * len(group.conjugacy_classes())


def test_orbit_stabiliser():
    group = delta27_orbifold()
    for entry in group.conjugacy_classes():
        centralizer = group.centralizer(entry[0])
        assert len(entry) * len(centralizer) == group.size


def test_an_abelian_group_is_all_pairs_and_all_singletons():
    group = z3_squared()
    assert group.is_abelian
    assert len(group.commuting_pairs()) == group.size**2
    assert all(len(entry) == 1 for entry in group.conjugacy_classes())


def test_generators_that_do_not_close_are_refused():
    shear = np.array([[1, 1], [0, 1]], dtype=np.int64)
    infinite = lattice_rotation(shear, UNIT, UNIT)
    with pytest.raises(ValueError, match="do not close"):
        close_group([(infinite, np.eye(3, dtype=complex))], cap=64)
    with pytest.raises(ValueError, match="at least one generator"):
        close_group([])


def test_an_action_outside_su3_is_refused():
    flip = np.diag([-1, -1, 1, 1, 1, 1]).astype(np.int64)
    with pytest.raises(ValueError, match="not Calabi-Yau"):
        close_group([(flip, np.diag([-1.0 + 0j, 1.0 + 0j, 1.0 + 0j]))])


# --------------------------------------------------------------------------
# the general path reproduces the verified one
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_the_general_euler_characteristic_agrees(name):
    action = MAKERS[name]()
    assert from_abelian(action).euler_characteristic() == pytest.approx(
        euler_characteristic(action)
    )


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_the_general_untwisted_hodge_agrees(name):
    action = MAKERS[name]()
    general = from_abelian(action).untwisted_hodge()
    abelian = untwisted_hodge(action)
    for key, value in abelian.items():
        assert general[key] == value


@pytest.mark.parametrize("name", sorted(MAKERS))
def test_the_bridge_preserves_the_group(name):
    action = MAKERS[name]()
    general = from_abelian(action)
    assert general.size == action.group.size
    assert general.is_abelian


# --------------------------------------------------------------------------
# what Delta(27) gives
# --------------------------------------------------------------------------


def test_delta27_stays_calabi_yau():
    group = delta27_orbifold()
    diamond = group.untwisted_hodge()
    assert diamond[(3, 0)] == 1
    assert diamond[(0, 0)] == 1
    for holomorphic in group.holomorphic:
        assert np.linalg.det(holomorphic) == pytest.approx(1.0)


def test_the_untwisted_forms_shrink_as_the_group_grows():
    """9 for Z_3, 3 for Z_3 x Z_3, 1 for Delta(27): the projection biting."""
    counts = [
        from_abelian(z3_orbifold()).untwisted_hodge()[(1, 1)],
        z3_squared().untwisted_hodge()[(1, 1)],
        delta27_orbifold().untwisted_hodge()[(1, 1)],
    ]
    assert counts == [9, 3, 1]


def test_delta27_euler_characteristic():
    group = delta27_orbifold()
    value = group.euler_characteristic()
    assert value == pytest.approx(72.0)
    assert round(value) % 2 == 0  # chi = 2(h11 - h21), so it cannot be odd


def test_every_diamond_has_the_symmetries_a_diamond_has():
    for group in (z3_squared(), delta27_orbifold()):
        diamond = group.untwisted_hodge()
        for p in range(4):
            for q in range(4):
                assert diamond[(p, q)] == diamond[(q, p)]
                assert diamond[(p, q)] == diamond[(3 - p, 3 - q)]


def test_the_group_action_validates_its_input():
    with pytest.raises(ValueError, match="one holomorphic matrix per"):
        GroupAction(lattice=(np.eye(6, dtype=np.int64),), holomorphic=())
    with pytest.raises(ValueError, match="at least the identity"):
        GroupAction(lattice=(), holomorphic=())
