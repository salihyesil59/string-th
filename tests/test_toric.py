"""Mirror symmetry: a polytope, its dual, and the two Hodge numbers swapping.

``hodge.py`` does this for orbifolds of a torus.  Hypersurfaces need Batyrev's
combinatorics instead, where the mirror is the same formula applied to the dual
polytope -- so what needs checking is not the exchange but the numbers, and the
Euler characteristic from Chern classes is what checks them.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from stringsim.compactification.toric import (
    Polytope,
    batyrev_count,
    greene_plesser_group,
    greene_plesser_order,
    hodge_numbers,
    hypersurface_euler,
    projective_space,
    weighted_projective,
)

FAMILIES = [
    ([1, 1, 1, 1, 1], 1, 101),
    ([1, 1, 1, 1, 2], 1, 103),
    ([1, 1, 1, 1, 4], 1, 149),
    ([1, 1, 1, 2, 5], 1, 145),
    ([1, 1, 1, 6, 9], 2, 272),
]


# --------------------------------------------------------------------------
# the polytope
# --------------------------------------------------------------------------


def test_the_quintic_polytope_counts_its_monomials() -> None:
    r"""Six lattice points in :math:`\Delta^*` and 126 in :math:`\Delta`.

    The 126 are the degree-5 monomials in five variables, ``C(9,4)``, which is
    what the dual polytope of a projective space is for.
    """
    delta_star = projective_space(4)
    assert delta_star.is_reflexive
    assert len(delta_star.lattice_points) == 6
    assert len(delta_star.dual().lattice_points) == 126
    assert len(delta_star.dual().lattice_points) == 126 == (9 * 8 * 7 * 6) // 24


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_duality_is_an_involution(dim: int) -> None:
    """The dual of the dual is the polytope again, up to the order of vertices."""
    polytope = projective_space(dim)
    twice = polytope.dual().dual()
    assert sorted(map(tuple, twice.vertices)) == sorted(map(tuple, polytope.vertices))


def test_the_faces_partition_the_lattice_points() -> None:
    r"""Grouping points by which facets they sit on *is* the face lattice.

    Every lattice point lands in exactly one group, the groups are the relative
    interiors, and the origin -- on no facet -- is the interior of the whole
    polytope.
    """
    polytope = projective_space(4)
    faces = polytope.faces
    assert sum(face.interior_count for face in faces.values()) == len(
        polytope.lattice_points
    )
    assert faces[()].dimension == 4
    assert faces[()].interior == ((0, 0, 0, 0),)
    # The simplex has six lattice points: the origin and the five vertices.  Its
    # facets are unimodular and carry nothing in their relative interiors, so
    # they do not appear at all -- which is what the sums in Batyrev's formula
    # want, since they run over the interior counts.
    assert len(polytope.faces_of_dimension(0)) == 5
    assert polytope.faces_of_dimension(3) == []
    for face in faces.values():
        assert 0 <= face.dimension <= 4

    # The dual does have facets with interior points, and they are what the
    # first correction term subtracts.
    dual = polytope.dual()
    facets = dual.faces_of_dimension(3)
    assert len(facets) == 5
    assert all(face.interior_count > 0 for face in facets)
    assert sum(face.interior_count for face in facets) == 5 * 4


def test_a_non_reflexive_polytope_is_refused() -> None:
    """Twice the simplex has a non-integral dual, and the code says so."""
    doubled = Polytope(2 * projective_space(3).vertices)
    assert not doubled.is_reflexive
    with pytest.raises(ValueError, match="not a lattice polytope"):
        doubled.dual()
    with pytest.raises(ValueError, match="reflexive"):
        batyrev_count(doubled)


# --------------------------------------------------------------------------
# the Hodge numbers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("weights", "h11", "h21"), FAMILIES)
def test_the_famous_families(weights, h11: int, h21: int) -> None:
    r"""``(1, 101)`` for the quintic, and the other one-parameter models.

    Nothing in the computation knows these numbers: it counts lattice points on
    faces of a polytope and of its dual.
    """
    polytope = weighted_projective(weights)
    assert polytope.is_reflexive
    numbers = hodge_numbers(polytope)
    assert (numbers.h11, numbers.h21) == (h11, h21)


def test_the_quintic_euler_characteristic_from_two_sides() -> None:
    r""":math:`\chi = -200`, from a polytope and from :math:`\int c_3`.

    The second route is the classical one: :math:`c(T) = (1+H)^5/(1+5H)`
    restricted to the hypersurface, with no polytope in it at all.
    """
    numbers = hodge_numbers(projective_space(4))
    assert numbers.euler == -200
    assert hypersurface_euler(5, 4) == -200
    assert numbers.euler == hypersurface_euler(5, 4)


def test_the_dual_polytope_is_the_mirror() -> None:
    r"""``(1, 101)`` and ``(101, 1)``, and the Euler characteristics opposite.

    The exchange is not discovered: :func:`hodge_numbers` calls one function on
    a polytope and on its dual, so the mirror is the shape of the formula.  What
    the test adds is that the pair is the *right* pair.
    """
    quintic = hodge_numbers(projective_space(4))
    mirror = hodge_numbers(projective_space(4).dual())
    assert (mirror.h11, mirror.h21) == (quintic.h21, quintic.h11)
    assert mirror.euler == -quintic.euler == 200
    assert quintic.mirror == mirror
    assert "chi" in str(quintic)


@pytest.mark.parametrize(("weights", "h11", "h21"), FAMILIES)
def test_every_mirror_pair_has_opposite_euler(weights, h11: int, h21: int) -> None:
    del h11, h21
    polytope = weighted_projective(weights)
    assert hodge_numbers(polytope).euler == -hodge_numbers(polytope.dual()).euler


# --------------------------------------------------------------------------
# where the formula stops
# --------------------------------------------------------------------------


def test_in_three_dimensions_it_is_not_the_hodge_number() -> None:
    r"""Batyrev's count for a K3 is the Picard number, and the Euler shows it.

    Every K3 has :math:`h^{1,1} = 20`.  The quartic's polytope gives 1 and its
    dual gives 19, so the two numbers are not Hodge numbers -- they are the part
    of :math:`H^{1,1}` the toric divisors reach.  The independent
    :math:`\chi = 24`, with :math:`h^{2,0} = 1`, forces :math:`b_2 = 22` and
    :math:`h^{1,1} = 20`.
    """
    quartic = projective_space(3)
    assert batyrev_count(quartic) == 1
    assert batyrev_count(quartic.dual()) == 19
    assert hypersurface_euler(4, 3) == 24
    assert 2 * (batyrev_count(quartic) - batyrev_count(quartic.dual())) != 24
    with pytest.raises(ValueError, match="four-dimensional"):
        hodge_numbers(quartic)


def test_the_two_k3_numbers_do_not_always_sum_to_twenty() -> None:
    """They do for the simplest families and not for all, and that is the point.

    ``P(1,1,2,4)`` gives 3 and 18.  The missing divisors are not visible to any
    polytope, which is why the module refuses to call these Hodge numbers.
    """
    sums = {}
    for weights in ([1, 1, 1, 1], [1, 1, 1, 3], [1, 1, 4, 6], [1, 1, 2, 4], [1, 2, 2, 5]):
        polytope = weighted_projective(weights)
        if not polytope.is_reflexive:  # pragma: no cover - all of these are
            continue
        sums[tuple(weights)] = batyrev_count(polytope) + batyrev_count(polytope.dual())
    assert sums[(1, 1, 1, 1)] == 20
    assert sums[(1, 1, 1, 3)] == 20
    assert sums[(1, 1, 4, 6)] == 20
    assert sums[(1, 1, 2, 4)] != 20
    assert sums[(1, 2, 2, 5)] != 20


@pytest.mark.parametrize(
    ("degree", "ambient", "expected"),
    [(4, 3, 24), (5, 4, -200), (3, 2, 0), (2, 3, 4), (6, 5, 2610)],
)
def test_the_classical_euler_characteristic(degree: int, ambient: int, expected: int) -> None:
    r"""``0`` for a plane cubic -- an elliptic curve -- and ``4`` for a quadric surface."""
    assert hypersurface_euler(degree, ambient) == expected


# --------------------------------------------------------------------------
# Greene and Plesser
# --------------------------------------------------------------------------


def test_the_quintic_orbifold_group_has_order_one_hundred_and_twenty_five() -> None:
    r"""``5^3``, and it really is a group of that size.

    The phases with :math:`\sum a_i \equiv 0 \pmod 5` number :math:`5^4`, and
    the five projective scalings are divided out.  Both are enumerated.
    """
    elements, scalings = greene_plesser_group([1, 1, 1, 1, 1])
    assert len(elements) == 5**4
    assert scalings == 5
    assert greene_plesser_order([1, 1, 1, 1, 1]) == 125 == 5**3


def test_the_phases_are_closed_under_addition() -> None:
    """A subgroup of the product of cyclic groups, checked rather than assumed."""
    elements, _ = greene_plesser_group([1, 1, 1, 1, 1])
    members = set(elements)
    assert (0, 0, 0, 0, 0) in members
    for left, right in itertools.islice(itertools.product(elements, repeat=2), 400):
        assert tuple((a + b) % 5 for a, b in zip(left, right, strict=True)) in members


def test_counting_the_scalings_is_not_the_smallest_exponent() -> None:
    r"""The obvious guess is right for the quintic and wrong elsewhere.

    ``P(1,1,1,2,5)`` has exponents ``(10,10,10,5,2)``; the smallest is 2 and
    there are 10 scalings.  Enumerating them costs nothing and removes the
    guess.
    """
    _, scalings = greene_plesser_group([1, 1, 1, 2, 5])
    assert scalings == 10
    assert min(10, 10, 10, 5, 2) == 2
    assert greene_plesser_order([1, 1, 1, 2, 5]) == 100


def test_bad_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        greene_plesser_order([1, 1, 0])
    with pytest.raises(ValueError, match="does not divide"):
        greene_plesser_order([1, 1, 4])
    with pytest.raises(ValueError, match="weight equal to 1"):
        weighted_projective([2, 2, 4])
    with pytest.raises(ValueError, match="two dimensions"):
        projective_space(1)
    with pytest.raises(ValueError, match="vertices"):
        Polytope(np.eye(3, dtype=np.int64))
