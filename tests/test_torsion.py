"""Discrete torsion: the phases the partition-function blocks may carry.

Two things are worth checking and both are checked here two ways. The number of
consistent phase assignments is found by brute enumeration over generator
values, filtered by the axioms; independently, the structure theorem says it
should be ``prod_{i<j} gcd(N_i, N_j)``, and the two must agree for every group
tried. And the axioms themselves are supposed to be equivalent to modular
invariance of the weights, so every pairing the enumeration returns is put back
through the ``T`` and ``S`` transformations on the whole grid of pairs.

The spectrum side is then arithmetic that has to close: the torsioned and
untorsioned projections of a twisted sector must be non-negative integers whose
sum is the projection by the twist's own subgroup, level by level.
"""

from __future__ import annotations

import numpy as np
import pytest

from stringsim.compactification.torsion import (
    TorsionGroup,
    projected_degeneracies,
    twisted_character,
)

# Z_2 x Z_2 on T^4: each generator flips a different pair of directions.
Z2Z2 = TorsionGroup((2, 2))
T4_PHASES = {
    (0, 0): [0.0, 0.0, 0.0, 0.0],
    (1, 0): [0.5, 0.5, 0.0, 0.0],
    (0, 1): [0.0, 0.0, 0.5, 0.5],
    (1, 1): [0.5, 0.5, 0.5, 0.5],
}


# --------------------------------------------------------------------------
# the group itself
# --------------------------------------------------------------------------


def test_group_arithmetic():
    group = TorsionGroup((2, 3))
    assert group.size == 6
    assert group.exponent == 6
    assert len(group.elements()) == 6
    assert group.elements()[0] == (0, 0)
    assert group.add((1, 2), (1, 2)) == (0, 1)
    assert group.inverse((1, 2)) == (1, 1)
    assert group.add((1, 2), group.inverse((1, 2))) == (0, 0)


def test_bad_orders_are_refused():
    with pytest.raises(ValueError, match="positive integers"):
        TorsionGroup(())
    with pytest.raises(ValueError, match="positive integers"):
        TorsionGroup((2, 0))


# --------------------------------------------------------------------------
# how many pairings there are
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "orders", [(2,), (5,), (6,), (2, 2), (2, 3), (2, 4), (3, 3), (4, 4), (2, 2, 2)]
)
def test_the_enumeration_matches_the_product_of_gcds(orders):
    """The count is a result of the search, and the structure theorem checks it."""
    group = TorsionGroup(orders)
    structure = group.torsion_group()
    expected = int(np.prod(structure)) if structure else 1
    assert len(group.pairings()) == expected


@pytest.mark.parametrize("orders", [(2,), (5,), (6,), (12,)])
def test_a_cyclic_orbifold_has_no_discrete_torsion(orders):
    group = TorsionGroup(orders)
    assert group.torsion_group() == ()
    pairings = group.pairings()
    assert len(pairings) == 1
    for g in group.elements():
        for h in group.elements():
            assert group.phase(pairings[0], g, h) == pytest.approx(1.0)


def test_z2_squared_is_the_smallest_group_with_a_choice():
    assert TorsionGroup((2, 2)).torsion_group() == (2,)
    trivial, torsion = TorsionGroup((2, 2)).pairings()
    group = TorsionGroup((2, 2))
    assert group.phase(trivial, (1, 0), (0, 1)) == pytest.approx(1.0)
    assert group.phase(torsion, (1, 0), (0, 1)) == pytest.approx(-1.0)
    assert group.phase(torsion, (0, 1), (1, 0)) == pytest.approx(-1.0)


@pytest.mark.parametrize("orders", [(2, 2), (2, 4), (3, 3), (2, 2, 2)])
def test_every_pairing_is_alternating_and_bilinear(orders):
    group = TorsionGroup(orders)
    elements = group.elements()
    for exponents in group.pairings():
        for g in elements:
            assert group.phase(exponents, g, g) == pytest.approx(1.0)
            for h in elements:
                assert group.phase(exponents, g, h) * group.phase(
                    exponents, h, g
                ) == pytest.approx(1.0)
                for k in elements:
                    combined = group.phase(exponents, group.add(g, h), k)
                    separate = group.phase(exponents, g, k) * group.phase(exponents, h, k)
                    assert combined == pytest.approx(separate)


@pytest.mark.parametrize("orders", [(2,), (2, 2), (2, 4), (3, 3), (2, 2, 2)])
def test_every_pairing_survives_T_and_S(orders):
    """The axioms are meant to *be* modular invariance; this is that claim."""
    group = TorsionGroup(orders)
    assert all(group.is_modular_consistent(exponents) for exponents in group.pairings())


def test_a_bad_pairing_is_rejected():
    """A phase that is not alternating fails both the search and the T/S check."""
    group = TorsionGroup((2, 2))
    bad = np.array([[1.0, 0.0], [0.0, 0.0]])  # epsilon(e1, e1) = -1
    assert group.phase(bad, (1, 0), (1, 0)) == pytest.approx(-1.0)
    assert not group.is_modular_consistent(bad)
    found = {tuple(exponents.ravel()) for exponents in group.pairings()}
    assert tuple(bad.ravel()) not in found


# --------------------------------------------------------------------------
# the projector
# --------------------------------------------------------------------------


def test_the_untwisted_projector_is_never_weighted():
    """epsilon(1, h) = 1, so torsion cannot touch the untwisted sector."""
    for orders in [(2, 2), (3, 3), (2, 2, 2)]:
        group = TorsionGroup(orders)
        identity = tuple([0] * len(orders))
        for exponents in group.pairings():
            weights = group.projector_weights(exponents, identity)
            assert all(weight == pytest.approx(1.0) for weight in weights)


def test_the_z2_squared_twisted_weights_are_plus_plus_minus_minus():
    group = TorsionGroup((2, 2))
    _, torsion = group.pairings()
    weights = [round(w.real) for w in group.projector_weights(torsion, (1, 0))]
    assert weights == [1, -1, 1, -1]  # over (1, g2, g1, g1g2) in element order
    assert [round(w.imag, 12) for w in group.projector_weights(torsion, (1, 0))] == [0] * 4


# --------------------------------------------------------------------------
# characters and counts
# --------------------------------------------------------------------------


def test_the_untwisted_character_is_the_ordinary_partition_series():
    """Four untwisted bosons: 1/prod(1-q^n)^4, so 1, 4, 14, 40, 105, ..."""
    series = twisted_character([0.0] * 4, [0.0] * 4, 6).real
    assert np.allclose(series, [1, 4, 14, 40, 105, 252, 574])


def test_a_half_integral_twist_moves_the_levels():
    """Two shifted directions: the first excited state sits at q^(1/2)."""
    series = twisted_character([0.5, 0.5], [0.0, 0.0], 4).real
    assert np.allclose(series, [1, 2, 3, 6, 9])


def test_the_character_of_a_reflection_is_the_signed_product():
    """One untwisted boson with h acting as -1 gives prod 1/(1 + q^n)."""
    n_max = 8
    series = twisted_character([0.0], [0.5], n_max)
    product = np.zeros(n_max + 1)
    product[0] = 1.0
    for mode in range(1, n_max + 1):  # multiply by 1/(1 + q^mode)
        for index in range(mode, n_max + 1):
            product[index] -= product[index - mode]
    assert np.allclose(series.real, product)
    assert np.max(np.abs(series.imag)) < 1e-12
    assert np.allclose(product[:6], [1, -1, 0, -1, 1, -1])


def test_mismatched_phase_lists_are_refused():
    with pytest.raises(ValueError, match="must match"):
        twisted_character([0.0, 0.5], [0.0], 4)


@pytest.mark.parametrize("element", [(1, 0), (0, 1), (1, 1)])
def test_torsion_keeps_the_other_half(element):
    """The two projections are complementary: they sum to the <g> projection."""
    trivial, torsion = Z2Z2.pairings()
    n_max = 12
    without = projected_degeneracies(Z2Z2, T4_PHASES, element, trivial, n_max)
    with_it = projected_degeneracies(Z2Z2, T4_PHASES, element, torsion, n_max)
    subgroup = np.rint(
        0.5
        * (
            twisted_character(T4_PHASES[element], T4_PHASES[(0, 0)], n_max).real
            + twisted_character(T4_PHASES[element], T4_PHASES[element], n_max).real
        )
    ).astype(int)
    assert np.array_equal(without + with_it, subgroup)
    assert not np.array_equal(without, with_it)
    assert np.min(without) >= 0
    assert np.min(with_it) >= 0


def test_the_untwisted_counts_do_not_move():
    trivial, torsion = Z2Z2.pairings()
    without = projected_degeneracies(Z2Z2, T4_PHASES, (0, 0), trivial, 10)
    with_it = projected_degeneracies(Z2Z2, T4_PHASES, (0, 0), torsion, 10)
    assert np.array_equal(without, with_it)
    assert without[0] == 1


def test_the_untwisted_projection_keeps_the_invariant_states():
    """Level by level it is the average of the four characters, and it is integral."""
    trivial, _ = Z2Z2.pairings()
    counts = projected_degeneracies(Z2Z2, T4_PHASES, (0, 0), trivial, 8)
    total = np.zeros(9)
    for element in Z2Z2.elements():
        total += twisted_character(T4_PHASES[(0, 0)], T4_PHASES[element], 8).real
    assert np.allclose(counts, total / 4.0)
    assert counts[0] == 1  # the vacuum is invariant


def test_torsion_of_a_cyclic_group_changes_nothing():
    group = TorsionGroup((2,))
    phases = {(0,): [0.0, 0.0], (1,): [0.5, 0.5]}
    (only,) = group.pairings()
    counts = projected_degeneracies(group, phases, (1,), only, 6)
    assert np.min(counts) >= 0
    assert counts[0] == 1
