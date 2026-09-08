"""``(p, q)`` strings: the duality group, and a junction that balances itself.

The tension formula is not written down here -- it is checked against the D1
tension the package already had, and derived again from an M2-brane wrapping a
cycle.  The junction's equilibrium is checked to be charge conservation and
nothing else.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.amplitudes.oneloop import in_fundamental_domain
from stringsim.branes.dbrane import dp_brane_tension
from stringsim.branes.pq import (
    Junction,
    act_on_tau,
    axio_dilaton,
    binding_energy,
    coupling_from_tau,
    duality_residual,
    einstein_tension,
    is_bound_state,
    junction_angles,
    junction_residual,
    membrane_residual,
    membrane_tension,
    reduce_coupling,
    tension,
    transform_charges,
)
from stringsim.units import Conventions

S = ((0, -1), (1, 0))
T = ((1, 1), (0, 1))
ELEMENTS = [S, T, ((2, 1), (1, 1)), ((1, 0), (3, 1)), ((3, 2), (4, 3))]
CHARGES = [(1, 0), (0, 1), (2, -3), (5, 7)]


# --------------------------------------------------------------------------
# the tension
# --------------------------------------------------------------------------


@pytest.mark.parametrize("g_s", [0.15, 0.35, 1.0, 2.4])
def test_the_two_ends_of_the_multiplet_are_what_the_package_already_had(g_s: float) -> None:
    r"""``(1,0)`` is the fundamental string and ``(0,1)`` is the D1-brane.

    Neither number is put in: the first is :math:`1/2\pi\alpha'` and the second
    is what :func:`~stringsim.branes.dbrane.dp_brane_tension` returns for
    ``p = 1``, computed with no reference to duality.
    """
    conv = Conventions()
    tau = axio_dilaton(g_s)
    assert tension(1, 0, tau, conv) == pytest.approx(
        1.0 / (2.0 * math.pi * conv.alpha_prime)
    )
    assert tension(0, 1, tau, conv) == pytest.approx(dp_brane_tension(1, g_s, conv))


def test_the_axion_shifts_the_fundamental_charge() -> None:
    r"""``T`` moves ``C_0`` by one and leaves a pure D1 with an extra F1 unit.

    Which is the Witten effect: a D1-brane in a background axion carries
    fundamental string charge.
    """
    assert transform_charges(T, 0, 1) == (1, 1)
    assert transform_charges(T, 1, 0) == (1, 0)
    assert transform_charges(S, 1, 0) == (0, 1)
    assert transform_charges(S, 0, 1) == (-1, 0)


@pytest.mark.parametrize("matrix", ELEMENTS)
@pytest.mark.parametrize(("p", "q"), CHARGES)
def test_the_einstein_frame_tension_is_invariant(matrix, p: int, q: int) -> None:
    r"""``|p + q tau| / sqrt(Im tau)`` does not move under the duality group.

    The charge transformation is read off the algebra rather than fitted, so
    this is a check.  The string-frame tension is *not* invariant, and should
    not be: a duality changes which string is being called fundamental.
    """
    tau = axio_dilaton(0.35, 0.4)
    assert duality_residual(matrix, p, q, tau) < 1e-12


def test_the_string_frame_tension_does_move() -> None:
    """Otherwise the previous test would be measuring nothing."""
    tau = axio_dilaton(0.35, 0.4)
    moved = tension(1, 0, act_on_tau(S, tau))
    here = tension(*transform_charges(S, 1, 0), tau)
    assert abs(moved / here - 1.0) > 0.5


def test_the_duality_elements_really_are_sl_two_z() -> None:
    """Integer, determinant one, and ``S`` squares to minus the identity."""
    for matrix in ELEMENTS:
        array = np.asarray(matrix, dtype=np.int64)
        assert round(float(np.linalg.det(array))) == 1
    s = np.asarray(S, dtype=np.int64)
    t = np.asarray(T, dtype=np.int64)
    assert np.array_equal(s @ s, -np.eye(2, dtype=np.int64))
    assert np.array_equal(np.linalg.matrix_power(s @ t, 3), -np.eye(2, dtype=np.int64))


# --------------------------------------------------------------------------
# eleven dimensions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("g_s", [0.2, 0.7, 1.5, 3.0])
@pytest.mark.parametrize(("p", "q"), [(1, 0), (0, 1), (3, 2), (5, -4)])
def test_a_wrapped_membrane_gives_the_same_tension(g_s: float, p: int, q: int) -> None:
    r"""An M2-brane on the :math:`(p,q)` cycle of the torus, and nothing fitted.

    The side length is forced: matching the :math:`(1,0)` string to the
    fundamental one gives :math:`L = 1/2\pi\alpha' T_{M2}`, and that is
    :math:`2\pi R_{11}` because :math:`\ell_p^3 = g_s \alpha'^{3/2}`.  Both
    factors come from :mod:`stringsim.branes.mtheory`, which has never heard of
    ``SL(2,Z)``.
    """
    tau = axio_dilaton(g_s)
    assert membrane_residual(p, q, tau) < 1e-12
    assert membrane_tension(p, q, tau) == pytest.approx(tension(p, q, tau), rel=1e-12)


def test_the_membrane_route_works_with_an_axion_too() -> None:
    """The cycle length picks up ``C_0`` the same way the tension does."""
    tau = axio_dilaton(0.4, 0.6)
    for p, q in CHARGES:
        assert membrane_residual(p, q, tau) < 1e-12


# --------------------------------------------------------------------------
# bound states
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "q", "bound"),
    [(1, 0, True), (0, 1, True), (1, 1, True), (2, 3, True), (2, 2, False), (3, 6, False)],
)
def test_coprime_charges_bind(p: int, q: int, bound: bool) -> None:
    r"""``gcd(p, q) = 1`` is one string; otherwise it is several at threshold."""
    assert is_bound_state(p, q) is bound


def test_the_bound_state_is_lighter_than_its_pieces() -> None:
    r"""Triangle inequality: :math:`|p + q\tau| < |p| + |q||\tau|`.

    Strictly, unless the two terms point the same way -- which for
    :math:`\tau` in the upper half plane means one of the charges is zero.
    """
    tau = axio_dilaton(0.35, 0.4)
    assert binding_energy(1, 1, tau) > 0.05
    assert binding_energy(2, 3, tau) > 0.1
    assert binding_energy(1, 0, tau) == pytest.approx(0.0, abs=1e-12)
    assert binding_energy(0, 1, tau) == pytest.approx(0.0, abs=1e-12)


def test_a_repeated_charge_sits_exactly_at_threshold() -> None:
    r"""``(2,2)`` is two ``(1,1)`` strings and weighs exactly twice as much."""
    tau = axio_dilaton(0.35, 0.4)
    assert binding_energy(2, 2, tau) == pytest.approx(0.0, abs=1e-12)
    assert tension(2, 2, tau) == pytest.approx(2.0 * tension(1, 1, tau))
    assert tension(3, 6, tau) == pytest.approx(3.0 * tension(1, 2, tau))


# --------------------------------------------------------------------------
# the junction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "charges",
    [
        ((1, 0), (0, 1), (-1, -1)),
        ((2, 1), (-1, 1), (-1, -2)),
        ((1, 0), (1, 1), (0, 1), (-2, -2)),
    ],
)
@pytest.mark.parametrize("g_s", [0.3, 1.0, 2.2])
def test_a_conserved_junction_balances_exactly(charges, g_s: float) -> None:
    r"""The net force vanishes because :math:`\sum(p_i + q_i\tau) = 0`.

    Nothing about angles is imposed anywhere: each string's direction is the
    phase of its own :math:`p + q\tau`, and the balance is a consequence.
    Mechanical equilibrium and charge conservation are one equation.
    """
    junction = Junction(charges=charges, tau=axio_dilaton(g_s, 0.3))
    assert junction.conserved
    assert junction_residual(junction) < 1e-15


def test_a_junction_that_violates_charge_does_not_balance() -> None:
    """And by an amount of order a string tension, not a rounding error."""
    junction = Junction(charges=((1, 0), (0, 1), (-1, 0)), tau=axio_dilaton(0.4))
    assert not junction.conserved
    assert junction.total_charge == (0, 1)
    assert junction_residual(junction) > 0.1
    assert "total" in str(junction)


def test_the_angles_are_the_coupling_and_not_a_choice() -> None:
    r"""Each string leaves along :math:`\arg(p + q\tau)`, so moving ``tau`` bends it.

    A junction is rigid.  Changing the coupling deforms it and it stays in
    equilibrium, because the charges have not changed.
    """
    charges = ((1, 0), (0, 1), (-1, -1))
    weak = Junction(charges=charges, tau=axio_dilaton(0.2))
    strong = Junction(charges=charges, tau=axio_dilaton(3.0))
    assert junction_angles(weak)[0] == pytest.approx(0.0)
    assert junction_angles(weak)[1] == pytest.approx(math.pi / 2)
    assert not np.allclose(junction_angles(weak), junction_angles(strong))
    assert junction_residual(weak) < 1e-15
    assert junction_residual(strong) < 1e-15


def test_a_junction_needs_three_strings() -> None:
    with pytest.raises(ValueError, match="three"):
        Junction(charges=((1, 0), (-1, 0)), tau=axio_dilaton(1.0))


# --------------------------------------------------------------------------
# the fundamental domain
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("g_s", "axion"), [(8.0, 0.0), (0.05, 2.3), (1.7, -3.4), (4.0, 0.5)])
def test_strong_coupling_is_weak_coupling_relabelled(g_s: float, axion: float) -> None:
    r"""The same ``SL(2,Z)`` that folds the worldsheet modulus folds the coupling.

    :func:`~stringsim.amplitudes.oneloop.fundamental_domain_representative` is
    reused unchanged.  There it says the string has no ultraviolet region; here
    it says a strongly-coupled type IIB vacuum is a weakly-coupled one with the
    strings relabelled, and the relabelling is
    :func:`~stringsim.branes.pq.transform_charges`.
    """
    tau = axio_dilaton(g_s, axion)
    domain = reduce_coupling(tau)
    assert in_fundamental_domain(domain.tau)
    assert coupling_from_tau(domain.tau) <= 2.0 / math.sqrt(3.0) + 1e-9
    for p, q in CHARGES:
        moved = einstein_tension(p, q, domain.tau)
        here = einstein_tension(*transform_charges(domain.matrix, p, q), tau)
        assert moved == pytest.approx(here, rel=1e-10)


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="coupling"):
        axio_dilaton(0.0)
    with pytest.raises(ValueError, match="upper half plane"):
        coupling_from_tau(1 - 1j)
    with pytest.raises(ValueError, match="not a string"):
        tension(0, 0, axio_dilaton(1.0))
    with pytest.raises(ValueError, match="singular"):
        act_on_tau(S, 0j)
