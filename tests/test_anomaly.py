"""Green-Schwarz anomaly cancellation: 496 and the two groups, computed.

The package already produces 496 from the lattice side, by counting roots.
These tests check the other side -- the ten-dimensional anomaly polynomial --
and that the two arrive at the same number without sharing a constant.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from stringsim.heterotic.anomaly import (
    ONE,
    TOP,
    Form,
    a_hat,
    anomaly_polynomial,
    cancels,
    candidates,
    combine,
    e8,
    factorise,
    gravitational_coefficient,
    hirzebruch,
    required_dimension,
    scan,
    self_dual_tensor,
    sixth_order_terms,
    so,
    spin_half,
    spin_three_half,
    symbol,
    type_iib_residual,
)
from stringsim.heterotic.lattice import GAUGE_DIMENSION, heterotic_lattices
from stringsim.heterotic.spectrum import anomaly_free_dimension

XS = [0.31, -0.47, 0.73, 1.09, -0.61]


def _evaluate(form: Form, scale: float) -> float:
    """Numeric value of a form at ``tr R^{2k} = 2(-1)^k s^{2k} sum x_j^{2k}``."""
    traces = {
        f"R{2 * k}": 2 * (-1) ** k * scale ** (2 * k) * sum(x ** (2 * k) for x in XS)
        for k in (1, 2, 3)
    }
    total = 0.0
    for monomial, coeff in form.items():
        term = float(coeff)
        for name, power in monomial:
            term *= traces[name] ** power
        total += term
    return total


# --------------------------------------------------------------------------
# the characteristic classes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "exact"),
    [
        ("a_hat", lambda s: math.prod((s * x / 2) / math.sinh(s * x / 2) for x in XS)),
        ("hirzebruch", lambda s: math.prod((s * x) / math.tanh(s * x) for x in XS)),
    ],
)
def test_series_match_the_product_formulas(name: str, exact) -> None:
    r"""The truncated series against :math:`\prod` evaluated directly.

    Both are cut off after the twelve-form, so the difference must fall like
    ``s^8``: a factor of 256 for every halving.  Getting the *rate* right is
    the check -- a wrong coefficient at any order would spoil it long before
    the constant did.
    """
    form = {"a_hat": a_hat(), "hirzebruch": hirzebruch()}[name]
    errors = [abs(_evaluate(form, s) - exact(s)) for s in (0.4, 0.2, 0.1)]
    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        assert 200.0 < coarse / fine < 320.0


def test_type_iib_anomaly_cancels_exactly() -> None:
    r"""Two gravitini, two dilatini and a self-dual tensor: nothing left.

    This is the calibration the rest of the module rests on.  Three twelve-form
    coefficients vanish at once with no free parameter, which is what fixes the
    relative normalisation of the three index densities.  In exact rationals,
    so ``0`` means zero.
    """
    residual = type_iib_residual()
    assert residual == 0
    assert isinstance(residual, Fraction)


def test_the_half_argument_hirzebruch_does_not_cancel() -> None:
    r"""The variant that is easy to write down by mistake, shown to be wrong.

    :math:`\prod (x_j/2)/\tanh(x_j/2)` in place of :math:`\prod x_j/\tanh x_j`
    leaves a type IIB anomaly behind.  Without this the choice would be an
    assertion.
    """
    assert type_iib_residual(half_argument=True) != 0


def test_the_pieces_are_not_separately_zero() -> None:
    """The IIB cancellation is between three non-vanishing contributions."""
    for piece in (spin_three_half(), spin_half(), self_dual_tensor()):
        assert piece.part(TOP).largest() > 0


# --------------------------------------------------------------------------
# 496
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 100, 496, 497, 1000])
def test_gravitational_coefficient_is_the_offset_from_496(n: int) -> None:
    """``(n - 496)/725760``, exactly, in rationals."""
    assert gravitational_coefficient(n) == Fraction(n - 496, 725760)


def test_the_coefficient_is_affine_in_the_dimension() -> None:
    """Second differences vanish -- ``n`` enters the character linearly."""
    values = [gravitational_coefficient(n) for n in range(5)]
    seconds = [c - 2 * b + a for a, b, c in zip(values, values[1:], values[2:], strict=False)]
    assert all(s == 0 for s in seconds)


def test_the_anomaly_and_the_lattice_agree_on_496() -> None:
    """Two routes, no shared constant.

    ``required_dimension`` solves the twelve-form; ``GAUGE_DIMENSION`` is 16
    Cartan directions plus 480 roots, and both heterotic lattices reproduce it
    by explicit enumeration.  Nothing connects them but the answer.
    """
    assert required_dimension() == 496
    assert anomaly_free_dimension() == required_dimension()
    assert GAUGE_DIMENSION == 496
    for lattice in heterotic_lattices().values():
        assert lattice.algebra_dimension == required_dimension()


# --------------------------------------------------------------------------
# the second condition, and the groups
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [16, 30, 31, 33, 34, 40])
def test_so_n_keeps_a_sixth_order_trace_unless_n_is_32(n: int) -> None:
    r"""``Tr F^6`` contains ``(N - 32) tr F^6``, and nothing can absorb it.

    No product of a four-form and an eight-form built from ``tr R^2``,
    ``tr F^2``, ``tr R^4`` and ``tr F^4`` produces a single ``tr F^6``, so a
    non-zero coefficient here ends the matter before factorisation is even
    tried.
    """
    leftovers = sixth_order_terms(anomaly_polynomial([so(n)]))
    assert leftovers.coefficient(("F6_0", 1)) == Fraction(n - 32, 1440)
    assert leftovers.largest() > 0


def test_so32_has_none() -> None:
    assert sixth_order_terms(anomaly_polynomial([so(32)])) == {}


def test_e8_has_no_independent_sixth_order_trace() -> None:
    """``E_8`` has no quartic or sextic Casimir, so it passes for free."""
    assert sixth_order_terms(anomaly_polynomial([e8(0), e8(1)])) == {}


@pytest.mark.parametrize(
    "factors",
    [
        pytest.param([so(32)], id="SO(32)"),
        pytest.param([e8(0), e8(1)], id="E8 x E8"),
    ],
)
def test_the_two_string_groups_cancel(factors) -> None:
    """Both work, and both give the same ``X_4``.

    ``b = 1/30`` for every factor is the textbook
    ``tr R^2 - (1/30) Tr F^2``.  It is not put in: it is read off one
    coefficient and then confirmed by the exactness of the division.
    """
    verdict, report = cancels(factors)
    assert verdict
    assert report["dimension"] == 496
    assert report["gravitational"] == 0
    assert report["sixth_order"] == {}
    assert report["factorisation"].residual == 0
    assert all(value == Fraction(1, 30) for _, value in report["factorisation"].coefficients)


def test_a_group_of_the_right_dimension_can_still_fail() -> None:
    """``SO(26) x SO(19)``: 496-dimensional, and still anomalous.

    This is what makes the two conditions worth stating separately.  It passes
    the gravitational one exactly and fails the gauge one on both factors.
    """
    factors = [so(26, 0), so(19, 1)]
    dimension, _ = combine(factors)
    assert dimension == 496
    poly = anomaly_polynomial(factors)
    assert poly.coefficient(("R6", 1)) == 0
    leftovers = sixth_order_terms(poly)
    assert leftovers.coefficient(("F6_0", 1)) == Fraction(26 - 32, 1440)
    assert leftovers.coefficient(("F6_1", 1)) == Fraction(19 - 32, 1440)
    assert not cancels(factors)[0]


def test_the_scan_finds_exactly_the_two() -> None:
    """Over 800 candidates in the family; two survive.

    A search over ``SO(N)`` and ``E_8`` factors, not a uniqueness proof --
    ``SU(N)``, ``Sp(N)``, the smaller exceptional algebras and the abelian
    solutions are all outside it.
    """
    assert len(candidates(max_so=40, max_factors=2)) > 800
    names = {" x ".join(f.name for f in group) for group in scan(max_so=40, max_factors=2)}
    assert names == {"SO(32)", "E8 x E8"}


def test_x_four_is_the_textbook_combination() -> None:
    r"""``X_4 = tr R^2 + tr F^2`` in the defining trace of ``SO(32)``.

    ``Tr F^2 = 30 tr F^2`` there, so the adjoint form of the same statement
    carries the familiar ``1/30``.  The sign is a convention -- the ratio is
    not.
    """
    result = factorise(anomaly_polynomial([so(32)]), [so(32)])
    assert result.works
    assert result.x_four.coefficient(("R2", 1)) == 1
    assert result.x_four.coefficient(("F2_0", 1)) == 1
    assert result.coefficients == (("SO(32)[0]", Fraction(1, 30)),)
    assert "factorises" in str(result)


def test_factorisation_agrees_with_substitution() -> None:
    r"""A second route to the same verdict.

    ``P = X_4 X_8`` with ``X_4 = tr R^2 + L`` holds exactly when substituting
    ``tr R^2 -> -L`` annihilates ``P``.  That never touches the division, so it
    checks it.
    """
    for factors in ([so(32)], [e8(0), e8(1)], [so(26, 0), so(19, 1)], [so(30)]):
        poly = anomaly_polynomial(factors)
        if not poly.coefficient(("R2", 1), ("R4", 1)):
            continue
        result = factorise(poly, factors)
        rest = result.x_four - symbol("R2")
        assert (poly.substitute("R2", rest * -1).largest() == 0) == result.works


# --------------------------------------------------------------------------
# the polynomial arithmetic itself
# --------------------------------------------------------------------------


def test_form_truncates_above_the_twelve_form() -> None:
    """Weight 4 and beyond cannot contribute to a ten-dimensional anomaly."""
    r2 = symbol("R2")
    assert (r2**3).coefficient(("R2", 3)) == 1
    assert r2**4 == {}
    assert (symbol("R4") * symbol("R4")) == {}


def test_form_weights_and_helpers() -> None:
    assert Form.weight((("R2", 1), ("R4", 1))) == 3
    assert Form.weight((("F6_2", 1),)) == 3
    assert (ONE * 3).coefficient() == 3
    assert (symbol("R2") * 0) == {}
    assert Form().largest() == 0
    assert (symbol("R2") - symbol("R2")) == {}


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        so(0)
    with pytest.raises(ValueError):
        candidates(max_factors=0)
    with pytest.raises(ValueError, match="tr R\\^2 tr R\\^4"):
        factorise(Form({(("F2_0", 1), ("F4_0", 1)): Fraction(1)}), [so(32)])
