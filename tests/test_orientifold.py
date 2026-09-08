"""Type I: the orientifold projection, the branes it needs, and SO(32) again.

The package reaches ``SO(32)`` three ways now -- an even self-dual lattice, the
ten-dimensional anomaly polynomial, and here by gauging worldsheet parity and
counting Chan-Paton factors.  These tests check the third and that it agrees
with the other two.
"""

from __future__ import annotations

import numpy as np
import pytest

from stringsim.classical.modes import ClosedString
from stringsim.heterotic.anomaly import required_dimension
from stringsim.heterotic.lattice import GAUGE_DIMENSION, heterotic_lattices
from stringsim.heterotic.spectrum import massless_content
from stringsim.superstring.orientifold import (
    Surface,
    antisymmetric_dimension,
    brane_count,
    chan_paton_dimension,
    closed_counts,
    euler_characteristic,
    gauge_group,
    massless_total,
    one_loop_surfaces,
    open_massless,
    parity_even,
    parity_image,
    parity_residual,
    project_closed,
    split_by_exchange,
    string_coupling_order,
    supersymmetric_sign,
    surfaces_for,
    symmetric_dimension,
    tadpole_structure,
)
from stringsim.superstring.typeii import massless_counts, type_ii_massless
from stringsim.units import Conventions

# --------------------------------------------------------------------------
# the projection
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 8, 16])
def test_the_two_halves_exhaust_the_product(n: int) -> None:
    assert symmetric_dimension(n) + antisymmetric_dimension(n) == n * n


@pytest.mark.parametrize("sector", ["NS-NS", "R-R"])
def test_the_split_is_found_not_assigned(sector: str) -> None:
    r"""36 and 28, and only one subset of the irreps can make 36.

    Both ``8_v x 8_v`` and ``8_s x 8_s`` decompose as ``1 + 28 + 35``, and in
    both the symmetric half is ``1 + 35``.  The function searches for the
    subset rather than being told, and raises if the dimensions leave it
    ambiguous.
    """
    fields = {s.name: s for s in type_ii_massless("IIB")}[sector].fields
    sym, anti = split_by_exchange(fields)
    assert sum(f.dimension for f in sym) == 36
    assert sum(f.dimension for f in anti) == 28
    assert {f.dimension for f in sym} == {1, 35}
    assert {f.dimension for f in anti} == {28}


def test_the_projection_keeps_the_right_fields() -> None:
    """Graviton and dilaton, the RR two-form, one gravitino and one dilatino.

    The Kalb-Ramond two-form is gone -- it is antisymmetric in NS-NS -- and so
    are ``C_0`` and the self-dual ``C_4``.  What is left of ``C_2`` is the field
    the D1-brane of type I couples to.
    """
    sectors = project_closed()
    names = {f.name for s in sectors for f in s.fields}
    assert "graviton" in names
    assert "dilaton" in names
    assert "RR 2-form C_2" in names
    assert "Kalb-Ramond B" not in names
    assert "RR scalar C_0" not in names
    assert "RR self-dual C_4" not in names
    assert len([f for s in sectors for f in s.fields if not f.is_boson]) == 2


def test_supersymmetry_picks_the_ramond_sign() -> None:
    r"""The sign is not a free choice, and the counting says so.

    ``+1`` leaves 72 bosons against 64 fermions.  ``-1`` leaves 64 and 64.  Only
    one of those is a supermultiplet, so the projection that defines type I is
    the only one available.
    """
    assert closed_counts(+1) == (72, 64)
    assert closed_counts(-1) == (64, 64)
    assert supersymmetric_sign() == -1


def test_the_closed_sector_is_ten_dimensional_n_equals_one_supergravity() -> None:
    """128 states, and the heterotic string's supergravity multiplet is the same.

    ``massless_content`` builds the heterotic massless level from a root lattice
    and splits off 128 supergravity states.  Nothing in that calculation knows
    about orientifolds.
    """
    bosons, fermions = closed_counts()
    assert (bosons, fermions) == (64, 64)
    for lattice in heterotic_lattices().values():
        assert massless_content(lattice).supergravity_states == bosons + fermions


def test_the_projection_halves_type_iib() -> None:
    """IIB has 256 massless states; gauging parity leaves 128 of them."""
    assert sum(massless_counts("IIB")) == 256
    assert sum(closed_counts()) == 128


# --------------------------------------------------------------------------
# the open sector
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "kind", "expected"), [
    (32, "SO", 496),
    (2, "SO", 1),
    (16, "SO", 120),
    (32, "SP", 528),
    (2, "SP", 3),
])
def test_chan_paton_dimensions(n: int, kind: str, expected: int) -> None:
    assert chan_paton_dimension(n, kind) == expected


def test_symplectic_needs_an_even_count() -> None:
    with pytest.raises(ValueError, match="even"):
        chan_paton_dimension(31, "SP")
    assert gauge_group(32, "SP") == "Sp(16)"


def test_the_brane_count_comes_from_the_anomaly() -> None:
    r"""``n(n-1)/2 = 496`` has one root, and 496 is not put in by hand.

    :func:`required_dimension` solves the twelve-form anomaly polynomial.  The
    orientifold contributes ``n(n-1)/2``.  Neither knows about the other.
    """
    assert required_dimension() == 496
    assert brane_count() == 32
    assert chan_paton_dimension(brane_count()) == required_dimension()
    assert gauge_group(brane_count()) == "SO(32)"


def test_the_symplectic_projection_has_no_solution() -> None:
    """``n(n+1)/2 = 496`` has no integer root, so ``Sp`` is not an option."""
    assert all(chan_paton_dimension(n, "SP") != 496 for n in range(2, 100, 2))
    with pytest.raises(ValueError, match="no SP group"):
        brane_count("SP")


def test_the_open_sector_is_supersymmetric_on_its_own() -> None:
    """``8_v`` and ``8_s`` per generator: equal counts, ``N = 1`` super Yang-Mills."""
    fields = open_massless(32)
    bosons = sum(f.dimension for f in fields if f.is_boson)
    fermions = sum(f.dimension for f in fields if not f.is_boson)
    assert bosons == fermions == 8 * 496


# --------------------------------------------------------------------------
# the whole spectrum, against the heterotic string
# --------------------------------------------------------------------------


def test_type_i_and_heterotic_so32_have_the_same_massless_level() -> None:
    """8064 either way, and the same split into 128 and 7936.

    One side gauges worldsheet parity on type IIB and counts Chan-Paton
    factors.  The other enumerates the roots of an even self-dual lattice.  The
    constructions have nothing in common; the agreement is the massless shadow
    of the duality between the two theories.
    """
    report = massless_total()
    assert report == {
        "branes": 32,
        "gauge_group": "SO(32)",
        "gauge_dimension": 496,
        "closed": 128,
        "open": 7936,
        "total": 8064,
        "bosons": 4032,
        "fermions": 4032,
    }
    heterotic = massless_content(heterotic_lattices()["Spin(32)/Z2"])
    assert heterotic.total == report["total"]
    assert heterotic.supergravity_states == report["closed"]
    assert heterotic.gauge_states == report["open"]
    assert heterotic.gauge_dimension == report["gauge_dimension"] == GAUGE_DIMENSION


def test_the_whole_spectrum_is_supersymmetric() -> None:
    report = massless_total()
    assert report["bosons"] == report["fermions"]


# --------------------------------------------------------------------------
# the surfaces
# --------------------------------------------------------------------------


def test_exactly_four_surfaces_are_one_loop() -> None:
    """Enumerated from ``2 - 2g - b - c = 0``, not listed."""
    found = one_loop_surfaces()
    assert len(found) == 4
    assert {s.name for s in found} == {"torus", "Klein bottle", "cylinder", "Mobius strip"}
    assert all(s.euler == 0 for s in found)


def test_which_theory_has_which() -> None:
    """Orientation and boundaries decide, and type I has all four."""
    assert [s.name for s in surfaces_for("closed oriented")] == ["torus"]
    assert {s.name for s in surfaces_for("closed unoriented")} == {"torus", "Klein bottle"}
    assert len(surfaces_for("open unoriented")) == 4
    with pytest.raises(ValueError):
        surfaces_for("open oriented")


def test_the_euler_characteristic_orders_perturbation_theory() -> None:
    """Sphere 2, disc and projective plane 1, one loop 0."""
    assert euler_characteristic(0, 0, 0) == 2
    assert euler_characteristic(0, 1, 0) == euler_characteristic(0, 0, 1) == 1
    assert [s.name for s in one_loop_surfaces(euler=2)] == ["sphere"]
    assert {s.name for s in one_loop_surfaces(euler=1)} == {"disc", "projective plane"}
    with pytest.raises(ValueError):
        euler_characteristic(-1, 0, 0)


def test_the_coupling_weights_the_surfaces() -> None:
    """``g_s^{-chi}``: tree level goes like ``1/g^2``, one loop like ``1``."""
    sphere = Surface("sphere", 0, 0, 0)
    disc = Surface("disc", 0, 1, 0)
    assert string_coupling_order(sphere, 0.1) == pytest.approx(100.0)
    assert string_coupling_order(disc, 0.1) == pytest.approx(10.0)
    for surface in one_loop_surfaces():
        assert string_coupling_order(surface, 0.1) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        string_coupling_order(sphere, 0.0)


# --------------------------------------------------------------------------
# the tadpole, and what is not computed about it
# --------------------------------------------------------------------------


def test_the_tadpole_is_a_perfect_square_with_a_double_root() -> None:
    r"""``(n - 32)^2``: a sum of ``<B|B>``, ``<C|C>`` and the cross term.

    The square is structural -- it is one inner product of one state with
    itself -- so the discriminant must vanish, and the double root is the
    charge that has to be cancelled.  The value of the crosscap charge is an
    input here, not a result; the test is that it agrees with the count the
    anomaly forces.
    """
    report = tadpole_structure(32)
    assert report["coefficients"] == (1, -64, 1024)
    assert report["discriminant"] == 0
    assert report["root"] == brane_count()
    assert report["charge"] == 0 and report["cancels"]
    assert tadpole_structure(16)["value"] == 256
    assert not tadpole_structure(16)["cancels"]


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        project_closed(rr_sign=0)
    with pytest.raises(ValueError):
        chan_paton_dimension(0)
    with pytest.raises(ValueError):
        chan_paton_dimension(4, "U")
    with pytest.raises(ValueError):
        symmetric_dimension(-1)
    with pytest.raises(ValueError, match="not a square of one space"):
        # IIA's R-R sector is 8_s x 8_c, a product of two *different* spaces.
        # Exchange is not even defined on it, and the function says so instead
        # of returning half of something.
        split_by_exchange(type_ii_massless("IIA")[1].fields)


# --------------------------------------------------------------------------
# the same projection, on a moving string
# --------------------------------------------------------------------------


def _travelling() -> ClosedString:
    """A closed string with unequal chiralities: a wave that goes round."""
    return ClosedString(
        conventions=Conventions(dim=4),
        alphas={1: np.array([0.0, 1.0, 0.4, 0.0]), 2: np.array([0.0, 0.3, 0.2, 0.0])},
        alphas_tilde={1: np.array([0.0, 0.2, 0.9, 0.0])},
    )


def test_parity_on_a_solution_is_the_reflection_of_sigma() -> None:
    r"""Swapping the mode coefficients really is :math:`\sigma \to -\sigma`.

    One statement is about which oscillators a state is built from, the other
    about a coordinate on the worldsheet.  They have to be the same thing, and
    this is where the discrete story and the continuum one meet.
    """
    string = _travelling()
    tau = np.linspace(0.0, 5.0, 7)[:, None]
    sigma = np.linspace(0.0, string.sigma_max, 33)[None, :]
    swapped = parity_image(string).position(tau, sigma)
    reflected = string.position(tau, -sigma)
    assert np.max(np.abs(swapped - reflected)) < 1e-12


def test_the_invariant_part_is_a_standing_wave() -> None:
    r"""An unoriented string cannot carry a wave that travels around it.

    A generic solution differs from its image by an amount of order its own
    size.  Averaging the two gives equal chiralities, which is a standing wave,
    and its residual is round-off.
    """
    string = _travelling()
    assert parity_residual(string) > 0.5
    even = parity_even(string)
    assert parity_residual(even) < 1e-12
    for n in set(even.alphas) | set(even.alphas_tilde):
        assert np.allclose(even.alphas[n], even.alphas_tilde[n])


def test_the_invariant_part_is_the_average_of_the_two() -> None:
    """Projecting the modes and averaging the solutions agree, as linearity says."""
    string = _travelling()
    tau = np.linspace(0.0, 4.0, 5)[:, None]
    sigma = np.linspace(0.0, string.sigma_max, 25)[None, :]
    average = 0.5 * (string.position(tau, sigma) + string.position(tau, -sigma))
    assert np.max(np.abs(parity_even(string).position(tau, sigma) - average)) < 1e-12


def test_a_string_that_was_already_standing_is_untouched() -> None:
    string = ClosedString(
        conventions=Conventions(dim=4),
        alphas={1: np.array([0.0, 1.0, 0.0, 0.0])},
        alphas_tilde={1: np.array([0.0, 1.0, 0.0, 0.0])},
    )
    assert parity_residual(string) < 1e-12
    assert parity_residual(parity_even(string)) < 1e-12
