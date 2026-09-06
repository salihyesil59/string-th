"""The RNS superstring: sectors, GSO, and the type II massless spectra.

The strongest checks here are the ones that cross a module boundary. The NS
degeneracies are counted by explicit state enumeration and required to match the
theta-function product already in ``quantum/partition.py``; the intercepts are
computed from cut-off mode sums and required to match their closed forms; and
the Ramond counting, which shares nothing with the NS counting, has to agree
with it level by level.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from stringsim.quantum.partition import (
    hagedorn_temperature,
    jacobi_identity_residual,
    superstring_degeneracies,
)
from stringsim.quantum.zeta import central_charge, critical_dimension
from stringsim.superstring.rns import (
    Sector,
    critical_dimension_from_intercept,
    fermion_mode_numbers,
    hagedorn_beta,
    hagedorn_species,
    intercept,
    ns_series_by_parity,
    open_superstring_levels,
    sector_degeneracies,
    supersymmetry_deficit,
    unprojected_ns_ground_state,
    zero_point_energy,
)
from stringsim.superstring.typeii import (
    form_dimension,
    massless_counts,
    open_superstring_massless,
    rr_form_ranks,
    spinor_spinor,
    stable_brane_ranks,
    type_ii_massless,
    vector_spinor,
    vector_vector,
)
from stringsim.units import Conventions

TEN = Conventions(alpha_prime=1.0, dim=10)


# -- sectors and mode numbers ------------------------------------------------


def test_mode_numbers_distinguish_the_sectors():
    """Ramond has a zero mode; Neveu-Schwarz does not, and that is the whole difference."""
    assert fermion_mode_numbers(Sector.NS, 4) == [
        Fraction(1, 2),
        Fraction(3, 2),
        Fraction(5, 2),
        Fraction(7, 2),
    ]
    assert fermion_mode_numbers(Sector.R, 4) == [Fraction(n) for n in range(4)]
    assert fermion_mode_numbers(Sector.R, 1)[0] == 0
    assert Sector.R.is_periodic and not Sector.NS.is_periodic
    with pytest.raises(ValueError):
        fermion_mode_numbers(Sector.NS, -1)


# -- ground-state energies ---------------------------------------------------


def test_intercepts_measured_from_cutoff_sums_match_the_closed_forms():
    """``a_NS = 1/2`` and ``a_R = 0``, from regularised mode sums."""
    assert intercept(Sector.NS, TEN, exact=False) == pytest.approx(0.5, abs=1e-6)
    assert intercept(Sector.R, TEN, exact=False) == pytest.approx(0.0, abs=1e-6)
    assert intercept(Sector.NS, TEN) == pytest.approx(0.5)
    assert intercept(Sector.R, TEN) == 0.0


def test_ramond_ground_state_is_massless():
    """``a_R = 0`` exactly: the bosonic and fermionic zero-point energies cancel."""
    assert zero_point_energy(Sector.R, TEN) == pytest.approx(0.0, abs=1e-6)


def test_ns_intercept_follows_the_general_dimension_formula():
    """``a_NS = (D-2)/16``, checked against the cut-off sum away from ``D = 10``."""
    for dim in (6, 10, 18):
        conventions = Conventions(dim=dim)
        assert intercept(Sector.NS, conventions) == pytest.approx((dim - 2) / 16)
        assert intercept(Sector.NS, conventions, exact=False) == pytest.approx(
            (dim - 2) / 16, abs=1e-5
        )


def test_critical_dimension_from_three_routes():
    """``a_NS = 1/2``, the anomaly ``c = 0``, and the existing zeta module all give 10."""
    assert critical_dimension_from_intercept() == 10
    assert critical_dimension("superstring") == 10
    assert central_charge(10, "superstring") == pytest.approx(0.0)


def test_the_unprojected_ns_ground_state_is_a_tachyon():
    """It sits at ``-1/2``, half as deep as the bosonic string's, and GSO removes it."""
    assert unprojected_ns_ground_state(TEN) == pytest.approx(-0.5)


# -- state counting ----------------------------------------------------------


def test_ns_enumeration_reproduces_the_theta_function_product():
    """Two unrelated routes to ``8, 128, 1152, 7680, ...``."""
    assert sector_degeneracies(Sector.NS, 6) == superstring_degeneracies(6)


def test_ramond_enumeration_reproduces_the_same_numbers():
    assert sector_degeneracies(Sector.R, 6) == superstring_degeneracies(6)


def test_massless_level_is_eight_and_eight():
    assert sector_degeneracies(Sector.NS, 0) == [8]
    assert sector_degeneracies(Sector.R, 0) == [8]


@pytest.mark.parametrize("n_max", [2, 5, 9])
def test_supersymmetry_holds_level_by_level(n_max):
    """Equal numbers of bosons and fermions at every mass -- the Jacobi identity, counted."""
    assert supersymmetry_deficit(n_max) == [0] * (n_max + 1)


def test_supersymmetry_agrees_with_the_abstract_theta_identity():
    """The same statement twice: once by counting states, once by the theta identity
    ``theta_3^4 = theta_2^4 + theta_4^4``."""
    assert all(coefficient == 0 for coefficient in jacobi_identity_residual(40))
    assert supersymmetry_deficit(6) == [0] * 7


def test_parity_split_accounts_for_every_state():
    """Even plus odd fermion number must be the unprojected total."""
    even, odd = ns_series_by_parity(8)
    # the unprojected NS count at u^k is even[k] + odd[k]; check the first few by hand
    assert even[0] == 1 and odd[0] == 0  # the ground state, F = 0
    assert even[1] == 0 and odd[1] == 8  # b_{-1/2}^i |0>, eight states, F = 1
    assert even[2] == 8 + 28  # bosonic n=1 (8), or two half-modes (C(8,2) = 28)
    assert odd[2] == 0


def test_gso_keeps_odd_fermion_number():
    """The projected counts are exactly the odd-parity entries at odd powers of ``u``."""
    _, odd = ns_series_by_parity(14)
    assert [odd[2 * n + 1] for n in range(6)] == sector_degeneracies(Sector.NS, 5)


def test_degeneracies_validate_their_arguments():
    with pytest.raises(ValueError):
        sector_degeneracies(Sector.NS, -1)
    with pytest.raises(ValueError):
        ns_series_by_parity(-1)
    with pytest.raises(ValueError):
        ns_series_by_parity(4, transverse=0)
    with pytest.raises(ValueError):
        sector_degeneracies(Sector.NS, 3, gso=False)


def test_open_superstring_levels_are_supersymmetric():
    levels = open_superstring_levels(4, TEN)
    assert all(level.is_supersymmetric for level in levels)
    assert levels[0].alpha_m2 == 0.0
    assert levels[0].degeneracy == 16  # 8 + 8, ten-dimensional N = 1 SYM
    assert [level.bosons for level in levels] == superstring_degeneracies(4)


# -- Hagedorn ----------------------------------------------------------------


def test_superstring_is_hotter_than_the_bosonic_string():
    r"""Fermionic modes count half, so ``c_eff = 12`` and ``beta_H = 2 pi sqrt(2)``."""
    assert hagedorn_species() == 12.0
    assert hagedorn_beta() == pytest.approx(2 * math.pi * math.sqrt(2))
    assert hagedorn_beta() < hagedorn_temperature(24, 1.0)  # 4 pi, the bosonic value


def test_hagedorn_scales_with_alpha_prime():
    for alpha_prime in (0.25, 1.0, 4.0):
        assert hagedorn_beta(alpha_prime) == pytest.approx(
            2 * math.pi * math.sqrt(2 * alpha_prime)
        )


# -- type II -----------------------------------------------------------------


def test_form_dimensions_are_binomials():
    assert [form_dimension(p) for p in range(5)] == [1, 8, 28, 56, 70]
    with pytest.raises(ValueError):
        form_dimension(9)


@pytest.mark.parametrize("kind", ["IIA", "IIB"])
def test_every_sector_has_sixty_four_states(kind):
    for sector in type_ii_massless(kind):
        assert sector.dimension == 64


@pytest.mark.parametrize("kind", ["IIA", "IIB"])
def test_type_ii_is_supersymmetric_at_the_massless_level(kind):
    bosons, fermions = massless_counts(kind)
    assert bosons == fermions == 128
    assert bosons + fermions == 256  # (8v + 8s) x (8v + 8s')


def test_the_ns_ns_sector_is_the_same_in_both_theories():
    """Graviton, ``B`` field and dilaton -- and identical to the bosonic string's trio."""
    fields = vector_vector()
    assert [field.dimension for field in fields] == [35, 28, 1]
    assert sum(field.dimension for field in fields) == 64
    assert type_ii_massless("IIA")[0].fields == type_ii_massless("IIB")[0].fields


def test_the_rr_sectors_are_what_differ():
    assert spinor_spinor(True) != spinor_spinor(False)
    assert [f.dimension for f in spinor_spinor(True)] == [1, 28, 35]  # IIB: C_0, C_2, C_4
    assert [f.dimension for f in spinor_spinor(False)] == [8, 56]  # IIA: C_1, C_3
    assert sum(f.dimension for f in spinor_spinor(True)) == 64
    assert sum(f.dimension for f in spinor_spinor(False)) == 64


def test_the_self_dual_four_form_is_half_of_seventy():
    """Self-duality is why ``C_4`` contributes 35 and not 70."""
    four_form = next(f for f in spinor_spinor(True) if "C_4" in f.name)
    assert four_form.dimension == form_dimension(4) // 2 == 35


def test_the_mixed_sectors_are_gravitino_plus_dilatino():
    fields = vector_spinor("NS-R")
    assert [f.dimension for f in fields] == [56, 8]
    assert all(f.statistics == "fermion" for f in fields)


def test_sector_statistics_are_uniform():
    for kind in ("IIA", "IIB"):
        sectors = type_ii_massless(kind)
        assert [s.statistics for s in sectors] == ["boson", "boson", "fermion", "fermion"]


def test_type_ii_validates_its_argument():
    with pytest.raises(ValueError):
        type_ii_massless("IIC")
    with pytest.raises(ValueError):
        rr_form_ranks("heterotic")


# -- which branes exist ------------------------------------------------------


def test_brane_ranks_have_the_right_parity():
    """IIA has even ``p`` and IIB odd -- the check that catches a slip in the duality."""
    assert stable_brane_ranks("IIA") == [0, 2, 4, 6]
    assert stable_brane_ranks("IIB") == [-1, 1, 3, 5, 7]
    assert all(p % 2 == 0 for p in stable_brane_ranks("IIA"))
    assert all(p % 2 != 0 for p in stable_brane_ranks("IIB"))


def test_brane_ranks_are_closed_under_magnetic_duality():
    """``p -> 6 - p`` must map each list onto itself."""
    for kind in ("IIA", "IIB"):
        ranks = set(stable_brane_ranks(kind))
        assert {6 - p for p in ranks} == ranks


def test_brane_ranks_follow_from_the_rr_form_ranks():
    for kind in ("IIA", "IIB"):
        electric = {rank - 1 for rank in rr_form_ranks(kind)}
        assert electric <= set(stable_brane_ranks(kind))


def test_the_branes_module_can_price_the_physical_ones():
    """Everything except the ``p = -1`` instanton, which has an action, not a tension."""
    from stringsim.branes.dbrane import dp_brane_tension

    for kind in ("IIA", "IIB"):
        for p in stable_brane_ranks(kind):
            if p < 0:
                with pytest.raises(ValueError):
                    dp_brane_tension(p, 0.1, TEN)
                continue
            assert dp_brane_tension(p, 0.1, TEN) > 0


def test_open_superstring_massless_is_sym():
    fields = open_superstring_massless()
    assert [f.dimension for f in fields] == [8, 8]
    assert [f.statistics for f in fields] == ["boson", "fermion"]
