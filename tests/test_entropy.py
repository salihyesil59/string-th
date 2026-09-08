"""Strominger-Vafa: the same entropy from a generating function and from a horizon.

Two calculations with no step in common.  One expands a product in exact
integers and takes a logarithm; the other multiplies three harmonic radii and
divides by Newton's constant.  These tests check that they agree, and that the
agreement is not a fit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.branes.entropy import (
    VOLUME_CONVENTIONS,
    Moduli,
    bekenstein_hawking,
    bps_degeneracies,
    cardy_entropy,
    central_charge,
    fit_cardy,
    harmonic_radii,
    horizon_area,
    microscopic_entropy,
    moduli_spread,
    newton_five,
    normalisation,
    species,
)
from stringsim.quantum.partition import oscillator_degeneracies

CHARGES = [(1, 1), (1, 2), (2, 2)]


# --------------------------------------------------------------------------
# the counting
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("q1", "q5"), CHARGES)
def test_the_central_charge_is_six_q1_q5(q1: int, q5: int) -> None:
    r"""``4 Q_1 Q_5`` bosons and as many fermions: ``c = 6 Q_1 Q_5``."""
    assert species(q1, q5) == 4 * q1 * q5
    assert central_charge(q1, q5) == 6 * q1 * q5
    assert central_charge(q1, q5) == species(q1, q5) + species(q1, q5) // 2


def test_the_degeneracies_are_exact_integers() -> None:
    """Enormous ones: they are what a logarithm is taken of."""
    table = bps_degeneracies(1, 1, 60)
    assert all(isinstance(d, int) for d in table)
    assert table[0] == 1
    assert all(a < b for a, b in zip(table[1:-1], table[2:], strict=True))
    assert len(str(table[60])) > 15


def test_switching_the_fermions_off_gives_the_bosonic_count() -> None:
    r"""``prod (1-q^n)^{-k}`` is what :mod:`stringsim.quantum.partition` counts.

    The D1-D5 generating function is the bosonic one times a numerator of
    ``(1 + q^n)``, so dropping the numerator has to land on the oscillator
    degeneracies the rest of the package uses.  Checked by building the
    numerator separately and dividing it out.
    """
    n_max, k = 30, 4
    full = bps_degeneracies(1, 1, n_max)
    fermionic = [0] * (n_max + 1)
    fermionic[0] = 1
    for _ in range(k):
        for n in range(1, n_max + 1):
            for i in range(n_max, n - 1, -1):
                fermionic[i] += fermionic[i - n]
    bosonic = oscillator_degeneracies(n_max, k)
    convolved = [
        sum(fermionic[i] * bosonic[level - i] for i in range(level + 1))
        for level in range(n_max + 1)
    ]
    assert convolved == full


@pytest.mark.parametrize(("q1", "q5", "n_max"), [(1, 1, 400), (1, 2, 400), (2, 2, 300)])
def test_the_cardy_exponent_comes_out_of_the_integers(q1: int, q5: int, n_max: int) -> None:
    r"""``2 pi sqrt(Q_1 Q_5 N)``, measured rather than quoted.

    Nothing in the fit knows the answer: it reads the slope off the logarithms
    of exact integers.
    """
    fit = fit_cardy(q1, q5, n_max)
    assert fit.expected == pytest.approx(2.0 * math.pi * math.sqrt(q1 * q5))
    assert fit.error < 2e-3
    assert fit.residual < 1e-4
    assert "against" in str(fit)


def test_the_log_term_is_not_a_correction() -> None:
    r"""Leaving it out costs two orders of magnitude.

    The same lesson as :func:`stringsim.quantum.partition.fit_hagedorn`: at any
    level that can be reached the subleading ``log N`` is comparable to what is
    being measured, and a two-term fit reports a slope that is per-cent wrong.
    """
    with_log = fit_cardy(1, 1, 400)
    without = fit_cardy(1, 1, 400, with_log=False)
    assert with_log.error < 1e-3
    assert without.error > 1e-2
    assert without.error / with_log.error > 20.0
    assert math.isnan(without.subleading)


def test_the_subleading_power_heads_for_minus_three_plus_k_over_four() -> None:
    r"""``b -> -(3 + 4 Q_1 Q_5)/4``, and it gets there slowly.

    The fit is asked where it is, not where it is going: at ``n_max = 200`` it
    reads ``-1.69`` against ``-1.75``, and the approach is monotone.
    """
    values = [fit_cardy(1, 1, n_max).subleading for n_max in (200, 400, 800)]
    assert all(a > b for a, b in zip(values[:-1], values[1:], strict=True))
    assert values[-1] < -1.70
    assert values[-1] > fit_cardy(1, 1, 200).subleading_expected


def test_the_honest_count_is_well_below_cardy_at_reachable_levels() -> None:
    """Which is why the comparison has to be a fit and not a ratio."""
    table = bps_degeneracies(1, 1, 60)
    assert microscopic_entropy(1, 1, 60, table) < 0.85 * cardy_entropy(1, 1, 60)
    assert microscopic_entropy(1, 1, 60, table) > 0.6 * cardy_entropy(1, 1, 60)


# --------------------------------------------------------------------------
# the horizon
# --------------------------------------------------------------------------


def test_the_entropy_forgets_every_modulus() -> None:
    r"""It counts states, so it cannot depend on a continuous parameter.

    The string coupling, the volume of the ``T^4``, the radius of the circle
    and ``alpha'`` all cancel between the horizon area and ``G_5``.  That needs
    the powers in all four places to be mutually consistent, and it is the part
    of the macroscopic side that is genuinely derived.
    """
    reference = bekenstein_hawking(3, 7, 200)
    for moduli in (
        Moduli(coupling=0.05, volume=6.9, radius=0.6, alpha_prime=2.3),
        Moduli(coupling=0.9, volume=0.4, radius=5.5, alpha_prime=0.7),
        Moduli(coupling=0.31, volume=3.3, radius=1.1, alpha_prime=1.9),
    ):
        assert bekenstein_hawking(3, 7, 200, moduli) == pytest.approx(reference, rel=1e-12)


def test_a_wrong_power_would_show() -> None:
    r"""The cancellation is not automatic: change one exponent and it breaks.

    ``newton_five`` scales as :math:`g_s^2\alpha'^4/RV` and the area as the
    same thing; multiplying the area by an extra power of the coupling makes
    the entropy move with it.
    """
    base = Moduli()
    other = Moduli(coupling=0.8, volume=5.0, radius=3.0, alpha_prime=2.0)
    broken = [
        horizon_area(3, 7, 200, m) * m.coupling / (4.0 * newton_five(m))
        for m in (base, other)
    ]
    assert abs(broken[0] / broken[1] - 1.0) > 0.5


@pytest.mark.parametrize(("q1", "q5", "n"), [(1, 1, 10), (3, 7, 200), (12, 5, 41)])
def test_the_two_entropies_agree(q1: int, q5: int, n: int) -> None:
    r"""``A / 4 G_5`` is :math:`2\pi\sqrt{Q_1Q_5N}`, exactly.

    One side came from a metric, the other from a central charge.
    """
    assert bekenstein_hawking(q1, q5, n) == pytest.approx(cardy_entropy(q1, q5, n), rel=1e-12)
    assert normalisation(q1, q5, n) == pytest.approx(1.0, rel=1e-12)


def test_the_agreement_is_over_determined() -> None:
    """Seven parameters vary and the ratio does not move.

    Three charges and four moduli against one number.  If the ratio depended on
    any of them the match would be a coincidence at one point rather than an
    agreement.
    """
    assert moduli_spread() < 1e-12


@pytest.mark.parametrize(
    ("convention", "expected"),
    [
        ("reduced", 1.0),
        ("mixed", 4.0 * math.pi**2),
        ("dimensionful", 16.0 * math.pi**4),
    ],
)
def test_the_volume_convention_is_visible(convention: str, expected: float) -> None:
    r"""Only one choice makes the two calculations agree, and it is stated.

    Whether the ``T^4`` enters the harmonic functions through ``V`` or through
    ``v = V/(2 pi)^4 alpha'^2`` changes the answer by a pure number and nothing
    else -- the moduli still cancel in all three.  The match picks ``reduced``,
    the same way the type IIB anomaly picks the Hirzebruch class in
    :mod:`stringsim.heterotic.anomaly`.
    """
    assert normalisation(3, 7, 200, convention=convention) == pytest.approx(expected, rel=1e-12)
    assert moduli_spread(convention=convention) < 1e-12


def test_the_radii_and_the_area() -> None:
    r"""``A = 2 pi^2 r_1 r_5 r_p`` with the radii each linear in their charge."""
    moduli = Moduli()
    one = harmonic_radii(2, 3, 5, moduli)
    two = harmonic_radii(4, 3, 5, moduli)
    assert two[0] == pytest.approx(2.0 * one[0])
    assert two[1] == pytest.approx(one[1])
    assert horizon_area(2, 3, 5, moduli) == pytest.approx(
        2.0 * math.pi**2 * math.sqrt(one[0] * one[1] * one[2])
    )
    assert np.all(np.array(one) > 0)


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        species(0, 3)
    with pytest.raises(ValueError, match="non-negative"):
        cardy_entropy(1, 1, -1)
    with pytest.raises(ValueError, match="n_max"):
        bps_degeneracies(1, 1, -1)
    with pytest.raises(ValueError, match="convention"):
        harmonic_radii(1, 1, 1, convention="whatever")
    with pytest.raises(ValueError, match="positive"):
        Moduli(coupling=0.0)
    assert set(VOLUME_CONVENTIONS) == {"reduced", "mixed", "dimensionful"}
