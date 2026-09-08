"""Vertex operators: the Veneziano amplitude out of a worldsheet integral.

``veneziano.py`` evaluates a Beta function.  This builds the same number from
where the vertex operators sit on the boundary of the disc, so the two are an
answer and a derivation of it rather than one calculation done twice.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.amplitudes.veneziano import veneziano, veneziano_residue
from stringsim.amplitudes.vertex import (
    converges,
    exponent_residual,
    exponents_from_momenta,
    faddeev_popov,
    fit_tachyon_pole,
    five_point_exponents,
    four_point_exponents,
    gauge_spread,
    koba_nielsen,
    mandelstam_sum,
    ordered_amplitude,
    tachyon_momenta,
)
from stringsim.units import minkowski

CONVERGENT = [(-2.0, -2.5), (-3.0, -1.6), (-2.2, -4.0), (-1.5, -1.8)]
FIVE = (-2.0, -2.2, -2.4, -2.1, -2.3)


# --------------------------------------------------------------------------
# the exponents
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("s", "t"), CONVERGENT)
def test_the_mass_shell_conditions_hold_exactly(s: float, t: float) -> None:
    r"""Every row sums to ``-2``, which is ``alpha' m^2 = -1`` in disguise.

    :math:`\sum_{j\neq i} 2\alpha' k_i\cdot k_j = -2\alpha' k_i^2`, so the row
    sums are the external mass-shell conditions.  Nothing imposes them here;
    they follow from ``s + t + u`` being what it is.
    """
    matrix = four_point_exponents(s, t)
    assert exponent_residual(matrix) < 1e-12
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix[0, 1] == pytest.approx(matrix[2, 3])
    assert matrix[1, 2] == pytest.approx(matrix[0, 3])


def test_the_invariants_are_the_ones_real_momenta_give() -> None:
    r"""Explicit on-shell vectors reproduce the exponent matrix.

    Built in the centre-of-mass frame at a *physical* ``(s, t)`` -- which is
    nowhere near where the integral converges, and that is the usual state of
    affairs.  What it establishes is that the invariants used everywhere else
    are the ones four real momenta actually have.
    """
    s, t, dim = 6.0, -1.5, 26
    momenta = tachyon_momenta(s, t, dim)
    eta = minkowski(dim)
    assert np.max(np.abs(momenta.sum(axis=0))) < 1e-12
    for k in momenta:
        assert float(np.sum(eta * k * k)) == pytest.approx(1.0)
    from_momenta = exponents_from_momenta(momenta)
    upper = np.triu_indices(4, 1)
    assert np.allclose(
        np.sort(from_momenta[upper]), np.sort(four_point_exponents(s, t)[upper])
    )


def test_the_mandelstam_sum_is_the_external_masses() -> None:
    assert mandelstam_sum(4, 1.0) == pytest.approx(-4.0)
    assert mandelstam_sum(4, 2.0) == pytest.approx(-2.0)
    with pytest.raises(ValueError):
        mandelstam_sum(5)


def test_five_point_exponents_solve_the_conditions() -> None:
    """Five adjacent invariants are free; the five chords are not.

    The mass-shell conditions are five linear equations for exactly the five
    non-adjacent exponents, and the function solves them.
    """
    matrix = five_point_exponents(FIVE)
    assert matrix.shape == (5, 5)
    assert exponent_residual(matrix) < 1e-12
    assert np.allclose(matrix, matrix.T)
    for slot, (i, j) in enumerate([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]):
        assert matrix[i, j] == pytest.approx(-FIVE[slot] - 2.0)


# --------------------------------------------------------------------------
# the amplitude
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("s", "t"), CONVERGENT)
def test_the_integral_is_the_veneziano_amplitude(s: float, t: float) -> None:
    """A quadrature over one puncture against a Beta function.

    The only thing the two share is the physics: one integrates
    ``prod |y_i - y_j|^{2 alpha' k_i k_j}`` over the boundary, the other calls
    ``gammaln``.
    """
    matrix = four_point_exponents(s, t)
    assert converges(matrix)
    value = ordered_amplitude(matrix, anchor=3.0)
    assert value == pytest.approx(float(veneziano(s, t)), rel=1e-9)


@pytest.mark.parametrize(("s", "t"), CONVERGENT)
def test_the_gauge_choice_does_not_matter(s: float, t: float) -> None:
    r"""Three punctures can be nailed down anywhere.

    Moving the third one from 1.5 to 6 changes the integrand completely -- only
    at :math:`R \to \infty` does it reduce to the Beta integrand -- and leaves
    the answer alone.
    """
    assert gauge_spread(four_point_exponents(s, t)) < 1e-10


@pytest.mark.parametrize(
    ("i", "j", "name"),
    [(0, 1, "0 and x"), (1, 2, "x and 1"), (1, 3, "x and R")],
)
@pytest.mark.parametrize("detuning", [0.01, 0.05])
def test_off_shell_breaks_the_gauge_invariance(i: int, j: int, name: str, detuning: float) -> None:
    """And it breaks in proportion, which is what says the condition is load-bearing.

    Nudge one exponent and two rows stop summing to ``-2``.  The spread over
    gauges then tracks the nudge: a per cent off shell is a per cent of gauge
    dependence.
    """
    del name
    matrix = four_point_exponents(-2.0, -2.5)
    matrix[i, j] += detuning
    matrix[j, i] += detuning
    assert exponent_residual(matrix) == pytest.approx(detuning)
    spread = gauge_spread(matrix)
    assert spread > 0.2 * detuning
    assert spread < 5.0 * detuning


def test_one_exponent_is_invisible_to_this_gauge() -> None:
    r"""The pair fixed a unit apart contributes :math:`1^{e} = 1`, whatever ``e``.

    Punctures 1 and 3 sit at 0 and 1 in every gauge of this family -- rescaling
    is itself an ``SL(2,R)`` map -- so detuning :math:`e_{13}` changes nothing,
    and :func:`gauge_spread` cannot see it going off shell.

    That is consistent rather than a hole: on shell :math:`e_{13}` is fixed by
    the others through the row sums, and the amplitude
    :math:`B(-\alpha(s), -\alpha(t))` has no independent ``u`` in it either.
    Knowing where a check is blind is part of the check.
    """
    matrix = four_point_exponents(-2.0, -2.5)
    clean = gauge_spread(matrix)
    matrix[0, 2] += 0.05
    matrix[2, 0] += 0.05
    assert exponent_residual(matrix) == pytest.approx(0.05)
    assert gauge_spread(matrix) == pytest.approx(clean, abs=1e-12)


def test_the_divergent_region_is_refused() -> None:
    r"""Past the first pole the ordered integral does not exist.

    The amplitude there is *defined* by the continuation the Beta function
    performs, and no quadrature reaches it.  Returning a number would be worse
    than refusing.
    """
    matrix = four_point_exponents(-0.5, -2.5)
    assert not converges(matrix)
    with pytest.raises(ValueError, match="diverges"):
        ordered_amplitude(matrix)
    assert np.isfinite(float(veneziano(-0.5, -2.5)))


# --------------------------------------------------------------------------
# the pole
# --------------------------------------------------------------------------


@pytest.mark.parametrize("t", [-3.0, -5.0])
def test_the_pole_comes_from_two_punctures_colliding(t: float) -> None:
    r"""Approaching :math:`\alpha(s) = 0`, the integral diverges at ``x = 0``.

    Nothing else in the integrand is singular there, so the tachyon pole is
    located in the geometry: it is where two vertex operators meet.  Fitting
    the divergence gives the residue, and it is ``-1`` for either ``t``,
    because the tachyon has no spin -- which is
    :func:`~stringsim.amplitudes.veneziano.veneziano_residue` at ``n = 0``.
    """
    fit = fit_tachyon_pole(t=t)
    assert fit.expected == float(veneziano_residue(0, t))
    assert fit.error < 1e-4
    # The offsets shrink towards the pole, so the amplitude grows, and
    # offset times amplitude closes on 1 -- which is the residue's size.
    assert np.all(np.diff(fit.amplitudes) > 0)
    assert abs(fit.samples[-1] * fit.amplitudes[-1] - 1.0) < 0.02
    assert "residue" in str(fit)


def test_the_pole_fit_rejects_a_bad_approach() -> None:
    with pytest.raises(ValueError, match="from below"):
        fit_tachyon_pole(offsets=(-0.01, 0.01))


# --------------------------------------------------------------------------
# five points, where there is nothing to look up
# --------------------------------------------------------------------------


def test_five_points_is_gauge_independent() -> None:
    """Two punctures integrated, no closed form, and the gauge still drops out.

    This is the only handle on whether the construction is right here, and it
    is the same handle that agreed with a Beta function at four points.
    """
    matrix = five_point_exponents(FIVE)
    assert converges(matrix, 5)
    assert gauge_spread(matrix, anchors=(2.0, 4.0)) < 1e-9


def test_five_points_has_the_symmetry_of_the_disc() -> None:
    """Cyclic rotations and the reflection leave the ordered amplitude alone.

    The punctures sit on a circle, so relabelling them around it is not a
    different diagram.  Nothing in the code enforces this -- the matrix is
    permuted and the integral redone.
    """
    matrix = five_point_exponents(FIVE)
    base = ordered_amplitude(matrix, 3.0)
    for shift in (1, 3):
        order = [(i + shift) % 5 for i in range(5)]
        rotated = matrix[np.ix_(order, order)]
        assert ordered_amplitude(rotated, 3.0) == pytest.approx(base, rel=1e-9)
    order = list(reversed(range(5)))
    assert ordered_amplitude(matrix[np.ix_(order, order)], 3.0) == pytest.approx(
        base, rel=1e-9
    )


# --------------------------------------------------------------------------
# the pieces
# --------------------------------------------------------------------------


def test_koba_nielsen_and_faddeev_popov() -> None:
    matrix = np.array([[0.0, 2.0], [2.0, 0.0]])
    assert koba_nielsen((0.0, 3.0), matrix) == pytest.approx(9.0)
    assert math.isinf(koba_nielsen((1.0, 1.0), matrix))
    assert faddeev_popov((0.0, 1.0, 3.0)) == pytest.approx(6.0)
    with pytest.raises(ValueError, match="punctures"):
        koba_nielsen((0.0, 1.0, 2.0), matrix)


def test_bad_inputs_are_rejected() -> None:
    matrix = four_point_exponents(-2.0, -2.5)
    with pytest.raises(ValueError, match="anchor"):
        ordered_amplitude(matrix, anchor=0.5)
    with pytest.raises(ValueError, match="four and five"):
        ordered_amplitude(np.zeros((6, 6)))
    with pytest.raises(ValueError, match="dimensions"):
        tachyon_momenta(6.0, -1.5, dim=3)
    with pytest.raises(ValueError, match="s > 0"):
        tachyon_momenta(-2.0, -2.5)
    with pytest.raises(ValueError, match="physical region"):
        tachyon_momenta(6.0, -40.0)
