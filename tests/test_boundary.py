"""Boundary states: a D-brane written as a closed string.

``dbrane.py`` describes a brane by what ends on it.  This describes the same
brane by what it emits -- a coherent state of closed strings -- and the two
have to agree.  The coherent state is an ansatz until the gluing conditions are
applied to it, which is what the explicit Fock space of ``quantum/fock.py`` is
for.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.branes.boundary import (
    boundary_coefficients,
    boundary_residual,
    classical_boundary_state,
    eta_factor,
    eta_residual,
    gluing_residual,
    gluing_signs,
    neumann_count,
    oscillator_overlap,
    overlap_closed_form,
    overlap_truncation,
)
from stringsim.classical.modes import ClosedString
from stringsim.quantum.partition import dedekind_eta, oscillator_degeneracies
from stringsim.units import Conventions

# --------------------------------------------------------------------------
# the gluing conditions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("p", "dim"), [(0, 4), (3, 10), (9, 10), (-1, 6)])
def test_gluing_signs_count_the_worldvolume(p: int, dim: int) -> None:
    """``-1`` along the brane, ``+1`` across it, and the brane is ``p + 1`` wide."""
    signs = gluing_signs(p, dim)
    assert neumann_count(signs) == p + 1
    assert set(np.unique(signs)) <= {-1.0, 1.0}
    assert len(signs) == dim


@pytest.mark.parametrize(("mode", "index"), [(1, 0), (1, 2), (2, 0), (2, 3), (3, 1)])
@pytest.mark.parametrize("p", [-1, 0, 1])
def test_the_coherent_state_satisfies_the_gluing(mode: int, index: int, p: int) -> None:
    r"""``(alpha_n - S alpha~_{-n})|B> = 0``, built and applied, not asserted.

    The state is constructed level by level in the Fock space and the operator
    is applied to it.  Exactly zero for every mode number, every direction and
    every brane dimension: the coefficients are rational and the cancellation
    is term by term rather than numerical.
    """
    dim = 4
    signs = gluing_signs(p, dim)
    assert gluing_residual(3, dim, signs, mode, index) == 0.0


def test_the_metric_in_the_coefficients_is_load_bearing() -> None:
    r"""Drop :math:`\eta_\mu` and only the timelike condition fails.

    The commutator :math:`[\alpha_n^\mu, \alpha_{-n}^\nu] = n \eta^{\mu\nu}`
    puts a metric factor in the exponent, and it is invisible in every spacelike
    direction because :math:`\eta = +1` there.  A check that looked at one
    spatial direction would pass a wrong state.
    """
    signs = gluing_signs(1, 4)
    without = [gluing_residual(3, 4, signs, 1, i, include_metric=False) for i in range(4)]
    assert without[0] == pytest.approx(2.0)
    assert all(value == 0.0 for value in without[1:])
    assert all(gluing_residual(3, 4, signs, 1, i) == 0.0 for i in range(4))


def test_the_coefficients_are_what_expanding_the_exponential_gives() -> None:
    r""":math:`(S\eta)^m / (n^m m!)`, worked out by hand for the low levels."""
    dim = 3
    signs = gluing_signs(0, dim)  # index 0 Neumann, 1 and 2 Dirichlet
    level_one = boundary_coefficients(1, dim, signs)
    # index 0 is timelike and Neumann: S = -1, eta = -1, so S eta = +1
    assert level_one[((1, 0),)] == pytest.approx(1.0)
    # index 1 is spacelike and Dirichlet: S = +1, eta = +1
    assert level_one[((1, 1),)] == pytest.approx(1.0)
    level_two = boundary_coefficients(2, dim, signs)
    assert level_two[((2, 1),)] == pytest.approx(1.0 / 2.0)
    assert level_two[((1, 1), (1, 1))] == pytest.approx(1.0 / math.factorial(2))
    assert boundary_coefficients(0, dim, signs)[()] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the norm
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("q", "cutoff"), [(0.1, 80), (0.3, 140), (0.5, 360)])
@pytest.mark.parametrize("dim", [8, 24])
def test_the_overlap_is_the_partition_function(q: float, cutoff: int, dim: int) -> None:
    r"""Summing over Fock states gives :math:`\prod (1-q^n)^{-\dim}`.

    Every diagonal state contributes 1, because the coefficient cancels against
    the norm; so the overlap is a plain sum over the oscillator degeneracies,
    and that sum is the product.  Two routes, no shared code.

    The cutoff is set per ``q`` and checked to have converged first, so the
    agreement is not being read off a sum that had not finished.
    """
    assert overlap_truncation(q, dim, cutoff) < 1e-13
    summed = oscillator_overlap(q, dim, level_max=cutoff)
    assert summed == pytest.approx(overlap_closed_form(q, dim), rel=1e-11)
    degeneracies = oscillator_degeneracies(cutoff, dim)
    assert summed == pytest.approx(sum(d * q**n for n, d in enumerate(degeneracies)))


def test_the_cutoff_needed_climbs_steeply_with_q() -> None:
    r"""Hagedorn growth, seen from the side that makes it inconvenient.

    The degeneracies rise like :math:`e^{4\pi\sqrt N}`, so the terms grow
    before they fall and the sum converges late.  At ``q = 0.1`` sixty levels
    reach machine precision; at 0.5 it takes several hundred, and a cutoff good
    for the first is off by a factor of two at the second.

    That is a fact about the density of states rather than about the code, and
    it is why every number from the sum is quoted next to
    :func:`overlap_truncation`.
    """
    assert overlap_truncation(0.1, 24, 60) < 1e-13
    assert overlap_truncation(0.5, 24, 60) > 0.1
    assert abs(oscillator_overlap(0.5, 24, 60) / overlap_closed_form(0.5, 24) - 1.0) > 0.5

    needed = []
    for q in (0.1, 0.3, 0.5):
        cutoff = next(
            level for level in range(20, 601, 20) if overlap_truncation(q, 24, level) < 1e-13
        )
        needed.append(cutoff)
    assert needed == sorted(needed)
    assert needed[-1] > 4 * needed[0]


@pytest.mark.parametrize("modulus", [0.4, 0.8, 1.3])
def test_the_norm_is_the_dedekind_eta(modulus: float) -> None:
    r"""The coherent state's oscillator factor is :math:`|\eta|^{-24}`.

    That is the factor
    :func:`~stringsim.amplitudes.oneloop.closed_channel_integrand` carries, so
    the cylinder's closed-channel oscillator content is the norm of a boundary
    state.  ``dedekind_eta`` is an independent implementation.
    """
    assert eta_residual(modulus) < 1e-12
    reference = abs(dedekind_eta(complex(0.0, modulus))) ** -24
    assert eta_factor(modulus, 24) == pytest.approx(reference, rel=1e-12)


# --------------------------------------------------------------------------
# the same conditions on a moving string
# --------------------------------------------------------------------------


def _modes(dim: int, seed: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    return {
        1: rng.normal(size=dim) + 1j * rng.normal(size=dim),
        2: rng.normal(size=dim),
    }


@pytest.mark.parametrize("p", [0, 2, 4])
def test_a_classical_boundary_state_lies_still_on_the_brane(p: int) -> None:
    r"""``Xdot = 0`` along the brane and ``X = y`` across it, at ``tau = 0``.

    The quantum gluing conditions are statements about oscillators; this is the
    same statement about a solution of the wave equation.  A closed string
    whose modes are glued is a string at rest on the brane, sitting at a point
    in the directions transverse to it.
    """
    conv = Conventions(dim=6)
    rng = np.random.default_rng(11)
    position = np.zeros(6)
    position[p + 1 :] = rng.normal(size=6 - p - 1)
    string = classical_boundary_state(_modes(6), p, position=position, conventions=conv)
    neumann, dirichlet = boundary_residual(string, p, position=position)
    assert neumann < 1e-12
    assert dirichlet < 1e-12


def test_a_generic_string_is_not_a_boundary_state() -> None:
    """Otherwise the previous test would be measuring nothing."""
    conv = Conventions(dim=6)
    rng = np.random.default_rng(4)
    string = ClosedString(
        conventions=conv,
        alphas={1: rng.normal(size=6)},
        alphas_tilde={1: rng.normal(size=6)},
    )
    neumann, dirichlet = boundary_residual(string, 2)
    assert neumann > 0.1
    assert dirichlet > 0.1


def test_the_conjugate_in_the_gluing_matters() -> None:
    r""":math:`\tilde\alpha_n = S\alpha_n^{*}`, and the star is not decoration.

    With real mode coefficients it is invisible.  Give a mode a phase and
    dropping it breaks both conditions, which is why
    :func:`classical_boundary_state` conjugates rather than just multiplying by
    the sign.
    """
    conv = Conventions(dim=6)
    modes = _modes(6)
    good = classical_boundary_state(modes, 2, conventions=conv)
    signs = gluing_signs(2, 6)
    naive = ClosedString(
        conventions=conv,
        alphas=dict(modes),
        alphas_tilde={n: signs * a for n, a in modes.items()},
    )
    assert max(boundary_residual(good, 2)) < 1e-12
    assert max(boundary_residual(naive, 2)) > 0.1


def test_real_modes_hide_the_conjugate() -> None:
    """The same construction with real coefficients cannot tell the two apart."""
    conv = Conventions(dim=6)
    rng = np.random.default_rng(9)
    modes = {1: rng.normal(size=6), 2: rng.normal(size=6)}
    signs = gluing_signs(2, 6)
    naive = ClosedString(
        conventions=conv,
        alphas=dict(modes),
        alphas_tilde={n: signs * a for n, a in modes.items()},
    )
    assert max(boundary_residual(naive, 2)) < 1e-12


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        gluing_signs(4, 4)
    with pytest.raises(ValueError, match="dim"):
        gluing_signs(0, 0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        oscillator_overlap(1.0, 24)
    with pytest.raises(ValueError, match="between 0 and 1"):
        overlap_closed_form(0.0, 24)
    with pytest.raises(ValueError, match="modulus"):
        eta_factor(0.0, 24)
    with pytest.raises(ValueError, match="mode number"):
        gluing_residual(2, 4, gluing_signs(0, 4), mode=0)
    with pytest.raises(ValueError, match="index"):
        gluing_residual(2, 4, gluing_signs(0, 4), index=9)
