"""The Narain partition function: modular invariance, and what it needs.

``torus.py`` builds the lattice and ``oneloop.py`` integrates a modular density;
this is the object that joins them.  The claim under test is the one
``narain_gram_matrix`` makes in a docstring -- that even self-duality is what
makes the one-loop amplitude modular invariant -- and the point of these tests
is that the two halves of it fail separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from stringsim.amplitudes.narain import (
    charge_vectors,
    compactified_integrand,
    duality_residual,
    evenness,
    lattice_norms,
    lattice_partition,
    level_degeneracies,
    modular_residual,
    root_multiplicity,
    theta_series,
    truncation_gap,
)
from stringsim.amplitudes.oneloop import torus_integrand
from stringsim.compactification.torus import (
    TorusBackground,
    b_shift,
    basis_change,
    factorized_duality,
    narain_momenta,
    root_vectors,
)

TAU = 0.3 + 1.1j
CUTOFF = 24.0


def _generic() -> TorusBackground:
    """A torus at no special point, with a B-field."""
    rng = np.random.default_rng(5)
    a = rng.normal(size=(2, 2))
    return TorusBackground(
        metric=a @ a.T + 1.5 * np.eye(2),
        b_field=np.array([[0.0, 0.4], [-0.4, 0.0]]),
    )


BACKGROUNDS = {
    "self-dual T1": TorusBackground(metric=np.eye(1)),
    "self-dual T2": TorusBackground(metric=np.eye(2)),
    "generic T2": _generic(),
}


# --------------------------------------------------------------------------
# the enumeration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(BACKGROUNDS))
def test_every_charge_is_inside_the_ball(name: str) -> None:
    """The enumeration is exact, so nothing outside the cutoff appears."""
    background = BACKGROUNDS[name]
    charges = charge_vectors(background, CUTOFF)
    left, right = lattice_norms(background, charges)
    assert np.all(left + right <= CUTOFF + 1e-9)
    assert len(charges) == len(set(map(tuple, charges)))
    assert (0,) * (2 * background.dim) in set(map(tuple, charges))


@pytest.mark.parametrize("name", list(BACKGROUNDS))
def test_the_two_routes_to_the_momenta_agree(name: str) -> None:
    r"""``Z^T H Z`` and ``Z^T eta Z`` against :math:`G^{-1/2}` built explicitly.

    :func:`lattice_norms` adds and subtracts the two invariant quadratic forms;
    :func:`~stringsim.compactification.torus.narain_momenta` constructs
    :math:`\ell_L` and :math:`\ell_R` themselves.  The vectorised route is what
    makes summing thousands of charges cheap, and this is what says it is the
    same quantity.
    """
    background = BACKGROUNDS[name]
    dim = background.dim
    charges = charge_vectors(background, 8.0)
    left, right = lattice_norms(background, charges)
    for i, z in enumerate(charges[:80]):
        l_left, l_right = narain_momenta(background, z[dim:], z[:dim])
        assert abs(float(l_left @ l_left) - left[i]) < 1e-10
        assert abs(float(l_right @ l_right) - right[i]) < 1e-10


@pytest.mark.parametrize("name", list(BACKGROUNDS))
@pytest.mark.parametrize("scale", [1, 2, 3])
def test_the_lattice_is_even(name: str, scale: int) -> None:
    r"""``l_L^2 - l_R^2 = 2 n.w``, an even integer, sublattices included.

    This is the whole of what ``T`` invariance needs, which is why restricting
    the momenta keeps it.
    """
    assert evenness(BACKGROUNDS[name], 12.0, scale) < 1e-10


@pytest.mark.parametrize("name", list(BACKGROUNDS))
def test_the_sum_has_converged(name: str) -> None:
    """Raising the cutoff by half again changes nothing at this precision.

    Every residual below is quoted against this: a discrepancy near the
    truncation gap is arithmetic, one far above it is physics.
    """
    assert truncation_gap(BACKGROUNDS[name], TAU, CUTOFF, extra=12.0) < 1e-12


@pytest.mark.parametrize("name", list(BACKGROUNDS))
def test_theta_is_real_where_the_phase_is(name: str) -> None:
    r"""Each term carries :math:`e^{2\pi i \tau_1 n\cdot w}`, and nothing else.

    So the series is real whenever :math:`2\tau_1` is an integer -- the phase
    is then :math:`\pm 1` -- and in particular on the imaginary axis, where
    every term is positive.  Symmetry under :math:`Z \to -Z` does *not* give
    this: it leaves :math:`n\cdot w` alone rather than flipping it.
    """
    background = BACKGROUNDS[name]
    for tau in (1.1j, 0.5 + 0.9j, 1.0 + 1.3j):
        value = theta_series(background, tau, CUTOFF)
        assert abs(value.imag) < 1e-12 * abs(value.real)


def test_theta_is_genuinely_complex_off_those_lines() -> None:
    r"""With a ``B``-field it is not real at generic :math:`\tau_1`, and that is real.

    The imaginary part is ``1.6e-8`` of the real one here, and it does not move
    when the cutoff is raised by two thirds, so it is the answer rather than the
    truncation.  With ``B = 0`` the map :math:`(n, w) \to (n, -w)` preserves
    :math:`\ell_L^2 + \ell_R^2` and flips :math:`n\cdot w`, which does make
    the series real everywhere.
    """
    metric = _generic().metric
    without = TorusBackground(metric=metric)
    with_field = _generic()
    assert abs(theta_series(without, TAU, CUTOFF).imag) < 1e-15
    tight = theta_series(with_field, TAU, CUTOFF)
    wide = theta_series(with_field, TAU, CUTOFF + 16.0)
    assert abs(tight.imag / tight.real) > 1e-9
    assert abs(tight.imag / wide.imag - 1.0) < 1e-6


@pytest.mark.parametrize("name", list(BACKGROUNDS))
def test_the_reflection_conjugates_it(name: str) -> None:
    r""":math:`\Theta(-\bar\tau) = \overline{\Theta(\tau)}`, exactly.

    This is the reality that matters: it is what makes the integral over the
    fundamental domain real even where the integrand is not.
    """
    background = BACKGROUNDS[name]
    for tau in (TAU, 0.5 + 0.9j):
        here = theta_series(background, tau, CUTOFF)
        reflected = theta_series(background, -tau.conjugate(), CUTOFF)
        assert abs(reflected - here.conjugate()) < 1e-12


# --------------------------------------------------------------------------
# modular invariance, and the two conditions behind it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(BACKGROUNDS))
def test_the_integrand_is_modular_invariant(name: str) -> None:
    """``T`` and ``S`` both leave it alone, to round-off."""
    shift, invert = modular_residual(BACKGROUNDS[name], TAU, CUTOFF)
    assert shift < 1e-10
    assert invert < 1e-10


@pytest.mark.parametrize("name", list(BACKGROUNDS))
@pytest.mark.parametrize("scale", [2, 3])
def test_a_sublattice_keeps_t_and_loses_s(name: str, scale: int) -> None:
    r"""Even is not enough.  Self-dual is the rest of it.

    Restricting the momenta to multiples of ``s`` leaves an even lattice of
    index :math:`s^d`.  ``T`` invariance survives untouched; ``S`` invariance
    fails by a wide margin, and the margin is nothing to do with truncation --
    the sum has converged to ``1e-12``.
    """
    background = BACKGROUNDS[name]
    shift, invert = modular_residual(background, TAU, CUTOFF, momentum_scale=scale)
    assert shift < 1e-10
    assert invert > 1e-3
    assert truncation_gap(background, TAU, CUTOFF, 12.0, momentum_scale=scale) < 1e-10


def test_the_failure_saturates_with_the_index() -> None:
    r"""Any index breaks ``S``, and a coarser one barely breaks it further.

    The naive guess is that the failure grows with the index.  It does not: it
    rises from zero at ``s = 1``, and from ``s = 3`` on it is flat to six
    digits.  Removing momentum modes only matters while they contribute, and at
    :math:`\tau_2 = 1.1` the lightest one already carries
    :math:`e^{-4\pi\tau_2} \approx 10^{-6}`.  So the sublattices differ from
    each other far less than any of them differs from the real lattice.

    Lowering :math:`\tau_2` makes those modes matter again and the saturation
    level rises, which is the same statement seen from the other side.
    """
    background = BACKGROUNDS["self-dual T2"]
    failures = [
        modular_residual(background, TAU, 60.0, momentum_scale=s)[1] for s in (1, 2, 3, 4)
    ]
    assert failures[0] < 1e-12
    assert failures[1] > 0.1
    assert failures == sorted(failures)
    assert abs(failures[3] / failures[2] - 1.0) < 1e-5

    colder = modular_residual(background, 0.3 + 0.35j, 60.0, momentum_scale=3)[1]
    assert colder > failures[2]


def test_the_theta_series_alone_is_not_invariant() -> None:
    r"""It carries weight: :math:`\Theta \to |\tau|^d \Theta` under ``S``.

    The invariance is a cancellation against :math:`|\eta|^{2d}`, so the theta
    series on its own must *fail*, and by exactly that factor.  Checking the
    factor is what makes the cancellation a result rather than a coincidence.
    """
    background = BACKGROUNDS["self-dual T2"]
    here = theta_series(background, TAU, CUTOFF)
    there = theta_series(background, -1.0 / TAU, CUTOFF)
    assert abs(there / here - 1.0) > 0.1
    assert abs(there / (abs(TAU) ** background.dim * here) - 1.0) < 1e-10


@pytest.mark.parametrize("name", list(BACKGROUNDS))
def test_the_ratio_against_eta_is_weight_zero(name: str) -> None:
    """``Theta / |eta|^{2d}`` is invariant on its own, before any measure."""
    background = BACKGROUNDS[name]
    here = lattice_partition(background, TAU, CUTOFF)
    inverted = lattice_partition(background, -1.0 / TAU, CUTOFF)
    assert abs(inverted / here - 1.0) < 1e-10


# --------------------------------------------------------------------------
# the two modules it joins
# --------------------------------------------------------------------------


def test_no_compact_directions_reproduces_the_uncompactified_integrand() -> None:
    """At ``d = 0`` the theta series is 1 and this is ``oneloop.torus_integrand``.

    The empty lattice has exactly one point, and the enumeration has to return
    it: a ``(1, 0)`` array whose ``size`` is zero, which is the trap.
    """
    background = TorusBackground(metric=np.zeros((0, 0)))
    assert charge_vectors(background, CUTOFF).shape == (1, 0)
    assert theta_series(background, TAU, CUTOFF) == pytest.approx(1.0)
    here = compactified_integrand(background, TAU, CUTOFF)
    assert abs(here.real / torus_integrand(TAU) - 1.0) < 1e-12


@pytest.mark.parametrize(
    ("label", "omega"),
    [
        ("factorized duality", factorized_duality(2, 0)),
        ("B-shift", b_shift(np.array([[0.0, 1.0], [-1.0, 0.0]]))),
        ("basis change", basis_change(np.array([[1, 1], [0, 1]]))),
    ],
)
def test_t_duality_leaves_the_partition_function_alone(label: str, omega) -> None:
    r"""``O(d,d;Z)`` moves the moduli and the charges together; the sum does not move.

    The spectrum-level statement is
    :func:`~stringsim.compactification.torus.spectrum_is_dual`, which had to
    follow states through the charge map because a truncated *box* is sheared
    by the duality.  Here the truncation is a ball in the invariant form, so
    the same terms are summed on both sides and the comparison is direct.
    """
    del label
    assert duality_residual(_generic(), omega, TAU, CUTOFF) < 1e-10


@pytest.mark.parametrize("name", list(BACKGROUNDS))
def test_the_root_count_matches_the_exact_solution(name: str) -> None:
    """Roots from the theta expansion against roots solved for exactly.

    :func:`~stringsim.compactification.torus.root_vectors` does not enumerate --
    it bounds the winding from ``w^T G w = 1`` and solves.  Agreement is a check
    on the enumeration the theta series relies on.
    """
    background = BACKGROUNDS[name]
    exact = sum(len(side) for side in root_vectors(background))
    assert root_multiplicity(background, CUTOFF) == exact


def test_the_self_dual_points_are_where_the_roots_are() -> None:
    """Four roots on the self-dual circle, eight on the self-dual square torus."""
    assert root_multiplicity(BACKGROUNDS["self-dual T1"], CUTOFF) == 4
    assert root_multiplicity(BACKGROUNDS["self-dual T2"], CUTOFF) == 8
    assert root_multiplicity(BACKGROUNDS["generic T2"], CUTOFF) == 0


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------


def test_level_degeneracies_partition_the_charges() -> None:
    background = BACKGROUNDS["self-dual T2"]
    counts = level_degeneracies(background, CUTOFF)
    assert sum(counts.values()) == len(charge_vectors(background, CUTOFF))
    assert counts[(0.0, 0.0)] == 1


def test_bad_inputs_are_rejected() -> None:
    background = BACKGROUNDS["self-dual T2"]
    with pytest.raises(ValueError, match="cutoff"):
        charge_vectors(background, 0.0)
    with pytest.raises(ValueError, match="momentum_scale"):
        charge_vectors(background, 4.0, momentum_scale=0)
    with pytest.raises(ValueError, match="upper half plane"):
        theta_series(background, 0.3 - 1.1j, CUTOFF)
    with pytest.raises(ValueError, match="upper half plane"):
        compactified_integrand(background, -1j, CUTOFF)
    with pytest.raises(ValueError, match="do not fit"):
        compactified_integrand(background, TAU, CUTOFF, dim=3)
