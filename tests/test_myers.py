"""The Myers effect: matrices that will not commute, and the sphere they make.

Three things are checked against something independent.

The ansatz ``Phi_i = (f/2) J_i`` is required to solve the *full* matrix
equations of motion, and the analytic gradient is checked against the
directional derivative of the potential along random Hermitian directions --
basis free, because contracting entry by entry probes only half of it. The
ordering of configurations is a statement about partitions -- ``sum N_a(N_a^2-1)``
is maximised by one block -- and is verified by enumerating them rather than
argued. And the spherical D2-brane with ``N`` units of flux, shrunk to zero
radius, has to weigh exactly ``N`` D0-branes, which follows from
``dp_brane_tension`` with nothing fitted in between.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.branes.dbrane import dp_brane_tension
from stringsim.branes.myers import (
    LEVI_CIVITA,
    algebra_residual,
    block_configuration,
    casimir,
    configuration_energies,
    fuzzy_radius,
    fuzzy_sphere,
    large_n_ratio,
    latitudes,
    myers_gradient,
    myers_potential,
    noncommutativity,
    partitions,
    shrunk_d2_energy,
    spherical_d2_energy,
    su2_generators,
    trace_j_squared,
)
from stringsim.units import Conventions

CONV = Conventions()
OTHER = Conventions(alpha_prime=0.63)
FLUX = 1.7


# --------------------------------------------------------------------------
# the algebra
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [1, 2, 3, 5, 9, 16])
def test_the_generators_satisfy_su2(dim):
    matrices = su2_generators(dim)
    assert matrices.shape == (3, dim, dim)
    assert algebra_residual(matrices) < 1e-12
    for matrix in matrices:
        assert np.allclose(matrix, matrix.conj().T)


@pytest.mark.parametrize("dim", [1, 2, 3, 5, 9, 16])
def test_the_casimir_is_the_closed_form(dim):
    matrices = su2_generators(dim)
    squared = sum(matrices[i] @ matrices[i] for i in range(3))
    assert np.allclose(squared, casimir(dim) * np.eye(dim))
    assert np.trace(squared).real == pytest.approx(trace_j_squared(dim))
    assert trace_j_squared(dim) == pytest.approx(dim * (dim**2 - 1) / 4.0)


def test_levi_civita_is_what_it_should_be():
    assert LEVI_CIVITA[0, 1, 2] == 1.0
    assert LEVI_CIVITA[0, 2, 1] == -1.0
    assert LEVI_CIVITA[1, 1, 2] == 0.0
    assert LEVI_CIVITA.sum() == pytest.approx(0.0)


def test_a_bad_dimension_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        su2_generators(0)


# --------------------------------------------------------------------------
# the potential and its gradient
# --------------------------------------------------------------------------


def directional_derivative(matrices, direction, flux: float, step: float = 1e-6) -> float:
    """``dV/ds`` along a Hermitian direction, by central differences."""
    matrices = np.asarray(matrices, dtype=complex)
    direction = np.asarray(direction, dtype=complex)
    forward = myers_potential(matrices + step * direction, flux)
    backward = myers_potential(matrices - step * direction, flux)
    return (forward - backward) / (2.0 * step)


def hermitian_triple(rng, dim: int, scale: float = 0.6) -> np.ndarray:
    raw = rng.normal(size=(3, dim, dim)) + 1j * rng.normal(size=(3, dim, dim))
    return np.array([(m + m.conj().T) / 2.0 for m in raw]) * scale


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_the_analytic_gradient_matches_finite_differences(dim):
    """Basis free: dV along any Hermitian direction must be Tr(G_i H_i).

    Contracting entry by entry would only probe the real symmetric directions --
    the gradient is Hermitian, so half of it would go unchecked.
    """
    rng = np.random.default_rng(7 + dim)
    for _ in range(4):
        matrices = hermitian_triple(rng, dim)
        gradient = myers_gradient(matrices, FLUX)
        assert np.allclose(
            gradient, np.array([g.conj().T for g in gradient])
        ), "the gradient must be Hermitian"
        for _ in range(3):
            direction = hermitian_triple(rng, dim, scale=1.0)
            predicted = sum(
                np.trace(gradient[i] @ direction[i]) for i in range(3)
            )
            assert abs(predicted.imag) < 1e-10
            assert directional_derivative(matrices, direction, FLUX) == pytest.approx(
                predicted.real, rel=1e-6, abs=1e-7
            )


@pytest.mark.parametrize("dim", [2, 3, 5, 9, 14])
def test_the_fuzzy_sphere_solves_the_equations_of_motion(dim):
    matrices = fuzzy_sphere(dim, FLUX)
    assert np.max(np.abs(myers_gradient(matrices, FLUX))) < 1e-11


@pytest.mark.parametrize("dim", [3, 5, 9])
def test_moving_off_the_solution_breaks_it(dim):
    matrices = fuzzy_sphere(dim, FLUX)
    for factor in (1.1, 1.3, 0.7):
        moved = matrices.copy()
        moved[0] = moved[0] * factor
        assert np.max(np.abs(myers_gradient(moved, FLUX))) > 0.05


@pytest.mark.parametrize("dim", [2, 3, 5, 9, 14])
def test_the_energy_is_the_closed_form(dim):
    matrices = fuzzy_sphere(dim, FLUX)
    assert myers_potential(matrices, FLUX) == pytest.approx(
        -(FLUX**4) * trace_j_squared(dim) / 96.0
    )


def test_the_ansatz_traces_out_the_expected_curve():
    """V(alpha) = Tr(J^2) (alpha^4/2 - f alpha^3/3), with the minimum at f/2."""
    generators = su2_generators(6)
    for alpha in (0.0, 0.3, FLUX / 2.0, FLUX, 2.0 * FLUX):
        expected = trace_j_squared(6) * (0.5 * alpha**4 - FLUX * alpha**3 / 3.0)
        assert myers_potential(alpha * generators, FLUX) == pytest.approx(expected)


def test_commuting_matrices_cost_nothing_and_are_not_the_minimum():
    diagonal = np.array([np.diag(np.arange(4.0)).astype(complex) for _ in range(3)])
    assert myers_potential(diagonal, FLUX) == pytest.approx(0.0, abs=1e-12)
    assert myers_potential(fuzzy_sphere(4, FLUX), FLUX) < 0.0


def test_the_potential_rejects_nonsense():
    with pytest.raises(ValueError, match="three square matrices"):
        myers_potential(np.zeros((2, 3, 3)))
    with pytest.raises(ValueError, match="Hermitian"):
        skew = np.zeros((3, 2, 2), dtype=complex)
        skew[0][0, 1] = 1.0
        skew[1][1, 0] = 1.0
        skew[2] = np.diag([1.0, -1.0])
        myers_potential(skew, FLUX)


# --------------------------------------------------------------------------
# which configuration wins
# --------------------------------------------------------------------------


@pytest.mark.parametrize("total", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_the_single_block_is_always_the_ground_state(total):
    found = configuration_energies(total, FLUX)
    assert found[0].partition == (total,)
    assert found[0].is_irreducible
    assert found[-1].partition == (1,) * total
    assert found[-1].energy == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("total", [1, 4, 7, 10])
def test_the_partitions_are_all_there(total):
    known = {1: 1, 4: 5, 7: 15, 10: 42}
    found = partitions(total)
    assert len(found) == known[total]
    assert len(set(found)) == len(found)
    assert all(sum(entry) == total for entry in found)
    assert all(list(entry) == sorted(entry, reverse=True) for entry in found)
    with pytest.raises(ValueError, match="at least 1"):
        partitions(0)


@pytest.mark.parametrize("total", [4, 6, 8])
def test_the_energy_ordering_is_the_casimir_ordering(total):
    """The physics reduces to an arithmetic fact about sums of N(N^2-1)."""
    found = configuration_energies(total, FLUX)
    traces = [entry.trace_j_squared for entry in found]
    assert traces == sorted(traces, reverse=True)
    for entry in found:
        assert entry.energy == pytest.approx(-(FLUX**4) * entry.trace_j_squared / 96.0)


def test_a_block_configuration_of_one_block_is_the_fuzzy_sphere():
    assert np.allclose(block_configuration([7], FLUX), fuzzy_sphere(7, FLUX))
    with pytest.raises(ValueError, match="positive integers"):
        block_configuration([2, 0])


# --------------------------------------------------------------------------
# it is a sphere
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 5, 12, 30, 80])
def test_the_radius_and_the_fuzziness(dim):
    matrices = fuzzy_sphere(dim, FLUX)
    assert fuzzy_radius(matrices) == pytest.approx(FLUX / 4.0 * math.sqrt(dim**2 - 1))
    assert noncommutativity(matrices) == pytest.approx(2.0 / dim, rel=0.35)


def test_the_sphere_becomes_classical():
    fuzz = [noncommutativity(fuzzy_sphere(dim, FLUX)) for dim in (2, 8, 32, 128)]
    assert fuzz == sorted(fuzz, reverse=True)
    ratios = [a / b for a, b in zip(fuzz[:-1], fuzz[1:], strict=True)]
    assert all(3.0 < ratio < 4.5 for ratio in ratios), ratios  # falls like 1/N
    assert noncommutativity(np.zeros((3, 4, 4))) == 0.0


@pytest.mark.parametrize("dim", [2, 5, 13])
def test_the_latitudes_are_the_eigenvalues_and_lie_on_the_sphere(dim):
    matrices = fuzzy_sphere(dim, FLUX)
    heights, radii = latitudes(matrices)
    assert len(heights) == dim
    assert np.allclose(heights, np.sort(heights))
    spacing = np.diff(heights)
    assert np.allclose(spacing, FLUX / 2.0)
    assert np.allclose(heights**2 + radii**2, fuzzy_radius(matrices) ** 2)


# --------------------------------------------------------------------------
# the other description
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_units", [1, 2, 5, 20, 100])
@pytest.mark.parametrize("g_s", [0.05, 0.3, 2.0])
@pytest.mark.parametrize("conv", [CONV, OTHER])
def test_a_shrunk_d2_with_n_flux_weighs_n_d0_branes(n_units, g_s, conv):
    assert shrunk_d2_energy(n_units, g_s, conv) == pytest.approx(
        n_units * dp_brane_tension(0, g_s, conv), rel=1e-13
    )


def test_the_d2_energy_grows_with_the_radius():
    values = [spherical_d2_energy(radius, 20, 0.3, CONV) for radius in (0.0, 0.5, 1.0, 2.0)]
    assert values == sorted(values)
    with pytest.raises(ValueError, match="non-negative"):
        spherical_d2_energy(-1.0, 5)
    with pytest.raises(ValueError, match="non-negative"):
        spherical_d2_energy(1.0, -5)


def test_the_leading_correction_to_the_continuum_is_exactly_one_over_n_squared():
    for dim in (2, 5, 20, 100):
        assert large_n_ratio(dim) == pytest.approx(1.0 - 1.0 / dim**2, abs=1e-14)
    assert large_n_ratio(1) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="at least 1"):
        large_n_ratio(0)


def test_the_quartic_expansion_of_the_d2_energy_matches_the_matrix_shape():
    """Both pictures give an energy that goes like R^4 above the flat answer."""
    n_units, g_s = 40, 0.3
    tension = dp_brane_tension(2, g_s, CONV)
    # E = 4 pi T_2 sqrt(R^4 + (pi alpha' N)^2) = N T_0 + 2 T_2 R^4 / (alpha' N) + ...
    coefficient = 2.0 * tension / (CONV.alpha_prime * n_units)
    for radius in (0.2, 0.4):
        above = spherical_d2_energy(radius, n_units, g_s, CONV) - shrunk_d2_energy(
            n_units, g_s, CONV
        )
        assert above == pytest.approx(coefficient * radius**4, rel=1e-3)
