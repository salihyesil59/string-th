"""The heterotic string on a torus: Gamma_{16+d,d} and Wilson lines.

Two reductions carry most of the weight here. At ``d = 0`` this module must
reproduce :mod:`stringsim.heterotic.spectrum` -- every root at
``(p_L^2, p_R^2) = (2, 0)`` -- and at zero gauge charge with no Wilson line the
compact part must reproduce :mod:`stringsim.compactification.torus` momentum for
momentum. Beyond that, the Wilson-line breakings are checked against the
standard table.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from stringsim.compactification.torus import (
    TorusBackground,
    b_shift,
    basis_change,
    factorized_duality,
    narain_momenta,
)
from stringsim.heterotic.compactified import (
    GAUGE_RANK,
    HeteroticBackground,
    gauge_algebra,
    gauge_rank,
    moduli_count,
    narain_form,
    narain_signature,
    unbroken_roots,
    wilson_boost,
)
from stringsim.heterotic.lattice import d16_plus, e8_squared
from stringsim.heterotic.spectrum import left_mass
from stringsim.units import Conventions

CONV = Conventions()


def _torus(dim: int, with_b: bool = True) -> TorusBackground:
    metric = np.eye(dim) * 1.7 + 0.2 * (np.ones((dim, dim)) - np.eye(dim))
    b_field = np.zeros((dim, dim))
    if with_b and dim > 1:
        b_field[0, dim - 1] = 0.35
        b_field[dim - 1, 0] = -0.35
    return TorusBackground(metric, b_field, CONV)


def _line(values, dim: int = 1) -> np.ndarray:
    """A Wilson line with the given leading gauge components."""
    a = np.zeros((GAUGE_RANK, dim))
    a[: len(values), 0] = values
    return a


# -- shape and counting ------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
def test_moduli_rank_and_signature(dim):
    """``d(d+16)`` moduli, rank ``16 + 2d``, signature ``(16+d, d)``."""
    assert moduli_count(dim) == dim * (dim + 16)
    # metric + B field + Wilson lines, counted separately
    assert moduli_count(dim) == dim * (dim + 1) // 2 + dim * (dim - 1) // 2 + 16 * dim
    assert gauge_rank(dim) == 16 + 2 * dim
    assert narain_signature(dim) == (16 + dim, dim)


def test_counting_helpers_reject_negative_dimensions():
    for function in (moduli_count, gauge_rank, narain_signature):
        with pytest.raises(ValueError):
            function(-1)
    with pytest.raises(ValueError):
        narain_form(-1)


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_background_reports_its_shape(dim):
    torus = None if dim == 0 else _torus(dim)
    background = HeteroticBackground(e8_squared(), torus)
    assert background.dim == dim
    assert background.spacetime_dimension == 10 - dim
    assert background.rank == 16 + 2 * dim
    assert background.moduli_count == dim * (dim + 16)
    assert background.signature == (16 + dim, dim)


def test_background_rejects_a_lattice_of_the_wrong_rank():
    from stringsim.heterotic.lattice import e8

    with pytest.raises(ValueError):
        HeteroticBackground(e8())  # eight-dimensional, not sixteen


# -- the lattice form --------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_narain_form_is_even_and_self_dual(dim):
    eta = narain_form(dim)
    assert eta.shape == (16 + 2 * dim, 16 + 2 * dim)
    assert abs(abs(np.linalg.det(eta)) - 1.0) < 1e-8
    eigenvalues = np.linalg.eigvalsh(eta)
    assert (int(np.sum(eigenvalues > 0)), int(np.sum(eigenvalues < 0))) == narain_signature(dim)


def test_the_form_is_even_on_the_lattice():
    """``pi^2 + 2 n.w`` is even: the gauge norm is even and the rest is twice something."""
    background = HeteroticBackground(e8_squared(), _torus(2))
    eta = background.narain_form()
    for root in background.gauge_lattice.roots[:20]:
        for winding in itertools.product((-1, 0, 1), repeat=2):
            for momentum in itertools.product((-1, 0, 2), repeat=2):
                z = background.charge_vector(root, winding, momentum)
                norm = float(z @ eta @ z)
                assert abs(norm - 2.0 * round(norm / 2.0)) < 1e-8


# -- Wilson lines as a rotation ---------------------------------------------


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_wilson_boost_preserves_the_lattice_form(dim):
    """The momentum shift is exactly what cancels the cross terms."""
    rng = np.random.default_rng(11)
    for _ in range(3):
        boost = wilson_boost(rng.normal(size=(GAUGE_RANK, dim)))
        eta = narain_form(dim)
        assert np.allclose(boost.T @ eta @ boost, eta, atol=1e-8)


def test_wilson_boost_composes():
    """``Omega_A Omega_B`` is again a boost, so Wilson lines add."""
    a = _line([0.3, -0.7])
    b = _line([1.1, 0.25])
    combined = wilson_boost(b) @ wilson_boost(a)
    eta = narain_form(1)
    assert np.allclose(combined.T @ eta @ combined, eta, atol=1e-8)


def test_wilson_boost_validates_its_shape():
    with pytest.raises(ValueError):
        wilson_boost(np.zeros(16))


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_generalized_metric_is_positive_definite_and_in_the_group(dim):
    rng = np.random.default_rng(5)
    torus = None if dim == 0 else _torus(dim)
    lines = None if dim == 0 else rng.normal(size=(GAUGE_RANK, dim)) * 0.4
    background = HeteroticBackground(e8_squared(), torus, lines)
    h = background.generalized_metric()
    eta = background.narain_form()
    assert np.allclose(h, h.T)
    assert np.all(np.linalg.eigvalsh(h) > 0)
    assert np.allclose(h @ np.linalg.inv(eta) @ h, eta, atol=1e-7)


def test_momenta_are_non_negative():
    rng = np.random.default_rng(2)
    background = HeteroticBackground(
        e8_squared(), _torus(2), rng.normal(size=(GAUGE_RANK, 2)) * 0.3
    )
    for root in background.gauge_lattice.roots[:30]:
        z = background.charge_vector(root, (1, -1), (2, 0))
        left, right = background.momenta_squared(z)
        assert left >= -1e-9
        assert right >= -1e-9


# -- the two reductions ------------------------------------------------------


def test_uncompactified_reduces_to_the_heterotic_spectrum_module():
    """Every root at ``(2, 0)``, and the mass formula the same one."""
    background = HeteroticBackground(e8_squared())
    for root in background.gauge_lattice.roots:
        left, right = background.momenta_squared(background.charge_vector(root))
        assert left == pytest.approx(2.0)
        assert right == pytest.approx(0.0, abs=1e-9)
    zero = background.charge_vector(np.zeros(16))
    # matches heterotic.spectrum.left_mass exactly
    assert background.alpha_m2(zero, left_level=0) == pytest.approx(4 * float(left_mass(0, 0)))
    assert background.alpha_m2(zero, left_level=1) == pytest.approx(4 * float(left_mass(1, 0)))
    root_charge = background.charge_vector(background.gauge_lattice.roots[0])
    assert background.alpha_m2(root_charge) == pytest.approx(4 * float(left_mass(0, 2)))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_zero_gauge_charge_reduces_to_the_torus_module(dim):
    """With no gauge charge and no Wilson line, the compact part is the torus."""
    torus = _torus(dim)
    background = HeteroticBackground(e8_squared(), torus)
    for momentum in itertools.product((-1, 0, 2), repeat=dim):
        for winding in itertools.product((-2, 0, 1), repeat=dim):
            z = background.charge_vector(np.zeros(16), winding, momentum)
            mine = background.momenta_squared(z)
            left, right = narain_momenta(torus, momentum, winding)
            assert mine[0] == pytest.approx(float(left @ left), abs=1e-10)
            assert mine[1] == pytest.approx(float(right @ right), abs=1e-10)


def test_torus_dualities_embed_and_still_preserve_the_form():
    """``O(d,d;Z)`` sits inside ``O(16+d,d;Z)`` acting trivially on the gauge part."""
    dim = 2
    eta = narain_form(dim)
    generators = [
        basis_change(np.array([[1, 1], [0, 1]])),
        b_shift(np.array([[0.0, 1.0], [-1.0, 0.0]])),
        factorized_duality(dim, 0),
    ]
    for omega in generators:
        embedded = np.block(
            [
                [np.eye(GAUGE_RANK), np.zeros((GAUGE_RANK, 2 * dim))],
                [np.zeros((2 * dim, GAUGE_RANK)), omega],
            ]
        )
        assert np.allclose(embedded.T @ eta @ embedded, eta, atol=1e-9)


# -- Wilson lines break the gauge group --------------------------------------


def test_no_wilson_line_leaves_the_group_unbroken():
    for lattice, name in ((e8_squared(), "e8 + e8"), (d16_plus(), "so(32)")):
        background = HeteroticBackground(lattice, _torus(1))
        assert len(unbroken_roots(background)) == 480
        assert gauge_algebra(background) == name


def test_uncompactified_has_nothing_to_break_it():
    background = HeteroticBackground(e8_squared())
    assert len(unbroken_roots(background)) == 480
    assert gauge_algebra(background) == "e8 + e8"


@pytest.mark.parametrize(
    "values, roots, algebra",
    [
        ([1.0], 352, "e8 + so(16)"),
        ([0.5, 0.5], 368, "e8 + e7 + su(2)"),
        ([1 / 3], 324, "e8 + so(14) + u(1)"),
    ],
)
def test_standard_e8_breakings(values, roots, algebra):
    """The textbook cases, with the surviving root count as a second check."""
    background = HeteroticBackground(e8_squared(), _torus(1), _line(values))
    assert len(unbroken_roots(background)) == roots
    assert gauge_algebra(background) == algebra


def test_standard_so32_breaking():
    """``A = (1/2^8; 0^8)`` gives ``so(16) + so(16)``: 112 roots from each half."""
    background = HeteroticBackground(d16_plus(), _torus(1), _line([0.5] * 8))
    assert len(unbroken_roots(background)) == 224
    assert gauge_algebra(background) == "so(16) + so(16)"


def test_a_wilson_line_in_the_lattice_acts_trivially():
    r"""``A -> A + lambda`` with ``lambda`` in the lattice leaves the group alone.

    The lattice is integral, so ``lambda . pi`` is an integer for every root and
    the condition ``A . pi in Z`` is untouched.  Wilson lines are therefore
    periodic, exactly as the ``B`` field is on the torus.
    """
    lattice = e8_squared()
    base = _line([0.5, 0.5])
    reference = gauge_algebra(HeteroticBackground(lattice, _torus(1), base))
    for shift in lattice.roots[:8]:
        moved = base + shift.reshape(GAUGE_RANK, 1)
        background = HeteroticBackground(lattice, _torus(1), moved)
        assert gauge_algebra(background) == reference


def test_breaking_never_adds_roots():
    """A Wilson line can only remove gauge bosons at zero winding."""
    rng = np.random.default_rng(17)
    for _ in range(5):
        lines = rng.normal(size=(GAUGE_RANK, 1)) * 0.7
        background = HeteroticBackground(e8_squared(), _torus(1), lines)
        assert len(unbroken_roots(background)) <= 480


def test_the_surviving_roots_really_satisfy_the_condition():
    """``A . pi`` integral, which is what ``p_R = 0`` with zero winding requires."""
    lines = _line([0.5, 0.5])
    background = HeteroticBackground(e8_squared(), _torus(1), lines)
    for root in unbroken_roots(background):
        product = float(root @ lines[:, 0])
        assert abs(product - round(product)) < 1e-9
        # and such a root is genuinely massless: p_R = 0, p_L^2 = 2
        momentum = (round(product),)
        z = background.charge_vector(root, (0,), momentum)
        left, right = background.momenta_squared(z)
        assert right == pytest.approx(0.0, abs=1e-9)
        assert left == pytest.approx(2.0)
        assert background.alpha_m2(z) == pytest.approx(0.0, abs=1e-8)


def test_a_broken_root_is_no_longer_massless():
    """One of the discarded roots, checked to have ``p_R != 0`` for every integer momentum."""
    lines = _line([1.0])
    background = HeteroticBackground(e8_squared(), _torus(1), lines)
    surviving = {tuple(np.round(r, 9)) for r in unbroken_roots(background)}
    discarded = [
        r for r in background.gauge_lattice.roots if tuple(np.round(r, 9)) not in surviving
    ]
    assert discarded, "expected some roots to be projected out"
    root = discarded[0]
    for momentum in range(-3, 4):
        z = background.charge_vector(root, (0,), (momentum,))
        _, right = background.momenta_squared(z)
        assert right > 1e-9


def test_level_matching_holds_for_a_massless_gauge_boson():
    """Left and right masses agree, in both right-moving sectors."""
    background = HeteroticBackground(e8_squared(), _torus(1))
    z = background.charge_vector(background.gauge_lattice.roots[0], (0,), (0,))
    # NS: N_R = 1/2 with a_R = 1/2;  R: N_R = 0 with a_R = 0
    assert background.level_matching_defect(z, 0, 0.5, 0.5) == pytest.approx(0.0)
    assert background.level_matching_defect(z, 0, 0.0, 0.0) == pytest.approx(0.0)
