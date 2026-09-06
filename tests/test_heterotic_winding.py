"""Winding enhancement in the compactified heterotic string.

``unbroken_roots`` answers the ``w = 0`` question and was documented as
incomplete. ``massless_vectors`` is the complete answer, and the value of these
tests is that completeness is checkable two ways: the enumeration is derived
from a bound (``w^T G w <= 1``, then a ball of radius ``sqrt 2``), and the
special radii are *solved for* rather than scanned -- so a brute scan of the
moduli space must find exactly the same points and nothing else.

The physics is then pinned by known answers: 480 roots at a generic radius,
482 at the self-dual one, ``so(18) + e8`` and ``so(34)`` where winding states
join the root system, and the whole of ``E8 x E8`` coming back at ``G = 1/8``
from a Wilson line that had broken it.
"""

from __future__ import annotations

import numpy as np
import pytest

from stringsim.compactification.torus import TorusBackground
from stringsim.heterotic.compactified import (
    GAUGE_RANK,
    HeteroticBackground,
    charge_lattice_gram,
    enhanced_algebra,
    enhancement_radii,
    gauge_vectors_near,
    left_momenta,
    massless_vectors,
    unbroken_roots,
)
from stringsim.heterotic.lattice import d16_plus, e8, e8_squared

GENERIC = TorusBackground(np.array([[2.37]]))
SELF_DUAL = TorusBackground(np.array([[1.0]]))


def line(values, dim: int = 1) -> np.ndarray:
    a = np.zeros((GAUGE_RANK, dim))
    a[: len(values), 0] = values
    return a


def circle(metric: float) -> TorusBackground:
    return TorusBackground(np.array([[float(metric)]]))


# --------------------------------------------------------------------------
# the ball enumeration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lattice", "expected"), [(e8(), 240), (e8_squared(), 480), (d16_plus(), 480)]
)
def test_the_ball_of_radius_two_holds_the_roots_and_the_origin(lattice, expected):
    points = gauge_vectors_near(lattice, np.zeros(lattice.dim), 2.0)
    norms = np.sum(points**2, axis=1)
    assert len(points) == expected + 1
    assert int(np.sum(np.abs(norms - 2.0) < 1e-9)) == expected
    assert int(np.sum(norms < 1e-9)) == 1


def test_every_point_found_is_in_the_lattice_and_in_the_ball():
    lattice = e8_squared()
    rng = np.random.default_rng(11)
    centre = rng.normal(size=GAUGE_RANK) * 0.7
    points = gauge_vectors_near(lattice, centre, 1.5)
    assert len(points) > 0
    assert np.all(np.sum((points - centre) ** 2, axis=1) < 1.5 + 1e-8)
    coordinates = np.linalg.solve(lattice.basis().T, points.T).T
    assert np.max(np.abs(coordinates - np.rint(coordinates))) < 1e-9


def test_a_smaller_ball_is_contained_in_a_larger_one():
    lattice = e8()
    centre = np.full(GAUGE_RANK // 2, 0.1)
    small = gauge_vectors_near(lattice, centre, 1.0)
    large = gauge_vectors_near(lattice, centre, 2.0)
    assert len(small) < len(large)
    rows = {tuple(np.round(v, 9)) for v in large}
    assert all(tuple(np.round(v, 9)) in rows for v in small)


def test_a_negative_radius_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        gauge_vectors_near(e8(), np.zeros(8), -1.0)


# --------------------------------------------------------------------------
# left momenta: the vectors behind the norms
# --------------------------------------------------------------------------


def test_left_momenta_reproduce_the_quadratic_forms():
    """The map is only useful if its norms are the ones H and eta already give."""
    rng = np.random.default_rng(5)
    for dim in (1, 2):
        metric = np.eye(dim) * 1.4 + 0.1 * (np.ones((dim, dim)) - np.eye(dim))
        lines = rng.normal(size=(GAUGE_RANK, dim)) * 0.3
        bg = HeteroticBackground(e8_squared(), TorusBackground(metric), lines)
        h, eta = bg.generalized_metric(), bg.narain_form()
        charges = np.array(
            [
                bg.charge_vector(
                    bg.gauge_lattice.roots[rng.integers(0, 480)],
                    rng.integers(-2, 3, dim),
                    rng.integers(-2, 3, dim),
                )
                for _ in range(6)
            ]
        )
        momenta = left_momenta(bg, charges)
        assert momenta.shape == (6, GAUGE_RANK + dim)
        for i, z1 in enumerate(charges):
            for j, z2 in enumerate(charges):
                expected = 0.5 * float(z1 @ h @ z2 + z1 @ eta @ z2)
                assert momenta[i] @ momenta[j] == pytest.approx(expected, abs=1e-10)


def test_left_momenta_at_zero_dimensions_are_the_gauge_charges():
    bg = HeteroticBackground(e8_squared())
    roots = bg.gauge_lattice.roots
    assert np.allclose(left_momenta(bg, roots), roots)


# --------------------------------------------------------------------------
# the complete enumeration
# --------------------------------------------------------------------------


def test_at_zero_dimensions_the_massless_vectors_are_the_roots():
    bg = HeteroticBackground(e8_squared())
    found = np.sort(massless_vectors(bg), axis=0)
    assert np.allclose(found, np.sort(bg.gauge_lattice.roots, axis=0))


@pytest.mark.parametrize("lattice", [e8_squared(), d16_plus()])
@pytest.mark.parametrize("values", [[], [1.0], [0.5, 0.5]])
def test_at_a_generic_radius_it_agrees_with_the_zero_winding_count(lattice, values):
    """Away from the special radii the winding states are all massive."""
    lines = line(values)
    bg = HeteroticBackground(lattice, GENERIC, lines)
    charges = massless_vectors(bg)
    assert len(charges) == len(unbroken_roots(bg))
    # the momentum need not vanish -- n = A . pi -- but the winding must
    assert np.max(np.abs(charges[:, GAUGE_RANK : GAUGE_RANK + bg.dim])) == 0.0


@pytest.mark.parametrize("lattice", [e8_squared(), d16_plus()])
def test_every_massless_vector_really_is_one(lattice):
    bg = HeteroticBackground(lattice, circle(0.125), line([0.5]))
    charges = massless_vectors(bg)
    metric, form = bg.generalized_metric(), bg.narain_form()
    totals = np.einsum("ij,jk,ik->i", charges, metric, charges)
    differences = np.einsum("ij,jk,ik->i", charges, form, charges)
    assert np.allclose(0.5 * (totals + differences), 2.0, atol=1e-9)
    assert np.allclose(0.5 * (totals - differences), 0.0, atol=1e-9)
    for charge in charges[:20]:
        left, right = bg.momenta_squared(charge)
        assert left == pytest.approx(2.0, abs=1e-9)
        assert right == pytest.approx(0.0, abs=1e-9)
        assert bg.alpha_m2(charge) == pytest.approx(0.0, abs=1e-8)


def test_the_self_dual_radius_adds_the_torus_su2():
    """Two extra states, pure winding and momentum, whatever the gauge lattice is."""
    for lattice, name in ((e8_squared(), "e8 + e8"), (d16_plus(), "so(32)")):
        bg = HeteroticBackground(lattice, SELF_DUAL)
        charges = massless_vectors(bg)
        assert len(charges) == 482
        extra = charges[np.abs(charges[:, GAUGE_RANK]) > 1e-9]
        assert len(extra) == 2
        assert np.max(np.abs(extra[:, :GAUGE_RANK])) == 0.0
        assert enhanced_algebra(bg) == f"{name} + su(2) + u(1)"


@pytest.mark.parametrize(
    ("lattice", "values", "metric", "roots", "algebra"),
    [
        (e8_squared(), [1.0], 0.5, 384, "so(18) + e8 + u(1)"),
        (e8_squared(), [0.5], 0.125, 480, "e8 + e8 + u(1) + u(1)"),
        (e8_squared(), [0.5], 0.375, 352, "e8 + so(16) + u(1) + u(1)"),
        (d16_plus(), [1.0], 0.5, 544, "so(34) + u(1)"),
        (d16_plus(), [0.25] * 16, 0.5, 244, "su(16) + su(2) + su(2) + u(1)"),
    ],
)
def test_named_enhancements(lattice, values, metric, roots, algebra):
    bg = HeteroticBackground(lattice, circle(metric), line(values))
    assert len(massless_vectors(bg)) == roots
    assert enhanced_algebra(bg) == algebra


def test_a_wilson_line_can_undo_itself():
    """A = (1/2, 0^15) breaks E8 x E8, and G = 1/8 puts all 480 roots back."""
    lines = line([0.5])
    broken = HeteroticBackground(e8_squared(), GENERIC, lines)
    restored = HeteroticBackground(e8_squared(), circle(0.125), lines)
    assert len(massless_vectors(broken)) == 324
    charges = massless_vectors(restored)
    assert len(charges) == 480
    assert int(np.sum(np.abs(charges[:, GAUGE_RANK]) > 1e-9)) == 156
    assert enhanced_algebra(restored) == enhanced_algebra(
        HeteroticBackground(e8_squared(), GENERIC)
    )


def test_the_rank_never_changes():
    """Enhancement grows the group but not its rank: that is fixed at 16 + 2d."""
    for metric in (0.125, 0.375, 1.0, 2.37):
        bg = HeteroticBackground(e8_squared(), circle(metric), line([0.5]))
        momenta = left_momenta(bg, massless_vectors(bg))
        assert np.linalg.matrix_rank(momenta, tol=1e-8) <= bg.rank


def test_two_dimensions_still_works():
    bg = HeteroticBackground(e8_squared(), TorusBackground(np.eye(2)))
    charges = massless_vectors(bg)
    # 480 gauge roots plus su(2) x su(2) from the two self-dual circles.
    assert len(charges) == 484
    # rank 20: the two extra u(1)s are the right-moving graviphotons, which
    # cannot enhance -- only the left-moving side carries a root system.
    assert enhanced_algebra(bg) == "e8 + e8 + su(2) + su(2) + u(1) + u(1)"


# --------------------------------------------------------------------------
# the special radii, solved for and then scanned for
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lattice", "values", "expected"),
    [
        (e8_squared(), [], [1.0]),
        (e8_squared(), [1.0], [0.5]),
        (e8_squared(), [0.5], [0.125, 0.375, 0.875]),
        (d16_plus(), [], [1.0]),
        (d16_plus(), [0.5] * 8, []),
    ],
)
def test_enhancement_radii_are_the_ones_a_scan_finds(lattice, values, expected):
    """The solved-for radii, and a brute scan of the moduli space that agrees."""
    lines = line(values)
    radii = enhancement_radii(lattice, lines[:, 0], winding_max=2)
    assert radii == pytest.approx(expected)

    baseline = len(unbroken_roots(HeteroticBackground(lattice, GENERIC, lines)))
    window = np.arange(0.1, 1.101, 0.025)
    scanned = [
        round(float(metric), 6)
        for metric in window
        if len(massless_vectors(HeteroticBackground(lattice, circle(metric), lines))) != baseline
    ]
    assert scanned == pytest.approx([r for r in expected if window[0] <= r <= window[-1]])


def test_winding_max_bounds_what_is_guaranteed():
    """Enhancement needs G w^2 <= 1, so everything above 1/winding_max^2 is found."""
    lines = line([0.5])
    small = enhancement_radii(e8_squared(), lines[:, 0], winding_max=1)
    large = enhancement_radii(e8_squared(), lines[:, 0], winding_max=4)
    assert set(small) <= set(large)
    assert all(radius in large for radius in small)
    assert [r for r in large if r > 1.0] == []
    # every radius above 1/1^2 that winding_max = 4 knows about, winding_max = 1 has too
    assert [r for r in large if r > 1.0] == [r for r in small if r > 1.0]


def test_enhancement_radii_needs_a_positive_winding_max():
    with pytest.raises(ValueError, match="winding_max"):
        enhancement_radii(e8_squared(), np.zeros(GAUGE_RANK), winding_max=0)


# --------------------------------------------------------------------------
# the two theories are one
# --------------------------------------------------------------------------


def test_the_charge_lattice_is_even_and_self_dual():
    for lattice in (e8_squared(), d16_plus()):
        for dim in (0, 1, 2):
            torus = None if dim == 0 else TorusBackground(np.eye(dim))
            gram = charge_lattice_gram(HeteroticBackground(lattice, torus))
            eigenvalues = np.linalg.eigvalsh(gram)
            assert gram.shape == (GAUGE_RANK + 2 * dim,) * 2
            assert np.max(np.abs(np.diag(gram) % 2)) < 1e-9
            assert abs(abs(np.linalg.det(gram)) - 1.0) < 1e-6
            assert (int(np.sum(eigenvalues > 0)), int(np.sum(eigenvalues < 0))) == (
                GAUGE_RANK + dim,
                dim,
            )


def test_the_two_heterotic_strings_meet_in_nine_dimensions():
    """so(16) + so(16) at every radius, from either lattice, with no enhancement."""
    pair = [
        (e8_squared(), line([1.0] + [0.0] * 7 + [1.0])),
        (d16_plus(), line([0.5] * 8)),
    ]
    for lattice, lines in pair:
        assert enhancement_radii(lattice, lines[:, 0], winding_max=4) == []
        for metric in (0.2, 0.5, 1.0, 2.37):
            bg = HeteroticBackground(lattice, circle(metric), lines)
            assert len(massless_vectors(bg)) == 224
            assert enhanced_algebra(bg) == "so(16) + so(16) + u(1) + u(1)"
