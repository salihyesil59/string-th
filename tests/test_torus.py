"""Toroidal compactification, the Narain lattice and ``O(d,d;Z)``.

The load-bearing checks here are the ones that tie this module to something
already trusted: at ``d = 1`` the torus spectrum must coincide, level by level,
with :mod:`stringsim.compactification.circle`; the quadratic form built out of
the momenta must equal the generalized metric assembled from ``(G, B)``; and
every ``O(d,d;Z)`` generator must leave each individual state's mass alone.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from stringsim.compactification import circle
from stringsim.compactification.torus import (
    TorusBackground,
    b_shift,
    basis_change,
    decompose_roots,
    extra_massless_states,
    factorized_duality,
    gauge_algebra,
    identify_algebra,
    is_odd_integer,
    massless_states,
    narain_gram_matrix,
    narain_momenta,
    odd_metric,
    root_vectors,
    spectrum,
    spectrum_is_dual,
    transform,
    transform_charges,
)
from stringsim.units import Conventions

CONV = Conventions(alpha_prime=1.0, dim=26)


def _generic(dim: int = 2) -> TorusBackground:
    """An asymmetric torus with a ``B`` field, at no special point."""
    if dim == 2:
        return TorusBackground(
            np.array([[1.7, 0.35], [0.35, 2.3]]),
            np.array([[0.0, 0.4], [-0.4, 0.0]]),
            CONV,
        )
    metric = np.eye(dim) * 1.9 + 0.23
    b = np.triu(np.ones((dim, dim)) * 0.31, 1)
    return TorusBackground(metric, b - b.T, CONV)


# -- background validation ---------------------------------------------------


def test_background_rejects_bad_moduli():
    with pytest.raises(ValueError):
        TorusBackground(np.array([[1.0, 0.2], [0.3, 1.0]]), conventions=CONV)  # not symmetric
    with pytest.raises(ValueError):
        TorusBackground(np.array([[1.0, 2.0], [2.0, 1.0]]), conventions=CONV)  # not positive
    with pytest.raises(ValueError):
        TorusBackground(np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]]), CONV)  # B symmetric
    with pytest.raises(ValueError):
        TorusBackground(np.eye(2), np.zeros((3, 3)), CONV)  # shape mismatch
    with pytest.raises(ValueError):
        TorusBackground(np.eye(30), conventions=CONV)  # more compact dims than D


def test_from_radii_converts_to_string_units():
    conv = Conventions(alpha_prime=4.0, dim=26)
    bg = TorusBackground.from_radii([2.0, 6.0], conv)
    assert np.allclose(np.diag(bg.metric), [(2 / 2) ** 2, (6 / 2) ** 2])
    with pytest.raises(ValueError):
        TorusBackground.from_radii([1.0, -1.0], conv)


def test_moduli_count_is_d_squared():
    """``d(d+1)/2`` from ``G`` plus ``d(d-1)/2`` from ``B``."""
    for d in (1, 2, 3, 4):
        bg = TorusBackground.self_dual(d, CONV)
        assert bg.moduli_count == d**2
        assert d * (d + 1) // 2 + d * (d - 1) // 2 == d**2


def test_volume_and_dimension():
    bg = TorusBackground.from_radii([1.0, 3.0], CONV)
    assert bg.dim == 2
    assert bg.volume == pytest.approx(3.0)


# -- the reduction to the circle ---------------------------------------------


@pytest.mark.parametrize("radius", [0.35, 0.8, 1.0, 1.7, 3.0])
def test_d_equals_one_reproduces_the_circle_module(radius):
    """The whole level-matched spectrum, not just a summary statistic."""
    bg = TorusBackground.from_radii([radius], CONV)
    mine = sorted(round(s.alpha_m2, 10) for s in spectrum(bg, charge_max=2, level_max=2))
    theirs = sorted(
        round(s.alpha_m2, 10)
        for s in circle.spectrum(radius, CONV, n_max=2, w_max=2, level_max=2)
    )
    assert mine == theirs


@pytest.mark.parametrize("alpha_prime", [0.5, 1.0, 3.0])
def test_circle_agreement_survives_a_change_of_alpha_prime(alpha_prime):
    conv = Conventions(alpha_prime=alpha_prime, dim=26)
    bg = TorusBackground.from_radii([1.3], conv)
    mine = sorted(round(s.alpha_m2, 10) for s in spectrum(bg, charge_max=2, level_max=1))
    theirs = sorted(
        round(s.alpha_m2, 10)
        for s in circle.spectrum(1.3, conv, n_max=2, w_max=2, level_max=1)
    )
    assert mine == theirs


def test_one_dimensional_duality_is_the_circle_radius_inversion():
    for radius in (0.3, 0.8, 2.5):
        bg = TorusBackground.from_radii([radius], CONV)
        dual = transform(bg, factorized_duality(1, 0))
        expected = circle.t_dual_radius(radius, CONV)
        assert float(np.sqrt(dual.metric[0, 0])) == pytest.approx(expected)


# -- momenta and the generalized metric --------------------------------------


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_generalized_metric_is_symmetric_positive_definite_and_in_odd(dim):
    bg = _generic(dim) if dim > 1 else TorusBackground.from_radii([1.4], CONV)
    h = bg.generalized_metric()
    eta = odd_metric(dim)
    assert np.allclose(h, h.T)
    assert np.all(np.linalg.eigvalsh(h) > 0)
    assert np.allclose(h @ eta @ h, eta)  # H lies in O(d,d)


def test_generalized_metric_round_trips_back_to_g_and_b():
    bg = _generic()
    recovered = TorusBackground.from_generalized_metric(bg.generalized_metric(), CONV)
    assert np.allclose(recovered.metric, bg.metric)
    assert np.allclose(recovered.b_field, bg.b_field)


def test_from_generalized_metric_rejects_a_matrix_of_the_wrong_form():
    with pytest.raises(ValueError):
        TorusBackground.from_generalized_metric(np.eye(3), CONV)  # odd size
    bogus = np.eye(4)
    bogus[0, 1] = bogus[1, 0] = 0.5  # symmetric, but not any H(G,B)
    with pytest.raises(ValueError):
        TorusBackground.from_generalized_metric(bogus, CONV)


def test_momenta_reproduce_the_two_quadratic_forms():
    """``l_L^2 + l_R^2 = Z^T H Z`` and ``l_L^2 - l_R^2 = 2 n.w``."""
    bg = _generic()
    h = bg.generalized_metric()
    for n in itertools.product((-2, 0, 1, 3), repeat=2):
        for w in itertools.product((-1, 0, 2), repeat=2):
            left, right = narain_momenta(bg, n, w)
            z = np.concatenate([w, n]).astype(float)
            assert float(left @ left + right @ right) == pytest.approx(float(z @ h @ z))
            assert float(left @ left - right @ right) == pytest.approx(2.0 * np.dot(n, w))


def test_level_matching_holds_for_every_enumerated_state():
    bg = _generic()
    for state in spectrum(bg, charge_max=2, level_max=2):
        expected = 2.0 * (state.level - state.level_tilde)
        assert state.p_left_sq - state.p_right_sq == pytest.approx(expected, abs=1e-9)


# -- the Narain lattice ------------------------------------------------------


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_narain_lattice_is_even_and_self_dual(dim):
    """Even norm and unit determinant -- what makes the torus amplitude modular."""
    gram = narain_gram_matrix(dim)
    assert abs(abs(np.linalg.det(gram)) - 1.0) < 1e-12  # self-dual
    for z in itertools.product((-2, -1, 0, 1, 2), repeat=2 * dim):
        norm = float(np.array(z) @ gram @ np.array(z))
        assert abs(norm % 2) < 1e-12  # even


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_narain_signature_is_d_comma_d(dim):
    eigenvalues = np.linalg.eigvalsh(narain_gram_matrix(dim))
    assert sum(eigenvalues > 0) == dim
    assert sum(eigenvalues < 0) == dim


def test_the_lattice_pairing_is_moduli_independent():
    """``l_L.l'_L - l_R.l'_R`` equals ``Z^T eta Z'`` whatever ``(G, B)`` are."""
    charges = [((1, 0), (0, 1)), ((2, -1), (1, 1)), ((0, 3), (-2, 0))]
    backgrounds = [_generic(), TorusBackground.self_dual(2, CONV), TorusBackground.su3_point(CONV)]
    eta = odd_metric(2)
    for bg in backgrounds:
        for (n, w), (n2, w2) in itertools.product(charges, repeat=2):
            left, right = narain_momenta(bg, n, w)
            left2, right2 = narain_momenta(bg, n2, w2)
            pairing = float(left @ left2 - right @ right2)
            z = np.concatenate([w, n]).astype(float)
            z2 = np.concatenate([w2, n2]).astype(float)
            assert pairing == pytest.approx(float(z @ eta @ z2))


# -- O(d,d;Z) ----------------------------------------------------------------


def _generators(dim: int = 2):
    """One generator of each kind.

    At ``d = 1`` there is no ``B`` field to shift -- a 1x1 antisymmetric matrix
    is zero -- and the only unimodular matrices are ``+/-1``, so the group is
    generated by the single radius inversion.
    """
    gens = {f"duality {k}": factorized_duality(dim, k) for k in range(dim)}
    if dim >= 2:
        unimodular = np.eye(dim, dtype=int)
        unimodular[0, 1] = 1
        theta = np.zeros((dim, dim))
        theta[0, 1] = 1.0
        theta[1, 0] = -1.0
        gens["basis change"] = basis_change(unimodular)
        gens["b shift"] = b_shift(theta)
    else:
        gens["basis change"] = basis_change(np.array([[-1]]))
    return gens


@pytest.mark.parametrize("dim", [2, 3])
def test_generators_lie_in_odd_over_the_integers(dim):
    for omega in _generators(dim).values():
        assert is_odd_integer(omega)


def test_products_and_inverses_stay_in_the_group():
    gens = list(_generators().values())
    product = gens[0] @ gens[1] @ gens[2]
    assert is_odd_integer(product)
    assert is_odd_integer(np.rint(np.linalg.inv(product)))


def test_is_odd_integer_rejects_the_obvious_impostors():
    assert not is_odd_integer(np.eye(3))  # odd size
    assert not is_odd_integer(0.5 * np.eye(4))  # not integer
    assert not is_odd_integer(np.diag([1.0, 2.0, 1.0, 1.0]))  # does not preserve eta


def test_generator_constructors_validate_their_input():
    with pytest.raises(ValueError):
        basis_change(np.array([[2, 0], [0, 1]]))  # det 2, inverse not integral
    with pytest.raises(ValueError):
        basis_change(np.array([[0.5, 0.0], [0.0, 2.0]]))  # not integer
    with pytest.raises(ValueError):
        b_shift(np.array([[0.0, 1.0], [1.0, 0.0]]))  # symmetric
    with pytest.raises(ValueError):
        b_shift(np.array([[0.0, 0.5], [-0.5, 0.0]]))  # not integer
    with pytest.raises(ValueError):
        factorized_duality(2, 5)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_every_generator_leaves_the_spectrum_alone(dim):
    bg = _generic(dim) if dim > 1 else TorusBackground.from_radii([1.4], CONV)
    for name, omega in _generators(dim).items():
        assert spectrum_is_dual(bg, omega, charge_max=2, level_max=1), name


def test_a_composition_of_generators_is_also_a_symmetry():
    bg = _generic()
    gens = _generators()
    product = gens["duality 0"] @ gens["b shift"] @ gens["basis change"]
    assert spectrum_is_dual(bg, product, charge_max=2, level_max=1)


def test_b_field_is_periodic_modulo_integers():
    """A ``B`` shift by an integer antisymmetric matrix is a relabelling, not a new theory."""
    bg = _generic()
    theta = np.array([[0.0, 1.0], [-1.0, 0.0]])
    shifted = transform(bg, b_shift(theta))
    assert np.allclose(shifted.metric, bg.metric)
    assert np.allclose(shifted.b_field, bg.b_field + theta)
    assert spectrum_is_dual(bg, b_shift(theta), charge_max=2, level_max=1)


def test_duality_twice_returns_to_the_start():
    bg = _generic()
    omega = factorized_duality(2, 0)
    back = transform(transform(bg, omega), omega)
    assert np.allclose(back.metric, bg.metric)
    assert np.allclose(back.b_field, bg.b_field)


def test_transform_rejects_a_matrix_outside_the_group():
    with pytest.raises(ValueError):
        transform(_generic(), np.diag([1.0, 2.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        spectrum_is_dual(_generic(), np.diag([1.0, 2.0, 1.0, 1.0]))


def test_transform_charges_matches_the_matrix_action():
    omega = factorized_duality(2, 1)
    n_new, w_new = transform_charges(omega, (1, 2), (3, 4))
    assert tuple(w_new) == (3, 2)  # w^1 <-> n_1 in direction 1
    assert tuple(n_new) == (1, 4)


# -- roots and the gauge algebra ---------------------------------------------


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_self_dual_torus_gives_su2_on_each_side(dim):
    r"""``2d`` roots per side, splitting into ``d`` orthogonal ``su(2)``s."""
    algebra = gauge_algebra(TorusBackground.self_dual(dim, CONV))
    assert algebra.roots_left == algebra.roots_right == 2 * dim
    expected = " + ".join(["su(2)"] * dim)
    assert algebra.name_left == algebra.name_right == expected
    assert algebra.dimension == 6 * dim  # su(2)^d has dimension 3d, twice over
    assert algebra.is_enhanced


def test_a2_point_gives_su3():
    algebra = gauge_algebra(TorusBackground.su3_point(CONV))
    assert algebra.roots_left == algebra.roots_right == 6
    assert algebra.name_left == algebra.name_right == "su(3)"
    assert algebra.dimension == 16  # 8 + 8


def test_generic_torus_has_no_enhancement():
    algebra = gauge_algebra(_generic())
    assert algebra.roots_left == algebra.roots_right == 0
    assert algebra.name_left == "u(1) + u(1)"
    assert not algebra.is_enhanced
    assert algebra.dimension == 4


def test_partial_enhancement_when_only_one_radius_is_self_dual():
    bg = TorusBackground.from_radii([1.0, 2.1], CONV)
    algebra = gauge_algebra(bg)
    assert algebra.roots_left == 2
    assert algebra.name_left == "su(2) + u(1)"


@pytest.mark.parametrize(
    "background",
    [
        TorusBackground.self_dual(1, CONV),
        TorusBackground.self_dual(2, CONV),
        TorusBackground.self_dual(3, CONV),
        TorusBackground.su3_point(CONV),
    ],
)
def test_every_root_has_squared_length_two(background):
    """Which is why only simply-laced algebras can appear here."""
    left, right = root_vectors(background)
    for vectors in (left, right):
        if len(vectors):
            assert np.allclose(np.sum(vectors * vectors, axis=1), 2.0)


def test_roots_come_in_plus_minus_pairs():
    left, _ = root_vectors(TorusBackground.su3_point(CONV))
    for vector in left:
        assert any(np.allclose(vector, -other) for other in left)


def test_roots_agree_with_the_massless_states_found_by_enumeration():
    """The exact root search and the truncated spectrum scan must not disagree."""
    for bg in (TorusBackground.self_dual(2, CONV), TorusBackground.su3_point(CONV)):
        left, right = root_vectors(bg)
        from_spectrum = [s for s in extra_massless_states(bg, charge_max=2) if s.is_root]
        # each root charge vector appears once for each admissible oscillator side
        assert len({(s.momentum, s.winding) for s in from_spectrum}) == len(left) + len(right)


def test_gauge_algebra_is_invariant_under_duality():
    bg = TorusBackground.su3_point(CONV)
    for omega in _generators().values():
        moved = gauge_algebra(transform(bg, omega))
        assert moved.name_left == "su(3)"
        assert moved.name_right == "su(3)"


def test_decompose_roots_separates_orthogonal_pieces():
    """This is what counting alone cannot do."""
    su2_cubed = root_vectors(TorusBackground.self_dual(3, CONV))[0]
    assert decompose_roots(su2_cubed) == [(1, 2), (1, 2), (1, 2)]
    su3 = root_vectors(TorusBackground.su3_point(CONV))[0]
    assert decompose_roots(su3) == [(2, 6)]
    assert decompose_roots(np.zeros((0, 2))) == []


def test_identify_algebra_reports_the_genuine_ambiguity():
    """Rank 3 with six roots is exactly the case the geometry is needed for."""
    assert identify_algebra(3, 6) == ("su(2) + su(2) + su(2)", "su(3) + u(1)")
    assert identify_algebra(2, 6) == ("su(3)",)
    assert identify_algebra(2, 4) == ("su(2) + su(2)",)
    assert identify_algebra(8, 240) == ("e8",)
    assert identify_algebra(4, 24) == ("so(8)",)
    assert identify_algebra(2, 0) == ("u(1) + u(1)",)
    assert identify_algebra(1, 4) == ()  # nothing has this shape
    with pytest.raises(ValueError):
        identify_algebra(-1, 0)


# -- massless spectrum -------------------------------------------------------


def test_generic_torus_has_the_expected_massless_count():
    """``(D-2)^2`` states at ``(n, w) = 0``, and nothing else."""
    bg = _generic()
    assert extra_massless_states(bg, charge_max=2) == []
    total = sum(s.degeneracy for s in massless_states(bg, charge_max=2))
    assert total == (CONV.dim - 2) ** 2


def test_self_dual_torus_has_extra_massless_states_of_two_kinds():
    bg = TorusBackground.self_dual(2, CONV)
    extra = extra_massless_states(bg, charge_max=2)
    roots = [s for s in extra if s.is_root]
    tower = [s for s in extra if not s.is_root]
    assert len(roots) == 8  # 4 left + 4 right
    # the rest are the bosonic tachyon tower passing through zero, as on the circle
    assert all(s.level == s.level_tilde == 0 for s in tower)
    assert all(
        s.p_left_sq == pytest.approx(2.0) and s.p_right_sq == pytest.approx(2.0) for s in tower
    )


def test_spectrum_rejects_negative_ranges():
    with pytest.raises(ValueError):
        spectrum(_generic(), charge_max=-1)
    with pytest.raises(ValueError):
        spectrum(_generic(), level_max=-1)


def test_masses_are_bounded_below_by_the_tachyon():
    """Nothing can be lighter than the ``alpha' M^2 = -4`` ground state."""
    for bg in (_generic(), TorusBackground.self_dual(2, CONV), TorusBackground.su3_point(CONV)):
        assert min(s.alpha_m2 for s in spectrum(bg, charge_max=2, level_max=1)) == pytest.approx(
            -4.0
        )


def test_state_repr_is_informative():
    state = spectrum(TorusBackground.self_dual(1, CONV), charge_max=1, level_max=1)[0]
    text = str(state)
    assert "alpha'M^2" in text and "l_L^2" in text


def test_inverse_sqrt_rejects_a_non_positive_matrix():
    from stringsim.compactification.torus import _inverse_sqrt

    with pytest.raises(ValueError):
        _inverse_sqrt(np.diag([1.0, -1.0]))


def test_su3_point_has_an_integral_e_matrix():
    """``E = G + B`` integer is precisely what lets ``n = E w`` be a momentum."""
    e = TorusBackground.su3_point(CONV).e_matrix
    assert np.allclose(e, np.rint(e))
    assert np.allclose(e, [[1.0, 0.0], [-1.0, 1.0]])


def test_su3_root_lattice_has_the_a2_gram_matrix():
    """Six roots at 60 degrees: the hexagon, not two orthogonal pairs."""
    left, _ = root_vectors(TorusBackground.su3_point(CONV))
    products = {round(float(a @ b), 6) for a in left for b in left}
    assert products == {2.0, 1.0, -1.0, -2.0}  # angles of 0, 60, 120, 180 degrees
    assert math.isclose(float(np.linalg.matrix_rank(left)), 2)
