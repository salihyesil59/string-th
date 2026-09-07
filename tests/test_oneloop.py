"""The one-loop moduli-space integral.

The whole module is checks. Modular invariance of the integrand is verified on
random points rather than deduced from the transformation of ``eta``; the
fundamental-domain walk has to land inside the domain, with an integer matrix of
determinant one that reproduces the move; the infrared growth has to give back
``alpha' M^2 = -4`` for the lightest closed string, which
``quantum/spectrum.py`` computed independently; the two channels of the cylinder
have to be the same function; and the superstring's integrand has to vanish
pointwise rather than after integration.

The one number worth stating twice: reducing a point with ``tau_2 = 2e-5``
returns ``tau_2 >= sqrt(3)/2``. That is not a regulator. It is the statement
that the region a field theory would call ultraviolet is a copy of one already
integrated over.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stringsim.amplitudes.oneloop import (
    FUNDAMENTAL_DOMAIN_FLOOR,
    abstruse_residual,
    annulus_integrand,
    channel_duality_residual,
    closed_channel_integrand,
    fundamental_domain_representative,
    in_fundamental_domain,
    jacobi_theta,
    large_tau_behaviour,
    modular_residual,
    modular_s,
    modular_t,
    superstring_annulus_integrand,
    superstring_torus_integrand,
    tachyon_alpha_m2,
    torus_amplitude,
    torus_integrand,
)
from stringsim.quantum.partition import dedekind_eta
from stringsim.quantum.spectrum import closed_bosonic_spectrum
from stringsim.units import Conventions

CONV = Conventions()


def sample(count: int = 40, seed: int = 5) -> list[complex]:
    rng = np.random.default_rng(seed)
    return [complex(rng.uniform(-0.9, 0.9), rng.uniform(0.4, 2.8)) for _ in range(count)]


# --------------------------------------------------------------------------
# theta functions
# --------------------------------------------------------------------------


def test_the_abstruse_identity_holds_as_functions():
    assert max(abstruse_residual(tau) for tau in sample()) < 1e-12


def test_theta_functions_have_the_expected_leading_behaviour():
    """At large tau_2 only the leading power of q survives each product."""
    tau = complex(0.0, 3.0)
    q = complex(math.exp(-2.0 * math.pi * 3.0), 0.0)
    assert jacobi_theta(2, tau).real == pytest.approx(2.0 * q**0.125, rel=1e-6)
    assert jacobi_theta(3, tau).real == pytest.approx(1.0 + 2.0 * q**0.5, rel=1e-6)
    assert jacobi_theta(4, tau).real == pytest.approx(1.0 - 2.0 * q**0.5, rel=1e-6)


def test_theta_input_is_validated():
    with pytest.raises(ValueError, match="must be 2, 3 or 4"):
        jacobi_theta(1, complex(0.0, 1.0))
    with pytest.raises(ValueError, match="upper half plane"):
        jacobi_theta(3, complex(0.0, -1.0))


# --------------------------------------------------------------------------
# the modular group
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [26, 10, 6])
def test_the_integrand_is_modular_invariant(dim):
    for tau in sample():
        under_t, under_s = modular_residual(tau, dim)
        assert under_t < 1e-11
        assert under_s < 1e-11


def test_the_generators_do_what_they_say():
    tau = complex(0.3, 1.4)
    assert modular_t(tau) == tau + 1.0
    assert modular_s(tau) == pytest.approx(-1.0 / tau)
    assert dedekind_eta(modular_s(tau)) == pytest.approx(
        (-1j * tau) ** 0.5 * dedekind_eta(tau)
    )


def test_the_domain_test_matches_its_definition():
    assert in_fundamental_domain(complex(0.0, 1.5))
    assert in_fundamental_domain(complex(0.5, 1.0))
    assert not in_fundamental_domain(complex(0.7, 1.5))
    assert not in_fundamental_domain(complex(0.0, 0.5))


@pytest.mark.parametrize(
    "tau",
    [
        complex(0.30, 0.05),
        complex(-2.70, 0.011),
        complex(0.13, 0.0007),
        complex(7.42, 0.00002),
        complex(0.0, 1.0),
        complex(-0.5, 0.9),
    ],
)
def test_the_walk_lands_in_the_domain(tau):
    found = fundamental_domain_representative(tau)
    assert in_fundamental_domain(found.tau)
    assert found.tau.imag >= FUNDAMENTAL_DOMAIN_FLOOR - 1e-9
    assert found.determinant == 1
    assert found.matrix.dtype.kind in "iu"
    assert abs(found.apply(tau) - found.tau) < 1e-8


def test_no_point_anywhere_reduces_below_the_floor():
    """This is the whole ultraviolet argument, as a number."""
    rng = np.random.default_rng(2)
    lowest = 1e9
    for _ in range(1500):
        tau = complex(rng.uniform(-9.0, 9.0), 10.0 ** rng.uniform(-5.0, 0.5))
        lowest = min(lowest, fundamental_domain_representative(tau).tau.imag)
    assert lowest >= FUNDAMENTAL_DOMAIN_FLOOR - 1e-9
    assert lowest < FUNDAMENTAL_DOMAIN_FLOOR + 0.05  # the bound is attained, not just true


def test_the_integrand_is_the_same_at_a_point_and_its_representative():
    """And the residual is a convergence question, not a physics one.

    The ``eta`` product needs ``q^n`` to become small, and ``|q| = e^{-2 pi
    tau_2}`` is close to 1 deep in the would-be ultraviolet.  With the default
    number of terms the two sides agree to a part in 1e4; asking for more brings
    them together, which is what says the gap was the truncation.
    """
    for tau in (complex(0.30, 0.05), complex(-2.70, 0.011)):
        found = fundamental_domain_representative(tau)
        coarse = abs(torus_integrand(tau) / torus_integrand(found.tau) - 1.0)
        fine = abs(
            torus_integrand(tau, 26, 4000) / torus_integrand(found.tau, 26, 4000) - 1.0
        )
        assert coarse < 1e-3
        assert fine < 1e-11


def test_the_walk_refuses_the_lower_half_plane():
    with pytest.raises(ValueError, match="upper half plane"):
        fundamental_domain_representative(complex(0.0, -1.0))


# --------------------------------------------------------------------------
# the infrared
# --------------------------------------------------------------------------


def test_the_growth_rate_is_the_tachyon_mass():
    """Measured from the amplitude, compared with the spectrum module."""
    ground = closed_bosonic_spectrum(1, CONV)[0]
    assert tachyon_alpha_m2(26) == pytest.approx(ground.alpha_m2, abs=1e-6)
    assert tachyon_alpha_m2(26) == pytest.approx(-4.0, abs=1e-6)


@pytest.mark.parametrize("dim", [26, 16, 10, 6])
def test_the_subleading_power_is_the_momentum_integral(dim):
    fit = large_tau_behaviour(dim)
    assert fit.log_power == pytest.approx(-(dim - 2) / 2.0, abs=1e-4)
    assert fit.alpha_m2 == pytest.approx(-4.0 * (dim - 2) / 24.0, abs=1e-6)
    assert fit.residual < 1e-6


def test_the_integral_diverges_with_the_cutoff():
    values = [torus_amplitude(tau2_max=cut) for cut in (3.0, 3.5, 4.0)]
    assert values[0] < values[1] < values[2]
    # each half-unit of tau_2 multiplies the answer by tens, and by more each time
    assert values[1] / values[0] > 10.0
    assert values[2] / values[1] > values[1] / values[0]
    with pytest.raises(ValueError, match="tau2_max"):
        torus_amplitude(tau2_max=0.5)


def test_the_integrand_rejects_bad_input():
    with pytest.raises(ValueError, match="upper half plane"):
        torus_integrand(complex(0.0, -1.0))
    with pytest.raises(ValueError, match="dim must exceed"):
        torus_integrand(complex(0.0, 1.0), dim=2)


# --------------------------------------------------------------------------
# the superstring
# --------------------------------------------------------------------------


def test_the_superstring_torus_integrand_vanishes_pointwise():
    assert max(superstring_torus_integrand(tau) for tau in sample()) < 1e-18


def test_the_superstring_cylinder_vanishes_too():
    for modulus in (0.3, 0.7, 1.0, 2.0, 3.0):
        superstring = superstring_annulus_integrand(modulus, 3)
        bosonic = annulus_integrand(modulus, 3, 1.3, CONV)
        assert superstring < 1e-10
        assert superstring / bosonic < 1e-13
    with pytest.raises(ValueError, match="must be positive"):
        superstring_annulus_integrand(0.0, 3)


# --------------------------------------------------------------------------
# the cylinder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0, 1, 3, 6])
@pytest.mark.parametrize("separation", [0.4, 1.3, 3.0])
def test_the_two_channels_are_the_same_function(p, separation):
    for modulus in (0.3, 0.6, 1.0, 1.7, 3.0):
        assert channel_duality_residual(modulus, p, separation, CONV) < 1e-12


def test_the_bosonic_cylinder_diverges_at_both_ends():
    """Two divergences, one tachyon seen from each channel.

    Large ``t`` is a long open-string loop and the open tachyon runs away in it;
    small ``t`` is a long *closed* cylinder, and there the closed tachyon is
    exchanged.  The duality says the second is the first read backwards, which
    is why they are the same number at ``t`` and ``1/t``.
    """
    middle = annulus_integrand(1.0, 3, 1.0, CONV)
    assert annulus_integrand(5.0, 3, 1.0, CONV) > 1e3 * middle
    assert annulus_integrand(0.05, 3, 1.0, CONV) > 1e3 * middle
    for modulus in (0.05, 5.0):
        assert closed_channel_integrand(1.0 / modulus, 3, 1.0, CONV) == pytest.approx(
            annulus_integrand(modulus, 3, 1.0, CONV)
        )


def test_separating_the_branes_suppresses_the_open_channel():
    """The exponential is the stretched string's tension times its length."""
    values = [annulus_integrand(1.0, 3, y, CONV) for y in (0.5, 1.5, 3.0)]
    assert values[0] > values[1] > values[2]
    ratio = math.log(values[0] / values[1])
    expected = (1.5**2 - 0.5**2) / (2.0 * math.pi * CONV.alpha_prime)
    assert ratio == pytest.approx(expected, rel=1e-12)


def test_the_cylinder_rejects_a_bad_modulus():
    with pytest.raises(ValueError, match="must be positive"):
        annulus_integrand(0.0, 3)
    with pytest.raises(ValueError, match="must be positive"):
        closed_channel_integrand(-1.0, 3)
