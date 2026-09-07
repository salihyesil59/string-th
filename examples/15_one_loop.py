"""One loop: the moduli-space integral, and why there is no ultraviolet.

Run:  python examples/15_one_loop.py

``examples/06_amplitudes.py`` computes a tree amplitude in closed form.  At one
loop the answer is an integral over the shape of a torus, and the interesting
content is the *shape of the integration region*: modular invariance means the
ultraviolet is not cut off, it is absent.

Writes ``figures/fundamental_domain.png``, ``figures/one_loop_integrand.png``,
``figures/channel_duality.png`` and ``figures/modular_reduction.gif``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stringsim.amplitudes.oneloop import (  # noqa: E402
    FUNDAMENTAL_DOMAIN_FLOOR,
    abstruse_residual,
    annulus_integrand,
    channel_duality_residual,
    closed_channel_integrand,
    fundamental_domain_representative,
    in_fundamental_domain,
    large_tau_behaviour,
    modular_residual,
    modular_s,
    superstring_annulus_integrand,
    superstring_torus_integrand,
    torus_amplitude,
    torus_integrand,
)
from stringsim.quantum.partition import jacobi_identity_residual  # noqa: E402
from stringsim.quantum.spectrum import closed_bosonic_spectrum  # noqa: E402
from stringsim.units import Conventions  # noqa: E402
from stringsim.viz.animate import animate_modular_reduction  # noqa: E402
from stringsim.viz.plots import (  # noqa: E402
    plot_channel_duality,
    plot_fundamental_domain,
    plot_one_loop_integrand,
)

FIG = Path(__file__).resolve().parents[1] / "figures"
CONV = Conventions()


def domain_outline(matrix=None, n_points: int = 160) -> np.ndarray:
    """The boundary of the fundamental domain, optionally moved by an SL(2,Z) element."""
    angles = np.linspace(math.pi / 3.0, 2.0 * math.pi / 3.0, n_points)
    arc = np.exp(1j * angles)
    top = 6.0
    boundary = np.concatenate(
        [arc, [complex(-0.5, top), complex(0.5, top)], arc[:1]]
    )
    if matrix is None:
        return boundary
    a, b = matrix[0]
    c, d = matrix[1]
    return (a * boundary + b) / (c * boundary + d)


def the_integrand_is_modular_invariant() -> None:
    print("=" * 72)
    print("The integrand, and the two moves that must not change it")
    print("=" * 72)
    print("  Z = int_F (d^2 tau / tau_2^2) (tau_2^(1/2) |eta(tau)|^2)^-(D-2)")
    print()
    rng = np.random.default_rng(5)
    points = [complex(rng.uniform(-0.8, 0.8), rng.uniform(0.4, 2.6)) for _ in range(200)]
    for dim in (26, 10, 6):
        worst = [max(residual) for residual in (modular_residual(p, dim) for p in points)]
        print(f"  D = {dim:>2d}: max relative change under T and S = {max(worst):.2e}")
    print()
    print("  The measure d^2 tau / tau_2^2 is invariant by inspection; the rest is")
    print("  because tau_2^(1/2) |eta|^2 is, which is checked here rather than")
    print("  deduced from eta(-1/tau) = sqrt(-i tau) eta(tau).")
    print()


def there_is_no_ultraviolet() -> None:
    print("=" * 72)
    print("Where the ultraviolet went")
    print("=" * 72)
    print("  A field theory integrates the Schwinger parameter down to zero and")
    print("  diverges there.  Here small tau_2 is not a region at all:")
    print()
    print(f"  {'starting tau':>22s} {'tau_2':>10s}  ->  {'reduced tau':>22s} {'tau_2':>10s}"
          f" {'steps':>6s}")
    for tau in (
        complex(0.30, 0.05),
        complex(-2.70, 0.011),
        complex(0.13, 0.0007),
        complex(7.42, 0.00002),
    ):
        found = fundamental_domain_representative(tau)
        print(f"  {f'{tau.real:+.4f}{tau.imag:+.5f}i':>22s} {tau.imag:>10.5f}  ->  "
              f"{f'{found.tau.real:+.4f}{found.tau.imag:+.5f}i':>22s} {found.tau.imag:>10.5f}"
              f" {found.steps:>6d}")
        assert in_fundamental_domain(found.tau)
        assert abs(found.apply(tau) - found.tau) < 1e-9
    print()
    rng = np.random.default_rng(1)
    lowest = min(
        fundamental_domain_representative(
            complex(rng.uniform(-8.0, 8.0), 10.0 ** rng.uniform(-5.0, 0.5))
        ).tau.imag
        for _ in range(4000)
    )
    print(f"  Over 4000 random starting points the smallest tau_2 reached is {lowest:.6f},")
    print(f"  against the corner value sqrt(3)/2 = {FUNDAMENTAL_DOMAIN_FLOOR:.6f}.")
    print("  The would-be ultraviolet is a copy of a region already counted, so")
    print("  there is nothing to regulate -- and every matrix above has det = 1,")
    print("  so nothing was thrown away either.")
    print()


def the_infrared_is_the_tachyon() -> None:
    print("=" * 72)
    print("What does diverge, and what it is")
    print("=" * 72)
    fit = large_tau_behaviour(26)
    print(f"  fit of log(integrand) at large tau_2:  {fit}")
    ground = closed_bosonic_spectrum(1, CONV)[0]
    print(f"  closed_bosonic_spectrum at N = 0:      alpha' M^2 = {ground.alpha_m2:+.6f}")
    print("  The exponential rate is the mass of the lightest closed string, read")
    print("  off an amplitude rather than a spectrum, and the two agree.")
    print()
    print("  The power of tau_2 is a second check: it has to be -(D-2)/2, the")
    print("  transverse momentum integral.")
    print(f"  {'D':>4s} {'alpha M^2':>12s} {'tau_2 power':>13s} {'-(D-2)/2':>10s}")
    for dim in (26, 10, 6):
        this = large_tau_behaviour(dim)
        print(f"  {dim:>4d} {this.alpha_m2:>12.6f} {this.log_power:>13.4f} {-(dim - 2) / 2:>10.1f}")
    print()
    for cutoff in (3.0, 4.0, 5.0):
        print(f"    integral over F with tau_2 <= {cutoff}: {torus_amplitude(tau2_max=cutoff):.4g}")
    print("  It runs away, as a theory with a tachyon should.  This is an infrared")
    print("  problem -- an unstable vacuum -- and not a short-distance one.")
    print()


def the_superstring_integrand_is_zero() -> None:
    print("=" * 72)
    print("The superstring: zero before the integral is done")
    print("=" * 72)
    rng = np.random.default_rng(3)
    points = [complex(rng.uniform(-0.5, 0.5), rng.uniform(0.5, 3.0)) for _ in range(200)]
    print(f"  max |theta_3^4 - theta_2^4 - theta_4^4| over 200 points = "
          f"{max(abstruse_residual(p) for p in points):.2e}")
    print(f"  max type II torus integrand                             = "
          f"{max(superstring_torus_integrand(p) for p in points):.2e}")
    print(f"  jacobi_identity_residual on the q-series (first terms)  = "
          f"{jacobi_identity_residual(12)[:6]}")
    print()
    print("  The one-loop cosmological constant vanishes pointwise on the upper")
    print("  half plane, not after some cancellation between regions.  It is the")
    print("  same identity the partition module proves on integer series, here as")
    print("  a statement about functions.")
    print()


def one_diagram_two_channels() -> None:
    print("=" * 72)
    print("The cylinder: an open loop and a closed exchange")
    print("=" * 72)
    print("  Between two Dp-branes the same diagram is a trace over open strings")
    print("  in t, and closed strings propagating in s = 1/t.  eta(i/t) =")
    print("  sqrt(t) eta(it) turns one integrand into the other:")
    print()
    print(f"  {'p':>3s} {'y':>5s} {'t':>7s} {'open channel':>16s} {'closed at 1/t':>16s}"
          f" {'relative':>10s}")
    for p in (0, 3, 6):
        for modulus in (0.4, 1.0, 2.5):
            here = annulus_integrand(modulus, p, 1.3, CONV)
            there = closed_channel_integrand(1.0 / modulus, p, 1.3, CONV)
            print(f"  {p:>3d} {1.3:>5.1f} {modulus:>7.2f} {here:>16.8g} {there:>16.8g}"
                  f" {channel_duality_residual(modulus, p, 1.3, CONV):>10.1e}")
    print()
    print("  That equality is why a one-loop open-string diagram is a statement")
    print("  about closed strings, and therefore about gravity: the force between")
    print("  branes comes out of a gauge-theory loop.")
    print()
    print("  For the superstring the same integrand vanishes:")
    print(f"  {'t':>7s} {'bosonic':>14s} {'superstring':>14s}")
    for modulus in (0.4, 1.0, 2.5):
        print(f"  {modulus:>7.2f} {annulus_integrand(modulus, 3, 1.3, CONV):>14.6g}"
              f" {superstring_annulus_integrand(modulus, 3):>14.3g}")
    print("  Parallel BPS branes exert no force.  In the open channel that is")
    print("  bosons cancelling fermions; in the closed channel it is NS-NS")
    print("  attraction cancelling R-R repulsion.  One identity, seen twice.")
    print()


def figures() -> None:
    generators = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 1], [0, 1]]),
        np.array([[1, -1], [0, 1]]),
        np.array([[0, -1], [1, 0]]),
        np.array([[0, -1], [1, 1]]),
        np.array([[0, -1], [1, -1]]),
        np.array([[1, 0], [1, 1]]),
        np.array([[1, 0], [-1, 1]]),
        np.array([[1, -1], [1, 0]]),
        np.array([[1, 1], [-1, 0]]),
    ]
    images = [domain_outline(matrix) for matrix in generators]
    marked, links = [], []
    for tau in (complex(0.30, 0.05), complex(-0.42, 0.18)):
        found = fundamental_domain_representative(tau)
        marked.append((rf"$\tau_2 = {tau.imag:.2f}$", tau))
        marked.append((rf"$\to {found.tau.imag:.2f}$", found.tau))
        links.append((tau, found.tau))
    plot_fundamental_domain(
        images, FIG / "fundamental_domain.png", points=marked, links=links
    )
    print(f"  wrote {FIG / 'fundamental_domain.png'}")

    heights = np.linspace(2.0, 9.0, 60)
    values = np.array([torus_integrand(complex(0.0, h)) for h in heights])
    fit = large_tau_behaviour(26)
    reference = math.exp(-math.pi * fit.alpha_m2 * heights[0]) * heights[0] ** fit.log_power
    scale = values[0] / reference
    fitted = scale * np.exp(-math.pi * fit.alpha_m2 * heights) * heights**fit.log_power
    plot_one_loop_integrand(heights, values, fitted, FIG / "one_loop_integrand.png")
    print(f"  wrote {FIG / 'one_loop_integrand.png'}")

    moduli = np.geomspace(0.25, 4.0, 120)
    open_channel = np.array([annulus_integrand(t, 3, 1.3, CONV) for t in moduli])
    closed = np.array([closed_channel_integrand(1.0 / t, 3, 1.3, CONV) for t in moduli])
    plot_channel_duality(moduli, open_channel, closed, FIG / "channel_duality.png")
    print(f"  wrote {FIG / 'channel_duality.png'}")

    animate_modular_reduction(
        complex(-2.70, 0.011),
        FIG / "modular_reduction.gif",
        domain_boundary=domain_outline(),
        title="from the would-be ultraviolet into the domain",
    )
    print(f"  wrote {FIG / 'modular_reduction.gif'}")
    print("    It starts at tau_2 = 0.011 and ends near 1.  Everything a field")
    print("    theory would call short distance is a copy of something already")
    print("    counted -- which is the whole reason there is nothing to regulate.")
    print()


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    the_integrand_is_modular_invariant()
    there_is_no_ultraviolet()
    the_infrared_is_the_tachyon()
    the_superstring_integrand_is_zero()
    one_diagram_two_channels()
    figures()
    print(f"  (S maps {complex(0.0, 0.05)} to {modular_s(complex(0.0, 0.05))}: "
          "the ultraviolet is the infrared)")
    print("done.")
