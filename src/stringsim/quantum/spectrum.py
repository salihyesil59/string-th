r"""The quantised mass spectrum: levels, masses and how many states sit at each.

Open bosonic string
    :math:`\alpha' M^2 = N - 1`, degeneracy :math:`d_N` from
    :func:`~stringsim.quantum.partition.oscillator_degeneracies` with
    ``D - 2 = 24`` species.

Closed bosonic string
    Left- and right-movers are independent but **level matched**,
    :math:`N = \tilde N`, and
    :math:`\alpha' M^2 = 4(N - 1)` with degeneracy :math:`d_N^2`.

Open superstring (GSO projected)
    :math:`\alpha' M^2 = N`, no tachyon, degeneracies
    ``8, 128, 1152, ...``.

Both bosonic theories start with a tachyon.  That is a genuine instability of
the bosonic string, not a bookkeeping error, and it is one of the reasons the
superstring is the theory people actually use.

The leading Regge trajectory carries the maximal spin at each level,
:math:`J = \alpha' M^2 + 1`.  Note the intercept: the classical rotating string
of :mod:`stringsim.classical.rotating` gives :math:`J = \alpha' M^2` exactly,
and the ``+1`` is the quantum normal-ordering constant of
:mod:`stringsim.quantum.zeta` showing up in a second, independent place.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..units import Conventions
from .partition import oscillator_degeneracies, superstring_degeneracies
from .zeta import normal_ordering_constant

__all__ = [
    "Level",
    "open_bosonic_spectrum",
    "closed_bosonic_spectrum",
    "open_superstring_spectrum",
    "leading_trajectory_spin",
]


@dataclass(frozen=True)
class Level:
    """One mass level of the string.

    Attributes
    ----------
    n:
        Oscillator level ``N``.
    alpha_m2:
        ``alpha' M^2``, the dimensionless mass.
    mass_squared:
        ``M^2`` in the units set by ``alpha'``.
    degeneracy:
        Number of physical states, exact integer.
    max_spin:
        Highest spin present, i.e. the leading Regge trajectory.
    """

    n: int
    alpha_m2: float
    mass_squared: float
    degeneracy: int
    max_spin: int

    @property
    def is_massless(self) -> bool:
        return self.alpha_m2 == 0.0

    @property
    def is_tachyonic(self) -> bool:
        return self.alpha_m2 < 0.0

    def __str__(self) -> str:  # pragma: no cover - display only
        tag = "tachyon" if self.is_tachyonic else ("massless" if self.is_massless else "massive")
        return (
            f"N={self.n:<3d} alpha'M^2={self.alpha_m2:>7.2f}  "
            f"states={self.degeneracy:<14d} J_max={self.max_spin:<3d} {tag}"
        )


def leading_trajectory_spin(alpha_m2: float, intercept: float = 1.0) -> float:
    r"""``J = alpha' M^2 + intercept``.

    The intercept is 1 for the open bosonic string (the normal-ordering
    constant) and 2 for the closed one, which is why the closed string's
    massless level contains a spin-2 state -- the graviton.
    """
    return alpha_m2 + intercept


def open_bosonic_spectrum(n_max: int = 6, conventions: Conventions | None = None) -> list[Level]:
    """Levels ``N = 0 .. n_max`` of the open bosonic string."""
    c = conventions or Conventions()
    a = normal_ordering_constant(c.dim, "bosonic")
    degen = oscillator_degeneracies(n_max, c.transverse_dim)
    out = []
    for n in range(n_max + 1):
        am2 = n - a
        out.append(
            Level(
                n=n,
                alpha_m2=am2,
                mass_squared=am2 / c.alpha_prime,
                degeneracy=degen[n],
                max_spin=n,
            )
        )
    return out


def closed_bosonic_spectrum(n_max: int = 4, conventions: Conventions | None = None) -> list[Level]:
    """Level-matched levels ``N = Ntilde = 0 .. n_max`` of the closed bosonic string."""
    c = conventions or Conventions()
    a = normal_ordering_constant(c.dim, "bosonic")
    degen = oscillator_degeneracies(n_max, c.transverse_dim)
    out = []
    for n in range(n_max + 1):
        am2 = 4.0 * (n - a)
        out.append(
            Level(
                n=n,
                alpha_m2=am2,
                mass_squared=am2 / c.alpha_prime,
                degeneracy=degen[n] ** 2,
                max_spin=2 * n,
            )
        )
    return out


def open_superstring_spectrum(
    n_max: int = 4, conventions: Conventions | None = None
) -> list[Level]:
    """GSO-projected open superstring, ``alpha' M^2 = N``, no tachyon.

    ``conventions.dim`` is ignored beyond a warning-free default; the GSO
    counting is specific to ``D = 10``.
    """
    c = conventions or Conventions(dim=10)
    degen = superstring_degeneracies(n_max)
    return [
        Level(
            n=n,
            alpha_m2=float(n),
            mass_squared=n / c.alpha_prime,
            degeneracy=degen[n],
            max_spin=n + 1,
        )
        for n in range(n_max + 1)
    ]
