r"""The RNS superstring: worldsheet fermions, the two sectors, and GSO.

Adding a worldsheet fermion :math:`\psi^\mu` to each boson :math:`X^\mu` changes
three things at once, and all three are computed here rather than quoted.

**Two sectors, because a fermion may be periodic or not.**  Nothing forces
:math:`\psi^\mu` to come back to itself around the string; only bilinears must.
So there are two boundary conditions:

===========  =====================  ============================
sector       mode numbers           ground state
===========  =====================  ============================
Neveu-Schwarz  :math:`r \in \mathbb{Z}+\tfrac12`   a spacetime **boson**
Ramond         :math:`n \in \mathbb{Z}` (zero modes!)  a spacetime **fermion**
===========  =====================  ============================

The Ramond zero modes :math:`\psi_0^\mu` satisfy a Clifford algebra, so the R
ground state is not a single state but a spinor -- which is where spacetime
fermions come from at all.

**The ground-state energies follow.**  A fermion's zero-point energy is
:math:`-\tfrac12\sum\omega` where a boson's is :math:`+\tfrac12\sum\omega`, and
:func:`stringsim.quantum.zeta.regularised_shifted_sum` supplies both sums:

.. math::
   a_{NS} = \frac{D-2}{16} = \tfrac12, \qquad a_R = 0 \quad (D = 10),

so :math:`\alpha' M^2 = N - \tfrac12` in NS and :math:`\alpha' M^2 = N` in R.
**The Ramond ground state is massless with no help from anything**, and the NS
ground state is a tachyon -- until GSO.

**The GSO projection.**  Keeping only odd worldsheet fermion number in NS
deletes that tachyon and promotes :math:`b_{-1/2}^i|0\rangle` (eight states, a
massless vector) to the bottom of the spectrum.  Keeping one chirality in R
leaves eight states there too.  Eight and eight, at every level after that as
well: :func:`supersymmetry_deficit` checks it level by level, and finding zero
is the concrete form of the Jacobi identity that
:func:`stringsim.quantum.partition.jacobi_identity_residual` proves abstractly.

The counting here is done by explicit state enumeration -- multiply out the
oscillator products, track fermion number, throw away the wrong parity -- which
is a completely different route from the theta-function product used in
:func:`stringsim.quantum.partition.superstring_degeneracies`.  The two are
required to agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from ..quantum.zeta import regularised_shifted_sum
from ..units import Conventions

__all__ = [
    "Sector",
    "fermion_mode_numbers",
    "zero_point_energy",
    "intercept",
    "critical_dimension_from_intercept",
    "ns_series_by_parity",
    "sector_degeneracies",
    "unprojected_ns_ground_state",
    "supersymmetry_deficit",
    "SuperLevel",
    "open_superstring_levels",
]


class Sector(str, Enum):
    """Which boundary condition the worldsheet fermions obey."""

    NS = "NS"
    R = "R"

    @property
    def is_periodic(self) -> bool:
        """True for Ramond, where the fermions come back to themselves."""
        return self is Sector.R


def fermion_mode_numbers(sector: Sector, count: int = 5) -> list[Fraction]:
    r"""The first ``count`` non-negative fermion mode numbers.

    ``NS`` gives ``1/2, 3/2, 5/2, ...`` and ``R`` gives ``0, 1, 2, ...``.  The
    zero in the Ramond list is the whole story of spacetime fermions: those
    modes have no energy cost, they satisfy
    :math:`\{\psi_0^\mu, \psi_0^\nu\} = \eta^{\mu\nu}`, and representing that
    Clifford algebra forces the ground state to be a spinor.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if sector is Sector.R:
        return [Fraction(n) for n in range(count)]
    return [Fraction(2 * n + 1, 2) for n in range(count)]


def zero_point_energy(sector: Sector, conventions: Conventions | None = None) -> float:
    r"""``E_0`` for the transverse oscillators, bosons and fermions together.

    A boson contributes :math:`+\tfrac12\zeta(-1,\phi)` and a fermion
    :math:`-\tfrac12\zeta(-1,\phi)`, with :math:`\phi = 1` for integer modes and
    :math:`\phi = \tfrac12` for half-integer ones.  Evaluated with
    :func:`~stringsim.quantum.zeta.regularised_shifted_sum`, i.e. from a cut-off
    sum rather than a closed form, so the answer is measured.

    In ``D = 10`` this returns ``-1/2`` for NS and ``0`` for R.
    """
    c = conventions or Conventions(dim=10)
    transverse = c.transverse_dim
    boson = 0.5 * regularised_shifted_sum(1.0).value
    fermion_shift = 1.0 if sector is Sector.R else 0.5
    fermion = 0.5 * regularised_shifted_sum(fermion_shift).value
    return transverse * boson - transverse * fermion


def intercept(sector: Sector, conventions: Conventions | None = None, exact: bool = True) -> float:
    r"""``a = -E_0``, so that ``alpha' M^2 = N - a``.

    ``exact=True`` returns the closed forms :math:`a_{NS} = (D-2)/16` and
    :math:`a_R = 0`; ``exact=False`` returns the value measured by
    :func:`zero_point_energy`.  The test suite requires them to agree, which is
    what makes the closed form a result rather than an assumption.
    """
    c = conventions or Conventions(dim=10)
    if not exact:
        return -zero_point_energy(sector, c)
    return 0.0 if sector is Sector.R else (c.dim - 2) / 16.0


def critical_dimension_from_intercept() -> int:
    r"""Solve ``a_NS = 1/2`` for ``D``, giving 10.

    The same Lorentz-invariance argument as in the bosonic string: the level-1/2
    NS states form a vector of ``SO(D-2)``, which can only be massless, so
    ``a_NS`` must be exactly ``1/2``.
    """
    for dim in range(3, 200):
        if abs(intercept(Sector.NS, Conventions(dim=dim)) - 0.5) < 1e-12:
            return dim
    raise RuntimeError("no critical dimension found")  # pragma: no cover


# ---------------------------------------------------------------------------
# state counting
# ---------------------------------------------------------------------------


def ns_series_by_parity(n_max: int, transverse: int = 8) -> tuple[list[int], list[int]]:
    r"""NS oscillator counts split by worldsheet fermion parity.

    Works in :math:`u = q^{1/2}` so that half-integer fermion modes have integer
    exponents.  Returns ``(even, odd)``, each indexed by the power of ``u``,
    i.e. by twice the level ``N``.

    * bosons contribute :math:`\prod_{n\geq1}(1-u^{2n})^{-c}` and never change
      the parity;
    * fermions contribute :math:`\prod_{k \text{ odd}}(1 + y\,u^{k})^{c}`, and
      each factor moves weight from one parity to the other.

    Tracking only the parity, rather than the full fermion number, is enough:
    GSO cares about :math:`(-1)^F` and nothing else.
    """
    if n_max < 0:
        raise ValueError("n_max must be non-negative")
    if transverse < 1:
        raise ValueError("transverse must be positive")
    even = [0] * (n_max + 1)
    odd = [0] * (n_max + 1)
    even[0] = 1

    # bosons: 1/(1 - u^{2n}) per transverse direction
    for n in range(1, n_max // 2 + 1):
        step = 2 * n
        for _ in range(transverse):
            for level in range(step, n_max + 1):
                even[level] += even[level - step]
                odd[level] += odd[level - step]

    # fermions: (1 + y u^k) for odd k, one factor per transverse direction
    for k in range(1, n_max + 1, 2):
        for _ in range(transverse):
            for level in range(n_max, k - 1, -1):
                even[level], odd[level] = (
                    even[level] + odd[level - k],
                    odd[level] + even[level - k],
                )
    return even, odd


def _ramond_series(n_max: int, transverse: int = 8) -> list[int]:
    r"""Coefficients of :math:`\prod_{n\geq1}\left[(1+q^n)/(1-q^n)\right]^{c}`.

    The Ramond oscillators: ``c`` bosons with integer modes and ``c`` fermions
    with integer modes.  The ``2^{c/2} = 16`` ground states from the zero modes,
    halved to 8 by the chirality projection, multiply this.
    """
    coefficients = [0] * (n_max + 1)
    coefficients[0] = 1
    for n in range(1, n_max + 1):
        for _ in range(transverse):
            for level in range(n, n_max + 1):  # 1/(1 - q^n)
                coefficients[level] += coefficients[level - n]
        for _ in range(transverse):
            for level in range(n_max, n - 1, -1):  # (1 + q^n)
                coefficients[level] += coefficients[level - n]
    return coefficients


def sector_degeneracies(
    sector: Sector, n_max: int = 6, transverse: int = 8, gso: bool = True
) -> list[int]:
    r"""States at :math:`\alpha' M^2 = 0, 1, \dots, n_{max}`, GSO projected.

    **NS.**  Keeping odd fermion number leaves levels at integer
    :math:`\alpha' M^2`: an odd number of half-odd-integer modes sums to a
    half-odd-integer, so :math:`N \in \mathbb{Z}+\tfrac12` and
    :math:`\alpha' M^2 = N - \tfrac12` is an integer.  With ``gso=False`` the
    projection is skipped and the tachyon at :math:`\alpha' M^2 = -1/2` comes
    back, so the returned list then starts half a unit lower -- see
    :func:`unprojected_ns_ground_state`.

    **R.**  The ``2^4 = 16`` zero-mode ground states are halved to 8 by the
    chirality projection, then dressed with oscillators.

    Returns ``8, 128, 1152, 7680, ...`` in both sectors, which is spacetime
    supersymmetry made arithmetic.
    """
    if n_max < 0:
        raise ValueError("n_max must be non-negative")
    if sector is Sector.R:
        ground = 2 ** (transverse // 2)
        if gso:
            ground //= 2
        return [ground * count for count in _ramond_series(n_max, transverse)]
    even, odd = ns_series_by_parity(2 * n_max + 2, transverse)
    if not gso:
        raise ValueError(
            "without GSO the NS levels are spaced by 1/2 and include a tachyon; "
            "use ns_series_by_parity or unprojected_ns_ground_state instead"
        )
    return [odd[2 * level + 1] for level in range(n_max + 1)]


def unprojected_ns_ground_state(conventions: Conventions | None = None) -> float:
    r"""``alpha' M^2`` of the NS ground state before GSO: ``-1/2``.

    It is a tachyon, and it is what the projection is for.  Compare the bosonic
    string, whose tachyon sits at ``-1`` and cannot be projected away.
    """
    return -intercept(Sector.NS, conventions)


def supersymmetry_deficit(n_max: int = 6, transverse: int = 8) -> list[int]:
    r"""``(NS count) - (R count)`` at each mass level; every entry is zero.

    Bosons and fermions in equal numbers at every mass, computed from two
    unrelated products.  This is the statement behind
    :math:`\theta_3^4 - \theta_2^4 - \theta_4^4 = 0`, arrived at by counting
    states instead of manipulating theta functions.
    """
    bosons = sector_degeneracies(Sector.NS, n_max, transverse)
    fermions = sector_degeneracies(Sector.R, n_max, transverse)
    return [b - f for b, f in zip(bosons, fermions, strict=True)]


@dataclass(frozen=True)
class SuperLevel:
    """One mass level of the open superstring, both sectors together."""

    n: int
    alpha_m2: float
    bosons: int
    fermions: int

    @property
    def degeneracy(self) -> int:
        return self.bosons + self.fermions

    @property
    def is_supersymmetric(self) -> bool:
        return self.bosons == self.fermions

    def __str__(self) -> str:  # pragma: no cover - display only
        tag = "massless" if self.alpha_m2 == 0 else "massive"
        return (
            f"alpha'M^2={self.alpha_m2:>5.1f}  "
            f"{self.bosons:>8d} bosons + {self.fermions:>8d} fermions  {tag}"
        )


def open_superstring_levels(
    n_max: int = 4, conventions: Conventions | None = None
) -> list[SuperLevel]:
    """The GSO-projected open superstring, bosons and fermions side by side.

    The massless level is ``8 + 8``: a gauge boson and a gaugino, i.e.
    ``N = 1`` super Yang-Mills in ten dimensions.
    """
    c = conventions or Conventions(dim=10)
    transverse = c.transverse_dim
    bosons = sector_degeneracies(Sector.NS, n_max, transverse)
    fermions = sector_degeneracies(Sector.R, n_max, transverse)
    return [
        SuperLevel(n=n, alpha_m2=float(n), bosons=bosons[n], fermions=fermions[n])
        for n in range(n_max + 1)
    ]


def hagedorn_species(transverse: int = 8) -> float:
    r"""Effective central charge of the superstring oscillators, ``c + c_f/2``.

    Eight bosons and eight fermions grow like ``12`` bosons would, so
    :math:`d_N \sim \exp(2\pi\sqrt{2N})` and
    :math:`\beta_H = 2\pi\sqrt{2\alpha'}` -- a *higher* Hagedorn temperature
    than the bosonic string's, because fermionic modes count half.
    """
    return transverse + transverse / 2.0


def hagedorn_beta(alpha_prime: float = 1.0, transverse: int = 8) -> float:
    r"""``beta_H = 2 pi sqrt(c_eff alpha' / 6)`` for the superstring."""
    return 2.0 * math.pi * math.sqrt(hagedorn_species(transverse) * alpha_prime / 6.0)


__all__ += ["hagedorn_species", "hagedorn_beta"]
