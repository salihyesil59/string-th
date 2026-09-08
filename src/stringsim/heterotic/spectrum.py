r"""The heterotic spectrum: level matching between two different theories.

A heterotic string is closed, and its two moving directions are not the same
theory.  Right-movers are the superstring; left-movers are the bosonic string
with sixteen of its directions on the lattice of
:mod:`stringsim.heterotic.lattice`.  Each side has its own mass formula,

.. math::
   \frac{\alpha' M^2}{4} = N_L + \frac{p^2}{2} - 1
   \qquad\text{and}\qquad
   \frac{\alpha' M^2}{4} = N_R - a_R ,

with :math:`p \in \Gamma_{16}` and :math:`a_R = \tfrac12` in NS, ``0`` in R,
and a physical state has to satisfy **both**.

That constraint does more work here than anywhere else in the package.

**It kills the tachyon without needing a projection.**  The lattice is even, so
:math:`p^2` is an even integer and the left-hand side is always an *integer*:
:math:`-1, 0, 1, \dots`.  The right-hand side in the NS sector is
:math:`N_R - \tfrac12`, always a *half-odd-integer or integer* starting at
:math:`-\tfrac12`.  The would-be tachyon at :math:`-1` on the left has nothing
to pair with, and the NS ground state at :math:`-\tfrac12` on the right has
nothing either.  The lightest matched state sits at exactly zero.  GSO is still
required for supersymmetry, but the tachyon is gone before it is applied --
:func:`level_matched_masses` shows this by enumeration.

**And it produces the gauge group.**  At zero mass the left side needs
:math:`N_L + p^2/2 = 1`, which happens two ways: one oscillator and no lattice
momentum (24 states, of which 16 are internal), or no oscillator and a lattice
vector of norm 2 (480 of them, the roots).  Sixteen Cartan directions plus 480
roots is 496 gauge bosons -- the gauge symmetry is not assumed anywhere, it is
what the massless level happens to contain.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..superstring.rns import Sector
from ..superstring.typeii import Field, vector_spinor, vector_vector
from ..units import Conventions
from .lattice import RootLattice

__all__ = [
    "LEFT_DIMENSION",
    "RIGHT_DIMENSION",
    "internal_dimension",
    "central_charges",
    "left_mass",
    "right_mass",
    "MatchedState",
    "level_matched_masses",
    "has_tachyon",
    "HeteroticMassless",
    "massless_content",
]

LEFT_DIMENSION = 26
"""The bosonic side's critical dimension."""

RIGHT_DIMENSION = 10
"""The superstring side's critical dimension."""


def internal_dimension() -> int:
    """``26 - 10 = 16``: the left-moving directions with nowhere to go."""
    return LEFT_DIMENSION - RIGHT_DIMENSION


def central_charges() -> tuple[float, float]:
    r"""``(c_L, c_R) = (26, 15)``.

    Left: 26 bosons.  Right: 10 bosons and 10 Majorana fermions, so
    :math:`10 + 10/2 = 15`, which is what a superconformal ghost system cancels.
    The two sides carry different central charges and that is the whole point of
    the construction -- "heterosis".
    """
    return float(LEFT_DIMENSION), RIGHT_DIMENSION * 1.5


def left_mass(level: int, p_squared: int) -> Fraction:
    r"""``alpha' M^2 / 4`` on the bosonic side: ``N_L + p^2/2 - 1``.

    ``p_squared`` must be even, which is exactly what makes the lattice even and
    what forces this to be an integer.
    """
    if level < 0:
        raise ValueError("level must be non-negative")
    if p_squared < 0 or p_squared % 2:
        raise ValueError("p^2 must be a non-negative even integer on an even lattice")
    return Fraction(level) + Fraction(p_squared, 2) - 1


def right_mass(level: Fraction | int, sector: Sector) -> Fraction:
    r"""``alpha' M^2 / 4`` on the superstring side: ``N_R - a_R``.

    ``N_R`` runs over half-integers in NS and integers in R, and ``a_R`` is
    :func:`stringsim.superstring.rns.intercept`.
    """
    level = Fraction(level)
    if level < 0:
        raise ValueError("level must be non-negative")
    if sector is Sector.NS and level.denominator not in (1, 2):
        raise ValueError("NS levels are multiples of 1/2")
    if sector is Sector.R and level.denominator != 1:
        raise ValueError("R levels are integers")
    a_r = Fraction(1, 2) if sector is Sector.NS else Fraction(0)
    return level - a_r


@dataclass(frozen=True)
class MatchedState:
    """One level-matched combination of a left and a right excitation."""

    alpha_m2: Fraction
    left_level: int
    p_squared: int
    right_level: Fraction
    sector: Sector

    @property
    def is_massless(self) -> bool:
        return self.alpha_m2 == 0

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"alpha'M^2 = {float(self.alpha_m2):>6.2f}   "
            f"N_L={self.left_level}, p^2={self.p_squared}   "
            f"N_R={str(self.right_level):>4s} ({self.sector.value})"
        )


def level_matched_masses(
    n_max: int = 3, p_squared_max: int = 6, gso: bool = True
) -> list[MatchedState]:
    r"""Every level-matched combination up to the given cut-offs, lightest first.

    Set ``gso=False`` to include the NS ground state, which the projection would
    remove.  It makes no difference to the lightest mass: the NS ground state
    sits at :math:`-\tfrac12` and the left side is always an integer, so it has
    nothing to pair with either way.  That is worth seeing rather than being
    told, which is why the switch is here.
    """
    if n_max < 0 or p_squared_max < 0:
        raise ValueError("cut-offs must be non-negative")
    out: list[MatchedState] = []
    for left_level in range(n_max + 1):
        for p_squared in range(0, p_squared_max + 1, 2):
            left = left_mass(left_level, p_squared)
            for sector in (Sector.NS, Sector.R):
                step = Fraction(1, 2) if sector is Sector.NS else Fraction(1)
                for index in range(int(n_max / step) + 1):
                    right_level = step * index
                    # GSO keeps odd worldsheet fermion number in NS, i.e. N_R a
                    # half-odd-integer.  The states it drops are the ones with
                    # integer N_R, whose mass N_R - 1/2 is a half-odd-integer --
                    # and the left side is always an integer, so they never match
                    # anyway.  That is why the switch changes no output, and it is
                    # the point: the tachyon dies from level matching, not GSO.
                    if gso and sector is Sector.NS and right_level.denominator != 2:
                        continue
                    if right_mass(right_level, sector) != left:
                        continue
                    out.append(
                        MatchedState(
                            alpha_m2=4 * left,
                            left_level=left_level,
                            p_squared=p_squared,
                            right_level=right_level,
                            sector=sector,
                        )
                    )
    return sorted(out, key=lambda s: (s.alpha_m2, s.left_level, s.p_squared))


def has_tachyon(n_max: int = 3, p_squared_max: int = 6, gso: bool = True) -> bool:
    """True if any level-matched state has negative mass squared.  It does not.

    The left side runs over integers from ``-1`` and the right side over
    half-integers from ``-1/2``; the only way to match below zero would be for
    both to be negative at the same value, and there is none.
    """
    return any(state.alpha_m2 < 0 for state in level_matched_masses(n_max, p_squared_max, gso))


@dataclass(frozen=True)
class HeteroticMassless:
    """The massless level, split into its supergravity and gauge parts."""

    lattice_name: str
    left_oscillator_states: int
    left_lattice_states: int
    right_states: int
    supergravity: tuple[Field, ...]
    gauge_algebra: str
    gauge_dimension: int

    @property
    def left_states(self) -> int:
        """``24 + n_roots``: 504 for either heterotic lattice."""
        return self.left_oscillator_states + self.left_lattice_states

    @property
    def total(self) -> int:
        """Left times right: 8064."""
        return self.left_states * self.right_states

    @property
    def supergravity_states(self) -> int:
        """``8 x 16 = 128``: the ten-dimensional ``N = 1`` supergravity multiplet."""
        return sum(field.dimension for field in self.supergravity)

    @property
    def gauge_states(self) -> int:
        """``dim G x 16``: a vector multiplet for every generator."""
        return self.gauge_dimension * self.right_states

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.lattice_name}: {self.total} massless states = "
            f"{self.supergravity_states} supergravity + {self.gauge_states} gauge "
            f"({self.gauge_algebra}, dim {self.gauge_dimension})"
        )


def massless_content(
    lattice: RootLattice, conventions: Conventions | None = None
) -> HeteroticMassless:
    r"""The massless level of the heterotic string on a given internal lattice.

    Right-movers contribute the ten-dimensional vector multiplet, ``8_v`` from
    NS and ``8_s`` from R, 16 states.  Left-movers contribute ``N_L = 1`` with
    no lattice momentum (``D - 2 = 24`` oscillators, eight of them spacetime and
    sixteen internal) plus ``N_L = 0`` with ``p^2 = 2`` (the roots).  Tensoring:

    * the eight spacetime oscillators give ``8 x 16 = 128`` states, which is
      exactly ``N = 1`` supergravity -- the same ``8_v x 8_v`` and ``8_v x 8_s``
      products as in :mod:`stringsim.superstring.typeii`;
    * the sixteen internal oscillators and the 480 roots give
      ``496 x 16`` states, a gauge multiplet for every generator.
    """
    c = conventions or Conventions(dim=RIGHT_DIMENSION)
    transverse = c.transverse_dim
    left_oscillators = LEFT_DIMENSION - 2
    right_states = 2 * transverse  # 8v from NS, 8s from R
    return HeteroticMassless(
        lattice_name=lattice.name,
        left_oscillator_states=left_oscillators,
        left_lattice_states=lattice.n_roots,
        right_states=right_states,
        supergravity=vector_vector(transverse) + vector_spinor("het", transverse),
        gauge_algebra=lattice.algebra,
        gauge_dimension=lattice.algebra_dimension,
    )


def anomaly_free_dimension() -> int:
    """``496``, what Green-Schwarz cancellation demands of ``dim G`` in ``D = 10``.

    Computed, not quoted: :func:`stringsim.heterotic.anomaly.required_dimension`
    assembles the ten-dimensional anomaly polynomial out of index densities and
    solves for the gauge dimension that kills ``tr R^6``.

    The lattice produces the same number from modular invariance alone, with no
    reference to anomalies -- ``GAUGE_DIMENSION`` is 16 Cartan directions plus
    480 roots.  Two unrelated consistency conditions agreeing is the reason the
    heterotic construction was taken seriously, and the test suite checks that
    the two routes agree rather than sharing a constant.
    """
    from .anomaly import required_dimension

    return required_dimension()


__all__ += ["anomaly_free_dimension"]
