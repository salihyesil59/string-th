r"""A closed string on a circle: momentum, winding, T-duality.

Compactify one direction, :math:`X^{25} \sim X^{25} + 2\pi R`.  Two things
change at once.  Momentum along the circle is quantised,
:math:`p = n/R`, exactly as for a point particle -- the Kaluza-Klein tower.  But
a *string* can also wrap the circle ``w`` times, and unwinding costs energy
proportional to the length wrapped, :math:`w R / \alpha'`.  A point particle has
no such option; this is the first place where a string is visibly not a
particle.

The closed-string mass is then

.. math::
   M^2 = \frac{n^2}{R^2} + \frac{w^2 R^2}{\alpha'^2}
         + \frac{2}{\alpha'}\left(N + \tilde N - 2\right),
   \qquad N - \tilde N = n w ,

the second relation being level matching, now sourced by momentum and winding.

**T-duality.**  The spectrum is invariant under

.. math::  R \to \frac{\alpha'}{R}, \qquad n \leftrightarrow w ,

because the exchange swaps the first two terms and flips the sign of ``nw``
consistently with :math:`N \leftrightarrow \tilde N`.  A circle of radius ``R``
and one of radius :math:`\alpha'/R` are the *same theory*: there is a shortest
distinguishable radius, the self-dual :math:`R = \sqrt{\alpha'}`, and shrinking
past it gets you nowhere new.  :func:`t_dual_radius` and
:func:`spectrum_is_t_dual` make that concrete.

At exactly the self-dual radius, states with :math:`(n, w) = (\pm1, \pm1)`
become massless and enlarge the gauge symmetry from
:math:`U(1)_L \times U(1)_R` to :math:`SU(2)_L \times SU(2)_R`.
:func:`extra_massless_states` finds them by enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..quantum.partition import oscillator_degeneracies
from ..units import Conventions

__all__ = [
    "CircleState",
    "self_dual_radius",
    "t_dual_radius",
    "spectrum",
    "massless_states",
    "extra_massless_states",
    "spectrum_is_t_dual",
    "lightest_mass",
]


@dataclass(frozen=True)
class CircleState:
    """One level-matched state of the closed string on a circle."""

    n: int
    w: int
    level: int
    level_tilde: int
    alpha_m2: float
    degeneracy: int

    @property
    def is_massless(self) -> bool:
        return abs(self.alpha_m2) < 1e-9

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"(n={self.n:+d}, w={self.w:+d}, N={self.level}, Nt={self.level_tilde}) "
            f"alpha'M^2 = {self.alpha_m2:+.4f}  x{self.degeneracy}"
        )


def self_dual_radius(conventions: Conventions | None = None) -> float:
    r"""``R = sqrt(alpha')``, the fixed point of T-duality."""
    return (conventions or Conventions()).string_length


def t_dual_radius(radius: float, conventions: Conventions | None = None) -> float:
    r"""``alpha' / R``."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    return (conventions or Conventions()).alpha_prime / radius


def _alpha_m2(n: int, w: int, level: int, level_tilde: int, radius: float, ap: float) -> float:
    return ap * (n / radius) ** 2 + (w * radius) ** 2 / ap + 2.0 * (level + level_tilde - 2)


def spectrum(
    radius: float,
    conventions: Conventions | None = None,
    n_max: int = 2,
    w_max: int = 2,
    level_max: int = 2,
) -> list[CircleState]:
    """Enumerate level-matched states, sorted by mass.

    Parameters
    ----------
    radius:
        Circle radius ``R`` in the same length units as ``sqrt(alpha')``.
    n_max, w_max:
        Ranges of momentum and winding number, inclusive of both signs.
    level_max:
        Highest oscillator level on either side.

    Notes
    -----
    Degeneracies use all ``D - 2 = 24`` transverse oscillators, including the
    one along the compact direction.  In the ``D - 1`` dimensional effective
    theory those split further into vectors and scalars; this function reports
    the total.
    """
    c = conventions or Conventions()
    if radius <= 0:
        raise ValueError("radius must be positive")
    degen = oscillator_degeneracies(level_max, c.transverse_dim)
    out: list[CircleState] = []
    for n in range(-n_max, n_max + 1):
        for w in range(-w_max, w_max + 1):
            for lvl in range(level_max + 1):
                lvl_t = lvl - n * w
                if not 0 <= lvl_t <= level_max:
                    continue
                out.append(
                    CircleState(
                        n=n,
                        w=w,
                        level=lvl,
                        level_tilde=lvl_t,
                        alpha_m2=_alpha_m2(n, w, lvl, lvl_t, radius, c.alpha_prime),
                        degeneracy=degen[lvl] * degen[lvl_t],
                    )
                )
    return sorted(out, key=lambda s: (s.alpha_m2, s.n, s.w))


def massless_states(
    radius: float, conventions: Conventions | None = None, **kw
) -> list[CircleState]:
    """Every state with ``alpha' M^2 = 0`` at this radius."""
    return [s for s in spectrum(radius, conventions, **kw) if s.is_massless]


def extra_massless_states(
    radius: float, conventions: Conventions | None = None, **kw
) -> list[CircleState]:
    r"""Massless states carrying momentum or winding, i.e. beyond the generic ones.

    At a generic radius the only massless states are the ``(n, w) = (0, 0)``,
    ``N = \tilde N = 1`` ones: the graviton, the ``B`` field, the dilaton, and
    the two ``U(1)`` gauge bosons from the metric and ``B`` field with one leg
    on the circle.  This function returns what is *left over*, which is empty
    except at special radii.

    At :math:`R = \sqrt{\alpha'}` it returns eight states: the four with
    :math:`(n,w) = (\pm1,\pm1)` and one oscillator, which supply the extra
    :math:`SU(2)_L \times SU(2)_R` gauge bosons, and the four with
    :math:`(n,w) = (\pm2, 0), (0, \pm2)` and no oscillators, which are the
    bosonic string's tachyon tower passing through zero mass.
    """
    return [s for s in massless_states(radius, conventions, **kw) if (s.n, s.w) != (0, 0)]


def spectrum_is_t_dual(
    radius: float, conventions: Conventions | None = None, tol: float = 1e-10, **kw
) -> bool:
    """True when the spectra at ``R`` and ``alpha'/R`` agree as multisets of masses."""
    c = conventions or Conventions()
    a = sorted(round(s.alpha_m2, 10) for s in spectrum(radius, c, **kw))
    b = sorted(round(s.alpha_m2, 10) for s in spectrum(t_dual_radius(radius, c), c, **kw))
    return len(a) == len(b) and all(abs(x - y) < tol for x, y in zip(a, b, strict=True))


def lightest_mass(radius: float, conventions: Conventions | None = None, **kw) -> CircleState:
    """The lightest *non-tachyonic, non-massless* state -- the first excitation.

    At large ``R`` this is a Kaluza-Klein mode (mass ``~ 1/R``); at small ``R``
    it is a winding mode (mass ``~ R/alpha'``).  Watching which one wins as
    ``R`` sweeps through :math:`\\sqrt{\\alpha'}` is T-duality in one plot.
    """
    states = [s for s in spectrum(radius, conventions, **kw) if s.alpha_m2 > 1e-9]
    if not states:
        raise ValueError("no massive states in the enumerated range")
    return min(states, key=lambda s: s.alpha_m2)


def kaluza_klein_mass(n: int, radius: float) -> float:
    """``|n| / R``: the Kaluza-Klein tower, identical to a point particle's."""
    return abs(n) / radius


def winding_mass(w: int, radius: float, conventions: Conventions | None = None) -> float:
    r"""``|w| R / alpha'``: the energy of a string wrapped ``w`` times.

    Has no point-particle counterpart.  It is what makes the small-radius limit
    of string theory look like the large-radius limit of another one.
    """
    c = conventions or Conventions()
    return abs(w) * radius / c.alpha_prime


__all__ += ["kaluza_klein_mass", "winding_mass"]
