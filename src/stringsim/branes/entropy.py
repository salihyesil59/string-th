r"""Strominger-Vafa: a black hole's entropy, counted and then measured.

The D1-D5-P system of type IIB on :math:`T^4 \times S^1` -- :math:`Q_1`
D1-branes on the circle, :math:`Q_5` D5-branes on the whole thing, and ``N``
units of momentum along the circle -- is a black hole in the five
non-compact directions.  Its entropy can be got at two ways that share no step.

**Counting.**  The bound state is a two-dimensional conformal field theory with
:math:`4Q_1Q_5` bosons and as many fermions, so
:math:`c = 4Q_1Q_5 + \tfrac12\cdot4Q_1Q_5 = 6Q_1Q_5`.  The momentum sits in the
left-movers at level ``N``, and the number of ways of putting it there is the
coefficient of :math:`q^N` in

.. math::
   \prod_{n\geq1}\Bigl(\frac{1+q^n}{1-q^n}\Bigr)^{4Q_1Q_5} ,

computed here in exact integers.  Cardy says
:math:`\log d_N \to 2\pi\sqrt{cN/6} = 2\pi\sqrt{Q_1Q_5N}`, and
:func:`fit_cardy` *measures* that exponent from the integers rather than
quoting it.  It converges slowly, for the same reason the Hagedorn fit in
:mod:`stringsim.quantum.partition` does: the subleading :math:`\log N` term is
not a small correction at reachable levels, and leaving it out of the fit
biases the answer by two per cent instead of two parts in ten thousand.

**Measuring.**  The same charges make an extremal black hole whose horizon is a
three-sphere.  Its area over :math:`4G_5` is the Bekenstein-Hawking entropy,
and it comes out :math:`2\pi\sqrt{Q_1Q_5N}` as well.

**What is derived and what is fixed by the match.**  The entropy of a black hole
counts states, so it must be a pure number: every continuous modulus -- the
string coupling, the volume of the :math:`T^4`, the radius of the circle,
:math:`\alpha'` -- has to drop out of :math:`A/4G_5`.  It does, exactly, and
that is a real check: it needs the powers in the three harmonic radii and in
:math:`G_5` to be mutually consistent, and any one of them wrong breaks it.

What the moduli cancellation does *not* fix is the overall constant, and that
turns on a convention: whether the :math:`T^4` enters the harmonic functions
through its dimensionful volume ``V`` or the dimensionless
:math:`v = V/(2\pi)^4\alpha'^2`.  The three choices give

===========================  ==========================
convention                   :math:`S_{\rm Cardy}/S_{\rm BH}`
===========================  ==========================
``V`` throughout             :math:`16\pi^4`
``v`` in :math:`r_1` only    :math:`4\pi^2`
``v`` in :math:`r_1, r_p``   :math:`1`
===========================  ==========================

and only the last makes the two calculations agree.  That is how the convention
is chosen here -- the same way the Hirzebruch class is fixed in
:mod:`stringsim.heterotic.anomaly`, by an over-determined match rather than by
memory.  The match has seven parameters in it (three charges, four moduli) and
one number to get right, so it is evidence rather than a fit.

Reference: A. Strominger and C. Vafa, *Microscopic origin of the
Bekenstein-Hawking entropy*, Phys. Lett. B **379** (1996) 99.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "species",
    "central_charge",
    "bps_degeneracies",
    "cardy_entropy",
    "microscopic_entropy",
    "CardyFit",
    "fit_cardy",
    "Moduli",
    "harmonic_radii",
    "horizon_area",
    "newton_five",
    "bekenstein_hawking",
    "normalisation",
    "moduli_spread",
    "VOLUME_CONVENTIONS",
]


def species(q1: int, q5: int) -> int:
    """``4 Q_1 Q_5`` -- the number of bosons, and of fermions, on the effective string."""
    if q1 < 1 or q5 < 1:
        raise ValueError("both charges must be positive")
    return 4 * q1 * q5


def central_charge(q1: int, q5: int) -> int:
    r"""``c = 6 Q_1 Q_5``: one for each boson, a half for each fermion."""
    return species(q1, q5) + species(q1, q5) // 2


def bps_degeneracies(q1: int, q5: int, n_max: int) -> list[int]:
    r"""Coefficients of :math:`\prod_n [(1+q^n)/(1-q^n)]^{4Q_1Q_5}`, exact integers.

    The numerator counts the fermions -- each mode occupied at most once -- and
    the denominator the bosons.  ``d_N`` is the number of ways of carrying ``N``
    units of momentum on the effective string.

    Exact arithmetic because the numbers are enormous: ``d_400`` for
    :math:`Q_1 = Q_5 = 1` has over a hundred digits, and the whole point is the
    logarithm of it.

    The cost is ``O(4 Q_1 Q_5 n_max^2)`` in big integers, so large charges want
    a small ``n_max`` and the other way round.
    """
    if n_max < 0:
        raise ValueError("n_max must be non-negative")
    out = [0] * (n_max + 1)
    out[0] = 1
    for _ in range(species(q1, q5)):
        for n in range(1, n_max + 1):  # (1 + q^n)
            for k in range(n_max, n - 1, -1):
                out[k] += out[k - n]
        for n in range(1, n_max + 1):  # 1 / (1 - q^n)
            for k in range(n, n_max + 1):
                out[k] += out[k - n]
    return out


def cardy_entropy(q1: int, q5: int, n: int) -> float:
    r""":math:`2\pi\sqrt{cN/6} = 2\pi\sqrt{Q_1Q_5N}`, from the central charge."""
    if n < 0:
        raise ValueError("the momentum must be non-negative")
    return 2.0 * math.pi * math.sqrt(central_charge(q1, q5) * n / 6.0)


def microscopic_entropy(q1: int, q5: int, n: int, degeneracies=None) -> float:
    """``log d_N``, the honest count at finite ``N``.

    Well below :func:`cardy_entropy` at any level that can be reached -- see
    :func:`fit_cardy`, which is why the comparison has to be a fit rather than a
    ratio.
    """
    table = bps_degeneracies(q1, q5, n) if degeneracies is None else degeneracies
    if table[n] <= 0:
        raise ValueError(f"no states at level {n}")
    return math.log(table[n])


@dataclass(frozen=True)
class CardyFit:
    r"""A fit of :math:`\log d_N = a\sqrt N + b\log N + c`."""

    slope: float
    expected: float
    subleading: float
    subleading_expected: float
    residual: float
    levels: np.ndarray

    @property
    def error(self) -> float:
        """Relative error of the measured Cardy exponent."""
        return abs(self.slope / self.expected - 1.0)

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"a = {self.slope:.6f} against {self.expected:.6f} "
            f"({self.error:.1e}), b = {self.subleading:+.4f} "
            f"heading for {self.subleading_expected:+.2f}"
        )


def fit_cardy(
    q1: int, q5: int, n_max: int = 400, n_fit: int | None = None, with_log: bool = True
) -> CardyFit:
    r"""Measure the Cardy exponent from the exact degeneracies.

    Fits :math:`\log d_N = a\sqrt N + b\log N + c` over the top half of the
    levels.  ``a`` should be :math:`2\pi\sqrt{Q_1Q_5}`.

    ``with_log = False`` drops the middle term, and that is worth doing once:
    it moves the answer from two parts in ten thousand to two per cent.  The
    subleading term is not a correction at these levels, which is the same
    lesson :func:`stringsim.quantum.partition.fit_hagedorn` learned.

    ``b`` tends to :math:`-(3 + k)/4` with :math:`k = 4Q_1Q_5` the number of
    species, and it gets there slowly -- ``-1.69`` at ``n_max = 200`` against
    ``-1.75``, ``-1.73`` at 1600.  The fit reports where it is, not where it is
    going.
    """
    table = bps_degeneracies(q1, q5, n_max)
    lowest = n_max // 2 if n_fit is None else max(n_max - n_fit + 1, 1)
    levels = np.arange(lowest, n_max + 1)
    values = np.array([math.log(table[int(n)]) for n in levels])
    columns = [np.sqrt(levels)]
    if with_log:
        columns.append(np.log(levels))
    columns.append(np.ones_like(levels, dtype=float))
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = float(np.sqrt(np.mean((design @ coefficients - values) ** 2)))
    return CardyFit(
        slope=float(coefficients[0]),
        expected=2.0 * math.pi * math.sqrt(q1 * q5),
        subleading=float(coefficients[1]) if with_log else float("nan"),
        subleading_expected=-(3.0 + species(q1, q5)) / 4.0,
        residual=residual,
        levels=levels,
    )


# ---------------------------------------------------------------------------
# the horizon
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Moduli:
    """Everything the entropy must turn out not to depend on."""

    coupling: float = 0.3
    volume: float = 2.0
    radius: float = 1.5
    alpha_prime: float = 1.0

    def __post_init__(self) -> None:
        for name in ("coupling", "volume", "radius", "alpha_prime"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def reduced_volume(self) -> float:
        r"""``v = V / (2 pi)^4 alpha'^2``, the dimensionless ``T^4`` volume."""
        return self.volume / ((2.0 * math.pi) ** 4 * self.alpha_prime**2)


#: Which radii use the dimensionless volume.  Only ``"reduced"`` makes the two
#: entropies agree; the others are kept so the choice is visible rather than
#: buried, and :func:`normalisation` reports what each gives.
VOLUME_CONVENTIONS = ("reduced", "mixed", "dimensionful")


def harmonic_radii(
    q1: int, q5: int, n: int, moduli: Moduli | None = None, convention: str = "reduced"
) -> tuple[float, float, float]:
    r"""The three squared radii in :math:`H_i = 1 + r_i^2/r^2`.

    With ``convention = "reduced"``:

    .. math::
       r_1^2 = \frac{g_s \alpha' Q_1}{v}, \quad
       r_5^2 = g_s \alpha' Q_5, \quad
       r_p^2 = \frac{g_s^2 \alpha'^2 N}{v R^2} .

    The other two entries of :data:`VOLUME_CONVENTIONS` put the dimensionful
    ``V`` in one or both of the first and third, which changes the answer by
    :math:`4\pi^2` or :math:`16\pi^4` and nothing else.
    """
    if convention not in VOLUME_CONVENTIONS:
        raise ValueError(f"convention must be one of {VOLUME_CONVENTIONS}")
    if n < 0:
        raise ValueError("the momentum must be non-negative")
    mod = moduli or Moduli()
    g, ap, r, big = mod.coupling, mod.alpha_prime, mod.radius, mod.volume
    small = mod.reduced_volume
    first = small if convention in ("reduced", "mixed") else big / ap**2
    third = small if convention == "reduced" else big / ap**2
    r1 = g * ap * q1 / first
    r5 = g * ap * q5
    rp = g**2 * ap**2 * n / (third * r**2)
    return r1, r5, rp


def horizon_area(
    q1: int, q5: int, n: int, moduli: Moduli | None = None, convention: str = "reduced"
) -> float:
    r"""``2 pi^2 r_1 r_5 r_p``, the area of the horizon three-sphere.

    Near :math:`r = 0` the angular part of the extremal metric becomes
    :math:`(r_1 r_5 r_p)^{2/3} d\Omega_3^2`, a sphere of radius
    :math:`(r_1r_5r_p)^{1/3}`, and a three-sphere of radius :math:`\rho` has
    area :math:`2\pi^2\rho^3`.
    """
    r1, r5, rp = harmonic_radii(q1, q5, n, moduli, convention)
    return 2.0 * math.pi**2 * math.sqrt(r1 * r5 * rp)


def newton_five(moduli: Moduli | None = None) -> float:
    r"""``G_5 = G_10 / (2 pi R V)`` with ``G_10 = 8 pi^6 g_s^2 alpha'^4``.

    The ten-dimensional constant is :math:`2\kappa_{10}^2 = (2\pi)^7 g_s^2
    \alpha'^4` divided by :math:`16\pi`, and reducing on
    :math:`T^4 \times S^1` divides by the compact volume.
    """
    mod = moduli or Moduli()
    ten = 8.0 * math.pi**6 * mod.coupling**2 * mod.alpha_prime**4
    return ten / (2.0 * math.pi * mod.radius * mod.volume)


def bekenstein_hawking(
    q1: int, q5: int, n: int, moduli: Moduli | None = None, convention: str = "reduced"
) -> float:
    """``A / 4 G_5`` -- the entropy read off the horizon."""
    return horizon_area(q1, q5, n, moduli, convention) / (4.0 * newton_five(moduli))


def normalisation(
    q1: int, q5: int, n: int, moduli: Moduli | None = None, convention: str = "reduced"
) -> float:
    """``S_Cardy / S_BH``.  One for the right convention, and a constant for all.

    The two entropies are computed from nothing in common -- one from a
    generating function, the other from a metric -- so this ratio being a pure
    number rather than a function of the charges is the content of the
    agreement, and its being *one* is what fixes the convention.
    """
    return cardy_entropy(q1, q5, n) / bekenstein_hawking(q1, q5, n, moduli, convention)


def moduli_spread(
    charges=((1, 1, 10), (3, 7, 200), (12, 5, 41)),
    moduli_list=None,
    convention: str = "reduced",
) -> float:
    """Relative spread of :func:`normalisation` over charges and moduli.

    Zero.  Seven parameters vary -- three charges and four moduli -- and the
    ratio does not move, which is the check that the two calculations are of the
    same thing.
    """
    if moduli_list is None:
        moduli_list = [
            Moduli(),
            Moduli(coupling=0.05, volume=6.9, radius=0.6, alpha_prime=2.3),
            Moduli(coupling=0.62, volume=5.3, radius=3.9, alpha_prime=1.5),
        ]
    values = [
        normalisation(q1, q5, n, mod, convention)
        for q1, q5, n in charges
        for mod in moduli_list
    ]
    middle = float(np.mean(values))
    return float(np.max(np.abs(np.array(values) - middle)) / abs(middle))
