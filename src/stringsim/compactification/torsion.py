r"""Discrete torsion: the phases you are free to put in front of the blocks.

An orbifold partition function is a sum over pairs of commuting group elements,
one for each boundary condition around the two cycles of the torus:

.. math::  Z = \frac{1}{|G|} \sum_{gh = hg} \varepsilon(g, h)\, Z[g, h] .

With :math:`\varepsilon \equiv 1` this is the usual projection.  The question is
what *else* is allowed, and the answer is fixed by how the modular group moves
the blocks around:

.. math::  T:\ Z[g,h] \to Z[g, gh], \qquad S:\ Z[g,h] \to Z[h, g^{-1}] .

Requiring the weights to survive both, together with the factorisation
:math:`\varepsilon(g_1 g_2, h) = \varepsilon(g_1,h)\varepsilon(g_2,h)` that
sewing two blocks demands, leaves exactly the **alternating bilinear pairings**
:math:`G \times G \to U(1)`:

.. math::
   \varepsilon(g_1g_2, h) = \varepsilon(g_1,h)\,\varepsilon(g_2,h), \qquad
   \varepsilon(g, g) = 1 ,

from which :math:`\varepsilon(g,h) = \varepsilon(h,g)^{-1}` follows.
:meth:`TorsionGroup.pairings` finds them by enumerating candidate phase
assignments on the generators and *testing* the axioms on every element, so the
answer is derived and not quoted, and :meth:`TorsionGroup.is_modular_consistent`
then checks the two transformations above on the whole grid of commuting pairs.

What comes out is

.. math::  H^2(G, U(1)) \;=\; \bigoplus_{i<j} \mathbb{Z}_{\gcd(N_i, N_j)} ,

so a **cyclic** orbifold has no discrete torsion at all -- there is nothing to
pair a generator with -- and the smallest group that has any is
:math:`\mathbb{Z}_2 \times \mathbb{Z}_2`, with two classes.

**What the phase does.**  In the ``g``-twisted sector the projector picks up the
weights, :math:`P_g = \frac{1}{|G|}\sum_h \varepsilon(g,h)\, h`.  The untwisted
sector is untouched, since :math:`\varepsilon(1,h) = 1` for every ``h``: torsion
can never change the untwisted massless spectrum.  In a twisted sector it flips
signs, and for :math:`\mathbb{Z}_2 \times \mathbb{Z}_2` the flip is total ---
:math:`P_{g_1} = \frac14(1 + g_1 - g_2 - g_1g_2)` keeps precisely the states the
untorsioned projector threw away.  :func:`projected_degeneracies` computes both
lists, and their sum is the unprojected count, which is what makes "the other
half" a statement and not a slogan.

**Scope.**  The character used here is the one of the twisted *oscillators*,
built from the mode shifts exactly as
:mod:`stringsim.compactification.orbifold` builds its own.  The fixed-point
multiplicity of a twisted sector, and the phase the group acts with on each
fixed point, are not included, so these are not Hodge numbers: the classic
statement that :math:`T^6/(\mathbb{Z}_2 \times \mathbb{Z}_2)` has
:math:`(h^{1,1}, h^{2,1}) = (51, 3)` without torsion and :math:`(3, 51)` with it
needs that extra data and is not computed here.  What is computed is the
mechanism underneath it.

Reference: Vafa, Nucl. Phys. B **273** (1986) 592.
"""

from __future__ import annotations

import cmath
import itertools
import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "TorsionGroup",
    "twisted_character",
    "projected_degeneracies",
]


@dataclass(frozen=True)
class TorsionGroup:
    """A finite abelian group ``Z_{N_1} x ... x Z_{N_k}``, written by its orders.

    Elements are tuples of integers, each reduced modulo its own order.
    """

    orders: tuple[int, ...]

    def __post_init__(self) -> None:
        orders = tuple(int(n) for n in self.orders)
        if not orders or any(n < 1 for n in orders):
            raise ValueError(f"orders must be positive integers, got {self.orders}")
        object.__setattr__(self, "orders", orders)

    @property
    def size(self) -> int:
        """``|G|``."""
        return int(np.prod(self.orders))

    @property
    def exponent(self) -> int:
        """The least common multiple of the orders."""
        return math.lcm(*self.orders)

    def elements(self) -> list[tuple[int, ...]]:
        """Every element, in lexicographic order, starting with the identity."""
        return list(itertools.product(*(range(n) for n in self.orders)))

    def add(self, g, h) -> tuple[int, ...]:
        """The group operation, written additively."""
        return tuple((a + b) % n for a, b, n in zip(g, h, self.orders, strict=True))

    def inverse(self, g) -> tuple[int, ...]:
        """``-g``."""
        return tuple((-a) % n for a, n in zip(g, self.orders, strict=True))

    # -- the pairings -------------------------------------------------------

    def phase(self, exponents: np.ndarray, g, h) -> complex:
        r"""``epsilon(g, h)`` from a table of generator exponents.

        ``exponents[i, j]`` is the integer ``k`` in
        :math:`\varepsilon(e_i, e_j) = e^{2\pi i k / M}`, with ``M`` the group
        exponent; bilinearity then fixes every other value.
        """
        exponents = np.asarray(exponents, dtype=float)
        total = float(np.asarray(g, dtype=float) @ exponents @ np.asarray(h, dtype=float))
        return cmath.exp(2j * math.pi * total / self.exponent)

    def _is_pairing(self, exponents: np.ndarray, tol: float = 1e-9) -> bool:
        """Bilinear, well defined modulo the orders, and trivial on the diagonal."""
        elements = self.elements()
        for g in elements:
            if abs(self.phase(exponents, g, g) - 1.0) > tol:
                return False
        for index, order in enumerate(self.orders):
            wrapped = [0] * len(self.orders)
            wrapped[index] = order
            for h in elements:
                if abs(self.phase(exponents, wrapped, h) - 1.0) > tol:
                    return False
                if abs(self.phase(exponents, h, wrapped) - 1.0) > tol:
                    return False
        return True

    def pairings(self) -> list[np.ndarray]:
        r"""Every consistent ``epsilon``, as a table of generator exponents.

        Enumerated over all assignments of exponents to generator pairs and
        filtered by the axioms, so the size of the answer is a result.  The
        first entry is always the trivial pairing.

        Distinct tables that give the same function on ``G x G`` are collapsed,
        which is why the count is
        :math:`\prod_{i<j}\gcd(N_i, N_j)` and not something larger.
        """
        rank = len(self.orders)
        modulus = self.exponent
        elements = self.elements()
        seen: dict[tuple, np.ndarray] = {}
        for values in itertools.product(range(modulus), repeat=rank * rank):
            exponents = np.array(values, dtype=float).reshape(rank, rank)
            if not self._is_pairing(exponents):
                continue
            signature = tuple(
                round(self.phase(exponents, g, h).real, 9)
                + 1j * round(self.phase(exponents, g, h).imag, 9)
                for g in elements
                for h in elements
            )
            seen.setdefault(signature, exponents)
        ordered = sorted(seen.items(), key=lambda item: tuple(item[1].ravel()))
        return [exponents for _, exponents in ordered]

    def torsion_group(self) -> tuple[int, ...]:
        r"""``gcd(N_i, N_j)`` for ``i < j``: the structure of ``H^2(G, U(1))``.

        The product of these is the number of :meth:`pairings`, which is how the
        two are checked against each other.  Empty for a cyclic group.
        """
        return tuple(
            math.gcd(self.orders[i], self.orders[j])
            for i in range(len(self.orders))
            for j in range(i + 1, len(self.orders))
        )

    # -- modular consistency ------------------------------------------------

    def is_modular_consistent(self, exponents: np.ndarray, tol: float = 1e-9) -> bool:
        r"""Check ``T`` and ``S`` on the weights of every pair of elements.

        ``T`` sends ``Z[g,h]`` to ``Z[g,gh]`` and ``S`` sends it to
        ``Z[h,g^{-1}]``; the weighted sum is only well defined if
        ``epsilon(g, gh) = epsilon(g, h)`` and
        ``epsilon(h, g^{-1}) = epsilon(g, h)``.  Both follow from the axioms, so
        this passing for every pairing is a consistency check on
        :meth:`pairings` rather than an extra filter.
        """
        for g in self.elements():
            for h in self.elements():
                base = self.phase(exponents, g, h)
                if abs(self.phase(exponents, g, self.add(g, h)) - base) > tol:
                    return False
                if abs(self.phase(exponents, h, self.inverse(g)) - base) > tol:
                    return False
        return True

    def projector_weights(self, exponents: np.ndarray, g) -> list[complex]:
        """``epsilon(g, h)`` for every ``h``, in :meth:`elements` order.

        These multiply the terms of the ``g``-twisted projector.  For ``g`` the
        identity they are all 1, which is why torsion leaves the untwisted
        sector alone.
        """
        return [self.phase(exponents, g, h) for h in self.elements()]

    def __str__(self) -> str:  # pragma: no cover - display only
        name = " x ".join(f"Z_{n}" for n in self.orders)
        torsion = self.torsion_group()
        classes = int(np.prod(torsion)) if torsion else 1
        return f"{name}: |G| = {self.size}, discrete torsion classes {classes}"


# ---------------------------------------------------------------------------
# what the phase does to a spectrum
# ---------------------------------------------------------------------------


def twisted_character(twist_phases, element_phases, n_max: int) -> np.ndarray:
    r"""``Tr_{g-twisted} ( h\, q^N )`` on the oscillators, as a ``q``-series.

    In the ``g``-twisted sector a boson has modes shifted to ``n + phi_g``, and
    ``h`` multiplies each by ``exp(2 pi i phi_h)``.  The trace is therefore

    .. math::
       \prod_{j}\prod_{n \geq 0, \; n + \phi_{g,j} > 0}
       \frac{1}{1 - e^{2\pi i \phi_{h,j}}\, q^{\,n + \phi_{g,j}}} ,

    and this returns the coefficients of ``q^{k/K}`` up to ``k = n_max``, where
    ``K`` is the common denominator of the shifts.  With ``h`` the identity the
    result is the ordinary twisted degeneracy series.

    ``twist_phases`` and ``element_phases`` are the ``phi`` of ``g`` and ``h``
    on the same directions, in the same order.
    """
    twist = np.asarray(twist_phases, dtype=float)
    acting = np.asarray(element_phases, dtype=float)
    if twist.shape != acting.shape:
        raise ValueError(
            f"the two phase lists must match: got {twist.shape} and {acting.shape}"
        )
    denominator = _common_denominator(twist)
    series = np.zeros(n_max + 1, dtype=complex)
    series[0] = 1.0
    for shift, angle in zip(twist, acting, strict=True):
        weight = cmath.exp(2j * math.pi * float(angle))
        level = 0
        while True:
            frequency = level + float(shift)
            units = int(round(frequency * denominator))
            if units > n_max:
                break
            if units > 0:
                series = _multiply_by_tower(series, units, weight, n_max)
            level += 1
    return series


def _common_denominator(phases: np.ndarray, cap: int = 720) -> int:
    """The smallest ``K`` making every phase a multiple of ``1/K``."""
    for denominator in range(1, cap + 1):
        scaled = phases * denominator
        if np.max(np.abs(scaled - np.rint(scaled))) < 1e-9:
            return denominator
    raise ValueError("twist phases are not rational with a small denominator")


def _multiply_by_tower(
    series: np.ndarray, units: int, weight: complex, n_max: int
) -> np.ndarray:
    """Multiply by ``1 / (1 - weight q^{units/K})``, in place of a geometric sum."""
    out = series.copy()
    for index in range(units, n_max + 1):
        out[index] += weight * out[index - units]
    return out


def projected_degeneracies(
    group: TorsionGroup,
    phase_table: dict,
    twist,
    exponents: np.ndarray,
    n_max: int = 8,
    tol: float = 1e-7,
) -> np.ndarray:
    r"""Level-by-level counts in the ``twist``-twisted sector, weights included.

    ``phase_table`` maps each group element to its twist phases ``phi``, which
    is the only geometry needed: the character of ``h`` on the ``g``-twisted
    oscillators follows from the two phase lists.  The projection is

    .. math::  d_N = \frac{1}{|G|}\sum_h \varepsilon(g,h)\,
               \mathrm{Tr}_{g}\!\left(h\, q^{N}\right)\Big|_{q^N} ,

    which must come out real and non-negative; both are checked, because a
    complex or negative count would mean the weights are wrong rather than
    interesting.
    """
    elements = group.elements()
    weights = group.projector_weights(exponents, tuple(twist))
    total = np.zeros(n_max + 1, dtype=complex)
    for element, weight in zip(elements, weights, strict=True):
        total += weight * twisted_character(phase_table[tuple(twist)], phase_table[element], n_max)
    total /= group.size
    worst = float(np.max(np.abs(total.imag)))
    if worst > tol:
        raise ValueError(f"projected counts came out complex (max imaginary part {worst:.2e})")
    counts = total.real
    if np.min(counts) < -tol:
        raise ValueError(f"projected counts came out negative (min {np.min(counts):.2e})")
    return np.rint(counts).astype(int)
