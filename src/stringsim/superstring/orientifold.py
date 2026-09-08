r"""Type I: the fifth superstring, as an orientifold of type IIB.

The package has IIA, IIB and both heterotic strings.  This is the one that was
missing, and it is not a new worldsheet -- it is type IIB with a gauge symmetry
imposed.

**Worldsheet parity.**  :math:`\Omega` exchanges left- and right-movers.  It
squares to one, so it can be gauged, and gauging it keeps only the invariant
states: the string becomes *unoriented*.  On the massless level of IIB that acts
sector by sector:

* **NS-NS**, :math:`8_v \otimes 8_v`: :math:`\Omega` swaps the two factors, so
  the **symmetric** part survives -- 36 states, the graviton and the dilaton.
  The Kalb-Ramond two-form is antisymmetric and is projected out.
* **R-R**, :math:`8_s \otimes 8_s`: the same swap, but with an extra sign from
  the Ramond zero modes, so the **antisymmetric** part survives -- the 28 of
  :math:`C_2`, while :math:`C_0` and the self-dual :math:`C_4` go.
* **NS-R and R-NS** are exchanged with each other, so one diagonal copy
  survives -- 64 states, one gravitino and one dilatino.

That extra sign in R-R is a choice, not a derivation, and it is the difference
between type I and type IIB orbifolded by :math:`\Omega(-1)^{F_L}`.  But it is
not a free choice: the *other* sign leaves 72 bosons against 64 fermions, which
no supermultiplet can be.  :func:`supersymmetric_sign` picks it by counting
rather than by fiat, and what comes out is exactly the ``N = 1`` supergravity
multiplet of ten dimensions, 64 + 64.

**The open sector is not optional.**  An unoriented closed string alone is
inconsistent: the Klein bottle carries a Ramond-Ramond tadpole, which is a
source with nowhere to go in a non-compact space and a violated Gauss law in a
compact one.  It is cancelled by adding D9-branes, whose Chan-Paton factors
carry the open strings.  :math:`\Omega` acts on those as
:math:`\lambda \to \pm \gamma \lambda^{T} \gamma^{-1}`, and the massless vector
survives with :math:`\lambda` antisymmetric, giving ``SO(n)`` for symmetric
:math:`\gamma` and ``Sp(n/2)`` for antisymmetric.

**How many branes.**  The tadpole condition gives 32, and computing it needs the
Klein bottle, the cylinder and the Mobius strip in their transverse channels --
which is *not* done here; see :func:`tadpole_structure` for what is.  What is
done is the equivalent condition: the tadpole and the ten-dimensional anomaly
cancel together, and :mod:`stringsim.heterotic.anomaly` already computes the
anomaly, giving ``dim G = 496``.  Solving ``n(n-1)/2 = 496`` gives ``n = 32``,
so the gauge group is ``SO(32)`` -- the same group the even self-dual lattice
gives the heterotic string, reached with no lattice anywhere in sight.

**And then the two theories have the same massless spectrum.**  Type I: 128
closed states plus ``496 x 16`` open ones.  Heterotic ``SO(32)``: ``504 x 16``.
Both 8064, from constructions with nothing in common -- see
:func:`massless_total`.  That they agree is the massless shadow of the
strong-weak duality between them.

**Why exactly four diagrams at one loop.**  A surface contributes at order
:math:`g_s^{-\chi}` with :math:`\chi = 2 - 2g - b - c` for genus ``g``, ``b``
boundaries and ``c`` crosscaps.  One loop is :math:`\chi = 0`, and
:func:`one_loop_surfaces` enumerates the solutions: torus, Klein bottle,
cylinder, Mobius strip.  An oriented closed theory has only the first; an
unoriented open one has all four.

**What the projection looks like on a moving string.**  On a classical
solution :math:`\Omega` is just :math:`\sigma \to -\sigma`, which swaps the
left- and right-moving mode coefficients.  An invariant solution therefore has
them equal, and a string with equal chiralities is a **standing** wave.  So the
statement that the unoriented string has half the states is the same statement
as: it cannot carry a wave that travels around it.  :func:`parity_image`,
:func:`parity_even` and :func:`parity_residual` do that on the
:class:`~stringsim.classical.modes.ClosedString` of section 1.

Reference: J. Polchinski, *String Theory* Vol. II, sections 6.5 and 10.8;
A. Sagnotti, *Open strings and their symmetry groups* (1987).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..classical.modes import ClosedString
from ..heterotic.anomaly import required_dimension
from .typeii import TRANSVERSE, Field, Sector, type_ii_massless

__all__ = [
    "symmetric_dimension",
    "antisymmetric_dimension",
    "split_by_exchange",
    "project_closed",
    "closed_counts",
    "supersymmetric_sign",
    "chan_paton_dimension",
    "gauge_group",
    "open_massless",
    "brane_count",
    "massless_total",
    "Surface",
    "euler_characteristic",
    "one_loop_surfaces",
    "surfaces_for",
    "tadpole_structure",
]


def symmetric_dimension(n: int) -> int:
    """``n(n+1)/2`` -- the symmetric square of an ``n``-dimensional space."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return n * (n + 1) // 2


def antisymmetric_dimension(n: int) -> int:
    """``n(n-1)/2`` -- the antisymmetric square."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return n * (n - 1) // 2


def split_by_exchange(
    fields: tuple[Field, ...], transverse: int = TRANSVERSE
) -> tuple[tuple[Field, ...], tuple[Field, ...]]:
    r"""Split a product ``8 x 8`` into its symmetric and antisymmetric halves.

    Which irreducible pieces go where is *found*, not assigned: the only subset
    of the pieces whose dimensions sum to ``n(n+1)/2`` is the symmetric one.
    For ``8_v \otimes 8_v = 1 + 28 + 35`` the split into 36 and 28 has exactly
    one solution, and the function checks that before returning it -- an
    ambiguity would mean the dimensions alone could not decide, and that is
    worth an error rather than a guess.

    Returns ``(symmetric, antisymmetric)``.
    """
    target = symmetric_dimension(transverse)
    total = sum(field.dimension for field in fields)
    if total != transverse**2:
        raise ValueError(f"expected a product of dimension {transverse**2}, got {total}")
    solutions = []
    for size in range(len(fields) + 1):
        for chosen in itertools.combinations(range(len(fields)), size):
            if sum(fields[i].dimension for i in chosen) == target:
                solutions.append(chosen)
    if not solutions:
        raise ValueError(
            f"no subset of {[f.dimension for f in fields]} sums to {target}: "
            "this product is not a square of one space, so exchange does not act on it"
        )
    if len(solutions) > 1:
        raise ValueError(
            f"dimensions do not determine the split: {len(solutions)} subsets sum to {target}"
        )
    picked = set(solutions[0])
    return (
        tuple(f for i, f in enumerate(fields) if i in picked),
        tuple(f for i, f in enumerate(fields) if i not in picked),
    )


def project_closed(
    rr_sign: int = -1, transverse: int = TRANSVERSE
) -> list[Sector]:
    r"""The closed-string states that survive :math:`\Omega`.

    ``rr_sign`` is the extra sign :math:`\Omega` carries in the Ramond-Ramond
    sector: ``-1`` keeps the antisymmetric part, ``+1`` the symmetric one.  Only
    ``-1`` gives a supermultiplet; :func:`supersymmetric_sign` establishes that
    rather than assuming it, and the default here is its answer.

    The two mixed sectors are exchanged rather than acted on internally, so one
    diagonal copy survives.  It is labelled ``NS-R + R-NS`` because it is
    neither one nor the other.
    """
    if rr_sign not in (-1, 1):
        raise ValueError("rr_sign must be +1 or -1")
    sectors = {s.name: s for s in type_ii_massless("IIB", transverse)}
    ns_sym, _ = split_by_exchange(sectors["NS-NS"].fields, transverse)
    rr_sym, rr_anti = split_by_exchange(sectors["R-R"].fields, transverse)
    return [
        Sector("NS-NS (symmetric)", ns_sym),
        Sector("R-R (antisymmetric)" if rr_sign < 0 else "R-R (symmetric)",
               rr_anti if rr_sign < 0 else rr_sym),
        Sector("NS-R + R-NS (diagonal)", sectors["NS-R"].fields),
    ]


def closed_counts(rr_sign: int = -1, transverse: int = TRANSVERSE) -> tuple[int, int]:
    """``(bosons, fermions)`` left in the closed sector after the projection."""
    sectors = project_closed(rr_sign, transverse)
    bosons = sum(s.dimension for s in sectors if s.statistics == "boson")
    fermions = sum(s.dimension for s in sectors if s.statistics == "fermion")
    return bosons, fermions


def supersymmetric_sign(transverse: int = TRANSVERSE) -> int:
    r"""Which Ramond-Ramond sign leaves a supermultiplet.  Tries both.

    ``+1`` keeps 36 R-R states on top of 36 NS-NS ones, 72 bosons against 64
    fermions: no supermultiplet.  ``-1`` keeps 28, giving 64 and 64.  The sign
    is conventionally quoted as part of the definition of type I; here it is the
    only one that can be.

    Raises
    ------
    RuntimeError
        If neither sign balances, or both do.
    """
    working = [s for s in (-1, 1) if len(set(closed_counts(s, transverse))) == 1]
    if len(working) != 1:
        raise RuntimeError(f"expected exactly one balanced sign, found {working}")
    return working[0]


def chan_paton_dimension(n: int, kind: str = "SO") -> int:
    r"""Dimension of the gauge algebra ``n`` Chan-Paton indices leave behind.

    :math:`\Omega` sends a Chan-Paton matrix to
    :math:`\pm\gamma\lambda^{T}\gamma^{-1}`, and the massless vector -- odd
    under :math:`\Omega` on its oscillator -- survives only for the matrices of
    the opposite symmetry.  Symmetric :math:`\gamma` therefore leaves the
    antisymmetric :math:`\lambda`, which is ``so(n)``; antisymmetric
    :math:`\gamma` leaves the symmetric ones, ``sp(n/2)``.
    """
    kind = kind.upper()
    if n < 1:
        raise ValueError("need at least one Chan-Paton index")
    if kind == "SO":
        return antisymmetric_dimension(n)
    if kind == "SP":
        if n % 2:
            raise ValueError("Sp needs an even number of Chan-Paton indices")
        return symmetric_dimension(n)
    raise ValueError("kind must be 'SO' or 'SP'")


def gauge_group(n: int, kind: str = "SO") -> str:
    """``'SO(32)'`` or ``'Sp(16)'`` -- the name, with ``Sp`` counted in pairs."""
    kind = kind.upper()
    chan_paton_dimension(n, kind)  # validates
    return f"SO({n})" if kind == "SO" else f"Sp({n // 2})"


def open_massless(n: int, kind: str = "SO", transverse: int = TRANSVERSE) -> tuple[Field, ...]:
    """The open sector: ``N = 1`` super Yang-Mills in the adjoint.

    ``8_v`` of gauge bosons and ``8_s`` of gauginos, one copy per generator.
    Equal counts -- the open sector is supersymmetric on its own, which the
    closed one had to be separately.
    """
    size = chan_paton_dimension(n, kind)
    group = gauge_group(n, kind)
    return (
        Field(f"gauge boson, {group}", "8v", transverse * size, "boson"),
        Field(f"gaugino, {group}", "8s", transverse * size, "fermion"),
    )


def brane_count(kind: str = "SO") -> int:
    r"""How many D9-branes the consistency conditions demand.

    Not from the tadpole, which is not computed here.  From the equivalent
    condition: :func:`stringsim.heterotic.anomaly.required_dimension` solves the
    ten-dimensional anomaly polynomial for ``dim G = 496``, and the orientifold
    gives ``dim G = n(n-1)/2``.  One integer solves it.

    Returns 32 for ``SO``.  ``Sp`` has no solution -- ``n(n+1)/2 = 496`` has no
    integer root -- which is the statement that the antisymmetric projection is
    not an option, and the code says so by raising.
    """
    wanted = required_dimension()
    for n in range(1, 4 * wanted):
        try:
            if chan_paton_dimension(n, kind) == wanted:
                return n
        except ValueError:
            continue
    raise ValueError(f"no {kind.upper()} group of dimension {wanted} from Chan-Paton factors")


def massless_total(n: int | None = None, kind: str = "SO", transverse: int = TRANSVERSE) -> dict:
    """Every massless state of type I, closed and open, counted.

    ``n`` defaults to :func:`brane_count`.  The total is compared in the test
    suite against the heterotic ``SO(32)`` massless level, which is built from a
    root lattice and knows nothing about orientifolds.
    """
    n = brane_count(kind) if n is None else n
    closed_b, closed_f = closed_counts(supersymmetric_sign(transverse), transverse)
    fields = open_massless(n, kind, transverse)
    open_b = sum(f.dimension for f in fields if f.is_boson)
    open_f = sum(f.dimension for f in fields if not f.is_boson)
    return {
        "branes": n,
        "gauge_group": gauge_group(n, kind),
        "gauge_dimension": chan_paton_dimension(n, kind),
        "closed": closed_b + closed_f,
        "open": open_b + open_f,
        "total": closed_b + closed_f + open_b + open_f,
        "bosons": closed_b + open_b,
        "fermions": closed_f + open_f,
    }


@dataclass(frozen=True)
class Surface:
    """A worldsheet: genus, boundaries, crosscaps."""

    name: str
    genus: int
    boundaries: int
    crosscaps: int

    @property
    def euler(self) -> int:
        """``2 - 2g - b - c``.  The amplitude carries ``g_s^{-euler}``."""
        return 2 - 2 * self.genus - self.boundaries - self.crosscaps

    @property
    def oriented(self) -> bool:
        """A crosscap is what makes a surface unoriented."""
        return self.crosscaps == 0

    @property
    def closed(self) -> bool:
        """No boundary means no open-string endpoints."""
        return self.boundaries == 0

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.name:<14s} g={self.genus} b={self.boundaries} c={self.crosscaps}  "
            f"chi={self.euler}"
        )


NAMES = {
    (1, 0, 0): "torus",
    (0, 0, 2): "Klein bottle",
    (0, 2, 0): "cylinder",
    (0, 1, 1): "Mobius strip",
    (0, 0, 0): "sphere",
    (0, 1, 0): "disc",
    (0, 0, 1): "projective plane",
}


def euler_characteristic(genus: int, boundaries: int, crosscaps: int) -> int:
    """``2 - 2g - b - c``, the exponent that orders string perturbation theory."""
    if min(genus, boundaries, crosscaps) < 0:
        raise ValueError("genus, boundaries and crosscaps must be non-negative")
    return 2 - 2 * genus - boundaries - crosscaps


def one_loop_surfaces(euler: int = 0, limit: int = 4) -> list[Surface]:
    """Every surface with the given Euler characteristic.

    At ``euler = 0`` there are exactly four, and they are the four one-loop
    diagrams: torus, Klein bottle, cylinder, Mobius strip.  Enumerated rather
    than listed, so the count is a result.
    """
    found = []
    for genus in range(limit + 1):
        for boundaries in range(limit + 1):
            for crosscaps in range(limit + 1):
                if euler_characteristic(genus, boundaries, crosscaps) != euler:
                    continue
                key = (genus, boundaries, crosscaps)
                found.append(Surface(NAMES.get(key, f"g{genus}b{boundaries}c{crosscaps}"),
                                     genus, boundaries, crosscaps))
    return found


def surfaces_for(theory: str, euler: int = 0) -> list[Surface]:
    """Which of them a given theory actually has.

    ``'closed oriented'`` (type II) keeps the torus alone; ``'closed
    unoriented'`` adds the Klein bottle; ``'open unoriented'`` (type I) has all
    four.  An open theory needs an unoriented one to be consistent, which is why
    there is no ``'open oriented'`` entry.
    """
    allowed = {
        "closed oriented": lambda s: s.closed and s.oriented,
        "closed unoriented": lambda s: s.closed,
        "open unoriented": lambda s: True,
    }
    if theory not in allowed:
        raise ValueError(f"theory must be one of {sorted(allowed)}")
    return [s for s in one_loop_surfaces(euler) if allowed[theory](s)]


def tadpole_structure(n: int, crosscap_charge: int = -32) -> dict:
    r"""The shape of the Ramond-Ramond tadpole, and what it costs to cancel.

    In the transverse channel each of the three unoriented one-loop diagrams is
    a closed string propagating between the sources it ends on: the cylinder is
    :math:`\langle B|B \rangle`, the Klein bottle
    :math:`\langle C|C \rangle`, and the Mobius strip the cross term.  Their sum
    is therefore

    .. math::
       \langle nB_1 + C \,|\, nB_1 + C \rangle
       \;\propto\; (n + q)^2 ,

    a perfect square in ``n`` whose double root is the total charge.  ``q`` is
    the crosscap's charge in units of one D9-brane's.

    **What is computed here and what is not.**  The square, its discriminant and
    its root are.  The value of ``q`` is not: extracting it needs the three
    amplitudes with their relative normalisations and the modular maps between
    the direct and transverse channels, which this package does not build for
    the superstring.  ``-32`` is the standard value, and it is passed in rather
    than derived.  :func:`brane_count` reaches the same 32 through a route the
    package *does* compute, and the agreement of the two is the only reason the
    number appears here at all.
    """
    quadratic = (1, 2 * crosscap_charge, crosscap_charge**2)
    a, b, c = quadratic
    return {
        "coefficients": quadratic,
        "discriminant": b * b - 4 * a * c,
        "root": -b // (2 * a),
        "charge": n + crosscap_charge,
        "cancels": n + crosscap_charge == 0,
        "value": (n + crosscap_charge) ** 2,
    }


def string_coupling_order(surface: Surface, coupling: float) -> float:
    """``g_s^{-chi}`` -- the weight a surface carries in perturbation theory."""
    if coupling <= 0:
        raise ValueError("the coupling must be positive")
    return math.pow(coupling, -surface.euler)


__all__ += ["string_coupling_order", "NAMES", "parity_image", "parity_even", "parity_residual"]


def parity_image(string: ClosedString) -> ClosedString:
    r"""The same solution seen with :math:`\sigma \to -\sigma`.

    Worldsheet parity exchanges left- and right-movers, so on a classical
    solution it simply swaps ``alphas`` and ``alphas_tilde``.  That the swap
    really is the coordinate reflection -- ``parity_image(s).position(tau,
    sigma) == s.position(tau, -sigma)`` -- is checked in the test suite rather
    than asserted here; it is the one place where the discrete statement about
    sectors and the continuum statement about waves have to agree.
    """
    return ClosedString(
        conventions=string.conventions,
        x0=string.x0,
        p=string.p,
        alphas=dict(string.alphas_tilde),
        alphas_tilde=dict(string.alphas),
    )


def _averaged(a: dict, b: dict, sign: float) -> dict:
    modes = set(a) | set(b)
    out = {}
    for n in modes:
        left = a.get(n, np.zeros(0))
        right = b.get(n, np.zeros(0))
        if left.size == 0:
            left = np.zeros_like(right)
        if right.size == 0:
            right = np.zeros_like(left)
        out[n] = 0.5 * (left + sign * right)
    return out


def parity_even(string: ClosedString) -> ClosedString:
    r"""The :math:`\Omega`-invariant part of a solution.

    Averaging the mode coefficients of the two chiralities gives
    :math:`\tfrac12[X(\tau,\sigma) + X(\tau,-\sigma)]`, which is a solution
    because the wave equation is linear.  Its left- and right-movers are equal,
    so it is a **standing** wave: an unoriented string cannot carry a travelling
    one, which is what the projection means when you watch it rather than count
    it.
    """
    shared = _averaged(string.alphas, string.alphas_tilde, +1.0)
    return ClosedString(
        conventions=string.conventions,
        x0=string.x0,
        p=string.p,
        alphas=shared,
        alphas_tilde={n: a.copy() for n, a in shared.items()},
    )


def parity_residual(
    string: ClosedString, n_tau: int = 24, n_sigma: int = 96
) -> float:
    r"""How far a solution is from being :math:`\Omega`-invariant.

    The largest value of :math:`|X(\tau,\sigma) - X(\tau,-\sigma)|` over a grid,
    normalised by the string's own extent so the number means something.  Zero
    for a standing wave, order one for a travelling one.
    """
    tau = np.linspace(0.0, 2.0 * math.pi, n_tau)[:, None]
    sigma = np.linspace(0.0, string.sigma_max, n_sigma)[None, :]
    here = string.position(tau, sigma)
    there = string.position(tau, -sigma)
    scale = max(float(np.max(np.abs(here - here.mean(axis=(0, 1))))), 1e-12)
    return float(np.max(np.abs(here - there)) / scale)
