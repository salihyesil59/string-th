r"""Toroidal orbifolds :math:`T^d/\mathbb{Z}_N`: twisted sectors and the projection.

An orbifold is a torus quotiented by a finite symmetry :math:`\theta`.  Two
things happen, and both are needed for consistency.

**The untwisted sector is projected.**  Only :math:`\theta`-invariant states
survive, counted by the character projector
:math:`P = \frac{1}{N}\sum_k \theta^k`.  This removes states, and what it
removes is physical: on :math:`S^1/\mathbb{Z}_2` the Kaluza-Klein gauge bosons
:math:`g_{\mu i}` and :math:`B_{\mu i}` are odd and disappear, so the
compactification has no massless vectors from the untwisted sector at a generic
radius.

**Twisted sectors appear.**  Strings that close only up to :math:`\theta^k`,
:math:`X(\sigma + 2\pi) = \theta^k X(\sigma)`, are new states with no
counterpart on the torus.  They are stuck at the fixed points of
:math:`\theta^k`, of which there are :math:`|\det(1 - \theta^k)|`, and their
oscillators carry *fractional* mode numbers.  That shifts the ground-state
energy: writing the eigenvalues of :math:`\theta^k` on the complexified compact
space as :math:`e^{2\pi i \phi_j}`,

.. math::
   a_k = 1 - \tfrac14 \sum_{j=1}^{d} \phi_j (1 - \phi_j), \qquad
   \frac{\alpha' M^2}{4} = N - a_k .

The ``1/4`` is not a guess: a boson with modes ``n + phi`` has zero-point energy
:math:`\tfrac12\zeta(-1,\phi)`, and
:func:`stringsim.quantum.zeta.regularised_shifted_sum` extracts that number from
a cut-off sum, exactly as ``-1/12`` was extracted for the untwisted string.  The
test suite checks the closed form above against that independent route.

**Which orbifolds exist is not a free choice.**  :math:`\theta` has to be an
automorphism of the lattice *and* preserve ``G`` and ``B``, which is verified
here by pushing it through the ``O(d,d;Z)`` machinery of
:mod:`stringsim.compactification.torus` and requiring the moduli to come back
unchanged.  In two dimensions the crystallographic restriction then leaves only
:math:`N = 1, 2, 3, 4, 6` -- :func:`crystallographic_orders` finds that by
search rather than quoting it.

**What is not here.**  Discrete torsion, asymmetric orbifolds (``theta`` acting
differently on left and right movers), and a full modular-invariance proof by
constructing :math:`Z[g,h]` for every pair.  The twisted-sector machinery below
assumes a *symmetric* action with isolated fixed points, and says so.

Reference: Dixon, Harvey, Vafa and Witten, Nucl. Phys. B **261** (1985) 678 and
B **274** (1986) 285.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from ..units import Conventions
from .torus import TorusBackground, basis_change, transform

__all__ = [
    "Orbifold",
    "crystallographic_orders",
    "matrix_order",
    "transverse_phases",
    "oscillator_trace_series",
    "untwisted_degeneracy",
    "UntwistedContent",
    "untwisted_massless_content",
    "TwistedLevel",
    "twisted_spectrum",
]

_TOL = 1e-9


def matrix_order(matrix: np.ndarray, max_order: int = 64) -> int | None:
    """Multiplicative order of an integer matrix, or ``None`` if larger than ``max_order``."""
    matrix = np.asarray(matrix, dtype=float)
    identity = np.eye(matrix.shape[0])
    power = identity.copy()
    for order in range(1, max_order + 1):
        power = power @ matrix
        if np.allclose(power, identity, atol=_TOL):
            return order
    return None


def crystallographic_orders(dim: int = 2, bound: int = 2, max_order: int = 24) -> set[int]:
    r"""Every finite order realised by an integer ``dim x dim`` matrix, found by search.

    In two dimensions the answer is ``{1, 2, 3, 4, 6}`` -- the crystallographic
    restriction.  The reason is short: a finite-order integer matrix has
    eigenvalues that are roots of unity, so its trace is both an algebraic
    integer sum :math:`2\cos(2\pi/N)` and an ordinary integer, which forces
    :math:`N \in \{1,2,3,4,6\}`.  That is why the bosonic string has
    :math:`T^2/\mathbb{Z}_N` orbifolds for exactly those ``N`` and no others.

    ``bound`` limits the entries searched; the answer is already complete at
    ``bound = 2`` for ``dim = 2``, since a larger entry forces a larger trace or
    determinant than a finite-order matrix can have.
    """
    found: set[int] = set()
    values = range(-bound, bound + 1)
    for entries in itertools.product(values, repeat=dim * dim):
        matrix = np.array(entries, dtype=float).reshape(dim, dim)
        if abs(abs(np.linalg.det(matrix)) - 1.0) > _TOL:
            continue  # a finite-order integer matrix is unimodular
        order = matrix_order(matrix, max_order)
        if order is not None:
            found.add(order)
    return found


@dataclass(frozen=True)
class Orbifold:
    r"""A torus together with a finite-order lattice automorphism.

    Parameters
    ----------
    background:
        The :math:`T^d` being quotiented.
    rotation:
        ``theta``, an integer ``d x d`` matrix in the lattice basis.  It must
        have finite order and preserve both ``G`` and ``B``; the constructor
        checks the latter by transforming the background with the corresponding
        ``O(d,d;Z)`` element and requiring the moduli to come back unchanged, so
        the check reuses code that is already tested rather than re-deriving the
        condition ``theta^T G theta = G``.
    """

    background: TorusBackground
    rotation: np.ndarray
    order: int = field(init=False)

    def __post_init__(self) -> None:
        theta = np.asarray(self.rotation, dtype=float)
        d = self.background.dim
        if theta.shape != (d, d):
            raise ValueError(f"rotation must be {d}x{d}, got {theta.shape}")
        if np.any(np.abs(theta - np.rint(theta)) > _TOL):
            raise ValueError("rotation must be an integer matrix in the lattice basis")
        order = matrix_order(theta)
        if order is None:
            raise ValueError("rotation must have finite order")
        moved = transform(self.background, basis_change(theta))
        if not (
            np.allclose(moved.metric, self.background.metric, atol=1e-8)
            and np.allclose(moved.b_field, self.background.b_field, atol=1e-8)
        ):
            raise ValueError("rotation does not preserve the metric and B field")
        object.__setattr__(self, "rotation", theta)
        object.__setattr__(self, "order", order)

    # -- constructors -------------------------------------------------------

    @classmethod
    def inversion(cls, background: TorusBackground) -> Orbifold:
        r"""``theta = -1``: the reflection orbifold, valid on any torus.

        At ``d = 1`` this is :math:`S^1/\mathbb{Z}_2`, an interval with two
        fixed points.
        """
        return cls(background, -np.eye(background.dim))

    @classmethod
    def z3_hexagonal(cls, conventions: Conventions | None = None) -> Orbifold:
        r""":math:`T^2/\mathbb{Z}_3` on the hexagonal lattice: three fixed points."""
        metric = np.array([[2.0, -1.0], [-1.0, 2.0]])
        background = TorusBackground(metric, conventions=conventions or Conventions())
        return cls(background, np.array([[0.0, -1.0], [1.0, -1.0]]))

    @classmethod
    def z4_square(cls, conventions: Conventions | None = None) -> Orbifold:
        r""":math:`T^2/\mathbb{Z}_4` on the square lattice: two fixed points at ``k=1``."""
        background = TorusBackground(np.eye(2), conventions=conventions or Conventions())
        return cls(background, np.array([[0.0, -1.0], [1.0, 0.0]]))

    @classmethod
    def z6_hexagonal(cls, conventions: Conventions | None = None) -> Orbifold:
        r""":math:`T^2/\mathbb{Z}_6` on the hexagonal lattice: a single fixed point."""
        metric = np.array([[2.0, -1.0], [-1.0, 2.0]])
        background = TorusBackground(metric, conventions=conventions or Conventions())
        return cls(background, np.array([[1.0, -1.0], [1.0, 0.0]]))

    # -- geometry -----------------------------------------------------------

    @property
    def dim(self) -> int:
        """``d``, the number of compact directions."""
        return self.background.dim

    def power(self, k: int) -> np.ndarray:
        """``theta^k`` as a float matrix."""
        return np.linalg.matrix_power(self.rotation, int(k) % self.order)

    def twist_phases(self, k: int = 1) -> np.ndarray:
        r"""``phi_j`` in ``[0, 1)`` from the eigenvalues ``exp(2 pi i phi_j)`` of ``theta^k``.

        Returned sorted, one entry per compact direction.  A phase of 0 means
        that direction is left alone by ``theta^k``, which is also what makes
        its fixed-point set non-isolated.
        """
        eigenvalues = np.linalg.eigvals(self.power(k))
        phases = np.angle(eigenvalues) / (2.0 * math.pi)
        return np.sort(np.mod(phases, 1.0))

    def intercept(self, k: int = 1) -> float:
        r"""``a_k = 1 - (1/4) sum_j phi_j (1 - phi_j)``, the ``k``-twisted ground-state energy.

        ``k = 0`` gives the untwisted value 1.  The formula follows from
        :math:`\tfrac12 \zeta(-1,\phi)` per twisted boson; see
        :func:`stringsim.quantum.zeta.regularised_shifted_sum`.
        """
        phases = self.twist_phases(k)
        return 1.0 - 0.25 * float(np.sum(phases * (1.0 - phases)))

    def fixed_points(self, k: int = 1) -> int:
        r"""``|det(1 - theta^k)|``, the number of fixed points of ``theta^k`` on the torus.

        Raises when ``theta^k`` has an eigenvalue 1, i.e. when the fixed locus
        is not a set of isolated points -- the determinant vanishes there and
        the count is not a number of points at all.
        """
        if int(k) % self.order == 0:
            raise ValueError("theta^0 fixes the whole torus; there is no finite count")
        matrix = np.eye(self.dim) - self.power(k)
        determinant = float(abs(np.linalg.det(matrix)))
        if determinant < _TOL:
            raise ValueError(
                f"theta^{k} has an eigenvalue 1, so its fixed locus is not isolated"
            )
        return int(round(determinant))

    def fixed_point_positions(self, k: int = 1) -> np.ndarray:
        r"""Where the fixed points sit, in lattice coordinates in ``[0, 1)``.

        A point of the torus is fixed when :math:`\theta^k x = x + \lambda` for
        some lattice vector, i.e. :math:`x \in (1-\theta^k)^{-1}\Lambda`, so the
        fixed points are the classes of that group modulo :math:`\Lambda` --
        exactly :math:`|\det(1-\theta^k)|` of them, which is why
        :meth:`fixed_points` is a determinant.

        The enumeration below is complete: :math:`\mathrm{adj}(1-\theta^k)`
        times :math:`(1-\theta^k)` is :math:`\pm D`, so :math:`D\Lambda` sits
        inside :math:`(1-\theta^k)\Lambda` and letting each component of
        :math:`\lambda` run over ``0 .. D-1`` reaches every class.
        """
        count = self.fixed_points(k)
        inverse = np.linalg.inv(np.eye(self.dim) - self.power(k))
        seen: dict[tuple[int, ...], np.ndarray] = {}
        for shift in itertools.product(range(count), repeat=self.dim):
            position = np.mod(inverse @ np.array(shift, dtype=float), 1.0)
            key = tuple(int(round(value * 1e6)) % 1_000_000 for value in position)
            seen.setdefault(key, position)
        found = np.array([seen[key] for key in sorted(seen)])
        if len(found) != count:
            raise RuntimeError(  # pragma: no cover - a bug guard, not a user error
                f"enumerated {len(found)} fixed points but det gives {count}"
            )
        return found

    def has_isolated_fixed_points(self, k: int = 1) -> bool:
        """True when ``theta^k`` leaves no direction untouched."""
        return bool(np.all(self.twist_phases(k) > _TOL))

    def charge_action(self, k: int = 1) -> np.ndarray:
        """The ``O(d,d;Z)`` matrix implementing ``theta^k`` on ``Z = (w, n)``."""
        return basis_change(self.power(k))

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"T^{self.dim}/Z_{self.order}: "
            f"phases {np.round(self.twist_phases(1), 4)}, "
            f"a_1 = {self.intercept(1):.6f}, "
            f"{self.fixed_points(1)} fixed points"
        )


# ---------------------------------------------------------------------------
# untwisted sector: the projection
# ---------------------------------------------------------------------------


def transverse_phases(orbifold: Orbifold, k: int = 1) -> np.ndarray:
    r"""Phases of ``theta^k`` on all ``D - 2`` transverse directions.

    The non-compact transverse directions are untouched, so they contribute
    ``D - 2 - d`` zeros; the compact ones contribute
    :meth:`Orbifold.twist_phases`.
    """
    transverse = orbifold.background.conventions.transverse_dim
    non_compact = transverse - orbifold.dim
    if non_compact < 0:
        raise ValueError("more compact directions than transverse ones")
    return np.concatenate([np.zeros(non_compact), orbifold.twist_phases(k)])


def oscillator_trace_series(phases, n_max: int) -> np.ndarray:
    r"""Coefficients of :math:`\prod_{n\geq1}\prod_j (1 - \lambda_j q^n)^{-1}`.

    With :math:`\lambda_j = e^{2\pi i \phi_j}` this is
    :math:`\mathrm{Tr}\,(\theta^k q^N)` over one chiral oscillator Fock space:
    the ``k = 0`` case reproduces the ordinary degeneracies ``1, 24, 324, ...``,
    and other ``k`` give the signed traces the projector needs.

    Returned as a complex array; the imaginary parts cancel only after the sum
    over ``k`` in :func:`untwisted_degeneracy`, so they are kept here.
    """
    phases = np.asarray(phases, dtype=float)
    eigenvalues = np.exp(2j * math.pi * phases)
    coefficients = np.zeros(n_max + 1, dtype=complex)
    coefficients[0] = 1.0
    for n in range(1, n_max + 1):
        for value in eigenvalues:
            # multiply by 1/(1 - lambda q^n) = sum_m lambda^m q^{nm}
            for level in range(n, n_max + 1):
                coefficients[level] += value * coefficients[level - n]
    return coefficients


def untwisted_degeneracy(orbifold: Orbifold, level: int, level_tilde: int) -> int:
    r"""Number of ``theta``-invariant states at oscillator level ``(N, Ntilde)``.

    Zero momentum and winding, which is where the massless states sit at a
    generic radius.  The count is

    .. math::
       \frac{1}{N}\sum_{k=0}^{N-1}
       \mathrm{Tr}_k(q^{N_L})\,\mathrm{Tr}_k(q^{N_R}) ,

    and it must come out a non-negative integer -- which the test suite checks
    at every level, since a sign or conjugation error would break exactly that.
    """
    if level < 0 or level_tilde < 0:
        raise ValueError("levels must be non-negative")
    n_max = max(level, level_tilde)
    total = 0.0 + 0.0j
    for k in range(orbifold.order):
        series = oscillator_trace_series(transverse_phases(orbifold, k), n_max)
        total += series[level] * series[level_tilde]
    value = total / orbifold.order
    if abs(value.imag) > 1e-6 or abs(value.real - round(value.real)) > 1e-6:
        raise ValueError(f"projection gave a non-integer count {value!r}; check the action")
    return int(round(value.real))


@dataclass(frozen=True)
class UntwistedContent:
    """Massless untwisted states of the orbifold, before and after the projection."""

    torus_states: int
    surviving: int
    graviton_sector: int
    moduli: int
    projected_out: int

    @property
    def removed(self) -> int:
        """How many states the projection deletes."""
        return self.torus_states - self.surviving

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.surviving} of {self.torus_states} survive "
            f"({self.graviton_sector} graviton/B/dilaton, {self.moduli} moduli, "
            f"{self.projected_out} vectors removed)"
        )


def untwisted_massless_content(orbifold: Orbifold) -> UntwistedContent:
    r"""The massless untwisted level, split into what survives and what does not.

    At ``(N, Ntilde) = (1, 1)`` the torus has ``(D-2)^2`` states.  Splitting the
    transverse index into ``D-2-d`` non-compact and ``d`` compact directions,
    and taking ``theta`` to act with eigenvalue ``-1`` on every compact one (the
    reflection orbifold), the surviving states are those with an even number of
    compact indices:

    * ``(D-2-d)^2`` with both indices non-compact -- the graviton, the
      Kalb-Ramond field and the dilaton of the lower-dimensional theory;
    * ``d^2`` with both compact -- the metric and ``B`` field moduli of the
      torus, which survive;
    * ``2d(D-2-d)`` with one of each -- the Kaluza-Klein and winding gauge
      bosons, which are **odd and projected out**.

    The counts are computed here by that index bookkeeping, and the test suite
    checks them against :func:`untwisted_degeneracy`, which arrives at the same
    numbers through the character projector without ever splitting an index.
    This function is therefore only valid for the reflection orbifold; for other
    rotations use :func:`untwisted_degeneracy`.
    """
    if not np.allclose(orbifold.rotation, -np.eye(orbifold.dim)):
        raise ValueError(
            "the index split above assumes theta = -1; use untwisted_degeneracy otherwise"
        )
    transverse = orbifold.background.conventions.transverse_dim
    d = orbifold.dim
    non_compact = transverse - d
    return UntwistedContent(
        torus_states=transverse**2,
        surviving=non_compact**2 + d**2,
        graviton_sector=non_compact**2,
        moduli=d**2,
        projected_out=2 * d * non_compact,
    )


# ---------------------------------------------------------------------------
# twisted sectors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwistedLevel:
    """One level of a twisted sector."""

    sector: int
    level: Fraction
    alpha_m2: float
    oscillator_states: int
    fixed_points: int

    @property
    def degeneracy(self) -> int:
        """States per side, times the number of fixed points they can sit at."""
        return self.oscillator_states * self.fixed_points

    @property
    def is_tachyonic(self) -> bool:
        return self.alpha_m2 < -_TOL

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"k={self.sector}  N={str(self.level):>6s}  "
            f"alpha'M^2 = {self.alpha_m2:+8.4f}  "
            f"x{self.oscillator_states} osc x{self.fixed_points} fixed points"
        )


def _twisted_oscillator_series(phases, n_untwisted: int, order: int, max_units: int):
    r"""Chiral partition function of the twisted oscillators, in units of ``q^{1/order}``.

    A direction with phase ``phi`` has creation operators at mode numbers
    ``n + phi`` for every ``n >= 0`` with ``n + phi > 0``; untwisted directions
    keep the ordinary ``n >= 1``.  Working in ``u = q^{1/order}`` turns all of
    those into integer powers, which is why the level is reported as a
    :class:`~fractions.Fraction` with denominator ``order``.
    """
    coefficients = np.zeros(max_units + 1, dtype=object)
    coefficients[0] = 1

    def multiply_by_tower(step_units: int) -> None:
        """Multiply by ``1/(1 - u^{step})``."""
        if step_units <= 0 or step_units > max_units:
            return
        for level in range(step_units, max_units + 1):
            coefficients[level] += coefficients[level - step_units]

    for _ in range(n_untwisted):
        for n in range(1, max_units // order + 1):
            multiply_by_tower(n * order)
    for phase in np.asarray(phases, dtype=float):
        shift_units = int(round(phase * order))
        for n in range(0, max_units // order + 2):
            step = n * order + shift_units
            if step > 0:
                multiply_by_tower(step)
    return coefficients


def twisted_spectrum(orbifold: Orbifold, sector: int = 1, n_levels: int = 4) -> list[TwistedLevel]:
    r"""Levels of the ``k``-twisted sector, from the ground state upwards.

    The mass is :math:`\alpha' M^2 / 4 = N - a_k`, with ``N`` running over the
    fractional levels the twisted modes allow and ``a_k`` from
    :meth:`Orbifold.intercept`.  Multiplicity is the oscillator degeneracy times
    the number of fixed points, since a twisted string can sit at any of them.

    Restricted to sectors with isolated fixed points -- if ``theta^k`` leaves a
    direction alone, that direction still carries momentum and winding and the
    counting below is incomplete rather than merely coarse, so it refuses.

    Level matching is ``N_L = N_R`` for a symmetric orbifold at zero charge, so
    the same fractional level appears on both sides and the mass formula above
    is the whole story.
    """
    if int(sector) % orbifold.order == 0:
        raise ValueError("sector 0 is the untwisted sector")
    if not orbifold.has_isolated_fixed_points(sector):
        raise ValueError(
            f"theta^{sector} leaves a direction untouched; that sector carries "
            "momentum and winding, which this function does not enumerate"
        )
    order = orbifold.order
    intercept = orbifold.intercept(sector)
    transverse = orbifold.background.conventions.transverse_dim
    phases = orbifold.twist_phases(sector)
    max_units = n_levels * order
    series = _twisted_oscillator_series(phases, transverse - orbifold.dim, order, max_units)
    points = orbifold.fixed_points(sector)

    out: list[TwistedLevel] = []
    for units, count in enumerate(series):
        if count == 0:
            continue
        level = Fraction(units, order)
        out.append(
            TwistedLevel(
                sector=int(sector),
                level=level,
                alpha_m2=4.0 * (float(level) - intercept),
                oscillator_states=int(count),
                fixed_points=points,
            )
        )
    return out


def massless_summary(orbifold: Orbifold) -> str:  # pragma: no cover - display only
    """A one-paragraph description of what survives, for the examples and CLI."""
    lines = [str(orbifold)]
    surviving = untwisted_degeneracy(orbifold, 1, 1)
    transverse = orbifold.background.conventions.transverse_dim
    lines.append(f"  untwisted massless: {surviving} of {transverse**2} survive")
    if np.allclose(orbifold.rotation, -np.eye(orbifold.dim)):
        lines.append(f"    breakdown: {untwisted_massless_content(orbifold)}")
    for k in range(1, orbifold.order):
        if not orbifold.has_isolated_fixed_points(k):
            lines.append(f"  sector k={k}: fixed locus not isolated, skipped")
            continue
        levels = twisted_spectrum(orbifold, k, n_levels=2)
        massless = [lv for lv in levels if abs(lv.alpha_m2) < 1e-9]
        lines.append(
            f"  sector k={k}: a = {orbifold.intercept(k):.6f}, "
            f"{orbifold.fixed_points(k)} fixed points, "
            f"ground state alpha'M^2 = {levels[0].alpha_m2:+.4f}, "
            f"{len(massless)} massless levels"
        )
    return "\n".join(lines)


__all__ += ["massless_summary"]
