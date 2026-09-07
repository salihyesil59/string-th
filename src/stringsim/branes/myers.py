r"""The Myers effect: N D0-branes that are secretly a sphere.

:mod:`stringsim.branes.dbi` gives one brane an action.  Put ``N`` of them on top
of each other and the transverse positions stop being numbers: they become
``N x N`` Hermitian matrices, and in a background Ramond-Ramond flux their
potential is

.. math::
   V(\Phi) = \mathrm{Tr}\left(
     -\tfrac14 [\Phi_i, \Phi_j][\Phi_i, \Phi_j]
     + \tfrac{i f}{3}\,\varepsilon_{ijk}\,\Phi_i\Phi_j\Phi_k \right),

with ``f`` fixing the strength of the flux.  The quartic term wants the matrices
to commute -- ordinary, separated branes -- and the cubic term does not.

**What wins is an SU(2) representation.**  Substituting
:math:`\Phi_i = \alpha J_i` with :math:`[J_i,J_j] = i\varepsilon_{ijk}J_k`
gives

.. math::
   V = \mathrm{Tr}(J^2)\left(\tfrac12\alpha^4 - \tfrac{f}{3}\alpha^3\right),
   \qquad
   \alpha_\star = \frac f2, \qquad
   V_\star = -\frac{f^4}{96}\,\mathrm{Tr}(J^2) ,

and :func:`myers_gradient` confirms it solves the *full* matrix equations of
motion, not just the equations restricted to the ansatz.  Since
:math:`V_\star` is proportional to :math:`-\mathrm{Tr}(J^2)`, the deepest
configuration is the one with the largest Casimir, and for a representation that
splits into blocks of sizes :math:`N_a` that is
:math:`\sum_a N_a(N_a^2-1)/4` -- maximised by a **single block**.
:func:`configuration_energies` enumerates every partition of ``N`` and finds the
irreducible one at the bottom every time.  Commuting matrices, the naive vacuum,
sit at :math:`V = 0`: they are not even a local minimum.

**And the single block is a sphere.**  Its physical radius is

.. math::
   R^2 = \frac1N \mathrm{Tr}(\Phi_i\Phi_i)
       = \alpha_\star^2\,j(j+1) = \frac{f^2}{16}(N^2-1) ,

so :math:`R \to fN/4`, while the relative size of the commutators --
:func:`noncommutativity` -- falls like :math:`1/N`.  A noncommutative sphere
becomes a classical one, and ``N`` point-like branes have turned into one
two-dimensional object.

**The other description, and a check with no free constants.**  That object is a
spherical D2-brane carrying ``N`` units of worldvolume flux.  Its Born-Infeld
energy is

.. math::
   E(R) = 4\pi T_2 \sqrt{R^4 + \pi^2\alpha'^2 N^2} ,

and at :math:`R = 0` this is :math:`4\pi^2\alpha' N T_2`, which is exactly
:math:`N T_0`: **a shrunk D2-brane with N units of flux weighs N D0-branes.**
:func:`shrunk_d2_energy` computes it and
:func:`stringsim.branes.dbrane.dp_brane_tension` supplies the comparison, with
nothing fitted in between.

Expanding at large ``N`` gives the same quartic-minus-cubic shape as the matrix
potential, so the two pictures describe one object from opposite ends.  Where
they differ is instructive: the continuum answer knows only :math:`N^3`, while
the matrices give :math:`N(N^2-1)`, so

.. math::  \frac{V_{\text{matrix}}}{V_{\text{continuum}}} = 1 - \frac1{N^2} ,

exact, not fitted -- :func:`large_n_ratio`.  The correction is the price of
having only ``N`` points on the sphere.

**Not here.**  The full non-abelian DBI beyond the leading commutator terms
(which is not unambiguously defined), the fermionic sector, and the
Chern-Simons coupling's normalisation, which is folded into ``f`` rather than
derived.  What is derived is everything that follows from the potential above.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..units import Conventions
from .dbrane import dp_brane_tension

__all__ = [
    "LEVI_CIVITA",
    "su2_generators",
    "algebra_residual",
    "casimir",
    "trace_j_squared",
    "myers_potential",
    "myers_gradient",
    "fuzzy_sphere",
    "fuzzy_radius",
    "noncommutativity",
    "latitudes",
    "partitions",
    "block_configuration",
    "Configuration",
    "configuration_energies",
    "spherical_d2_energy",
    "shrunk_d2_energy",
    "large_n_ratio",
]


def _levi_civita() -> np.ndarray:
    tensor = np.zeros((3, 3, 3))
    for permutation in itertools.permutations(range(3)):
        tensor[permutation] = np.sign(np.linalg.det(np.eye(3)[list(permutation)]))
    return tensor


#: ``epsilon_ijk`` in three dimensions, built from determinants of permutations.
LEVI_CIVITA = _levi_civita()


# ---------------------------------------------------------------------------
# the algebra
# ---------------------------------------------------------------------------


def su2_generators(dim: int) -> np.ndarray:
    r"""The ``dim``-dimensional irreducible representation, shape ``(3, dim, dim)``.

    Spin :math:`j = (N-1)/2`, with :math:`J_3` diagonal and :math:`J_\pm`
    carrying :math:`\sqrt{j(j+1) - m(m\pm1)}`.  Normalised so that
    :math:`[J_i, J_j] = i\varepsilon_{ijk}J_k`, which
    :func:`algebra_residual` checks.
    """
    if dim < 1:
        raise ValueError(f"dim must be at least 1, got {dim}")
    spin = (dim - 1) / 2.0
    weights = np.array([spin - step for step in range(dim)])
    raising = np.zeros((dim, dim), dtype=complex)
    if dim > 1:
        upper = weights[1:]
        raising[np.arange(dim - 1), np.arange(1, dim)] = np.sqrt(
            spin * (spin + 1) - upper * (upper + 1)
        )
    lowering = raising.conj().T
    return np.array(
        [
            (raising + lowering) / 2.0,
            (raising - lowering) / 2.0j,
            np.diag(weights).astype(complex),
        ]
    )


def algebra_residual(matrices: np.ndarray) -> float:
    r"""``max |[J_i, J_j] - i eps_ijk J_k|``: zero for a genuine representation."""
    matrices = np.asarray(matrices, dtype=complex)
    worst = 0.0
    for i, j in itertools.product(range(3), repeat=2):
        commutator = matrices[i] @ matrices[j] - matrices[j] @ matrices[i]
        expected = sum(1j * LEVI_CIVITA[i, j, k] * matrices[k] for k in range(3))
        worst = max(worst, float(np.max(np.abs(commutator - expected))))
    return worst


def casimir(dim: int) -> float:
    r"""``j(j+1)`` for the ``dim``-dimensional representation."""
    spin = (dim - 1) / 2.0
    return spin * (spin + 1.0)


def trace_j_squared(dim: int) -> float:
    r"""``Tr(J^2) = N (N^2 - 1) / 4``, which is what the depth of the minimum tracks."""
    return dim * (dim**2 - 1) / 4.0


# ---------------------------------------------------------------------------
# the potential
# ---------------------------------------------------------------------------


def _as_triple(matrices) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=complex)
    if matrices.ndim != 3 or matrices.shape[0] != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(f"expected three square matrices, got shape {matrices.shape}")
    return matrices


def myers_potential(matrices, flux: float = 1.0) -> float:
    r"""``Tr(-1/4 [Phi_i,Phi_j][Phi_i,Phi_j] + (i f/3) eps_ijk Phi_i Phi_j Phi_k)``.

    Real for Hermitian ``matrices``; the imaginary part is discarded only after
    it has been produced, so a non-Hermitian input will not silently pass.
    """
    matrices = _as_triple(matrices)
    quartic = 0.0
    for i, j in itertools.product(range(3), repeat=2):
        commutator = matrices[i] @ matrices[j] - matrices[j] @ matrices[i]
        quartic += -0.25 * np.trace(commutator @ commutator)
    cubic = 0.0
    for i, j, k in itertools.product(range(3), repeat=3):
        sign = LEVI_CIVITA[i, j, k]
        if sign:
            cubic += sign * np.trace(matrices[i] @ matrices[j] @ matrices[k])
    total = quartic + 1j * flux / 3.0 * cubic
    if abs(total.imag) > 1e-8 * max(1.0, abs(total.real)):
        raise ValueError(f"the potential came out complex ({total}); are the matrices Hermitian?")
    return float(total.real)


def myers_gradient(matrices, flux: float = 1.0) -> np.ndarray:
    r"""``dV/dPhi_i = [[Phi_i,Phi_j],Phi_j] + i f eps_ijk Phi_j Phi_k``.

    The equations of motion.  Written analytically here and checked against a
    finite-difference gradient of :func:`myers_potential` in the test suite, so
    neither is trusted on its own.
    """
    matrices = _as_triple(matrices)
    out = np.zeros_like(matrices)
    for i in range(3):
        for j in range(3):
            commutator = matrices[i] @ matrices[j] - matrices[j] @ matrices[i]
            out[i] += commutator @ matrices[j] - matrices[j] @ commutator
        for j, k in itertools.product(range(3), repeat=2):
            sign = LEVI_CIVITA[i, j, k]
            if sign:
                out[i] += 1j * flux * sign * matrices[j] @ matrices[k]
    return out


def fuzzy_sphere(dim: int, flux: float = 1.0) -> np.ndarray:
    r"""``Phi_i = (f/2) J_i``: the solution the cubic term forces."""
    return (flux / 2.0) * su2_generators(dim)


def fuzzy_radius(matrices) -> float:
    r"""``sqrt(Tr(Phi_i Phi_i) / N)``, the physical radius of the configuration."""
    matrices = _as_triple(matrices)
    total = sum(np.trace(matrices[i] @ matrices[i]).real for i in range(3))
    return float(math.sqrt(total / matrices.shape[1]))


def noncommutativity(matrices) -> float:
    r"""``max |[Phi_i, Phi_j]| / R^2``: how far the sphere is from being classical.

    For the fuzzy sphere the commutators are :math:`(f/2)^2` times the
    generators, whose entries grow like :math:`N`, while :math:`R^2` grows like
    :math:`N^2` -- so this falls like :math:`1/N` and the noncommutative sphere
    turns into an ordinary one.
    """
    matrices = _as_triple(matrices)
    radius = fuzzy_radius(matrices)
    if radius == 0.0:
        return 0.0
    worst = max(
        float(np.max(np.abs(matrices[i] @ matrices[j] - matrices[j] @ matrices[i])))
        for i, j in itertools.product(range(3), repeat=2)
    )
    return worst / radius**2


def latitudes(matrices) -> tuple[np.ndarray, np.ndarray]:
    r"""The circles the fuzzy sphere is made of: heights and their radii.

    :math:`\Phi_3` is diagonalisable, and its eigenvalues are the heights at
    which the sphere has support -- ``N`` of them, evenly spaced.  Each sits at
    the radius the constraint :math:`\sum_i \Phi_i^2 = R^2` allows,
    :math:`\rho = \sqrt{R^2 - z^2}`.

    So a fuzzy sphere is genuinely a stack of ``N`` circles, and it looks like a
    sphere only when ``N`` is large.  Returned sorted by height.
    """
    matrices = _as_triple(matrices)
    heights = np.sort(np.linalg.eigvalsh(matrices[2]))
    radius = fuzzy_radius(matrices)
    return heights, np.sqrt(np.maximum(radius**2 - heights**2, 0.0))


# ---------------------------------------------------------------------------
# which configuration wins
# ---------------------------------------------------------------------------


def partitions(total: int) -> list[tuple[int, ...]]:
    """Every way of splitting ``N`` branes into irreducible blocks, largest first."""
    if total < 1:
        raise ValueError(f"total must be at least 1, got {total}")

    def walk(remaining: int, cap: int):
        if remaining == 0:
            yield ()
            return
        for size in range(min(remaining, cap), 0, -1):
            for rest in walk(remaining - size, size):
                yield (size, *rest)

    return list(walk(total, total))


def block_configuration(partition, flux: float = 1.0) -> np.ndarray:
    """Block-diagonal ``Phi``: one fuzzy sphere per block, all at once."""
    sizes = [int(size) for size in partition]
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError(f"partition must be positive integers, got {partition}")
    total = sum(sizes)
    out = np.zeros((3, total, total), dtype=complex)
    offset = 0
    for size in sizes:
        block = fuzzy_sphere(size, flux)
        for i in range(3):
            out[i][offset : offset + size, offset : offset + size] = block[i]
        offset += size
    return out


@dataclass(frozen=True)
class Configuration:
    """One block structure and what it costs."""

    partition: tuple[int, ...]
    energy: float
    trace_j_squared: float
    radius: float

    @property
    def is_irreducible(self) -> bool:
        """True for the single-block configuration, the fuzzy sphere."""
        return len(self.partition) == 1

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{str(self.partition):>22s}  V = {self.energy:+.6f}  "
            f"Tr J^2 = {self.trace_j_squared:>8.2f}  R = {self.radius:.4f}"
        )


def configuration_energies(total: int, flux: float = 1.0) -> list[Configuration]:
    """Every partition of ``N``, evaluated and sorted by energy.

    The first entry is always the single block: the depth of the minimum tracks
    ``sum N_a (N_a^2 - 1)``, which one big block maximises.  The last is
    ``(1, 1, ..., 1)``, commuting matrices at ``V = 0`` -- the configuration one
    would have called the vacuum.
    """
    found = []
    for partition in partitions(total):
        matrices = block_configuration(partition, flux)
        found.append(
            Configuration(
                partition=partition,
                energy=myers_potential(matrices, flux),
                trace_j_squared=sum(trace_j_squared(size) for size in partition),
                radius=fuzzy_radius(matrices),
            )
        )
    return sorted(found, key=lambda entry: entry.energy)


# ---------------------------------------------------------------------------
# the other description
# ---------------------------------------------------------------------------


def spherical_d2_energy(
    radius: float, n_units: int, g_s: float = 1.0, conventions: Conventions | None = None
) -> float:
    r"""``4 pi T_2 sqrt(R^4 + pi^2 alpha'^2 N^2)`` for a sphere carrying ``N`` flux quanta.

    The Born-Infeld energy of a D2-brane wrapping a sphere of radius ``R`` with
    :math:`\int F = 2\pi N`: the determinant of :math:`g + 2\pi\alpha'F` is
    :math:`\sin^2\theta\,(R^4 + \pi^2\alpha'^2N^2)`, and the angular integral
    gives :math:`4\pi`.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if n_units < 0:
        raise ValueError("the flux quantum number must be non-negative")
    conv = conventions or Conventions()
    tension = dp_brane_tension(2, g_s, conv)
    inside = radius**4 + (math.pi * conv.alpha_prime * n_units) ** 2
    return 4.0 * math.pi * tension * math.sqrt(inside)


def shrunk_d2_energy(
    n_units: int, g_s: float = 1.0, conventions: Conventions | None = None
) -> float:
    r"""The same at ``R = 0``, which had better be ``N T_0``.

    :math:`4\pi^2\alpha' T_2 = T_0` follows from the tension formula with no
    freedom in it, so this is a check with nothing fitted: **a D2-brane with N
    units of flux, shrunk to a point, weighs N D0-branes**.  That is why the
    matrix description and the sphere description are of the same object.
    """
    return spherical_d2_energy(0.0, n_units, g_s, conventions)


def large_n_ratio(dim: int) -> float:
    r"""``1 - 1/N^2``: matrix energy over the continuum answer.

    The fuzzy sphere gives :math:`\mathrm{Tr}(J^2) = N(N^2-1)/4` where the
    continuum D2-brane knows only :math:`N^3/4`.  The gap is exact and is the
    cost of building a sphere out of ``N`` points.
    """
    if dim < 1:
        raise ValueError(f"dim must be at least 1, got {dim}")
    return trace_j_squared(dim) / (dim**3 / 4.0)
