r"""Hodge numbers of a toroidal orbifold, and what discrete torsion does to them.

:mod:`stringsim.compactification.torsion` derives the phases
:math:`\varepsilon(g,h)` the partition function may carry, and says that the
classic consequence -- :math:`T^6/(\mathbb{Z}_2\times\mathbb{Z}_2)` having
:math:`(h^{1,1}, h^{2,1}) = (51, 3)` without torsion and :math:`(3, 51)` with
it -- was not computed there.  It is computed here.

**The Euler characteristic is a fixed-point count, and nothing else.**

.. math::
   \chi = \frac{1}{|G|} \sum_{gh = hg} \varepsilon(g,h)\;
          \chi\!\left(M^{g,h}\right) ,

where :math:`M^{g,h}` is the part of the torus both elements hold still.  On a
torus that set is a disjoint union of subtori, so its Euler characteristic is
zero unless the subtori are points -- and then it is how many.  Both facts come
out of one integer computation: stacking :math:`1-g` and :math:`1-h` into a
:math:`2d \times d` matrix, the fixed set has

* dimension :math:`d - \mathrm{rank}`, and
* :math:`\gcd` of all maximal minors many components,

because those are the Smith invariants -- :func:`fixed_set`.  For
:math:`T^6/(\mathbb{Z}_2\times\mathbb{Z}_2)` every pair of *distinct* non-trivial
elements gives 64 points and every other pair gives a positive-dimensional set,
so :math:`\chi = 6 \times 64 / 4 = 96`.  The non-trivial pairing is
:math:`-1` on exactly those six pairs, so with torsion :math:`\chi = -96`.

**The untwisted Hodge numbers are a character.**  On :math:`\Lambda^p \otimes
\bar\Lambda^q` of :math:`\mathbb{C}^3` the trace of ``g`` is
:math:`e_p(\lambda)\,\overline{e_q(\lambda)}` with :math:`\lambda` its
holomorphic eigenvalues and :math:`e_k` the elementary symmetric polynomials, so

.. math::
   h^{p,q}_{\text{untwisted}}
     = \frac1{|G|}\sum_g e_p(\lambda_g)\,\overline{e_q(\lambda_g)} .

:func:`untwisted_hodge` evaluates it.  That :math:`h^{3,0} = 1` comes out is the
Calabi-Yau condition :math:`\prod_j \lambda_j = 1`, checked rather than assumed.

**One input, stated.**  Each singular locus contributes one blow-up modulus --
an :math:`A_1` curve has one exceptional divisor, a :math:`\mathbb{C}^3/Z_3`
point has one -- and loci are counted once per pair :math:`\{g, g^{-1}\}`, since
an element and its inverse hold the same set still.  That is
:func:`blowup_moduli`, and it is geometry put in rather than derived.  With it,

.. math::
   h^{1,1} + h^{2,1} = (\text{untwisted total}) + (\text{blow-up moduli}),
   \qquad h^{1,1} - h^{2,1} = \chi/2 ,

which :func:`hodge_numbers` solves.  The two sides were computed from unrelated
things -- a lattice minor count and a character sum -- so the answers coming out
as non-negative integers is a real check, and it passes for all three standard
orbifolds:

===========================  ==========  ================  ============
orbifold                     ``chi``     untwisted         result
===========================  ==========  ================  ============
:math:`T^6/\mathbb{Z}_3`      ``+72``     ``(9, 0)``        ``(36, 0)``
:math:`T^6/\mathbb{Z}_4`      ``+48``     ``(5, 1)``        ``(31, 7)``
:math:`T^6/(Z_2\times Z_2)`   ``+96``     ``(3, 3)``        ``(51, 3)``
   ... with discrete torsion  ``-96``     ``(3, 3)``        ``(3, 51)``
===========================  ==========  ================  ============

**Not here.**  Non-abelian orbifold groups, orbifolds with fixed loci that
intersect in ways the one-modulus-per-locus rule does not cover, and the
resolution itself -- this counts moduli, it does not build the manifold.
"""

from __future__ import annotations

import cmath
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .torsion import TorsionGroup

__all__ = [
    "FixedSet",
    "minor_gcd",
    "fixed_set",
    "OrbifoldAction",
    "lattice_rotation",
    "fixed_locus",
    "joint_euler",
    "euler_characteristic",
    "untwisted_hodge",
    "blowup_moduli",
    "HodgeNumbers",
    "hodge_numbers",
    "z2_z2_orbifold",
    "z3_orbifold",
    "z4_orbifold",
]

_TOL = 1e-8


# ---------------------------------------------------------------------------
# fixed sets on a torus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedSet:
    """What part of the torus a set of rotations holds still.

    Attributes
    ----------
    components:
        How many connected pieces.
    dimension:
        The dimension of each; they are all subtori of the same size.
    """

    components: int
    dimension: int

    @property
    def euler_characteristic(self) -> int:
        """Zero unless the pieces are points, since a torus has none."""
        return self.components if self.dimension == 0 else 0

    def __str__(self) -> str:  # pragma: no cover - display only
        if self.dimension:
            return f"{self.components} T^{self.dimension}"
        return f"{self.components} point" + ("" if self.components == 1 else "s")


def minor_gcd(matrix, size: int) -> int:
    r"""``gcd`` of every ``size x size`` minor, which is a Smith invariant.

    The product of the first ``k`` elementary divisors of an integer matrix is
    the gcd of its ``k x k`` minors, so for ``k`` equal to the rank this is the
    product of *all* the non-zero ones -- exactly the number of components of
    the solution set.  Enumerating the minors is slow in principle and instant
    at the sizes an orbifold uses.
    """
    matrix = np.rint(np.asarray(matrix, dtype=float)).astype(np.int64)
    if size < 0 or size > min(matrix.shape):
        raise ValueError(f"size must lie in [0, {min(matrix.shape)}], got {size}")
    if size == 0:
        return 1
    found = 0
    for rows in itertools.combinations(range(matrix.shape[0]), size):
        for columns in itertools.combinations(range(matrix.shape[1]), size):
            block = matrix[np.ix_(rows, columns)].astype(float)
            found = math.gcd(found, abs(int(round(np.linalg.det(block)))))
            if found == 1:
                return 1  # nothing smaller is possible, so stop enumerating
    return found


def fixed_set(matrix) -> FixedSet:
    r"""``{x : M x in Z^n} / Z^d`` for an integer matrix, as a count and a dimension.

    Put ``M`` in Smith normal form, ``M = U S V``.  The condition becomes
    ``s_i y_i in Z`` for ``y = V x``, so each non-zero invariant contributes
    ``|s_i|`` choices and each zero one a free circle.  Hence the count is the
    product of the non-zero invariants -- :func:`minor_gcd` at the rank -- and
    the dimension is the nullity.
    """
    matrix = np.rint(np.asarray(matrix, dtype=float)).astype(np.int64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {matrix.shape}")
    rank = int(np.linalg.matrix_rank(matrix.astype(float)))
    return FixedSet(
        components=minor_gcd(matrix, rank), dimension=matrix.shape[1] - rank
    )


# ---------------------------------------------------------------------------
# the action
# ---------------------------------------------------------------------------


def lattice_rotation(*blocks) -> np.ndarray:
    """Block-diagonal integer rotation built from two-dimensional pieces."""
    mats = [np.rint(np.asarray(block, dtype=float)).astype(np.int64) for block in blocks]
    size = sum(m.shape[0] for m in mats)
    out = np.zeros((size, size), dtype=np.int64)
    offset = 0
    for m in mats:
        out[offset : offset + m.shape[0], offset : offset + m.shape[1]] = m
        offset += m.shape[0]
    return out


@dataclass(frozen=True)
class OrbifoldAction:
    r"""An abelian group acting on :math:`T^6`, with its complex structure.

    Parameters
    ----------
    group:
        The abstract group, as a :class:`stringsim.compactification.torsion.TorsionGroup`.
    rotations:
        ``element -> integer 6 x 6 matrix`` in the lattice basis.
    phases:
        ``element -> three holomorphic eigenvalues``.  The complex structure has
        to be supplied: a lattice rotation has eigenvalues in conjugate pairs,
        and which of each pair is holomorphic is a choice.  Requiring the
        product to be 1 is the Calabi-Yau condition and is enforced.
    """

    group: TorsionGroup
    rotations: dict = field(default_factory=dict)
    phases: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        elements = self.group.elements()
        rotations = {
            tuple(key): np.rint(np.asarray(value, dtype=float)).astype(np.int64)
            for key, value in self.rotations.items()
        }
        phases = {
            tuple(key): tuple(complex(v) for v in value)
            for key, value in self.phases.items()
        }
        if set(rotations) != set(elements) or set(phases) != set(elements):
            raise ValueError("give a rotation and a phase triple for every group element")
        identity = tuple([0] * len(self.group.orders))
        size = rotations[identity].shape[0]
        for g in elements:
            if rotations[g].shape != (size, size):
                raise ValueError(f"rotation for {g} has the wrong shape")
            if len(phases[g]) != size // 2:
                raise ValueError(f"expected {size // 2} holomorphic phases for {g}")
            if abs(np.prod(phases[g]) - 1.0) > _TOL:
                raise ValueError(
                    f"the phases for {g} multiply to {np.prod(phases[g])}, not 1: "
                    "the action is not in SU(3) and the quotient is not Calabi-Yau"
                )
            eigenvalues = np.sort_complex(np.linalg.eigvals(rotations[g].astype(float)))
            expected = np.sort_complex(
                np.array([*phases[g], *[np.conj(p) for p in phases[g]]])
            )
            if np.max(np.abs(eigenvalues - expected)) > 1e-6:
                raise ValueError(f"the phases for {g} are not the eigenvalues of its rotation")
            for h in elements:
                product = rotations[g] @ rotations[h]
                if not np.array_equal(product, rotations[self.group.add(g, h)]):
                    raise ValueError(f"the rotations are not a representation at {g}, {h}")
        object.__setattr__(self, "rotations", rotations)
        object.__setattr__(self, "phases", phases)

    @property
    def dim(self) -> int:
        """Real dimension of the torus."""
        return next(iter(self.rotations.values())).shape[0]

    @property
    def identity(self) -> tuple[int, ...]:
        return tuple([0] * len(self.group.orders))

    def __str__(self) -> str:  # pragma: no cover - display only
        name = " x ".join(f"Z_{n}" for n in self.group.orders)
        return f"T^{self.dim}/({name})"


def fixed_locus(action: OrbifoldAction, element) -> FixedSet:
    """What a single element holds still."""
    identity = np.eye(action.dim, dtype=np.int64)
    return fixed_set(identity - action.rotations[tuple(element)])


def joint_euler(action: OrbifoldAction, first, second) -> int:
    r"""``chi`` of the set both elements hold still.

    Stack :math:`1-g` above :math:`1-h`; the answer is zero unless the common
    fixed set is a finite set of points, and then it is how many.
    """
    identity = np.eye(action.dim, dtype=np.int64)
    stacked = np.vstack(
        [identity - action.rotations[tuple(first)], identity - action.rotations[tuple(second)]]
    )
    # A positive-dimensional fixed set contributes nothing, so there is no need
    # to count its components -- and counting them is the expensive part.
    if int(np.linalg.matrix_rank(stacked.astype(float))) < action.dim:
        return 0
    return fixed_set(stacked).euler_characteristic


def euler_characteristic(action: OrbifoldAction, exponents=None) -> float:
    r"""``(1/|G|) sum_{gh=hg} eps(g,h) chi(M^{g,h})``.

    ``exponents`` is a discrete-torsion pairing from
    :meth:`stringsim.compactification.torsion.TorsionGroup.pairings`; the
    default is the trivial one.  The group is abelian, so every pair commutes
    and the sum runs over all of ``G x G``.
    """
    group = action.group
    if exponents is None:
        exponents = group.pairings()[0]
    total = 0.0
    for first in group.elements():
        for second in group.elements():
            weight = group.phase(exponents, first, second)
            total += weight.real * joint_euler(action, first, second)
    return total / group.size


# ---------------------------------------------------------------------------
# the Hodge numbers
# ---------------------------------------------------------------------------


def _elementary_symmetric(order: int, values) -> complex:
    if order == 0:
        return 1.0 + 0.0j
    return sum(
        complex(np.prod(choice)) for choice in itertools.combinations(values, order)
    )


def untwisted_hodge(action: OrbifoldAction) -> dict[tuple[int, int], int]:
    r"""``h^{p,q}`` of the invariant forms, by averaging characters.

    The trace of ``g`` on :math:`\Lambda^p \otimes \bar\Lambda^q` is
    :math:`e_p(\lambda)\overline{e_q(\lambda)}`, so the invariant dimension is
    its average over the group.  Returns the whole diamond, ``p, q`` from 0 to
    3, with ``h^{3,0} = 1`` when the action really is Calabi-Yau.
    """
    group = action.group
    out: dict[tuple[int, int], int] = {}
    for p, q in itertools.product(range(4), repeat=2):
        total = sum(
            _elementary_symmetric(p, action.phases[g])
            * np.conj(_elementary_symmetric(q, action.phases[g]))
            for g in group.elements()
        ) / group.size
        if abs(total.imag) > _TOL:
            raise ValueError(f"h^({p},{q}) came out complex: {total}")
        rounded = round(total.real)
        if abs(total.real - rounded) > _TOL:
            raise ValueError(f"h^({p},{q}) came out non-integral: {total.real}")
        out[(p, q)] = rounded
    return out


def blowup_moduli(action: OrbifoldAction) -> int:
    r"""Number of blow-up moduli: one per singular locus.

    An element and its inverse hold the same set still, so the loci are counted
    once per pair :math:`\{g, g^{-1}\}`, and each contributes as many moduli as
    it has components.

    **This is geometry put in, not derived.**  It is the statement that an
    :math:`A_1` curve resolves with one exceptional divisor and a
    :math:`\mathbb{C}^3/\mathbb{Z}_3` point with one, and it holds for the
    orbifolds in this module -- which the consistency check in
    :func:`hodge_numbers` is there to notice if it ever stops holding.
    """
    group = action.group
    seen: set[tuple[int, ...]] = set()
    total = 0
    for element in group.elements():
        if element == action.identity or element in seen:
            continue
        seen.add(element)
        seen.add(group.inverse(element))
        total += fixed_locus(action, element).components
    return total


@dataclass(frozen=True)
class HodgeNumbers:
    """The two independent Hodge numbers of a Calabi-Yau threefold."""

    h11: int
    h21: int
    euler: int
    untwisted: tuple[int, int]
    twisted: int

    @property
    def total(self) -> int:
        """``h11 + h21``."""
        return self.h11 + self.h21

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"(h11, h21) = ({self.h11}, {self.h21}), chi = {self.euler:+d}   "
            f"[untwisted {self.untwisted}, {self.twisted} blow-up moduli]"
        )


def hodge_numbers(action: OrbifoldAction, exponents=None) -> HodgeNumbers:
    r"""Solve ``h11 - h21 = chi/2`` and ``h11 + h21 = untwisted + blow-ups``.

    The difference comes from a count of lattice points held still; the sum from
    a character average plus the blow-up rule.  Nothing connects the two
    calculations, so both answers coming out as non-negative integers is a
    check, and it is raised as an error rather than rounded away.
    """
    chi = euler_characteristic(action, exponents)
    if abs(chi - round(chi)) > _TOL or round(chi) % 2:
        raise ValueError(f"the Euler characteristic came out at {chi}, not an even integer")
    untwisted = untwisted_hodge(action)
    base = (untwisted[(1, 1)], untwisted[(2, 1)])
    twisted = blowup_moduli(action)
    total = base[0] + base[1] + twisted
    difference = round(chi) // 2
    if (total + difference) % 2:
        raise ValueError(
            f"h11 + h21 = {total} and h11 - h21 = {difference} have different parities; "
            "the blow-up count and the fixed-point count disagree"
        )
    h11, h21 = (total + difference) // 2, (total - difference) // 2
    if h11 < 0 or h21 < 0:
        raise ValueError(f"got a negative Hodge number: ({h11}, {h21})")
    return HodgeNumbers(
        h11=h11, h21=h21, euler=round(chi), untwisted=base, twisted=twisted
    )


# ---------------------------------------------------------------------------
# the standard examples
# ---------------------------------------------------------------------------

_ROTATIONS = {
    2: np.array([[-1, 0], [0, -1]], dtype=np.int64),
    3: np.array([[0, -1], [1, -1]], dtype=np.int64),
    4: np.array([[0, -1], [1, 0]], dtype=np.int64),
    6: np.array([[1, -1], [1, 0]], dtype=np.int64),
}


def _powers(group: TorsionGroup, generators, phases) -> tuple[dict, dict]:
    rotations, holomorphic = {}, {}
    for element in group.elements():
        matrix = np.eye(generators[0].shape[0], dtype=np.int64)
        triple = np.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j])
        pieces = zip(generators, phases, strict=True)
        for power, (generator, phase) in zip(element, pieces, strict=True):
            matrix = matrix @ np.linalg.matrix_power(generator, power).astype(np.int64)
            triple = triple * np.array(phase) ** power
        rotations[element] = matrix
        holomorphic[element] = tuple(triple)
    return rotations, holomorphic


def z3_orbifold() -> OrbifoldAction:
    r""":math:`T^6/\mathbb{Z}_3` with ``theta = (w, w, w)``, three hexagonal tori."""
    group = TorsionGroup((3,))
    generator = lattice_rotation(_ROTATIONS[3], _ROTATIONS[3], _ROTATIONS[3])
    omega = cmath.exp(2j * cmath.pi / 3.0)
    rotations, phases = _powers(group, [generator], [(omega, omega, omega)])
    return OrbifoldAction(group, rotations, phases)


def z4_orbifold() -> OrbifoldAction:
    r""":math:`T^6/\mathbb{Z}_4` with ``theta = (i, i, -1)``."""
    group = TorsionGroup((4,))
    generator = lattice_rotation(_ROTATIONS[4], _ROTATIONS[4], _ROTATIONS[2])
    rotations, phases = _powers(group, [generator], [(1j, 1j, -1.0 + 0j)])
    return OrbifoldAction(group, rotations, phases)


def z2_z2_orbifold() -> OrbifoldAction:
    r""":math:`T^6/(\mathbb{Z}_2\times\mathbb{Z}_2)`, each generator flipping two tori."""
    group = TorsionGroup((2, 2))
    unit = np.eye(2, dtype=np.int64)
    first = lattice_rotation(unit, _ROTATIONS[2], _ROTATIONS[2])
    second = lattice_rotation(_ROTATIONS[2], unit, _ROTATIONS[2])
    rotations, phases = _powers(
        group, [first, second], [(1.0 + 0j, -1.0 + 0j, -1.0 + 0j), (-1.0 + 0j, 1.0 + 0j, -1.0 + 0j)]
    )
    return OrbifoldAction(group, rotations, phases)
