r"""Asymmetric orbifolds: twists that act differently on left- and right-movers.

:mod:`stringsim.compactification.orbifold` quotients a torus by a rotation of
*space*, so left- and right-movers are turned by the same angle.  Nothing
requires that.  On the Narain lattice a twist is an element
:math:`\Omega \in O(d,d;\mathbb{Z})` that also fixes the moduli,

.. math::  \Omega^{T}\eta\,\Omega = \eta, \qquad \Omega^{T}\mathcal{H}\,\Omega
           = \mathcal{H},

and because it preserves both forms it preserves :math:`\ell_L^2` and
:math:`\ell_R^2` separately: in the momentum frame it is a *pair* of rotations
:math:`(R_L, R_R) \in O(d) \times O(d)`.  :func:`momentum_action` extracts them.
When the two differ the twist is asymmetric and has no description as a motion
of the torus at all.

**Which twists exist is a lattice question, and it is answered here by search.**
:math:`\mathcal{H}` is positive definite, so the automorphism group is finite
and can be built by backtracking over images of basis vectors --
:func:`narain_automorphisms`.  At a fully self-dual :math:`T^d` it comes out as

.. math::  |\mathrm{Aut}| = 2^{2d}\, d!, \qquad
           |\mathrm{Aut}_{\text{geometric}}| = 2^{d}\, d! ,

the second being the signed permutations, which is :math:`\mathrm{Aut}` of
:math:`\mathbb{Z}^d` itself.  The ratio is :math:`2^d`, the order of the Weyl
group of one side's enhanced algebra: at a generic point in the moduli space
*every* automorphism is geometric, and asymmetric twists appear exactly where
:func:`stringsim.compactification.torus.gauge_algebra` reports roots.  On the
hexagonal :math:`T^2`, which has no enhanced symmetry, the search returns 12
automorphisms and not one of them is asymmetric.

**Geometric means one thing precisely.**  With :math:`Z = (w, n)` a
diffeomorphism sends :math:`w \to Aw` and a :math:`B`-shift changes only the
momentum, so both leave the upper-right block of :math:`\Omega` zero: winding
never comes from momentum.  :func:`is_geometric` is that test, and T-duality,
whose whole content is :math:`n \leftrightarrow w`, fails it.

**Level matching.**  A twisted sector is consistent only if
:math:`L_0 - \bar L_0` is quantised in units of :math:`1/N`, which for an
order-``N`` twist with no shift is

.. math::  N\,(E_L - E_R) = N\,(a_R - a_L) \in \mathbb{Z},

with the intercepts built from the twist phases exactly as
:class:`stringsim.compactification.orbifold.Orbifold` builds its own.  Applied
to every automorphism of every background below, the condition turns out to
hold **precisely when the two phase multisets agree** -- the left and right
rotations may be different rotations, but they must turn by the same angles.
That is an observation from the enumeration, not an input to it, and the test
suite re-derives it rather than trusting this paragraph.

For the twists that pass, :math:`|\det(1 - \Omega)|` is always a perfect square,
and its square root is the twisted-sector degeneracy: the count of fixed points
of a Narain-lattice twist enters the partition function under a square root
because left- and right-movers each supply half of it.  A zero determinant means
:math:`\Omega` fixes a direction, so the fixed set is continuous and the count
does not apply; :attr:`AsymmetricTwist.twisted_degeneracy` returns ``None``
there rather than a wrong number.

**Not here.**  Shifts, which are what rescue most asymmetric rotations that fail
level matching on their own, and the twisted spectra themselves.  This module
enumerates the candidate twists and applies the consistency conditions to them;
building the states is a further step.

Reference: Narain, Sarmadi and Vafa, Nucl. Phys. B **288** (1987) 551.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .torus import (
    TorusBackground,
    integer_points_in_ball,
    narain_momenta,
    odd_metric,
)

__all__ = [
    "momentum_frame",
    "narain_automorphisms",
    "momentum_action",
    "is_geometric",
    "twist_phases",
    "phase_intercept",
    "AsymmetricTwist",
    "classify_automorphisms",
]

_TOL = 1e-8


def momentum_frame(background: TorusBackground) -> np.ndarray:
    r"""The linear map ``Z = (w, n) -> (l_L, l_R)``, shape ``(2d, 2d)``.

    Built column by column from
    :func:`stringsim.compactification.torus.narain_momenta`, so it inherits that
    function's normalisation and nothing is re-derived here.  In this frame
    ``H`` is the identity and ``eta`` is ``diag(I, -I)``.
    """
    dim = background.dim
    columns = []
    for index in range(2 * dim):
        charge = np.zeros(2 * dim)
        charge[index] = 1.0
        left, right = narain_momenta(background, charge[dim:], charge[:dim])
        columns.append(np.concatenate([left, right]))
    return np.array(columns).T


def narain_automorphisms(background: TorusBackground, tol: float = 1e-7) -> list[np.ndarray]:
    r"""Every ``Omega`` preserving both the lattice form and the moduli.

    The group is finite because :math:`\mathcal{H}` is positive definite, so
    each basis vector can only go to one of the finitely many integer vectors of
    the same :math:`\mathcal{H}`-norm --
    :func:`stringsim.compactification.torus.integer_points_in_ball` lists them.
    The search then fixes the images one at a time, keeping only those that
    reproduce every inner product under *both* forms; a partial assignment that
    already fails is abandoned, which is what makes the enumeration finish.

    Returns
    -------
    list[ndarray]
        Integer matrices, sorted, including the identity.  The result is a group
        under multiplication, which the tests check.
    """
    dim = background.dim
    metric = background.generalized_metric()
    form = odd_metric(dim)
    size = 2 * dim

    wanted = [float(metric[i, i]) for i in range(size)]
    pool = integer_points_in_ball(metric, np.zeros(size), max(wanted), tol)
    norms = np.einsum("ij,jk,ik->i", pool, metric, pool)
    candidates = [pool[np.abs(norms - value) < tol] for value in wanted]

    found: list[np.ndarray] = []
    columns = np.zeros((size, size))

    def extend(index: int) -> None:
        if index == size:
            found.append(columns.copy())
            return
        for image in candidates[index]:
            chosen = columns[:, :index]
            if index and (
                np.max(np.abs(chosen.T @ metric @ image - metric[:index, index])) > tol
                or np.max(np.abs(chosen.T @ form @ image - form[:index, index])) > tol
            ):
                continue
            columns[:, index] = image
            extend(index + 1)
        columns[:, index] = 0.0

    extend(0)
    return sorted(found, key=lambda omega: tuple(omega.ravel()))


def momentum_action(
    background: TorusBackground, omega: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""``(R_L, R_R)``: how ``omega`` turns the left and right momenta.

    Conjugating by :func:`momentum_frame` must give a block-diagonal matrix,
    since preserving ``H`` and ``eta`` means preserving ``l_L^2`` and ``l_R^2``
    one at a time.  That the off-diagonal blocks vanish is checked, not assumed:
    a non-zero one would mean ``omega`` is not a symmetry of this background.
    """
    dim = background.dim
    frame = momentum_frame(background)
    acted = frame @ np.asarray(omega, dtype=float) @ np.linalg.inv(frame)
    if (
        np.max(np.abs(acted[:dim, dim:])) > 1e-7
        or np.max(np.abs(acted[dim:, :dim])) > 1e-7
    ):
        raise ValueError("omega mixes left and right momenta, so it does not fix the moduli")
    return acted[:dim, :dim], acted[dim:, dim:]


def is_geometric(omega: np.ndarray, tol: float = _TOL) -> bool:
    r"""True when ``omega`` is a diffeomorphism of the torus, possibly with a ``B``-shift.

    In the ``Z = (w, n)`` basis a relabelling of the lattice sends
    ``w -> A w`` and a ``B``-shift adds to the momentum only, so in both cases
    the new winding depends on the old winding alone: the upper-right block is
    zero.  T-duality exchanges ``n`` and ``w`` and therefore fails, which is the
    whole point -- it acts on the fields, not on the space.
    """
    omega = np.asarray(omega, dtype=float)
    dim = omega.shape[0] // 2
    return bool(np.max(np.abs(omega[:dim, dim:])) < tol)


def twist_phases(rotation: np.ndarray) -> np.ndarray:
    r"""``phi_j`` in ``[0, 1)`` from the eigenvalues ``exp(2 pi i phi_j)``, sorted."""
    eigenvalues = np.linalg.eigvals(np.asarray(rotation, dtype=float))
    return np.sort(np.mod(np.angle(eigenvalues) / (2.0 * math.pi), 1.0))


def phase_intercept(phases) -> float:
    r"""``a = 1 - (1/4) sum_j phi_j (1 - phi_j)`` for one moving side.

    The same expression :class:`stringsim.compactification.orbifold.Orbifold`
    uses, where it is checked against the Hurwitz zeta function rather than
    quoted.  Here it is applied to each side separately, which is the only thing
    an asymmetric twist changes.
    """
    phases = np.asarray(phases, dtype=float)
    return 1.0 - 0.25 * float(np.sum(phases * (1.0 - phases)))


def _matrix_order(omega: np.ndarray, cap: int = 64) -> int | None:
    identity = np.eye(omega.shape[0])
    power = identity.copy()
    for step in range(1, cap + 1):
        power = power @ omega
        if np.allclose(power, identity, atol=1e-8):
            return step
    return None


@dataclass(frozen=True)
class AsymmetricTwist:
    """One candidate twist of a Narain background, with its consistency data.

    Parameters
    ----------
    background:
        The :math:`T^d`, through its moduli.
    omega:
        An element of :math:`O(d,d;\\mathbb{Z})` fixing those moduli --
        :func:`narain_automorphisms` produces them.
    """

    background: TorusBackground
    omega: np.ndarray
    order: int = field(init=False)

    def __post_init__(self) -> None:
        omega = np.asarray(self.omega, dtype=float)
        size = 2 * self.background.dim
        if omega.shape != (size, size):
            raise ValueError(f"omega must be {size}x{size}, got {omega.shape}")
        if np.max(np.abs(omega - np.rint(omega))) > _TOL:
            raise ValueError("omega must be an integer matrix")
        order = _matrix_order(omega)
        if order is None:
            raise ValueError("omega must have finite order")
        momentum_action(self.background, omega)  # raises if the moduli are not fixed
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "order", order)

    # -- the two rotations --------------------------------------------------

    def rotations(self) -> tuple[np.ndarray, np.ndarray]:
        """``(R_L, R_R)``."""
        return momentum_action(self.background, self.omega)

    @property
    def left_phases(self) -> np.ndarray:
        """``phi_L``, one per compact direction."""
        return twist_phases(self.rotations()[0])

    @property
    def right_phases(self) -> np.ndarray:
        """``phi_R``."""
        return twist_phases(self.rotations()[1])

    @property
    def is_geometric(self) -> bool:
        """True when the twist is a motion of the torus rather than of the fields."""
        return is_geometric(self.omega)

    @property
    def is_asymmetric(self) -> bool:
        """True when it is not geometric."""
        return not self.is_geometric

    @property
    def has_equal_phases(self) -> bool:
        """True when the left and right rotations turn by the same angles.

        Distinct from :attr:`is_geometric`: an asymmetric twist can rotate both
        sides by the same angles while being a different rotation on each.
        """
        return bool(np.allclose(self.left_phases, self.right_phases, atol=1e-7))

    # -- consistency --------------------------------------------------------

    @property
    def left_intercept(self) -> float:
        """``a_L``."""
        return phase_intercept(self.left_phases)

    @property
    def right_intercept(self) -> float:
        """``a_R``."""
        return phase_intercept(self.right_phases)

    @property
    def level_matching_defect(self) -> float:
        r"""``N (a_R - a_L)`` reduced to ``[0, 1/2]``; zero when the twist is consistent.

        The ground-state value of :math:`L_0 - \bar L_0` in the twisted sector is
        :math:`E_L - E_R = a_R - a_L`, and it must be a multiple of ``1/N`` for
        :math:`T^N` to act trivially on the sector.  A geometric twist has
        ``a_L = a_R`` identically and so cannot fail.
        """
        value = (self.order * (self.right_intercept - self.left_intercept)) % 1.0
        return float(min(value, 1.0 - value))

    @property
    def is_level_matched(self) -> bool:
        """True when :attr:`level_matching_defect` vanishes."""
        return self.level_matching_defect < 1e-9

    # -- the fixed set ------------------------------------------------------

    @property
    def fixed_points(self) -> float:
        r"""``|det(1 - Omega)|`` on the charge lattice.

        Zero means ``Omega`` fixes a direction, so the fixed set is a continuum
        rather than a finite number of points.
        """
        size = 2 * self.background.dim
        return float(abs(np.linalg.det(np.eye(size) - self.omega)))

    @property
    def twisted_degeneracy(self) -> int | None:
        r"""``sqrt(|det(1 - Omega)|)``, or ``None`` when that is not available.

        Left- and right-movers each contribute half of the fixed-point count on
        the Narain lattice, so the multiplicity of the twisted sector is the
        square root.  ``None`` when the determinant vanishes (a continuous fixed
        set) or is not a perfect square (which never happens for a level-matched
        twist, and the tests check that).
        """
        determinant = self.fixed_points
        if determinant < 1e-9:
            return None
        rounded = round(determinant)
        if abs(determinant - rounded) > 1e-6:
            return None
        root = math.isqrt(rounded)
        return root if root * root == rounded else None

    def __str__(self) -> str:  # pragma: no cover - display only
        kind = "geometric" if self.is_geometric else "asymmetric"
        matched = "level-matched" if self.is_level_matched else (
            f"defect {self.level_matching_defect:.4f}"
        )
        return (
            f"order {self.order} {kind}: phi_L {np.round(self.left_phases, 4)}, "
            f"phi_R {np.round(self.right_phases, 4)}, {matched}"
        )


def classify_automorphisms(background: TorusBackground) -> list[AsymmetricTwist]:
    """Every automorphism of the background, wrapped for inspection.

    A convenience over :func:`narain_automorphisms`; the interesting quantities
    are then attributes rather than separate calls.
    """
    return [AsymmetricTwist(background, omega) for omega in narain_automorphisms(background)]
