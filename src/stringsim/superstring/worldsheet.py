r"""Numerical evolution of the worldsheet fermions.

:mod:`stringsim.superstring.rns` counts states.  This module does for
:math:`\psi^\mu` what :mod:`stringsim.classical.evolve` does for :math:`X^\mu`:
put the field on a grid, step it forward, and read the physics back off the
result instead of quoting it.

**The equation is first order, and that changes everything.**  In conformal
gauge the Dirac equation splits into two decoupled transport equations,

.. math::
   (\partial_\tau + \partial_\sigma)\,\psi_- = 0, \qquad
   (\partial_\tau - \partial_\sigma)\,\psi_+ = 0,

so :math:`\psi_-` is a rigid profile sliding toward larger :math:`\sigma` and
:math:`\psi_+` one sliding the other way.  Nothing oscillates; nothing
disperses.  All of the content is in what happens at the ends.

**The open string is one chiral field on a doubled circle.**  At
:math:`\sigma = 0` the two components must agree, and at :math:`\sigma = \pi`
they may agree up to a sign:

.. math::  \psi_+(0) = \psi_-(0), \qquad \psi_+(\pi) = \eta\,\psi_-(\pi),
           \qquad \eta = \pm 1.

Fold the interval open with :math:`\psi(\sigma) = \psi_-(\sigma)` on
:math:`[0,\pi]` and :math:`\psi(\sigma) = \eta\,\psi_+(2\pi - \sigma)` on
:math:`[\pi, 2\pi]`, and the pair becomes a *single* right-moving field on a
circle of circumference :math:`2\pi` obeying

.. math::  \psi(\sigma + 2\pi) = \eta\,\psi(\sigma).

That one line is the whole of the sector story.  Periodic (:math:`\eta = +1`,
Ramond) admits :math:`e^{ir\sigma}` with :math:`r` integral -- **including
:math:`r = 0`, a piece of the field that never moves**, which is the Clifford
zero mode that makes the Ramond ground state a spinor.  Antiperiodic
(:math:`\eta = -1`, Neveu-Schwarz) admits only :math:`r \in \mathbb{Z} +
\tfrac12`, and has no zero mode at all.  :func:`fermion_mode_spectrum` gets the
mode numbers this way, from a snapshot, rather than from
:func:`stringsim.superstring.rns.fermion_mode_numbers`, which asserts them.

The same sign shows up in time: after :math:`\Delta\tau = 2\pi` the profile has
gone once around, so :math:`\psi(\tau + 2\pi) = \eta\,\psi(\tau)`.  An NS
fermion comes back as **minus itself** and needs :math:`4\pi` to return.  At
Courant number 1 the scheme is an exact shift, so this comes out at round-off.

**The closed string has two independent fields**, each with its own sign, hence
the four spin structures -- see :data:`SPIN_STRUCTURES`.

**Supersymmetry picks the sector.**  Under :math:`\delta\psi_\mp \propto
\partial_\mp X` with a constant parameter, Neumann data (:math:`\partial_\sigma
X = 0` at the ends, so :math:`\partial_+ X = \partial_- X` there) is carried
into fermion data with :math:`\psi_+ = \psi_-` at *both* ends: the Ramond
condition.  :func:`susy_variation` builds it and
:func:`boundary_residual` shows it satisfies R and fails NS -- the NS sector
needs an antiperiodic supersymmetry parameter, and here that is measured rather
than asserted.

Scope
-----
:math:`\psi` is evolved as a real commuting field.  That is exact for
everything above -- the equation of motion, the boundary conditions, the mode
numbers, the period doubling, and the supercurrent :math:`G_- = \psi_- \cdot
\partial_- X`, which is bilinear in *different* fields.  It is not the Grassmann
field, so the fermion bilinear in :math:`T_{\pm\pm}` and the anticommutator
algebra are outside what a c-number grid can represent; those stay algebraic, in
:mod:`stringsim.superstring.rns`.

Scheme
------
Lax-Wendroff, second order, stable for ``|courant| <= 1`` and an exact
one-cell shift at ``courant = 1``.  Unlike the leapfrog in
:mod:`stringsim.classical.evolve` it is slightly dissipative below 1, which
:func:`norm` will show: that is the scheme, not the string.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rns import Sector

__all__ = [
    "SPIN_STRUCTURES",
    "sector_twist",
    "FermionEvolution",
    "unfold",
    "fold",
    "boundary_residual",
    "open_fermion_modes",
    "evolve_open_fermion",
    "evolve_closed_fermion",
    "fermion_mode_spectrum",
    "reconstruct_from_modes",
    "zero_mode",
    "norm",
    "susy_variation",
    "supercurrent",
    "supercurrent_residual",
]

#: The four closed-string spin structures, as ``name -> (twist_-, twist_+)``.
SPIN_STRUCTURES = {
    "NS-NS": (-1, -1),
    "NS-R": (-1, +1),
    "R-NS": (+1, -1),
    "R-R": (+1, +1),
}

_TWO_PI = 2.0 * math.pi


def sector_twist(sector: Sector) -> int:
    """``+1`` for Ramond (periodic), ``-1`` for Neveu-Schwarz (antiperiodic)."""
    return 1 if Sector(sector) is Sector.R else -1


def _check_twist(twist: int) -> int:
    twist = int(twist)
    if twist not in (1, -1):
        raise ValueError(f"twist must be +1 or -1, got {twist}")
    return twist


def _shift(psi: np.ndarray, k: int, twist: int) -> np.ndarray:
    """``psi[(j + k) mod N]`` with the sign the twisted periodicity implies."""
    out = np.roll(psi, -k, axis=0)
    if twist == -1 and k:
        if k > 0:
            out[-k:] *= -1.0
        else:
            out[:-k] *= -1.0
    return out


def _lax_wendroff(psi: np.ndarray, lam: float, twist: int) -> np.ndarray:
    """One step of ``d_tau psi = -c d_sigma psi`` with ``lam = c dtau / h``."""
    up, down = _shift(psi, 1, twist), _shift(psi, -1, twist)
    return psi - 0.5 * lam * (up - down) + 0.5 * lam**2 * (up - 2.0 * psi + down)


@dataclass(frozen=True)
class FermionEvolution:
    """The result of :func:`evolve_open_fermion` or :func:`evolve_closed_fermion`.

    Attributes
    ----------
    tau:
        Times, shape ``(n_steps + 1,)``.
    sigma:
        Grid points of the *physical* string: ``[0, pi]`` inclusive for the open
        string, ``[0, 2 pi)`` for the closed one.
    psi_minus, psi_plus:
        The two components, shape ``(n_steps + 1, n_points, n_target)``.
        ``psi_minus`` moves toward larger ``sigma``.
    twists:
        ``(twist_-, twist_+)``.  Equal for the open string, where a single sign
        relates the two ends.
    boundary:
        ``"open"`` or ``"closed"``.
    unfolded:
        Open string only: the single chiral field on the doubled circle, shape
        ``(n_steps + 1, 2 * (n_points - 1), n_target)``.  ``None`` for the
        closed string, which has no folding to undo.
    """

    tau: np.ndarray
    sigma: np.ndarray
    psi_minus: np.ndarray
    psi_plus: np.ndarray
    twists: tuple[int, int]
    boundary: str
    unfolded: np.ndarray | None = None

    @property
    def courant(self) -> float:
        """``dtau / dsigma``."""
        return float((self.tau[1] - self.tau[0]) / (self.sigma[1] - self.sigma[0]))

    @property
    def sector(self) -> Sector:
        """The open-string sector.  Raises for the closed string, which has two."""
        if self.boundary != "open":
            raise ValueError("a closed string has a spin structure, not a single sector")
        return Sector.R if self.twists[0] == 1 else Sector.NS

    def at(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """``(psi_-, psi_+)`` at one time, each shape ``(n_points, n_target)``."""
        return self.psi_minus[index], self.psi_plus[index]


def unfold(
    psi_minus: np.ndarray, psi_plus: np.ndarray, twist: int, tolerance: float = 1e-9
) -> np.ndarray:
    r"""Fold the open string open into one chiral field on ``[0, 2 pi)``.

    ``psi_minus`` and ``psi_plus`` are given on ``n + 1`` points spanning
    ``[0, pi]`` inclusive; the result has ``2n`` points spanning ``[0, 2 pi)``.
    The two endpoint conditions are checked, not assumed --
    :func:`boundary_residual` reports them, and ``tolerance`` is the relative
    slack allowed.  Data built analytically satisfies them to round-off; data
    that came out of another numerical evolution, such as
    :func:`susy_variation`, misses by its own discretisation error and needs a
    tolerance to match.
    """
    twist = _check_twist(twist)
    minus = np.asarray(psi_minus, float)
    plus = np.asarray(psi_plus, float)
    if minus.ndim != 2 or minus.shape != plus.shape:
        raise ValueError(
            f"psi_minus and psi_plus must both have shape (n_points, n_target); "
            f"got {minus.shape} and {plus.shape}"
        )
    at_zero, at_pi = boundary_residual(minus, plus, twist)
    scale = max(1.0, float(np.max(np.abs(minus))), float(np.max(np.abs(plus))))
    if max(at_zero, at_pi) > tolerance * scale:
        raise ValueError(
            "initial data does not satisfy the boundary conditions of this sector "
            f"(residual {at_zero:.2e} at sigma = 0, {at_pi:.2e} at sigma = pi, "
            f"twist {twist:+d}); build admissible data with open_fermion_modes"
        )
    # psi(sigma) = psi_-(sigma) on [0, pi], and twist * psi_+(2 pi - sigma) beyond.
    return np.concatenate([minus[:-1], twist * plus[:0:-1]], axis=0)


def fold(psi: np.ndarray, twist: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`unfold`: recover ``(psi_-, psi_+)`` on ``[0, pi]``."""
    twist = _check_twist(twist)
    psi = np.asarray(psi, float)
    n_points = psi.shape[0]
    if n_points % 2:
        raise ValueError(f"the doubled circle needs an even number of points, got {n_points}")
    half = n_points // 2
    # psi_+(sigma_j) = twist * psi(sigma_{N-j}), with psi(sigma_N) = twist * psi(0).
    extended = np.concatenate([psi, twist * psi[:1]], axis=0)
    minus = psi[: half + 1].copy()
    plus = twist * extended[n_points : n_points - half - 1 : -1]
    return minus, np.ascontiguousarray(plus)


def boundary_residual(
    psi_minus: np.ndarray, psi_plus: np.ndarray, twist: int
) -> tuple[float, float]:
    r"""How badly the data misses ``psi_+(0) = psi_-(0)`` and ``psi_+(pi) = eta psi_-(pi)``.

    Returns the two residuals as a pair.  The second is the one that knows about
    the sector: the same data cannot be small in both signs unless it vanishes at
    the end.
    """
    twist = _check_twist(twist)
    minus, plus = np.asarray(psi_minus, float), np.asarray(psi_plus, float)
    return (
        float(np.max(np.abs(plus[0] - minus[0]))),
        float(np.max(np.abs(plus[-1] - twist * minus[-1]))),
    )


def open_fermion_modes(
    modes: dict[float, np.ndarray], sector: Sector, tau: float, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""The exact open-string solution built from mode amplitudes.

    .. math::
       \psi_\mp(\tau, \sigma)
         = \mathrm{Re} \sum_r b_r \, e^{-ir(\tau \mp \sigma)}

    with :math:`r` integral in the Ramond sector and half-integral in
    Neveu-Schwarz; ``modes`` maps :math:`r` to a complex amplitude vector
    :math:`b_r`.  Mode numbers belonging to the other sector are rejected, so
    this both *generates* admissible data for :func:`evolve_open_fermion` and
    provides the exact answer to compare the evolution against.
    """
    twist = sector_twist(sector)
    sigma = np.asarray(sigma, float)
    items = []
    for r, amplitude in modes.items():
        r = float(r)
        doubled = 2.0 * r
        if abs(doubled - round(doubled)) > 1e-12 or r < 0.0:
            raise ValueError(f"mode number {r} is neither integral nor half-integral, or negative")
        integral = abs(r - round(r)) < 1e-12
        if integral != (twist == 1):
            wanted = "integral" if twist == 1 else "half-integral"
            raise ValueError(f"the {Sector(sector).value} sector has {wanted} modes; got r = {r}")
        if twist == -1 and r == 0.0:
            raise ValueError("the NS sector has no zero mode")
        items.append((r, np.asarray(amplitude, complex).reshape(-1)))
    if not items:
        raise ValueError("give at least one mode")
    width = items[0][1].size
    if any(vec.size != width for _, vec in items):
        raise ValueError("all amplitudes must have the same number of components")

    minus = np.zeros(sigma.shape + (width,))
    plus = np.zeros_like(minus)
    for r, vec in items:
        minus += np.real(np.exp(-1j * r * (tau - sigma))[..., None] * vec)
        plus += np.real(np.exp(-1j * r * (tau + sigma))[..., None] * vec)
    return minus, plus


def _run(psi: np.ndarray, lam: float, twist: int, n_steps: int) -> np.ndarray:
    out = np.empty((n_steps + 1,) + psi.shape)
    out[0] = psi
    for k in range(1, n_steps + 1):
        psi = _lax_wendroff(psi, lam, twist)
        out[k] = psi
    return out


def evolve_open_fermion(
    psi_minus0: np.ndarray,
    psi_plus0: np.ndarray,
    *,
    sector: Sector = Sector.NS,
    n_steps: int = 200,
    courant: float = 1.0,
    tolerance: float = 1e-9,
) -> FermionEvolution:
    """Evolve the open-string fermion by transporting the unfolded chiral field.

    Parameters
    ----------
    psi_minus0, psi_plus0:
        Initial data on ``n + 1`` points spanning ``[0, pi]``, shape
        ``(n_points, n_target)``.  It must satisfy the boundary conditions of
        ``sector``; :func:`open_fermion_modes` builds data that does.
    sector:
        :attr:`~stringsim.superstring.rns.Sector.NS` or ``.R``.
    n_steps, courant:
        Time steps and ``dtau / dsigma``; ``courant = 1`` is an exact shift.
    tolerance:
        Relative slack in the boundary check; see :func:`unfold`.

    Returns
    -------
    FermionEvolution
        With :attr:`~FermionEvolution.unfolded` filled in.
    """
    if not 0 < courant <= 1.0:
        raise ValueError(f"courant must lie in (0, 1], got {courant}")
    twist = sector_twist(sector)
    doubled = unfold(psi_minus0, psi_plus0, twist, tolerance)
    n_points = doubled.shape[0]
    h = _TWO_PI / n_points
    history = _run(doubled, courant, twist, n_steps)

    folded = [fold(snapshot, twist) for snapshot in history]
    return FermionEvolution(
        tau=np.arange(n_steps + 1) * (courant * h),
        sigma=np.linspace(0.0, math.pi, n_points // 2 + 1),
        psi_minus=np.stack([m for m, _ in folded]),
        psi_plus=np.stack([p for _, p in folded]),
        twists=(twist, twist),
        boundary="open",
        unfolded=history,
    )


def evolve_closed_fermion(
    psi_minus0: np.ndarray,
    psi_plus0: np.ndarray,
    *,
    spin_structure: str | tuple[int, int] = "NS-NS",
    n_steps: int = 200,
    courant: float = 1.0,
) -> FermionEvolution:
    """Evolve the two independent closed-string fermions.

    ``psi_minus0`` and ``psi_plus0`` live on ``n`` points spanning
    ``[0, 2 pi)``.  Each carries its own periodicity, and the four combinations
    are the spin structures of :data:`SPIN_STRUCTURES`; a name from that mapping
    or an explicit ``(twist_-, twist_+)`` pair may be given.  Nothing couples the
    two, which is exactly why a closed string can be Ramond on one side and
    Neveu-Schwarz on the other.
    """
    if not 0 < courant <= 1.0:
        raise ValueError(f"courant must lie in (0, 1], got {courant}")
    if isinstance(spin_structure, str):
        if spin_structure not in SPIN_STRUCTURES:
            raise ValueError(
                f"unknown spin structure {spin_structure!r}; "
                f"expected one of {sorted(SPIN_STRUCTURES)}"
            )
        twists = SPIN_STRUCTURES[spin_structure]
    else:
        twists = tuple(_check_twist(t) for t in spin_structure)

    minus = np.asarray(psi_minus0, float)
    plus = np.asarray(psi_plus0, float)
    if minus.ndim != 2 or minus.shape != plus.shape:
        raise ValueError(
            f"psi_minus0 and psi_plus0 must both have shape (n_points, n_target); "
            f"got {minus.shape} and {plus.shape}"
        )
    n_points = minus.shape[0]
    h = _TWO_PI / n_points
    return FermionEvolution(
        tau=np.arange(n_steps + 1) * (courant * h),
        sigma=np.arange(n_points) * h,
        psi_minus=_run(minus, courant, twists[0], n_steps),
        psi_plus=_run(plus, -courant, twists[1], n_steps),
        twists=(twists[0], twists[1]),
        boundary="closed",
    )


def _mode_numbers(twist: int, n_modes: int) -> np.ndarray:
    """``0, 1, 2, ...`` when periodic; ``1/2, 3/2, ...`` when antiperiodic."""
    return np.arange(n_modes) + (0.0 if _check_twist(twist) == 1 else 0.5)


def fermion_mode_spectrum(field: np.ndarray, twist: int, n_modes: int = 8) -> np.ndarray:
    r"""Read the mode content off a snapshot of the chiral field.

    ``field`` is the field on ``n`` points spanning ``[0, 2 pi)`` -- the
    :attr:`~FermionEvolution.unfolded` array for an open string, or one
    component of a closed one.  Returns

    .. math::  c_r = \frac{1}{N}\sum_j \psi_j e^{-ir\sigma_j}

    for the ``n_modes`` non-negative mode numbers the periodicity allows, shape
    ``(n_modes, n_target)``.  For a real field the physical amplitude is
    :math:`2c_r` (and :math:`c_0` for the Ramond zero mode, which has no partner).

    These basis functions are orthogonal on the grid *for the matching twist*.
    Projecting onto the wrong sector's mode numbers is not a small error but a
    different vector space, which is what :func:`reconstruct_from_modes` makes
    visible.
    """
    field = np.asarray(field, float)
    n_points = field.shape[0]
    sigma = np.arange(n_points) * (_TWO_PI / n_points)
    basis = np.exp(-1j * np.outer(_mode_numbers(twist, n_modes), sigma))
    return (basis @ field) / n_points


def reconstruct_from_modes(coefficients: np.ndarray, twist: int, n_points: int) -> np.ndarray:
    """Rebuild the field from :func:`fermion_mode_spectrum`, on ``n_points``."""
    coefficients = np.asarray(coefficients, complex)
    numbers = _mode_numbers(twist, coefficients.shape[0])
    sigma = np.arange(n_points) * (_TWO_PI / n_points)
    phases = np.exp(1j * np.outer(numbers, sigma))
    weights = np.where(numbers == 0.0, 1.0, 2.0)[:, None]
    return np.real(phases.T @ (weights * coefficients))


def zero_mode(field: np.ndarray, twist: int) -> np.ndarray:
    r"""The non-propagating part of the field, or zeros when there is none.

    For a periodic (Ramond) field this is the mean, the :math:`r = 0` mode: a
    constant that the transport equation moves nowhere.  It is the Clifford zero
    mode :math:`\psi_0^\mu` whose algebra makes the Ramond ground state a
    spinor.  An antiperiodic (NS) field has no such mode, and this returns zeros
    -- its mean over the circle is a discretisation artefact, not a state.
    """
    field = np.asarray(field, float)
    if _check_twist(twist) == -1:
        return np.zeros(field.shape[1:])
    return field.mean(axis=0)


def norm(field: np.ndarray) -> float:
    r"""``(1 / 2 pi) \int \psi \cdot \psi \, d\sigma`` on the ``[0, 2 pi)`` grid.

    Conserved exactly by the transport, and by the scheme at ``courant = 1``.
    Below 1 Lax-Wendroff damps short wavelengths, so this decays slightly: a
    property of the discretisation, and it shrinks like ``h^2``.
    """
    field = np.asarray(field, float)
    return float(np.mean(np.sum(field**2, axis=-1)))


def susy_variation(x_evolution, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    r"""Turn a bosonic snapshot into fermion data by a supersymmetry variation.

    With a constant parameter the variation is :math:`\delta\psi_\mp \propto
    \partial_\mp X`, where :math:`\partial_\mp = \tfrac12(\partial_\tau \mp
    \partial_\sigma)`.  ``x_evolution`` is a
    :class:`stringsim.classical.evolve.Evolution`; the time derivative is
    centred, so ``index`` must not be an endpoint of the run.

    Returns ``(psi_-, psi_+)`` on the same ``[0, pi]`` grid.  Feed the result to
    :func:`boundary_residual`: for Neumann ``X`` it satisfies the Ramond
    condition and violates the Neveu-Schwarz one, which is the statement that
    unbroken worldsheet supersymmetry with a periodic parameter lands in R.
    """
    tau, sigma, X = x_evolution.tau, x_evolution.sigma, x_evolution.X
    if not 0 < index < len(tau) - 1:
        raise ValueError(f"index must be interior to the run, got {index} of {len(tau)}")
    d_tau = (X[index + 1] - X[index - 1]) / (tau[index + 1] - tau[index - 1])
    d_sigma = np.gradient(X[index], sigma, axis=0, edge_order=2)
    return 0.5 * (d_tau - d_sigma), 0.5 * (d_tau + d_sigma)


def supercurrent(x_evolution, fermion_evolution: FermionEvolution) -> np.ndarray:
    r"""``G_- = psi_- . d_- X`` on the shared ``(tau, sigma)`` grid.

    Both evolutions must use the same ``sigma`` grid and the same step, which
    happens automatically when the fermion runs on ``2n`` points of the doubled
    circle and the boson on ``n + 1`` points of ``[0, pi]`` at the same Courant
    number.  Returns shape ``(n_steps - 1, n_points)``, dropping the first and
    last time so the ``tau`` derivative can be centred.
    """
    if fermion_evolution.boundary != "open":
        raise ValueError("the supercurrent here is built for the open string")
    sigma, X = x_evolution.sigma, x_evolution.X
    if len(sigma) != len(fermion_evolution.sigma) or not np.allclose(
        sigma, fermion_evolution.sigma
    ):
        raise ValueError(
            f"the two evolutions disagree about sigma: {len(sigma)} points against "
            f"{len(fermion_evolution.sigma)}"
        )
    steps = min(len(x_evolution.tau), len(fermion_evolution.tau))
    tau = x_evolution.tau[:steps]
    if not np.allclose(tau, fermion_evolution.tau[:steps]):
        raise ValueError("the two evolutions disagree about tau; use the same Courant number")

    d_tau = (X[2:steps] - X[0 : steps - 2]) / (tau[2] - tau[0])
    d_sigma = np.gradient(X[1 : steps - 1], sigma, axis=1, edge_order=2)
    d_minus = 0.5 * (d_tau - d_sigma)
    return np.sum(fermion_evolution.psi_minus[1 : steps - 1] * d_minus, axis=-1)


def supercurrent_residual(x_evolution, fermion_evolution: FermionEvolution) -> float:
    r"""``max |d_+ G_-|`` over the interior of the grid.

    The supercurrent is chiral: :math:`\psi_-` and :math:`\partial_- X` both
    depend on :math:`\tau - \sigma` alone, so :math:`\partial_+ G_- = 0`
    identically.  What is left here is discretisation error, and it falls like
    ``h^2``.  This is the fermionic counterpart of the Virasoro residual in
    :mod:`stringsim.classical.constraints`.
    """
    g = supercurrent(x_evolution, fermion_evolution)
    sigma = x_evolution.sigma
    dt = float(x_evolution.tau[1] - x_evolution.tau[0])
    d_tau = (g[2:] - g[:-2]) / (2.0 * dt)
    d_sigma = np.gradient(g[1:-1], sigma, axis=1, edge_order=2)
    interior = slice(1, -1)
    return float(np.max(np.abs(0.5 * (d_tau + d_sigma))[:, interior]))
