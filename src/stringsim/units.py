"""Conventions and unit conversions for the bosonic string.

Signature is mostly-plus, ``eta = diag(-1, +1, ..., +1)``.  The Regge slope
``alpha'`` sets every scale; setting ``alpha_prime = 1`` makes masses
dimensionless in string units.

Reference conventions: Polchinski, *String Theory* Vol. I, ch. 1; Zwiebach,
*A First Course in String Theory*, 2nd ed., ch. 12-13.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Conventions", "minkowski", "dot"]


@dataclass(frozen=True)
class Conventions:
    """Global choices: the Regge slope and the spacetime dimension.

    Parameters
    ----------
    alpha_prime:
        Regge slope ``alpha'`` in units where ``hbar = c = 1``.  The string
        tension is ``T = 1 / (2 pi alpha')`` and the string length is
        ``l_s = sqrt(alpha')``.
    dim:
        Number of spacetime dimensions ``D``.  The critical bosonic value is
        26; nothing here forces it, so non-critical values may be explored, but
        the quantum mass formula is only consistent at ``D = 26``.
    """

    alpha_prime: float = 1.0
    dim: int = 26

    def __post_init__(self) -> None:
        if self.alpha_prime <= 0:
            raise ValueError("alpha_prime must be positive")
        if self.dim < 3:
            raise ValueError("dim must be at least 3 (need two transverse directions)")

    @property
    def string_length(self) -> float:
        """``l_s = sqrt(alpha')``."""
        return math.sqrt(self.alpha_prime)

    @property
    def tension(self) -> float:
        """``T = 1 / (2 pi alpha')``, the mass per unit length."""
        return 1.0 / (2.0 * math.pi * self.alpha_prime)

    @property
    def transverse_dim(self) -> int:
        """``D - 2``: the number of physical (light-cone transverse) directions."""
        return self.dim - 2

    @property
    def metric(self) -> np.ndarray:
        """The flat metric ``diag(-1, +1, ..., +1)`` as a 1-D array of signs."""
        return minkowski(self.dim)


def minkowski(dim: int) -> np.ndarray:
    """Diagonal of the mostly-plus Minkowski metric in ``dim`` dimensions."""
    eta = np.ones(dim)
    eta[0] = -1.0
    return eta


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Lorentz-invariant contraction over the *last* axis, mostly-plus signature.

    Works on stacks: ``a`` and ``b`` of shape ``(..., D)`` give shape ``(...)``.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(f"dimension mismatch: {a.shape[-1]} vs {b.shape[-1]}")
    eta = minkowski(a.shape[-1])
    return np.sum(a * b * eta, axis=-1)
