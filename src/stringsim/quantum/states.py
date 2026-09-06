r"""Which particle is which vibration.

A state of the string is a list of oscillators acting on the ground state, and
the particle it describes is fixed by how those oscillators transform under the
little group -- ``SO(D-2)`` for a massless state, ``SO(D-1)`` for a massive one.
That single fact turns the degeneracies of :mod:`stringsim.quantum.partition`
into a particle table.

The two results worth remembering:

* **Open string, level 1.**  ``alpha_{-1}^i |0>`` is a vector of ``SO(24)``:
  24 states, massless, spin 1.  A photon.  It *has* to be massless, because a
  massive vector in ``D = 26`` would need 25 polarisations and there are only
  24 oscillators to make them from -- which is the Lorentz-invariance argument
  that forces ``a = 1`` and ``D = 26``.

* **Closed string, level 1.**  ``alpha_{-1}^i \tilde\alpha_{-1}^j |0>`` is
  ``24 \otimes 24 = 576`` states, and the product splits into

  ==========================  =====  =========================================
  symmetric traceless          299   the **graviton** :math:`g_{\mu\nu}`
  antisymmetric                276   the **Kalb-Ramond** field :math:`B_{\mu\nu}`
  trace                          1   the **dilaton** :math:`\Phi`
  ==========================  =====  =========================================

  The graviton is not put in by hand: every closed string has this state, which
  is why string theory *contains* gravity rather than being compatible with it.

Everything below computes those dimensions from ``D``, so changing the
dimension changes the table rather than contradicting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

__all__ = [
    "ParticleContent",
    "sym_traceless_dim",
    "antisym_dim",
    "little_group",
    "open_level_content",
    "closed_massless_content",
]


def sym_traceless_dim(n: int, rank: int) -> int:
    """Dimension of the rank-``rank`` symmetric traceless tensor of ``SO(n)``."""
    if rank < 0:
        raise ValueError("rank must be non-negative")
    if rank == 0:
        return 1
    if rank == 1:
        return n
    return comb(n + rank - 1, rank) - comb(n + rank - 3, rank - 2)


def antisym_dim(n: int, rank: int) -> int:
    """Dimension of the rank-``rank`` antisymmetric tensor of ``SO(n)``."""
    if not 0 <= rank <= n:
        raise ValueError(f"rank must lie in [0, {n}]")
    return comb(n, rank)


@dataclass(frozen=True)
class ParticleContent:
    """One irreducible piece of a mass level."""

    name: str
    tensor: str
    dimension: int
    spin: str
    little_group: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.name:<16s} {self.tensor:<28s} {self.dimension:>8d}  spin {self.spin}"


def little_group(alpha_m2: float, dim: int = 26) -> tuple[str, int]:
    """The little group and the size of its vector representation.

    * massless: ``('SO(D-2)', D-2)`` -- transverse polarisations only;
    * massive:  ``('SO(D-1)', D-1)`` -- the rest frame's rotations;
    * tachyonic: ``('SO(D-2,1)', D-1)`` -- there is no rest frame, so the
      stabiliser of a spacelike momentum is non-compact.  Its vector index
      still runs over ``D-1`` values, which is what the dimension counting
      needs, but calling it ``SO(D-1)`` would be wrong.

    The massless/massive mismatch is exactly why the string's level-1 states
    are forced to be massless: ``D-2`` oscillators cannot fill a massive
    vector.
    """
    if alpha_m2 == 0.0:
        return f"SO({dim - 2})", dim - 2
    if alpha_m2 < 0.0:
        return f"SO({dim - 2},1)", dim - 1
    return f"SO({dim - 1})", dim - 1


def open_level_content(n: int, dim: int = 26) -> list[ParticleContent]:
    """Named ``SO`` content of open-string level ``n``, for ``n <= 3``.

    Raises ``NotImplementedError`` above level 3: the decompositions exist but
    grow into long lists of mixed-symmetry tableaux, and quoting them without
    deriving them would be guesswork.  The degeneracy itself is available at
    any level from
    :func:`~stringsim.quantum.spectrum.open_bosonic_spectrum`.
    """
    alpha_m2 = float(n - (dim - 2) / 24.0)
    group, m = little_group(alpha_m2, dim)
    if n == 0:
        return [ParticleContent("tachyon", "scalar", 1, "0", group)]
    if n == 1:
        return [ParticleContent("photon", "vector A_mu", m, "1", group)]
    if n == 2:
        return [
            ParticleContent(
                "massive spin 2", "symmetric traceless", sym_traceless_dim(m, 2), "2", group
            )
        ]
    if n == 3:
        return [
            ParticleContent(
                "massive spin 3", "symmetric traceless", sym_traceless_dim(m, 3), "3", group
            ),
            ParticleContent("massive 2-form", "antisymmetric", antisym_dim(m, 2), "1", group),
        ]
    raise NotImplementedError(
        f"named content is only tabulated up to level 3 (asked for {n}); "
        "use open_bosonic_spectrum for the degeneracy"
    )


def closed_massless_content(dim: int = 26) -> list[ParticleContent]:
    r"""The massless closed-string level: graviton, Kalb-Ramond field, dilaton.

    The three dimensions sum to ``(D-2)^2``, the size of the
    ``alpha_{-1} \tilde\alpha_{-1}`` product, which the test suite checks.
    """
    m = dim - 2
    group = f"SO({m})"
    return [
        ParticleContent(
            "graviton", "symmetric traceless g_mu nu", sym_traceless_dim(m, 2), "2", group
        ),
        ParticleContent("Kalb-Ramond", "antisymmetric B_mu nu", antisym_dim(m, 2), "1", group),
        ParticleContent("dilaton", "trace Phi", 1, "0", group),
    ]
