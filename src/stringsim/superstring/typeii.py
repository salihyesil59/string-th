r"""Type IIA and IIB: the massless spectrum, and which D-branes each theory has.

A closed superstring has two sets of worldsheet fermions, one for each moving
direction, and each set independently chooses a sector.  The massless level is
therefore a product of four pieces,

.. math::
   (8_v \oplus 8_s) \otimes (8_v \oplus 8_{s'}) = 256 \text{ states},

and which spinor :math:`8_{s'}` is -- the same chirality as :math:`8_s` or the
opposite -- is the *only* difference between the two theories:

* **IIB** takes the same chirality on both sides.  Chiral, and its Ramond-Ramond
  fields are forms of even rank.
* **IIA** takes opposite chiralities.  Non-chiral, and its RR fields have odd
  rank.

Everything else follows from four ``SO(8)`` tensor products, and every dimension
below is computed from a binomial coefficient rather than written down:

.. math::
   8_v \otimes 8_v = 1 \oplus 28 \oplus 35_v, \qquad
   8_s \otimes 8_s = 1 \oplus 28 \oplus 35_s,

.. math::
   8_s \otimes 8_c = 8_v \oplus 56_v, \qquad
   8_v \otimes 8_s = 8_c \oplus 56_s .

Both theories end up with 128 bosons and 128 fermions, which is the massless
shadow of the level-by-level equality that
:func:`stringsim.superstring.rns.supersymmetry_deficit` checks.

**The RR ranks decide the branes.**  A Dp-brane is charged under a rank-``p+1``
RR potential, so IIA's odd-rank forms give branes of even ``p`` and IIB's
even-rank forms give odd ``p``.  :func:`stable_brane_ranks` returns those lists,
and they are what :mod:`stringsim.branes.dbrane` computes tensions for.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

__all__ = [
    "TRANSVERSE",
    "form_dimension",
    "Field",
    "Sector",
    "vector_vector",
    "spinor_spinor",
    "vector_spinor",
    "type_ii_massless",
    "massless_counts",
    "stable_brane_ranks",
    "rr_form_ranks",
    "open_superstring_massless",
]

TRANSVERSE = 8
"""``D - 2`` in ten dimensions: the little group of a massless state is ``SO(8)``."""


def form_dimension(rank: int, transverse: int = TRANSVERSE) -> int:
    r"""Number of components of a rank-``p`` antisymmetric form of ``SO(n)``.

    ``comb(n, p)``.  At ``n = 8`` this gives ``1, 8, 28, 56, 70`` for
    ``p = 0..4``; the middle one is where self-duality can halve 70 to 35.
    """
    if not 0 <= rank <= transverse:
        raise ValueError(f"rank must lie in [0, {transverse}]")
    return comb(transverse, rank)


def _graviton_dimension(transverse: int = TRANSVERSE) -> int:
    """Symmetric traceless two-tensor: ``n(n+1)/2 - 1``, which is 35 at ``n = 8``."""
    return transverse * (transverse + 1) // 2 - 1


@dataclass(frozen=True)
class Field:
    """One irreducible piece of the massless spectrum."""

    name: str
    rep: str
    dimension: int
    statistics: str

    @property
    def is_boson(self) -> bool:
        return self.statistics == "boson"

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.name:<24s} {self.rep:<6s} {self.dimension:>4d}  {self.statistics}"


@dataclass(frozen=True)
class Sector:
    """One of the four products, e.g. ``NS-NS``."""

    name: str
    fields: tuple[Field, ...]

    @property
    def dimension(self) -> int:
        """Total states, which must be 64 for every sector."""
        return sum(field.dimension for field in self.fields)

    @property
    def statistics(self) -> str:
        """``'boson'`` or ``'fermion'`` -- a sector is never mixed."""
        kinds = {field.statistics for field in self.fields}
        if len(kinds) != 1:  # pragma: no cover - a bug guard
            raise RuntimeError(f"sector {self.name} mixes statistics: {kinds}")
        return kinds.pop()

    def __str__(self) -> str:  # pragma: no cover - display only
        rows = "\n".join(f"    {field}" for field in self.fields)
        return f"  {self.name} ({self.dimension} states, {self.statistics}s)\n{rows}"


def vector_vector(transverse: int = TRANSVERSE) -> tuple[Field, ...]:
    r"""``8_v x 8_v = 1 + 28 + 35_v``: the NS-NS sector, identical in IIA and IIB."""
    return (
        Field("graviton", "35v", _graviton_dimension(transverse), "boson"),
        Field("Kalb-Ramond B", "28", form_dimension(2, transverse), "boson"),
        Field("dilaton", "1", 1, "boson"),
    )


def spinor_spinor(same_chirality: bool, transverse: int = TRANSVERSE) -> tuple[Field, ...]:
    r"""The Ramond-Ramond sector.

    ``same_chirality`` (IIB): :math:`8_s \otimes 8_s = 1 \oplus 28 \oplus 35_s`,
    i.e. potentials ``C_0``, ``C_2`` and a **self-dual** ``C_4`` -- self-duality
    is why the 70 components of a 4-form appear as 35.

    Otherwise (IIA): :math:`8_s \otimes 8_c = 8_v \oplus 56_v`, i.e. ``C_1`` and
    ``C_3``.
    """
    if same_chirality:
        return (
            Field("RR scalar C_0", "1", form_dimension(0, transverse), "boson"),
            Field("RR 2-form C_2", "28", form_dimension(2, transverse), "boson"),
            Field(
                "RR self-dual C_4",
                "35s",
                form_dimension(4, transverse) // 2,
                "boson",
            ),
        )
    return (
        Field("RR 1-form C_1", "8v", form_dimension(1, transverse), "boson"),
        Field("RR 3-form C_3", "56v", form_dimension(3, transverse), "boson"),
    )


def vector_spinor(label: str, transverse: int = TRANSVERSE) -> tuple[Field, ...]:
    r"""``8_v x 8_s = 8_c + 56_s``: a gravitino and a dilatino, 64 fermionic states."""
    return (
        Field(f"gravitino ({label})", "56", form_dimension(3, transverse), "fermion"),
        Field(f"dilatino ({label})", "8", form_dimension(1, transverse), "fermion"),
    )


def type_ii_massless(kind: str = "IIB", transverse: int = TRANSVERSE) -> list[Sector]:
    """The four massless sectors of type IIA or type IIB.

    ``kind`` is ``"IIA"`` or ``"IIB"``.  The NS-NS and the two mixed sectors are
    the same in both; only R-R differs, and that single choice is what makes IIB
    chiral and IIA not.
    """
    kind = kind.upper()
    if kind not in ("IIA", "IIB"):
        raise ValueError("kind must be 'IIA' or 'IIB'")
    return [
        Sector("NS-NS", vector_vector(transverse)),
        Sector("R-R", spinor_spinor(kind == "IIB", transverse)),
        Sector("NS-R", vector_spinor("NS-R", transverse)),
        Sector("R-NS", vector_spinor("R-NS", transverse)),
    ]


def massless_counts(kind: str = "IIB", transverse: int = TRANSVERSE) -> tuple[int, int]:
    """``(bosons, fermions)`` at the massless level: ``(128, 128)`` for both theories."""
    sectors = type_ii_massless(kind, transverse)
    bosons = sum(s.dimension for s in sectors if s.statistics == "boson")
    fermions = sum(s.dimension for s in sectors if s.statistics == "fermion")
    return bosons, fermions


def rr_form_ranks(kind: str = "IIB") -> list[int]:
    """Ranks of the Ramond-Ramond potentials: ``[0, 2, 4]`` for IIB, ``[1, 3]`` for IIA."""
    kind = kind.upper()
    if kind not in ("IIA", "IIB"):
        raise ValueError("kind must be 'IIA' or 'IIB'")
    return [0, 2, 4] if kind == "IIB" else [1, 3]


def stable_brane_ranks(kind: str = "IIB") -> list[int]:
    r"""Which ``Dp``-branes the theory has.

    A ``Dp``-brane carries electric charge under the rank-``p+1`` potential
    ``C_{p+1}``.  Its field strength ``F_{p+2}`` dualises in ten dimensions to
    ``F_{8-p}``, hence to a potential ``C_{7-p}``, which is the electric
    potential of a ``D(6-p)``-brane.  Closing :func:`rr_form_ranks` under
    ``p -> 6 - p`` therefore gives

    * IIA: ``0, 2, 4, 6`` -- even, from ``C_1`` and ``C_3``;
    * IIB: ``-1, 1, 3, 5, 7`` -- odd, from ``C_0``, ``C_2`` and ``C_4``.

    ``p = -1`` is the D-instanton: a point in *time* as well as space, with an
    action rather than a tension, so :func:`stringsim.branes.dbrane.dp_brane_tension`
    will refuse it.  The IIA ``D8`` is likewise absent here on purpose -- it
    couples to a nine-form whose field strength is non-dynamical, and it exists
    only in the massive (Romans) deformation of IIA, which this closure knows
    nothing about.

    Feed the rest to :func:`stringsim.branes.dbrane.dp_brane_tension` for the
    masses.
    """
    ranks: set[int] = set()
    for rank in rr_form_ranks(kind):
        electric = rank - 1
        ranks.add(electric)
        ranks.add(6 - electric)
    return sorted(ranks)


def open_superstring_massless(transverse: int = TRANSVERSE) -> tuple[Field, ...]:
    """The open superstring's massless level: ``8 + 8``, ten-dimensional ``N = 1`` SYM."""
    return (
        Field("gauge boson", "8v", transverse, "boson"),
        Field("gaugino", "8s", transverse, "fermion"),
    )
