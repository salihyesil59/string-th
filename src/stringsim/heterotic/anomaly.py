r"""Green-Schwarz anomaly cancellation, computed rather than quoted.

The heterotic section of this package produces **496** gauge bosons by counting
lattice vectors: sixteen Cartan directions plus 480 roots.  The usual next
sentence is that 496 is separately what anomaly cancellation demands in ten
dimensions, and that two unrelated consistency conditions agreeing is why the
construction was taken seriously.  This module computes the second one.

**The anomaly polynomial is assembled, not looked up.**  In ten dimensions the
hexagon anomaly of a chiral field is the twelve-form part of an index density:

.. math::
   \hat I_{1/2} = \hat A(R)\,\mathrm{ch}(F), \qquad
   \hat I_{3/2} = \hat A(R)\,\bigl[\mathrm{tr}\,e^{iR/2\pi} - 1\bigr],
   \qquad \hat I_{A} = -\tfrac18 L(R),

with :math:`\hat A = \prod (x_j/2)/\sinh(x_j/2)` and
:math:`L = \prod x_j/\tanh x_j`.  Everything here is a power series in the
symmetric functions of the :math:`x_j`, truncated at twelve-form order, with
exact rational coefficients -- see :class:`Form`.

**The conventions are fixed by a check, not by assertion.**  The relative
normalisation of the three densities, and whether ``L`` carries a half-argument,
are exactly the places where a remembered formula goes wrong.  Type IIB
supergravity settles it: two gravitini, two dilatini of the opposite chirality
and one self-dual four-form, and the total anomaly vanishes identically.  That
is three equations -- the coefficients of ``tr R^6``, ``tr R^4 tr R^2`` and
``(tr R^2)^3`` -- with no free parameter, and :func:`type_iib_residual` returns
exactly zero.  Only then is the same machinery pointed at ``N = 1``.

**Then 496 comes out.**  For ``N = 1`` supergravity coupled to super-Yang-Mills
with a gauge group of dimension ``n``, the coefficient of ``tr R^6`` is

.. math::
   \frac{n - 496}{725760} ,

and nothing can cancel it: no gauge field appears in a pure-gravity term.  See
:func:`gravitational_coefficient` and :func:`required_dimension`.

**And the group is pinned twice.**  Killing ``tr R^6`` fixes only the
*dimension*.  A second, independent condition kills ``tr F^6``: the remaining
twelve-form has to factorise as ``X_4 X_8`` so that a ``B \wedge X_8``
counterterm can cancel it, and a surviving sixth-order gauge trace makes that
impossible -- no product of a four-form and an eight-form built from ``tr R^2``,
``tr F^2``, ``tr R^4`` and ``tr F^4`` can produce one.  For ``SO(N)`` the
adjoint trace identity is

.. math::
   \mathrm{Tr}\,F^6 = (N - 32)\,\mathrm{tr}\,F^6
                      + 15\,\mathrm{tr}\,F^2\,\mathrm{tr}\,F^4 ,

so the dangerous term vanishes at ``N = 32`` and nowhere else.  ``dim SO(32) =
496``.  The two conditions come from different halves of the anomaly and land on
the same group.

``SO(26) x SO(19)`` is the test that this has teeth: dimension 496, passes the
gravitational condition, and fails on ``tr F^6``.

Scope: ``SO(N)`` and ``E_8`` factors.  ``SU(N)``, ``Sp(N)`` and the other
exceptional algebras are not implemented, so the scan below is a search over a
family, not a proof of uniqueness.  The two further known solutions,
``E_8 x U(1)^248`` and ``U(1)^496``, are also outside it; neither has a known
string realisation.

Reference: M. B. Green and J. H. Schwarz, *Anomaly cancellations in
supersymmetric D = 10 gauge theory and superstring theory*, Phys. Lett. B
**149** (1984) 117; L. Alvarez-Gaume and E. Witten, *Gravitational anomalies*,
Nucl. Phys. B **234** (1984) 269.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

__all__ = [
    "TOP",
    "Form",
    "symbol",
    "a_hat",
    "hirzebruch",
    "spin_half",
    "spin_three_half",
    "self_dual_tensor",
    "type_iib_residual",
    "GaugeFactor",
    "so",
    "e8",
    "combine",
    "anomaly_polynomial",
    "gravitational_coefficient",
    "required_dimension",
    "sixth_order_terms",
    "Factorisation",
    "factorise",
    "cancels",
    "candidates",
    "scan",
]

#: Highest weight kept.  One unit of weight is a four-form, so ``TOP = 3`` is
#: the twelve-form -- the whole content of a ten-dimensional anomaly.
TOP = 3


def _weight(name: str) -> int:
    """``R2`` and ``F2_1`` weigh 1, ``R4`` weighs 2, ``R6`` weighs 3."""
    head = name.split("_")[0]
    return int(head[-1]) // 2


class Form(dict):
    """A characteristic-class polynomial with exact rational coefficients.

    Keys are monomials -- sorted tuples of ``(symbol, exponent)`` -- and values
    are :class:`~fractions.Fraction`.  Anything above :data:`TOP` is dropped on
    multiplication, since it is beyond the twelve-form and cannot contribute.

    Exact arithmetic is not a luxury here.  The whole result is that certain
    coefficients are *zero*, and in floating point ``496 - n`` would come out as
    ``1e-16`` for a group of the wrong dimension just as readily as for the
    right one.
    """

    @staticmethod
    def weight(monomial) -> int:
        return sum(_weight(s) * e for s, e in monomial)

    def __add__(self, other: Form) -> Form:
        out = Form(self)
        for m, c in other.items():
            total = out.get(m, Fraction(0)) + c
            if total:
                out[m] = total
            else:
                out.pop(m, None)
        return out

    def __sub__(self, other: Form) -> Form:
        return self + other * -1

    def __mul__(self, other) -> Form:
        if not isinstance(other, Form):
            factor = Fraction(other)
            return Form({m: c * factor for m, c in self.items()}) if factor else Form()
        out: dict = {}
        for left, a in self.items():
            for right, b in other.items():
                powers: dict[str, int] = {}
                for s, e in left + right:
                    powers[s] = powers.get(s, 0) + e
                monomial = tuple(sorted(powers.items()))
                if Form.weight(monomial) <= TOP:
                    out[monomial] = out.get(monomial, Fraction(0)) + a * b
        return Form({m: c for m, c in out.items() if c})

    __rmul__ = __mul__

    def __pow__(self, n: int) -> Form:
        out = ONE
        for _ in range(n):
            out = out * self
        return out

    def part(self, weight: int) -> Form:
        """The piece of the given weight; ``part(3)`` is the twelve-form."""
        return Form({m: c for m, c in self.items() if Form.weight(m) == weight})

    def coefficient(self, *powers: tuple[str, int]) -> Fraction:
        """Coefficient of one monomial, written as ``("R2", 2), ("R4", 1)``."""
        return self.get(tuple(sorted(p for p in powers if p[1])), Fraction(0))

    def substitute(self, name: str, value: Form) -> Form:
        """Replace one symbol everywhere.  Used to set ``tr R^2 -> -X``."""
        out = Form()
        for monomial, c in self.items():
            rest = Form({tuple((s, e) for s, e in monomial if s != name): c})
            power = dict(monomial).get(name, 0)
            out = out + rest * (value**power)
        return out

    def largest(self) -> Fraction:
        """Largest coefficient in absolute value; zero for the zero form."""
        return max((abs(c) for c in self.values()), default=Fraction(0))


ONE = Form({(): Fraction(1)})


def symbol(name: str) -> Form:
    """One generator, e.g. ``symbol("R4")`` for ``tr R^4``."""
    _weight(name)  # validates the name
    return Form({((name, 1),): Fraction(1)})


R2, R4, R6 = symbol("R2"), symbol("R4"), symbol("R6")


def _exp(series: Form) -> Form:
    """``exp`` of a form with no constant term, truncated at :data:`TOP`."""
    term, out = ONE, Form(ONE)
    for k in (1, 2, 3):
        term = term * series
        out = out + term * Fraction(1, (1, 1, 2, 6)[k])
    return out


def a_hat() -> Form:
    r""":math:`\hat A(R) = \prod (x_j/2)/\sinh(x_j/2)`, to twelve-form order.

    From :math:`\log\frac{u}{\sinh u} = -\frac{u^2}{6} + \frac{u^4}{180} -
    \frac{u^6}{2835}` at :math:`u = x/2`, so that the exponent is
    :math:`-p_1/24 + p_2/2880 - p_3/181440` in the power sums
    :math:`p_k = \sum_j x_j^{2k}`.  Those are rewritten with
    :math:`p_1 = -\tfrac12 \mathrm{tr} R^2`, :math:`p_2 = \tfrac12
    \mathrm{tr} R^4`, :math:`p_3 = -\tfrac12 \mathrm{tr} R^6` -- the traces are
    over the vector, and the block form of an antisymmetric matrix gives
    ``tr R^{2k} = 2(-1)^k p_k``.
    """
    return _exp(R2 * Fraction(1, 48) + R4 * Fraction(1, 5760) + R6 * Fraction(1, 362880))


def hirzebruch(half_argument: bool = False) -> Form:
    r""":math:`L(R) = \prod x_j/\tanh x_j`, to twelve-form order.

    From :math:`\log\frac{x}{\tanh x} = \frac{x^2}{3} - \frac{7x^4}{90} +
    \frac{62 x^6}{2835}`, in the same power sums and the same substitution.

    ``half_argument`` builds :math:`\prod (x_j/2)/\tanh(x_j/2)` instead.  That
    variant is the one that is easy to write down by mistake, and it is here so
    the mistake can be *shown* to be one: pass it through
    :func:`type_iib_residual` and the type IIB anomaly no longer cancels.
    """
    if half_argument:
        return _exp(
            R2 * Fraction(-1, 24) + R4 * Fraction(-7, 2880) + R6 * Fraction(-31, 181440)
        )
    return _exp(R2 * Fraction(-1, 6) + R4 * Fraction(-7, 180) + R6 * Fraction(-31, 2835))


def spin_half(character: Form | None = None) -> Form:
    r""":math:`\hat A(R)\,\mathrm{ch}(F)` -- a Weyl fermion in a representation.

    ``character`` is :math:`\mathrm{ch}(F) = \mathrm{Tr}\,e^{iF/2\pi}`; omit it
    for a singlet.
    """
    return a_hat() * (ONE if character is None else character)


def spin_three_half(dim: int = 10) -> Form:
    r""":math:`\hat A(R)\,[\mathrm{tr}\,e^{iR/2\pi} - 1]` -- the gravitino.

    The trace runs over the ``dim``-dimensional vector,
    :math:`\sum_j 2\cosh x_j = D + p_1 + p_2/12 + p_3/360`, and the ``-1``
    removes the ghost.  ``dim`` is the only place the spacetime dimension
    enters explicitly.
    """
    vector = (
        ONE * (dim - 1)
        + R2 * Fraction(-1, 2)
        + R4 * Fraction(1, 24)
        + R6 * Fraction(-1, 720)
    )
    return a_hat() * vector


def self_dual_tensor(half_argument: bool = False) -> Form:
    r""":math:`-\tfrac18 L(R)` -- a self-dual antisymmetric tensor."""
    return hirzebruch(half_argument) * Fraction(-1, 8)


def type_iib_residual(dim: int = 10, half_argument: bool = False) -> Fraction:
    r"""The type IIB total anomaly, which must vanish identically.

    Two gravitini, two dilatini of the opposite chirality and one self-dual
    four-form; the Majorana-Weyl halves cancel between the pairs, leaving
    :math:`\hat I_{3/2} - \hat I_{1/2} + \hat I_A`.

    Returns the largest twelve-form coefficient in absolute value.  It is
    exactly zero, and that single fact fixes every relative normalisation used
    below -- three coefficients vanishing together, with nothing to tune.
    """
    total = spin_three_half(dim) - spin_half() + self_dual_tensor(half_argument)
    return total.part(TOP).largest()


@dataclass(frozen=True)
class GaugeFactor:
    """One simple factor: its dimension and its adjoint trace identities.

    ``traces`` maps ``2, 4, 6`` to :math:`\\mathrm{Tr}\\,F^{2k}` written in the
    factor's own symbols.  For ``SO(N)`` those are the defining-rep traces
    ``F2_i, F4_i, F6_i``; for ``E_8`` the adjoint is the defining
    representation and everything reduces to powers of ``F2_i``.
    """

    name: str
    dimension: int
    traces: dict[int, Form]

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.name} ({self.dimension})"


def so(n: int, index: int = 0) -> GaugeFactor:
    r"""``SO(n)`` with the standard adjoint trace identities.

    .. math::
       \mathrm{Tr}F^2 = (n-2)\,\mathrm{tr}F^2, \quad
       \mathrm{Tr}F^4 = (n-8)\,\mathrm{tr}F^4 + 3(\mathrm{tr}F^2)^2, \quad
       \mathrm{Tr}F^6 = (n-32)\,\mathrm{tr}F^6 + 15\,\mathrm{tr}F^2\,
       \mathrm{tr}F^4 .

    ``index`` distinguishes the factors of a product, giving each its own field
    strength.
    """
    if n < 1:
        raise ValueError("SO(n) needs n >= 1")
    f2, f4, f6 = (symbol(f"F{k}_{index}") for k in (2, 4, 6))
    return GaugeFactor(
        name=f"SO({n})",
        dimension=n * (n - 1) // 2,
        traces={
            2: f2 * (n - 2),
            4: f4 * (n - 8) + f2 * f2 * 3,
            6: f6 * (n - 32) + f2 * f4 * 15,
        },
    )


def e8(index: int = 0) -> GaugeFactor:
    r"""``E_8``: no independent quartic or sextic Casimir.

    .. math::
       \mathrm{Tr}F^4 = \tfrac{1}{100}(\mathrm{Tr}F^2)^2, \qquad
       \mathrm{Tr}F^6 = \tfrac{1}{7200}(\mathrm{Tr}F^2)^3 .

    There is no ``tr F^6`` to cancel, so ``E_8`` passes the second condition
    for free -- and being 248-dimensional, two of them pass the first.
    """
    f2 = symbol(f"F2_{index}")
    return GaugeFactor(
        name="E8",
        dimension=248,
        traces={2: f2, 4: f2 * f2 * Fraction(1, 100), 6: f2**3 * Fraction(1, 7200)},
    )


def combine(factors) -> tuple[int, dict[int, Form]]:
    """Total dimension and total adjoint traces of a product of factors."""
    factors = list(factors)
    total = {k: Form() for k in (2, 4, 6)}
    for factor in factors:
        for k in (2, 4, 6):
            total[k] = total[k] + factor.traces[k]
    return sum(f.dimension for f in factors), total


def anomaly_polynomial(factors, dim: int = 10) -> Form:
    r"""The full ``N = 1`` twelve-form anomaly.

    One gravitino, one dilatino of the opposite chirality, and gaugini in the
    adjoint:

    .. math::
       I_{12} = \tfrac12\bigl[\hat I_{3/2} - \hat I_{1/2}
                + \hat I_{1/2}(\mathrm{adj})\bigr] ,

    the half being Majorana-Weyl.  ``ch(adj) = n + \mathrm{Tr}F^2/2 +
    \mathrm{Tr}F^4/24 + \mathrm{Tr}F^6/720``; odd traces vanish for these
    representations.

    Pass an empty list for pure supergravity, which is what
    :func:`gravitational_coefficient` uses.
    """
    dimension, traces = combine(factors)
    character = (
        ONE * dimension
        + traces[2] * Fraction(1, 2)
        + traces[4] * Fraction(1, 24)
        + traces[6] * Fraction(1, 720)
    )
    total = spin_three_half(dim) - spin_half() + spin_half(character)
    return (total * Fraction(1, 2)).part(TOP)


def gravitational_coefficient(dimension: int, dim: int = 10) -> Fraction:
    r"""Coefficient of ``tr R^6`` for a gauge group of the given dimension.

    Comes out ``(n - 496)/725760``.  Nothing else in the twelve-form contains
    ``tr R^6``, and no counterterm can cancel a pure-gravity term, so this must
    vanish on its own.
    """
    empty = GaugeFactor("trivial", dimension, {k: Form() for k in (2, 4, 6)})
    return anomaly_polynomial([empty], dim).coefficient(("R6", 1))


def required_dimension(dim: int = 10) -> int:
    """The gauge-group dimension that kills ``tr R^6``.  Solved, not quoted.

    ``n`` enters the character linearly, so the coefficient is affine in it.
    That is checked here rather than assumed -- three samples, second difference
    zero -- and then the root is taken exactly in rationals.  Had the field
    content been different, the answer would move with it.

    Raises
    ------
    RuntimeError
        If the coefficient turns out not to be affine in ``n``, or does not
        depend on it, or has no non-negative integer root.
    """
    c0, c1, c2 = (gravitational_coefficient(n, dim) for n in (0, 1, 2))
    if (c2 - c1) != (c1 - c0):
        raise RuntimeError("tr R^6 coefficient is not affine in the gauge dimension")
    slope = c1 - c0
    if slope == 0:
        raise RuntimeError("tr R^6 coefficient does not depend on the gauge dimension")
    root = -c0 / slope
    if root.denominator != 1 or root < 0:
        raise RuntimeError(f"no non-negative integer dimension kills tr R^6 (got {root})")
    return int(root)


def sixth_order_terms(poly: Form) -> Form:
    """The parts of a twelve-form that are a single sixth-order trace.

    ``tr R^6`` and each factor's ``tr F^6``.  These are precisely the terms no
    ``X_4 X_8`` product can reach, so they have to vanish before factorisation
    is even worth asking about.
    """
    return Form(
        {
            m: c
            for m, c in poly.part(TOP).items()
            if len(m) == 1 and m[0][1] == 1 and _weight(m[0][0]) == TOP
        }
    )


@dataclass(frozen=True)
class Factorisation:
    """The result of trying to write the twelve-form as ``X_4 X_8``."""

    x_four: Form
    x_eight: Form
    residual: Fraction
    coefficients: tuple[tuple[str, Fraction], ...]

    @property
    def works(self) -> bool:
        """``True`` when the division is exact."""
        return self.residual == 0

    def __str__(self) -> str:  # pragma: no cover - display only
        terms = ", ".join(f"{k} -> {v}" for k, v in self.coefficients)
        return f"{'factorises' if self.works else 'does not factorise'} ({terms})"


def _divide(poly: Form, x_four: Form) -> tuple[Form, Form]:
    """Divide by ``R2 + L``, treating ``R2`` as the leading variable."""
    quotient, remainder = Form(), Form(poly)
    while True:
        leading = [(m, c) for m, c in remainder.items() if dict(m).get("R2", 0) >= 1]
        if not leading:
            return quotient, remainder
        monomial, coeff = max(leading, key=lambda pair: dict(pair[0]).get("R2", 0))
        powers = dict(monomial)
        powers["R2"] -= 1
        term = Form({tuple(sorted((s, e) for s, e in powers.items() if e)): coeff})
        quotient = quotient + term
        remainder = remainder - term * x_four


def factorise(poly: Form, factors) -> Factorisation:
    r"""Try to write the twelve-form as :math:`X_4 X_8`.

    The four-form is :math:`X_4 = \mathrm{tr}R^2 + \sum_a b_a
    \mathrm{Tr}F_a^2`, and each :math:`b_a` is *determined* by a single
    coefficient -- the one multiplying :math:`\mathrm{Tr}F_a^2\,\mathrm{tr}R^4`
    against the one multiplying :math:`\mathrm{tr}R^2\,\mathrm{tr}R^4`.  Every
    other coefficient is then a prediction, and the residual of the division is
    what checks them.

    Returns the pieces and the residual.  ``b_a`` comes out ``1/30`` for both
    ``SO(32)`` and ``E_8 x E_8``, which is the textbook
    :math:`\mathrm{tr}R^2 - \frac{1}{30}\mathrm{Tr}F^2` -- the sign is a
    convention here, the ``1/30`` is not.
    """
    top = poly.part(TOP)
    reference = top.coefficient(("R2", 1), ("R4", 1))
    if reference == 0:
        raise ValueError("no tr R^2 tr R^4 term to normalise against")

    x_four, coefficients = Form(R2), []
    for slot, factor in enumerate(factors):
        traces = factor.traces[2]
        if not traces:
            continue
        # Tr F_a^2 tr R^4 sits in the twelve-form with coefficient b_a times
        # the tr R^2 tr R^4 one.  Read it off any symbol that Tr F_a^2 uses.
        name, scale = next(iter(traces.items()))
        gauge_symbol = name[0][0]
        value = top.coefficient((gauge_symbol, 1), ("R4", 1)) / (reference * scale)
        coefficients.append((f"{factor.name}[{slot}]", value))
        x_four = x_four + traces * value

    quotient, remainder = _divide(top, x_four)
    return Factorisation(
        x_four=x_four,
        x_eight=quotient,
        residual=remainder.largest(),
        coefficients=tuple(coefficients),
    )


def cancels(factors, dim: int = 10) -> tuple[bool, dict]:
    """Does the Green-Schwarz mechanism cancel this group's anomaly?

    Returns ``(verdict, report)`` with the report naming which of the three
    conditions failed: the dimension, a surviving sixth-order trace, or the
    factorisation.  They are genuinely separate -- ``SO(26) x SO(19)`` passes
    the first and fails the second.
    """
    factors = list(factors)
    dimension, _ = combine(factors)
    poly = anomaly_polynomial(factors, dim)
    leftovers = sixth_order_terms(poly)
    report = {
        "dimension": dimension,
        "gravitational": poly.coefficient(("R6", 1)),
        "sixth_order": leftovers,
        "factorisation": None,
    }
    if report["gravitational"] != 0 or leftovers:
        return False, report
    result = factorise(poly, factors)
    report["factorisation"] = result
    return result.works, report


def candidates(max_so: int = 40, max_factors: int = 2) -> list[list[GaugeFactor]]:
    """Products of ``SO(N)`` and ``E_8`` factors, up to ``max_factors`` of them.

    ``SO(N)`` for ``3 <= N <= max_so`` plus ``E_8``.  Only groups whose total
    dimension can reach 496 are worth listing, so nothing smaller than a single
    factor is generated; the scan below filters on the conditions.
    """
    if max_factors < 1:
        raise ValueError("need at least one factor")
    singles = [("E8", None)] + [("SO", n) for n in range(3, max_so + 1)]

    def build(chosen):
        out = []
        for index, (kind, n) in enumerate(chosen):
            out.append(e8(index) if kind == "E8" else so(n, index))
        return out

    found, seen = [], set()
    stack = [[]]
    while stack:
        chosen = stack.pop()
        if chosen:
            key = tuple(sorted(chosen, key=str))
            if key not in seen:
                seen.add(key)
                found.append(build(chosen))
        if len(chosen) < max_factors:
            start = singles.index(chosen[-1]) if chosen else 0
            stack.extend(chosen + [single] for single in singles[start:])
    return found


def scan(max_so: int = 40, max_factors: int = 2, dim: int = 10) -> list[list[GaugeFactor]]:
    """Every candidate in the family whose anomaly actually cancels.

    Comes back with ``SO(32)`` and ``E_8 x E_8`` and nothing else.  That is a
    search over a family, not a uniqueness proof: ``SU(N)``, ``Sp(N)`` and the
    smaller exceptional algebras are not implemented, and neither are the
    abelian solutions.
    """
    return [c for c in candidates(max_so, max_factors) if cancels(c, dim)[0]]
