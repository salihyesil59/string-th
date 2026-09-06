# stringsim

A simulation toolkit for string theory. It solves the worldsheet equations
numerically, counts the quantum states, identifies which particle each vibration
is, and reproduces the classic results — the critical dimension, the Regge
trajectory, T-duality, orbifold twisted sectors, the type II spectra and the two
heterotic strings — as computed output rather than quoted facts.

It starts with the bosonic string, where every step can be watched, and builds
up: circle, torus, orbifold, superstring, heterotic, heterotic on a torus. Both
worldsheet fields are simulated -- the bosons obey a wave equation and are
integrated with a leapfrog, the fermions obey a transport equation and are
integrated with Lax-Wendroff, and the two sectors of the superstring come out of
a single sign at the end of the string. Each
layer is required to reproduce the one below it — the torus at `d = 1` must give
the circle module's spectrum state for state, and it is tested that way.

The design rule throughout: **anything that can be checked two ways is checked
two ways.** The mode expansion is validated against a finite-difference solution
of the same wave equation; the light-cone mass formula against the integrated
Noether charges; the amplitude's poles against the independently counted
spectrum; the critical dimension against the conformal anomaly. The test suite
is where those cross-checks live.

```
python -m stringsim                       # summary of everything, in one screen
python examples/01_vibrating_string.py    # animations + constraint residuals
python -m pytest                          # 432 checks
```

---

## Install

No install is needed to run the examples or the tests — both put `src/` on the
path themselves. For an editable install:

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10, numpy, scipy and matplotlib.

---

## Conventions

| quantity | convention |
|---|---|
| metric | mostly plus, `diag(-1, +1, ..., +1)` |
| Regge slope | `alpha'`, free; `alpha_prime = 1` by default |
| string length | `l_s = sqrt(alpha')` |
| tension | `T = 1 / (2 pi alpha')` |
| open string | `sigma` in `[0, pi]`, `alpha_0 = sqrt(2 alpha') p` |
| closed string | `sigma` in `[0, 2 pi)`, `alpha_0 = alpha_tilde_0 = sqrt(alpha'/2) p` |
| dimension | `D = 26` by default; nothing hard-codes it |

Everything is set by one frozen dataclass:

```python
from stringsim import Conventions
conv = Conventions(alpha_prime=1.0, dim=26)
```

---

## What is in it

### 1. Classical worldsheet dynamics — `stringsim.classical`

In conformal gauge the embedding satisfies the free wave equation, so every
solution is a superposition of normal modes. Two independent routes are
provided, and they agree.

**Analytic mode expansions** (`modes.py`) for the open (Neumann) and closed
(periodic) string, with `X`, `dX/dtau` and `dX/dsigma` in closed form.

**Light-cone gauge** (`lightcone.py`) is where the physics enters. Fixing
`X^+ = 2 alpha' p^+ tau` turns the Virasoro constraints from conditions into
*definitions* of the minus components,

```
alpha_n^-  =  (1 / sqrt(2 alpha') p^+) . (1/2) sum_m alpha_{n-m}^i alpha_m^i
```

so the transverse amplitudes are free data and the string is physical by
construction. The `n = 0` component of that same equation is the mass shell, and
it returns, with nothing else put in,

```
alpha' M^2  =  sum_{m >= 1} |alpha_m^i|^2  =  N
```

Measured on the assembled `D`-dimensional solution, the constraint residuals
come out at **5e-16** and the momentum integral reproduces `p^mu` to better
than **1e-13**.

**Finite-difference evolution** (`evolve.py`) goes the other way: pluck a string,
release it, and integrate. Neumann, Dirichlet (an endpoint on a D-brane), mixed
and periodic ends are supported. The scheme is second-order — measured
convergence ratios 4.00, 4.00, 4.00 under grid halving — and exact to round-off
at Courant number 1, where it becomes d'Alembert's solution.
`mode_spectrum()` then reads the harmonics back off a snapshot: a midpoint pluck
gives `n = 0, 2, 6, 10, ...` falling like `1/n^2`, the classical guitar-string
answer, because it is the same equation.

**The rotating string** (`rotating.py`) is the exact solution

```
X^0 = A tau,   X^1 = A cos(tau) cos(sigma),   X^2 = A sin(tau) cos(sigma)
```

Its endpoints move at exactly the speed of light (measured: `1.0000`), and
integrating the Noether currents gives `M = A/(2 alpha')`, `J = A^2/(4 alpha')`,
so

```
J = alpha' M^2        (agreement to 1e-14 over a range of amplitudes)
```

That is where the Regge slope gets its name — and it is the same `alpha'` that
sets the spectrum spacing and the amplitude's pole spacing.

### 2. The quantum spectrum — `stringsim.quantum`

**Counting** (`partition.py`). The number of states at level `N` is the
coefficient of `q^N` in `prod (1-q^n)^{-(D-2)}`, computed in exact integer
arithmetic:

```
d_N  =  1, 24, 324, 3200, 25650, 176256, 1073720, ...
```

`d_400` has 92 digits, which is why floats are not used. The Dedekind eta
function is included and its modular transformation
`eta(-1/tau) = sqrt(-i tau) eta(tau)` verified to **3e-16**.

**Where D = 26 comes from** (`zeta.py`). The zero-point energy needs
`sum_n n`, and `zeta(-1) = -1/12` is extracted here from a genuine convergent
sum with an exponential cutoff, by removing the `1/eps^2` divergence and fitting
what is left:

```
measured  -0.083333333328     exact  -0.083333333333     error  4.9e-12
```

Hence `a = (D-2)/24`, and `a = 1` — forced by Lorentz invariance, since a
massive vector cannot be built from only `D-2` oscillators — gives `D = 26`. The
conformal-anomaly route, `c = D - 26 = 0`, is implemented separately and agrees;
the superstring versions give `D = 10` by both routes.

**Which particle is which vibration** (`states.py`). The little group fixes the
identification — `SO(D-2)` for a massless state, `SO(D-1)` for a massive one:

| level | `alpha' M^2` | states | content |
|---|---|---|---|
| 0 | −1 | 1 | tachyon |
| 1 | 0 | 24 | **photon**, vector of `SO(24)` |
| 2 | 1 | 324 | massive spin 2, symmetric traceless of `SO(25)` |
| 3 | 2 | 3200 | spin 3 (2900) ⊕ 2-form (300) |

and for the closed string at the massless level, `24 x 24 = 576` splits as

| piece | dimension | field |
|---|---|---|
| symmetric traceless | 299 | **graviton** `g_mu nu` |
| antisymmetric | 276 | Kalb–Ramond `B_mu nu` |
| trace | 1 | dilaton `Phi` |

The graviton is not inserted by hand. Every closed string has that state, which
is the reason string theory *contains* gravity. The tests check that the named
irreducible pieces exhaust the independently counted degeneracy — nothing left
over — at every level where a decomposition is given.

**Hagedorn temperature.** Degeneracies growing like `exp(4 pi sqrt(N))` mean the
canonical partition function diverges above a finite temperature. Fitted from
the computed degeneracies at `N <= 400`:

```
beta_H measured 12.5298   against  4 pi = 12.5664     (0.3%)
```

**Superstring.** The GSO-projected degeneracies `8, 128, 1152, 7680, ...` are
computed, and Jacobi's *aequatio identica satis abstrusa*
`theta_3^4 = theta_2^4 + theta_4^4` is verified to **exactly zero in integers**
over 80 orders — the statement that bosons and fermions are equinumerous at
every mass level, the first fingerprint of spacetime supersymmetry.

### 3. A circle: winding and T-duality — `stringsim.compactification.circle`

Put one direction on a circle of radius `R`. Momentum is quantised, `n/R`, as it
would be for a point particle; but a string can also *wind*, at energy
`wR/alpha'`, and that has no point-particle counterpart:

```
M^2 = (n/R)^2 + (wR/alpha')^2 + (2/alpha')(N + Ntilde - 2),    N - Ntilde = n w
```

The spectrum is invariant under `R -> alpha'/R` with `n <-> w` — verified as a
multiset identity across radii and values of `alpha'`. A circle smaller than
`sqrt(alpha')` is a circle you have already seen; there is a shortest
distinguishable radius.

At the self-dual radius the code finds, by enumeration, eight extra massless
states: four with `(n,w) = (±1,±1)` carrying one oscillator, which enlarge
`U(1)_L x U(1)_R` to `SU(2)_L x SU(2)_R`, and four with `(n,w) = (±2,0), (0,±2)`
and no oscillators — the bosonic string's tachyon tower passing through zero
mass. The two kinds are labelled separately rather than lumped together.

### 4. A torus: the Narain lattice and `O(d,d;Z)` — `stringsim.compactification.torus`

Compactifying `d` directions is not just `d` copies of the circle. A constant
metric `G_ij` **and** a constant antisymmetric `B_ij` are now available — `d^2`
moduli in all — and the duality group grows from `R -> alpha'/R` to the full
`O(d,d;Z)`. With `Z = (w, n)` the charge vector and `E = G + B`,

```
l_L = (1/sqrt2) G^{-1/2} (n + E^T w),      l_R = (1/sqrt2) G^{-1/2} (n - E w)

alpha' M^2 = l_L^2 + l_R^2 + 2(N + Ntilde - 2),   l_L^2 - l_R^2 = 2 n.w = 2(N - Ntilde)
```

**Two quadratic forms on one lattice.** The mass is `Z^T H Z` with the
generalized metric `H(G,B)`, positive definite and moduli-dependent; level
matching is `Z^T eta Z` with `eta = [[0,I],[I,0]]`, indefinite and *fixed*. The
Narain lattice `Gamma_{d,d}` is even (norms `2 n.w`) and self-dual
(`|det eta| = 1`) at every point in moduli space, which is what keeps the
one-loop amplitude modular invariant while the moduli vary.

**The first thing checked is the old case.** At `d = 1` the torus spectrum
agrees with `circle.py` state for state, across radii and across values of
`alpha'` — 43 level-matched states each time, identical multisets. A
generalisation that cannot reproduce what it generalises is not one.

**T-duality.** Three kinds of generator: a lattice basis change, an integer
shift of `B` (so `B` is periodic — a modulus with no large-field limit), and
the factorized duality exchanging `w^k <-> n_k`, which at `d = 1` is exactly
`R -> alpha'/R`. Each is verified to be an integer matrix preserving `eta`, and
to leave the spectrum invariant.

> A trap worth recording. The obvious test — enumerate the spectrum before and
> after, compare as multisets — **gives a false negative** for the basis change
> and the `B` shift. The enumeration is truncated to a box `|n|,|w| <= k`, and
> those generators shear the box, so states near the edge leave the window. The
> factorized dualities merely permute components and pass. `spectrum_is_dual`
> therefore follows each state through the charge map instead; the masses then
> agree to `1e-15` for every generator and for their products.

**The gauge group is a root system.** A massless vector needs
`(l_L^2, l_R^2) = (2, 0)` or `(0, 2)` — a lattice vector of squared length 2,
which is the standard normalisation for the roots of a simply laced algebra.
Because all roots have the *same* length, only `A`, `D`, `E` can appear from a
plain torus; `B_n`, `C_n`, `G_2`, `F_4` need orbifolds or Wilson lines.

The search is complete, not truncated: `l_R = 0` forces `n = E w` and
`l_L^2 = 2 w^T G w`, so `w^T G w = 1` bounds `w` outright.

| point in moduli space | roots per side | algebra |
|---|---|---|
| generic `T^2` | 0 | `u(1)^2` |
| one radius self-dual | 2 | `su(2) + u(1)` |
| self-dual `T^d` | `2d` | `su(2)^d` |
| `A_2` point of `T^2` | 6 | `su(3)` |

The `A_2` point is `G = [[1, -1/2], [-1/2, 1]]`, `B = [[0, 1/2], [-1/2, 0]]`,
chosen so that `w^T G w = 1` has six solutions *and* `E = G + B` is an integer
matrix, which is what makes `n = E w` an allowed momentum for each of them.

**Counting alone is not always enough**, and the code says so rather than
guessing: rank 3 with six roots is `su(2)^3` *or* `su(3) + u(1)`, and
`identify_algebra(3, 6)` returns both. `gauge_algebra` settles it from the
geometry — it splits the roots into connected components under
non-orthogonality, finding three orthogonal pairs rather than six coplanar
vectors. `figures/roots_su3.png` is that distinction drawn: a hexagon at 60
degrees, not two perpendicular pairs.

### 5. An orbifold: twisted sectors and the projection — `stringsim.compactification.orbifold`

Quotient the torus by a finite symmetry `theta` and two things happen at once.

**Untwisted states are projected.** Only `theta`-invariant states survive,
counted by the character projector `P = (1/N) sum_k theta^k`. What that removes
is physical: on `S^1/Z_2` the Kaluza-Klein gauge bosons `g_{mu i}` and `B_{mu i}`
carry one compact index, are odd, and disappear. Of the torus's 576 massless
states, 530 survive — 529 graviton/`B`/dilaton plus the radius modulus, with the
**46 vectors gone**. The compactification has no massless vectors from the
untwisted sector.

That number is computed twice: once by the character projector, which never
splits an index, and once by counting `(D-2-d)^2 + d^2` by hand. They agree for
`S^1/Z_2`, `T^2/Z_2` and `T^4/Z_2` (530, 488, 416). For a *rotation* rather than
a reflection, only the one compact pair with `lambda lambda* = 1` survives, and
`T^2/Z_3`, `Z_4`, `Z_6` all give `22^2 + 2 = 486`.

**Twisted sectors appear.** Strings closing only up to `theta^k` live at its
fixed points — `|det(1 - theta^k)|` of them — and carry fractional oscillator
modes, which shifts the ground-state energy to

```
a_k = 1 - (1/4) sum_j phi_j (1 - phi_j),      alpha' M^2 / 4 = N - a_k
```

with `exp(2 pi i phi_j)` the eigenvalues of `theta^k`. The `1/4` is not asserted:
a boson with modes `n + phi` has zero-point energy `(1/2) zeta(-1, phi)`, and
`regularised_shifted_sum` extracts `zeta(-1, a) = -B_2(a)/2` from a cut-off sum
exactly as `-1/12` was extracted for the untwisted string. Summing that over the
24 transverse bosons reproduces the closed form to `1e-7`, for every sector of
every orbifold in the table.

| orbifold | phases | `a_1` | fixed points |
|---|---|---|---|
| `S^1/Z_2` | 1/2 | 15/16 | 2 |
| `T^2/Z_2` | 1/2, 1/2 | 7/8 | 4 |
| `T^4/Z_2` | 1/2 (x4) | 3/4 | 16 |
| `T^2/Z_3` | 1/3, 2/3 | 8/9 | 3 |
| `T^2/Z_4` | 1/4, 3/4 | 29/32 | 2 |
| `T^2/Z_6` | 1/6, 5/6 | 67/72 | 1 |

The fixed points are located as well as counted — they are the classes of
`(1 - theta^k)^{-1} Lambda / Lambda` — and `figures/fixed_points_z3.png` draws
them in the hexagonal cell.

**Which orbifolds exist is not a free choice.** `theta` must be a lattice
automorphism that also preserves `G` and `B`, which is checked by pushing it
through the `O(d,d;Z)` machinery of the torus module and demanding the moduli
come back unchanged — a 90-degree rotation is fine on a square lattice and
rejected on a rectangular one. In two dimensions the crystallographic
restriction then leaves only `N = 1, 2, 3, 4, 6`, and
`crystallographic_orders` finds that by searching integer matrices rather than
quoting it.

**An honest negative result.** None of these orbifolds has a massless twisted
state: `a_k` never lands on the level lattice, so every twisted level is either
tachyonic or massive. The bosonic string keeps its instability, and the
quotient does not cure it.

### 6. The superstring: worldsheet fermions and GSO — `stringsim.superstring`

Give each boson `X^mu` a fermionic partner `psi^mu` and three things change at
once.

**Two sectors.** Nothing forces the worldsheet fermion to come back to itself
around the string, so it may be antiperiodic (Neveu-Schwarz, half-integer modes)
or periodic (Ramond, **integer modes including a zero mode**). Those zero modes
obey a Clifford algebra, which forces the R ground state to be a spinor. That is
where spacetime fermions come from at all.

**The ground-state energies follow, and are measured.** A fermion contributes
`-(1/2) zeta(-1, phi)` where a boson contributes `+(1/2) zeta(-1, phi)`, and
`regularised_shifted_sum` supplies both:

```
a_NS = (D-2)/16 = 1/2,        a_R = 0        (D = 10)
```

so `alpha' M^2 = N - 1/2` in NS and `alpha' M^2 = N` in R. **The Ramond ground
state comes out massless with nothing put in.** The NS ground state is a tachyon
at `-1/2` — half as deep as the bosonic string's — and the projection is what
removes it.

**GSO, by explicit enumeration.** Keeping odd worldsheet fermion number deletes
the NS tachyon and leaves `b_{-1/2}^i|0>`, eight states, a massless vector.
Keeping one chirality in R leaves eight there too. The counting is done by
multiplying out the oscillator products and tracking parity — an entirely
different route from the theta-function ratio in `quantum/partition.py` — and
the two are required to agree:

```
8, 128, 1152, 7680, 42112, 200448, 855552      (NS, enumerated)
8, 128, 1152, 7680, 42112, 200448, 855552      (R,  enumerated)
8, 128, 1152, 7680, 42112, 200448, 855552      (theta-function product)
```

Equal bosons and fermions at every mass level is the concrete form of
`theta_3^4 = theta_2^4 + theta_4^4`, which section 2 proves as an identity.
`supersymmetry_deficit` returns zeros.

**Type IIA and IIB.** Two sets of fermions means four sectors, and the only
difference between the theories is whether the two Ramond spinors have the same
chirality:

| sector | content | IIA | IIB |
|---|---|---|---|
| NS-NS | graviton 35, `B` 28, dilaton 1 | same | same |
| R-R | | `C_1` 8, `C_3` 56 | `C_0` 1, `C_2` 28, self-dual `C_4` 35 |
| NS-R, R-NS | gravitino 56, dilatino 8 | same | same |

64 states each, 128 bosons and 128 fermions in both. Every dimension is a
binomial coefficient computed on the spot, self-duality included — which is why
`C_4` contributes 35 and not 70.

**And that decides the branes.** A `Dp`-brane couples to `C_{p+1}`, whose dual
is `C_{7-p}`, the potential of a `D(6-p)`-brane. Closing the RR ranks under
`p -> 6 - p` gives

```
IIA:  p = 0, 2, 4, 6        IIB:  p = -1, 1, 3, 5, 7
```

even for IIA and odd for IIB, which `stable_brane_ranks` derives rather than
tabulates — and a test asserts the parity, because getting the dual off by one
gives a plausible-looking wrong list. Feed the result to section 7's
`dp_brane_tension` for the masses; `p = -1` is the D-instanton, which has an
action rather than a tension and is correctly refused.

**The superstring runs hotter.** Eight bosons and eight fermions grow like
twelve bosons would, not twenty-four, so `beta_H = 2 pi sqrt(2) = 8.886` against
the bosonic `4 pi = 12.566`.

**The fermions are also simulated, not only counted** —
`stringsim.superstring.worldsheet`. The Dirac equation in conformal gauge is
first order, so `psi_-` and `psi_+` are rigid profiles sliding in opposite
directions; all of the content is at the ends. Fold the open string open with
`psi(sigma) = psi_-(sigma)` on `[0, pi]` and `eta psi_+(2 pi - sigma)` beyond,
and the pair becomes **one** right-moving field on a circle of circumference
`2 pi` with

```
psi(sigma + 2 pi) = eta psi(sigma)
```

which is the whole sector story in one line. Everything else is then measured on
the grid rather than quoted:

| measured from the evolution | NS (`eta = -1`) | R (`eta = +1`) |
|---|---|---|
| mode numbers that rebuild a snapshot | `1/2, 3/2, 5/2, ...` | `0, 1, 2, ...` |
| rebuilt with the *other* sector's modes | error `1.1` | error `0.79` |
| non-propagating (zero) mode | none | one, conserved to `1e-13` |
| `psi(tau + 2 pi)` | `-psi(tau)`, so `4 pi` to return | `+psi(tau)` |

The wrong sector's mode numbers are not a small error but a different vector
space, which is why the second row is `O(1)` and not `O(h^2)`. The scheme is
Lax-Wendroff: second order (error ratio `3.83, 3.97, 3.99` on halving) and an
exact one-cell shift at Courant number 1, where a full run reproduces the
analytic mode solution to `6e-16`.

The closed string has two independent fields and therefore the four spin
structures NS-NS, NS-R, R-NS, R-R — and a constant survives only where the field
is periodic, so only Ramond sides carry zero modes.

**Supersymmetry picks the sector, and the supercurrent is a constraint.** Under
`delta psi_-+ = d_-+ X` with a constant parameter, Neumann data has
`d_+ X = d_- X` at the ends, so the varied fermion satisfies `psi_+ = psi_-`
at *both* — the Ramond condition. Running `classical/evolve.py` and this module
on the same grid:

```
    n     R at 0    R at pi   NS at pi  max |d_+ G_-|
   32   1.19e-03   1.19e-03   3.87e-01      3.893e-03
   64   1.48e-04   1.48e-04   3.91e-01      1.076e-03  (/3.6)
  128   1.39e-05   1.39e-05   3.95e-01      2.757e-04  (/3.9)
  256   2.02e-06   2.02e-06   3.94e-01      6.924e-05  (/4.0)
```

The R residuals go to zero with the grid; the NS one sits at `0.4` and stays.
The last column is the chirality of the supercurrent `G_- = psi_- . d_- X`, the
fermionic counterpart of the Virasoro residual in section 1, falling by four
each time.

`psi` is evolved as a real commuting field. That is exact for the equation of
motion, the boundary conditions, the mode numbers, the period doubling and
`G_-`, which is bilinear in different fields. It is not the Grassmann field, so
the fermion bilinear in `T_{++}` and the anticommutator algebra stay algebraic,
in `rns.py`.

### 7. Heterotic strings: two theories, and only two — `stringsim.heterotic`

A heterotic string is closed, and its two moving directions are *different
theories*: right-movers are the superstring (`c_R = 15`), left-movers the
bosonic string (`c_L = 26`). Ten left-moving directions are spacetime; the
remaining

```
26 - 10 = 16
```

have nowhere to go, and modular invariance forces them onto a lattice that is
**even** and **self-dual**. Such lattices exist only in dimensions divisible by
8 — and 16 is on that list. Had the two critical dimensions differed by
anything else there would be no heterotic string at all.

In sixteen dimensions there are exactly two, so there are exactly two heterotic
strings. Both are constructed here and checked from an explicit basis (found by
integer elimination, so the determinant is exact):

| lattice | roots | covolume | even | `dim G` | algebra |
|---|---|---|---|---|---|
| `E8 + E8` | 480 | 1 | yes | 496 | `e8 + e8` |
| `D16+` | 480 | 1 | yes | 496 | `so(32)` |

**They are genuinely hard to tell apart.** Rank 16 with 480 roots is `e8 + e8`
*or* `so(32)`, and `identify_algebra(16, 480)` returns both rather than picking
one. What separates them is connectivity: the `E8 + E8` roots split into two
mutually orthogonal families of 240, the `D16+` roots form one connected system
of 480. `decompose_roots` — written for the torus, reused unchanged — settles
it, and `figures/heterotic_roots.png` is that distinction drawn as an adjacency
matrix. (A two-dimensional projection of the roots was tried first and shows
nothing: both are shapeless clouds.)

**`D16` alone is not enough.** Its covolume is 2, so it is even but not
self-dual. Adding the spinor coset `(1/2, ..., 1/2)` halves the covolume to 1.
That vector has norm 4, so it is *not* a root and contributes no gauge boson —
it only fixes self-duality, and it is why the group is `Spin(32)/Z_2` rather
than `SO(32)`.

**Level matching removes the tachyon, before GSO.** Each side has its own mass
formula and a physical state must satisfy both:

```
alpha' M^2 / 4  =  N_L + p^2/2 - 1   =   N_R - a_R
```

The lattice is even, so `p^2` is even and the left side is always an *integer*,
lowest value `-1`. The NS ground state on the right sits at `-1/2`. Neither has
a partner, so the lightest matched state is exactly massless — and
`level_matched_masses(gso=False)` returns the same answer as `gso=True`, which
is the point.

**And the gauge group is what the massless level happens to contain.** At zero
mass the left side needs `N_L + p^2/2 = 1`: either one oscillator and no lattice
momentum (24 states, 16 of them internal) or no oscillator and a root (480 of
them). Sixteen Cartan directions plus 480 roots is **496 gauge bosons**. That
number is separately what Green-Schwarz anomaly cancellation demands in ten
dimensions. The lattice knows nothing about anomalies; two unrelated
consistency conditions agreeing on 496 is why the construction was taken
seriously.

The full massless level is `504 x 16 = 8064` states: `128` of `N = 1`
supergravity (the same graviton/`B`/dilaton/gravitino/dilatino reps as section
6) plus `496 x 16 = 7936` gauge.

**What is not proved here.** That there are *only* two even self-dual lattices
in sixteen dimensions is a theorem, not something this code establishes. What
the code shows is that both candidates satisfy every condition and that they are
inequivalent.

**On a torus, the two constructions merge.** Compactify ``d`` more directions
and the charge lattice becomes `Gamma_{16+d,d}` — the gauge lattice and the
Narain lattice of section 4, side by side. What is new is a third kind of
modulus: a **Wilson line** `A_i^I`, one gauge vector per compact direction, so

```
d(d+1)/2  +  d(d-1)/2  +  16d  =  d(d + 16)
```

moduli in all, the dimension of `O(16+d,d)/(O(16+d) x O(d))`. The gauge rank is
`16 + 2d`. A Wilson line acts on charges as a shift `pi -> pi + A w` together
with a compensating shift of the momentum, and that pair is an `O(16+d,d)`
rotation — built by `wilson_boost` and verified to preserve the lattice form,
since the shift on its own would spoil it and the cancellation is the point.

**Wilson lines break the gauge group, by one condition.** A gauge boson needs
`p_R = 0` and `p_L^2 = 2`. With no winding the first forces `n_i = A_i . pi`,
and `n` is an integer, so a root survives only when `A_i . pi` is an integer:

| lattice | Wilson line | roots | unbroken algebra |
|---|---|---|---|
| `E8 x E8` | `0` | 480 | `e8 + e8` |
| `E8 x E8` | `(1, 0^7; 0^8)` | 352 | `e8 + so(16)` |
| `E8 x E8` | `(1/2, 1/2, 0^6; 0^8)` | 368 | `e8 + e7 + su(2)` |
| `E8 x E8` | `(1/3, 0^7; 0^8)` | 324 | `e8 + so(14) + u(1)` |
| `Spin(32)/Z2` | `(1/2^8; 0^8)` | 224 | `so(16) + so(16)` |

`A = (1/2^8; 0^8)` leaves `E8 x E8` untouched — every `E8` vector has even
coordinate sum, so `A . pi` is always an integer — while the same Wilson line
cuts `Spin(32)/Z2` in half. And shifting `A` by a lattice vector changes
nothing, so Wilson lines are periodic, exactly as `B` is on the torus.

**Both reductions are tested, not assumed.** At `d = 0` every root gives
`(p_L^2, p_R^2) = (2, 0)` and the mass formula is the one in `spectrum.py`; at
zero gauge charge and no Wilson line the compact part reproduces the torus
module's momenta to `1e-14`.

Not enumerated: extra massless vectors carrying winding, which enhance the group
again at special radii. Finding those needs gauge charges of norm above 2, a
search of a different size, and `unbroken_roots` says so rather than quietly
returning a partial answer.

### 8. D-branes — `stringsim.branes`

Tension `T_p = 1/((2 pi)^p g_s alpha'^{(p+1)/2})`. The single power of `1/g_s`
is the point: heavy at weak coupling, light at strong coupling — unlike a field
theory soliton, which would go as `1/g_s^2`.

A string stretched between branes a distance `d` apart has

```
M^2 = (d / 2 pi alpha')^2 + (N - 1)/alpha'
```

so at `d = 0` the level-1 states are massless vectors and `N` coincident branes
carry `U(N)`; separating them gives the off-diagonal vectors a mass proportional
to the distance. The Higgs mechanism, with the vacuum expectation value read as
a length:

```
[0,0,0,0]   -> U(4)                    16 massless vectors
[0,0,0,2.5] -> U(3) x U(1)             10
[0,1,2,3]   -> U(1) x U(1) x U(1) x U(1)   4
```

### 9. Amplitudes — `stringsim.amplitudes`

The Veneziano amplitude `A(s,t) = B(-alpha(s), -alpha(t))`, `alpha(x) = 1 +
alpha' x`, with three things checked numerically:

* **Poles at the spectrum.** `alpha' s = -1, 0, 1, 2, ...` — the same numbers as
  `alpha' M^2 = N - 1` produced by the counting module, from a completely
  different calculation.
* **Residues.** The numerical limit `(alpha(s) - n) A` matches
  `-(1/n!) prod_{k=1}^{n} (alpha(t)+k)`, a polynomial of degree exactly `n`:
  level `n` exchanges spin up to `n` and no higher.
* **High energy.** The fixed-`t` Regge limit is approached monotonically
  (`|A|/|asymptotic|` = 0.9902, 0.9990, 0.9999, 1.0000), and at fixed `t/s` the
  amplitude falls **exponentially**, with the measured slope matching Stirling to
  a few parts in 10^4. Extended objects have no point-like hard core.

The closed-string Virasoro–Shapiro amplitude is included, with poles at
`alpha' s = 4(N-1)` — four times the open spacing, matching the closed spectrum.

Everything is evaluated through `gammaln`/`gammasgn`, not `gamma`: at
`s = -2000` the amplitude is far past what double precision can represent, and
`veneziano_log_abs` is the only honest way to look at it.

### 10. Figures and animations — `stringsim.viz`

GIFs are written with matplotlib's Pillow writer, so no external binary is
needed. `examples/` produces:

| file | what it shows |
|---|---|
| `rotating_string.gif` | the rotating solution, endpoint trail at `v = c` |
| `level2_string.gif` | a three-mode excited state, `N = 6` |
| `plucked_string.gif` | a triangle released from rest, evolved numerically |
| `snapshots.png` | the same motion as still frames |
| `plucked_modes.png` | harmonics read off the pluck |
| `regge_trajectory.png` | `J` against `alpha' M^2` |
| `open_spectrum.png`, `closed_spectrum.png` | mass ladders with degeneracies |
| `hagedorn.png` | `log d_N` against `sqrt(N)` with the fitted slope |
| `tduality.png` | the two towers crossing at the self-dual radius |
| `roots_su2.png`, `roots_su3.png` | root systems of the enhanced gauge groups |
| `torus_enhancement.png` | where in the `T^2` moduli space the symmetry grows |
| `fixed_points_z3.png`, `fixed_points_z4.png` | orbifold fixed points in the torus cell |
| `orbifold_intercepts.png` | how twisting lowers `a_k` |
| `supersymmetry.png` | equal boson and fermion counts, and the two Hagedorn slopes |
| `heterotic_roots.png` | root connectivity: two blocks against one |
| `wilson_breaking.png` | the same picture before and after a Wilson line |
| `fermion_reflection.png` | a pulse bouncing: NS colours alternate, R do not |
| `brane_separation.png` | levels rising as branes separate |
| `veneziano.png` | the amplitude and its poles |

---

## Examples

```bash
python examples/01_vibrating_string.py    # constraints, rotation, pluck, animations
python examples/02_particle_spectrum.py   # spectra and particle content
python examples/03_critical_dimension.py  # zeta, D=26, modularity, Hagedorn
python examples/04_tduality.py            # winding, duality, enhanced symmetry
python examples/05_dbranes.py             # tensions, stretched strings, U(N)
python examples/06_amplitudes.py          # poles, residues, Regge, hard scattering
python examples/07_torus.py               # Narain lattice, O(d,d;Z), root systems
python examples/08_orbifold.py            # projection, twisted sectors, fixed points
python examples/09_superstring.py         # NS and R, GSO, type IIA/IIB, which branes
python examples/10_heterotic.py           # the two lattices, 496, and no tachyon
python examples/11_heterotic_compactified.py   # Gamma_{16+d,d} and Wilson lines
python examples/12_worldsheet_fermions.py      # fermion transport, reflection, sectors
```

Each prints its numbers and writes its figures into `figures/`.

---

## Tests

```bash
python -m pytest
```

432 checks, about 45 seconds. They are cross-checks rather than regression
snapshots — the value of a test here is that it would fail if the physics were
wrong, not merely if the code changed. A representative sample:

* light-cone states satisfy `(Xdot ± X')^2 = 0` to `1e-11` at five different
  excitation patterns, and `alpha' M^2 = N` independently of `p^+` and of a
  transverse boost;
* the evolver's error falls by a factor of 4 under each grid halving, and is
  exact at Courant 1;
* the named `SO(n)` content of each level sums to the independently counted
  degeneracy;
* `theta_3^4 - theta_2^4 - theta_4^4 = 0` exactly, in integers;
* `q^{1/24}/eta(tau)` reproduces the partition-number generating function;
* the T-dual spectra agree as multisets, and state `(n,w)` at `R` matches
  `(w,n)` at `alpha'/R`;
* the `T^d` spectrum at `d = 1` is identical to the circle module's, and every
  `O(d,d;Z)` generator moves each state onto another of the same mass;
* the roots at the `A_2` point have pairwise products in `{2, 1, -1, -2}` — the
  hexagon — while the self-dual `T^3` roots split into three orthogonal pairs;
* the twisted intercept from the closed form matches the one from cut-off mode
  sums, and the character projector matches index counting, for every orbifold;
* the projected state count is a non-negative integer at every level — which is
  precisely what a sign or conjugation slip in the character sum would break;
* the NS degeneracies counted by explicit enumeration equal the ones from the
  theta-function product, and the Ramond counting equals both;
* the D-brane ranks are all even for IIA and all odd for IIB, and each list is
  closed under `p -> 6 - p`;
* both heterotic lattices come out even, unimodular and 496-dimensional, and
  `D16` without its spinor coset comes out even but *not* unimodular;
* the Wilson-line boost preserves the lattice form for random `A` in every `d`,
  and the compactified module reproduces both the `d = 0` heterotic spectrum and
  the zero-charge torus momenta;
* `U(N) -> U(k) x U(N-k)` never increases the number of massless vectors;
* the Virasoro–Shapiro residues are stable under halving the offset, which is
  what "simple pole" means.

---

## What is deliberately not here

* **Grassmann worldsheet fermions.** `superstring/worldsheet.py` evolves
  `psi^mu` as a real commuting field, which is exact for the transport, the
  boundary conditions, the mode numbers and the supercurrent, but cannot
  represent the anticommutator algebra or the fermion bilinear in `T_{++}`.
  Those stay algebraic.
* **M-theory and branes beyond Dp.** M2/M5 branes, the eleven-dimensional
  picture and the DBI action are absent; `branes/` covers the tension, the
  stretched-string spectrum and the gauge group only.
* **Calabi–Yau compactification.** The torus and its `Z_N` orbifolds are
  implemented; discrete torsion, asymmetric orbifolds and curved
  compactifications are not.
* **Interacting worldsheets.** Amplitudes are the known closed forms, not a
  moduli-space integral; there is no genus expansion and no loop calculation.

---

## References

* J. Polchinski, *String Theory*, Vol. I, chapters 1–8 (conventions followed
  here).
* B. Zwiebach, *A First Course in String Theory*, 2nd ed. — light-cone gauge,
  D-branes, the rotating string.
* K. Becker, M. Becker, J. Schwarz, *String Theory and M-Theory*, chapters 2–6.
* G. Veneziano, *Construction of a crossing-symmetric, Regge-behaved amplitude*,
  Nuovo Cim. A **57** (1968) 190.
* R. Hagedorn, *Statistical thermodynamics of strong interactions*, Nuovo Cim.
  Suppl. **3** (1965) 147.
* K. S. Narain, *New heterotic string theories in uncompactified dimensions*,
  Phys. Lett. B **169** (1986) 41.
* A. Giveon, M. Porrati and E. Rabinovici, *Target space duality in string
  theory*, Phys. Rept. **244** (1994) 77.
* L. Dixon, J. Harvey, C. Vafa and E. Witten, *Strings on orbifolds*, Nucl.
  Phys. B **261** (1985) 678 and B **274** (1986) 285.
* F. Gliozzi, J. Scherk and D. Olive, *Supersymmetry, supergravity theories and
  the dual spinor model*, Nucl. Phys. B **122** (1977) 253.
* J. Polchinski, *String Theory*, Vol. II, chapters 10-13.
* D. Gross, J. Harvey, E. Martinec and R. Rohm, *Heterotic string*, Phys. Rev.
  Lett. **54** (1985) 502.
* J. H. Conway and N. J. A. Sloane, *Sphere Packings, Lattices and Groups*,
  ch. 4 — the even self-dual lattices and their classification.

---

## Related work

This is the simulation counterpart to two existing repositories:
[`wljs-gr-toolkit`](https://github.com/salihyesil59/wljs-gr-toolkit), which
derives general-relativity results symbolically in Wolfram, and
[`CosmoFit`](https://github.com/salihyesil59/CosmoFit), which fits cosmological
models to data. Same discipline — every result cross-checked by an independent
route — applied to a different question.

## License

MIT. See [LICENSE](LICENSE).
