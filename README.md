# stringsim

A simulation toolkit for the bosonic string. It solves the worldsheet equations
numerically, counts the quantum states, identifies which particle each vibration
is, and reproduces the classic results — the critical dimension, the Regge
trajectory, T-duality, D-brane gauge symmetry and the Veneziano amplitude — as
computed output rather than quoted facts.

The design rule throughout: **anything that can be checked two ways is checked
two ways.** The mode expansion is validated against a finite-difference solution
of the same wave equation; the light-cone mass formula against the integrated
Noether charges; the amplitude's poles against the independently counted
spectrum; the critical dimension against the conformal anomaly. The test suite
is where those cross-checks live.

```
python -m stringsim                       # summary of everything, in one screen
python examples/01_vibrating_string.py    # animations + constraint residuals
python -m pytest                          # 225 checks
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

### 5. D-branes — `stringsim.branes`

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

### 6. Amplitudes — `stringsim.amplitudes`

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

### 7. Figures and animations — `stringsim.viz`

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
```

Each prints its numbers and writes its figures into `figures/`.

---

## Tests

```bash
python -m pytest
```

225 checks, about 25 seconds. They are cross-checks rather than regression
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
* `U(N) -> U(k) x U(N-k)` never increases the number of massless vectors;
* the Virasoro–Shapiro residues are stable under halving the offset, which is
  what "simple pole" means.

---

## What is deliberately not here

* **Superstring worldsheet dynamics.** The GSO counting, the critical dimension
  and the Jacobi identity are implemented, but the RNS fermions themselves are
  not simulated.
* **M-theory and branes beyond Dp.** M2/M5 branes, the eleven-dimensional
  picture and the DBI action are absent; `branes/` covers the tension, the
  stretched-string spectrum and the gauge group only.
* **Calabi–Yau compactification and orbifolds.** The torus is implemented, with its full
  `O(d,d;Z)`; quotienting it to reach chiral spectra, and curved
  compactifications, are not.
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
