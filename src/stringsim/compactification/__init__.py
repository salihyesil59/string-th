"""Compactification: circles, tori and their orbifolds.

``circle`` is the one-dimensional case, written out explicitly: Kaluza-Klein
momenta, winding, ``R -> alpha'/R`` and the enhanced symmetry at the self-dual
radius.

``torus`` is the general smooth case, ``T^d`` with a metric and a ``B`` field.
It brings in the Narain lattice, the full ``O(d,d;Z)`` duality group, and gauge
symmetry read off a root system.  At ``d = 1`` it reproduces ``circle`` state
for state, which is how the two are kept honest.

``orbifold`` quotients a torus by a finite symmetry: the untwisted sector is
projected onto invariant states and new twisted sectors appear at the fixed
points, with fractional oscillator modes and a shifted ground-state energy.  It
reuses ``torus``'s ``O(d,d;Z)`` machinery to check that a proposed rotation
really is a symmetry of the moduli.
"""
