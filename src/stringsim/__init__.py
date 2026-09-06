"""stringsim -- a simulation toolkit for the bosonic string.

Four areas, each usable on its own:

``stringsim.classical``
    Solutions of the worldsheet wave equation: analytic mode expansions, a
    finite-difference evolver with Neumann / Dirichlet / periodic ends, the
    light-cone construction that solves the Virasoro constraints exactly, and
    the rigidly rotating string that traces out the leading Regge trajectory.

``stringsim.quantum``
    The quantised spectrum: level degeneracies from the oscillator partition
    function, the normal-ordering constant from zeta regularisation, the
    critical dimension, the particle content of the low levels (tachyon,
    photon, graviton / Kalb-Ramond / dilaton), and the Hagedorn temperature.

``stringsim.compactification``
    A closed string on a circle: Kaluza-Klein momenta, winding, T-duality and
    the enhanced gauge symmetry at the self-dual radius.

``stringsim.branes`` and ``stringsim.amplitudes``
    D-brane tensions, the spectrum of strings stretched between branes and the
    resulting gauge group; the Veneziano and Virasoro-Shapiro amplitudes with
    their poles and Regge behaviour.

``stringsim.viz`` draws and animates all of it.
"""

from __future__ import annotations

from .units import Conventions, dot, minkowski

__version__ = "0.1.0"

__all__ = ["Conventions", "dot", "minkowski", "__version__"]
