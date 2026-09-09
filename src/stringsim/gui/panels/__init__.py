"""The panels, in the order the window lists them.

Adding one means writing a module here and appending it to :data:`REGISTRY`.
Nothing else in the GUI knows how many there are or what they contain: the app
reads the title, the blurb and the controls off whatever is in this tuple.

The order is deliberate rather than alphabetical -- it is the order in which
the ideas are worth meeting, the same order the README uses.
"""

from __future__ import annotations

from .branes import BranePanel
from .critical import CriticalDimensionPanel
from .hagedorn import HagedornPanel
from .mirror import MirrorPanel
from .myers import MyersPanel
from .orbifold import OrbifoldPanel
from .pq import PQPanel
from .tduality import TDualityPanel
from .veneziano import VenezianoPanel

__all__ = [
    "REGISTRY",
    "TDualityPanel",
    "CriticalDimensionPanel",
    "BranePanel",
    "PQPanel",
    "VenezianoPanel",
    "HagedornPanel",
    "MyersPanel",
    "MirrorPanel",
    "OrbifoldPanel",
]

REGISTRY = (
    CriticalDimensionPanel(),
    HagedornPanel(),
    TDualityPanel(),
    OrbifoldPanel(),
    BranePanel(),
    MyersPanel(),
    PQPanel(),
    VenezianoPanel(),
    MirrorPanel(),
)
