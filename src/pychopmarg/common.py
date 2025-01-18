"""
Definitions common to all PyChOpMarg modules.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from enum       import Enum
from pathlib    import Path
from typing     import TypeVar, TypeAlias

import numpy        as np  # type: ignore
import numpy.typing as npt  # type: ignore
import skrf         as rf  # type: ignore

Real = TypeVar('Real', np.float64, np.float64)
Comp = TypeVar('Comp', np.complex64, np.complex128)
Rvec: TypeAlias = npt.NDArray[Real]
Cvec: TypeAlias = npt.NDArray[Comp]
Rmat: TypeAlias = npt.NDArray[Real]
Cmat: TypeAlias = npt.NDArray[Comp]

COMFiles: TypeAlias = str | list[str]
COMNtwk: TypeAlias = tuple[rf.Network, str]
COMChnl: TypeAlias = tuple[COMNtwk, Cvec]

PI: float = np.pi
TWOPI: float = 2 * np.pi

ChnlGrpName: TypeAlias = str  # channel group name
ChnlSetName: TypeAlias = str  # channel set name (the stem of the thru channel s4p file name)
ChnlTypName: TypeAlias = str  # channel type name ("thru", "next", or "fext")
ChnlSetComp: TypeAlias = list[Path]
ChnlSet:     TypeAlias = dict[ChnlTypName, ChnlSetComp]

class OptMode(Enum):
    "Linear equalization optimization mode."
    PRZF = 1
    MMSE = 2

class NormMode(Enum):
    "Tap weight normalization mode."

    P8023dj   = 1
    "As per standard (i.e. - clip then renormalize for unit amplitude pulse response.)"

    Scaled    = 2
    "Uniformly and minimally scaled to bring tap weights just within their limits."

    Unaltered = 3
    "Use constrained optimization solution, unchanged."

    UnitDcGain = 4
    "Tap weights are uniformly scaled, to yield unity gain at d.c."
