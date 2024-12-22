"""
Definitions common to all PyChOpMarg modules.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from typing import TypeVar, TypeAlias

import numpy        as np  # type: ignore
import numpy.typing as npt  # type: ignore
import skrf         as rf  # type: ignore

Real = TypeVar('Real', np.float64, np.float64)
Comp = TypeVar('Comp', np.complex64, np.complex64)
Rvec: TypeAlias = npt.NDArray[Real]
Cvec: TypeAlias = npt.NDArray[Comp]
Rmat: TypeAlias = npt.NDArray[Real]
Cmat: TypeAlias = npt.NDArray[Comp]

COMFiles: TypeAlias = str | list[str]
COMNtwk: TypeAlias = tuple[rf.Network, str]
COMChnl: TypeAlias = tuple[COMNtwk, Cvec]

PI: float = np.pi
TWOPI: float = 2 * np.pi
