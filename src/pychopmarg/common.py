"""
Definitions common to all PyChOpMarg modules.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024 (Copied from `pybert.utility`.)

Copyright (c) 2024 David Banas; all rights reserved World wide.

ToDo:
    1. Specify ``COMParams`` concretely.
"""

from typing import TypeVar, TypeAlias, Any

import numpy as np  # type: ignore
import numpy.typing as npt  # type: ignore

Real = TypeVar('Real', float, float)
Comp = TypeVar('Comp', complex, complex)
Rvec: TypeAlias = npt.NDArray[Real]
Cvec: TypeAlias = npt.NDArray[Comp]
Rmat: TypeAlias = npt.NDArray[Rvec]  # type: ignore
Cmat: TypeAlias = npt.NDArray[Cvec]  # type: ignore
COMParams: TypeAlias = dict[str, Any]  # ToDo: Specify this concretely, perhaps in `standards` module.
COMFiles: TypeAlias = str | list[str]

PI: float = np.pi
TWOPI: float = 2 * np.pi
