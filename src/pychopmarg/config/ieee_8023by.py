"""
COM parameter definitions for IEEE 802.3dj

Original author: David Banas <capn.freako@gmail.com>

Original date:   November 6, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

import numpy as np
from pychopmarg.config.template import COMParams

fB = 25 * 66. / 64.

IEEE_8023by = COMParams(
    fB, 0.01, 40e9, 2, 32, 1e-5, 0.010, 1.0, 0.4, 0.4, 0.6, 50,
    0.05, 27, 5.2e-8, 0.01,
    fB / 4, fB / 4, fB, 1.0, [float(-n) for n in range(13)], [0.],
    [-1.] * 6, [1.] * 6, [0.25] * 6, 0.,
    0.75, np.array([-1.0]), np.array([1.0]), np.array([]), np.array([]), 0,
    np.array([55]), [4.0e-5, 9.0e-4, 1.1e-4], [0.0], [1.8e-4], [0.13, 0.15, 0.14], [87.5, 92.5],
    [12., 33.], 5.0e-4, 8.9e-4, 2.0e-4, 6.141e-3)
