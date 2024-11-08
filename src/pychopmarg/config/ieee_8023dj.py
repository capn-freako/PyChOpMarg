"""
COM parameter definitions for IEEE 802.3dj

Original author: David Banas <capn.freako@gmail.com>

Original date:   November 6, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from pychopmarg.config.template import COMParams

IEEE_8023dj = COMParams(
    106.25, 10e6, 4, 32, 2e-4, 0.004, 0.95, 0.413, 0.413, 0.45, 50,
    0.02, 33, 6.0e-9, 0.01,
    42.5, 42.5, 106.25, 1.328125, [-n for n in range(16)], [-n / 2 for n in range(11)],
    [-0.06, 0.0, -0.34, -0.2, 0.0, 0.0], [0.0, 0.12, 0.0, 0.0, 0.0, 0.0], [0.005, 0.005, 0.005, 0.005, 0.0, 0.0], 0.5,
    0.58, [0.0], [0.85], [-1] * 16, [1] * 16, 5,
    50, [0.04, 0.09, 0.11], 0.03, 0.04, [0.13, 0.15, 0.14], [87.5, 92.5], [33,    1.8], 5.0e-4, 8.9e-4, 2.0e-4, 6.141e-3)
