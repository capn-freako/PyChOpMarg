"""
MS Excel CSV COM configuration file reader for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   November 8, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

import pandas as pd


def get_com_params(cfg_file: Path) -> COMParams:
    """
    Read COM configuration parameters from CSV file
    and create an equivalent ``COMParams`` instance.
    """

    data = pd.read_csv(cfg_file)
