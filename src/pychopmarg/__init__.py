"""
A Python implementation of COM, as per IEEE 802.3-22 Annex 93A/178A.

.. moduleauthor:: David Banas <capn.freako@gmail.com>

Original Author: David Banas <capn.freako@gmail.com>

Original Date:   29 February 2024

Copyright (c) 2024 by David Banas; all rights reserved World wide.
"""

from importlib.metadata import version as _get_version

# Set PEP396 version attribute
try:
    __version__ = _get_version("PyChOpMarg")
except Exception:  # pylint: disable=broad-exception-caught
    __version__ = "(dev)"

__date__ = "January 20, 2025"
__authors__ = "David Banas"
__copy__ = "Copyright (c) 2024 David Banas"
