"""
Jupyter utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   January 23, 2025

Copyright (c) 2025 David Banas; all rights reserved World wide.
"""

import click                        # type: ignore
import os
from   pathlib      import Path
import subprocess
from typing         import Optional


def execute_notebook(notebook_path: Path) -> None:
    """
    Executes a Jupyter notebook.

    Args:
        notebook_path: Path to the notebook.
    """

    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"The notebook {notebook_path} was not found!")
    
    subprocess.run([
        'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', notebook_path],
        check=True)


def convert_notebook_to_html(notebook_path: Path, results_path: Optional[Path] = None) -> None:
    """
    Converts a Jupyter notebook to an HTML file, excluding the input cells (code).

    Args:
        notebook_path: Path to the notebook.

    Keyword Args:
        results_path: Result file path.
            Default: None (Use stem of ``notebook_path`` with ``.html`` suffix.)
    """

    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"The notebook {notebook_path} was not found!")
    
    html_path = results_path or Path(notebook_path.name).with_suffix('.html')
    subprocess.run(
        ['jupyter', 'nbconvert', '--to', 'html', '--no-input', '--output', html_path, notebook_path],
        check=True)
