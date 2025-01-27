"""
Main Entry Point for PyChOpMarg when using the CLI.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 25, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from pathlib import Path
from typing  import Optional

import click  # type: ignore

from pychopmarg import __version__
# from pychopmarg.com import COM
from pychopmarg.utility.jupyter import execute_notebook, convert_notebook_to_html


# ToDo: Activate when new `PyChOpMargUI` package is ready.
# @click.group(invoke_without_command=True, context_settings=dict(help_option_names=["-h", "--help"]))
@click.group(invoke_without_command=False, context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
@click.version_option(version=__version__)
def cli(ctx):
    """
    PyChOpMarg command line interface.
    """

    if ctx.invoked_subcommand is None:  # No sub-command like `sim` given open the GUI like default.
        # theCOM = COM()

        # # Show the GUI.
        # theCOM.configure_traits(view=traits_view)
        raise RuntimeError("Sorry, the PyChOpMarg GUI is currently non-functional.")


# @cli.command(context_settings=dict(help_option_names=["-h", "--help"]))
# @click.argument("config-file", type=click.Path(exists=True))
# @click.option("--results", "-r", type=click.Path(), help="Override the results filename.")
# def run(config_file, results):
#     """
#     Run a COM calculation without opening the GUI.

#     Will load the CONFIG_FILE from the given filepath, run the
#     COM calculation and then save the results into a file with the same name
#     but a different extension as the configuration file.
#     """
#     theCOM = COM()
#     theCOM.load_configuration(config_file)
#     theCOM.simulate(initial_run=True, update_plots=True)
#     if not results:
#         results = Path(config_file).with_suffix(".pychopmarg_results")
#     theCOM.save_results(results)


@cli.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("notebook-file", type=click.Path(exists=True))
def run_notebook(notebook_file: Path):
    """
    Run a Jupyter notebook in place w/o opening a browser tab.
    """
    execute_notebook(notebook_file)

@cli.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("notebook-file", type=click.Path(exists=True))
@click.option("--out-file", "-o", type=click.Path(), help="Override the output filename.")
def convert_notebook(notebook_file: Path, out_file: Optional[Path] = None):
    """
    Convert a Jupyter notebook to an HTML report, omitting all code.
    """
    convert_notebook_to_html(Path(notebook_file), results_path=out_file)
    