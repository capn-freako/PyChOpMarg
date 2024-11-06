# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyChOpMarg'
copyright = '2024, David Banas'
author = 'David Banas'
release = 'v2.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

sys.path.insert(0,  os.path.abspath('../src'))
extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon', 
    'sphinx.ext.todo', 
    'sphinx.ext.autosummary', 
    'sphinx.ext.viewcode', 
]
auto_doc_default_options = {
   'undoc-members': True,
   'inherited-members': False,
   'show-inheritance': True,
}
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_mock_imports = [
   "matplotlib",
]
autodata_content = 'both'
autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
todo_include_todos = True
