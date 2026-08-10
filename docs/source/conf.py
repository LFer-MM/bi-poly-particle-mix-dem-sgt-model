# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Bidisperse Polyhedrical Particle Mixing DEM Surrogate Model'
copyright = '2026, Fernando Martinez'
author = 'Fernando Martinez'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))  # points to your repo root

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Google-style docstrings in the package.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
}

# Allow docs to build without heavyweight/optional runtime deps.
autodoc_mock_imports = [
    "tensorflow",
    "yade",
    "yade._polyhedra_utils",
    "yade.utils",
    "yade.wrapper",
]

html_theme = 'sphinx_rtd_theme'

templates_path = ['_templates']

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for LaTeX / PDF output -------------------------------------------------
# Sphinx's pdflatex default uses TeX Gyre (tgtermes / tgheros), which needs
# extra Debian packages (e.g. tex-gyre). Latin Modern ships with
# texlive-latex-recommended and keeps CI minimal while producing valid PDFs.
latex_elements = {
    'fontpkg': r'\usepackage{lmodern}',
}
