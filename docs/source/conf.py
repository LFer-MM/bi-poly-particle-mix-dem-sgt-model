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
]

# Allow docs to build without heavyweight/optional runtime deps.
autodoc_mock_imports = [
    "tensorflow",
    "yade",
]

html_theme = 'sphinx_rtd_theme'

templates_path = ['_templates']

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
