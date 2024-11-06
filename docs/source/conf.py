# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
# Add project root to sys.path
import os
import sys

project = "mle_training"
copyright = "2024, Sreevathsava"
author = "Sreevathsava"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
]

templates_path = ["_templates"]
exclude_patterns = []

sys.path.insert(0, os.path.abspath(".."))

# Enable Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Include todos in output
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"  # or 'alabaster', if you prefer
html_static_path = ["_static"]
