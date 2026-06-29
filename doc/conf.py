# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# The monitoring modules live alongside this file, one level up.
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'pyOMA-Monitoring'
copyright = '2024-2025, Simon Marwitz, Volkmar Zabel'  # pylint: disable=redefined-builtin
author = 'Simon Marwitz, Volkmar Zabel'

# -- General configuration ---------------------------------------------------

extensions = [
    'autoclasstoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'myst_nb',
]

nitpicky = False
suppress_warnings = [
    'ref.citation',
    'docutils',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_logo = '_static/logo.png'

html_theme_options = {
    'navbar_align': 'left',
    'navbar_center': ['navbar-nav'],
    'collapse_navigation': True,
    'navigation_depth': 3,
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/pyOMA-dev/pyOMA-Monitoring',
            'icon': 'fa-brands fa-github',
        },
    ],
    'show_prev_next': True,
}

html_static_path = ['_static']

todo_include_todos = True

autodoc_default_flags = ['members']
autosummary_generate = True
autosummary_generate_overwrite = False

nb_execution_mode = 'off'
