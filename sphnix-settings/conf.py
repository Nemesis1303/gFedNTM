
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'gFedNTM'
copyright = '2022, L. Calvo-Bartolomé,  J. Arenas-García,'
author = 'L. Calvo-Bartolomé, J. Arenas-García'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.extlinks',
    # 'sphinx.ext.graphviz',
    # 'sphinx.ext.inheritance_diagram',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Autodoc settings
autodoc_default_options = {
    'special-members': "__init__",
    # 'special-members': True,
    # 'private-members': True,
    'member-order': 'bysource',
}
autoclass_content = "both"

# Napoleon settings
napoleon_google_docstring = False
# napoleon_use_param = False
# napoleon_use_rtype = True
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If true, sectionauthor and moduleauthor directives will be shown in the output.
# Ignore by default.
show_authors = True

# # If true, the current module name will be prepended to all description
# # unit titles (such as .. function::).
# add_module_names = False

# The name for this set of Sphinx documents.
# "<project> v<release> documentation" by default.
#
html_title = "Topic Modeler documentation"

# If true, the index is split into individual pages for each letter.
#
# html_split_index = False