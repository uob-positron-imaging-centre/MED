#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : conf.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 29.06.2020


# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.


import os
import sys

sys.path.insert(0, os.path.abspath('../../medeq'))
os.environ['MPLBACKEND'] = 'Agg'  # avoid tkinter import errors on rtfd.io


# -- Project information -----------------------------------------------------

project = 'MedEQ'
copyright = '2023, the MedEQ developers'

# The full version, including alpha/beta/rc tags
# Load the package's __version__.py module as a dictionary.
about = {}
with open('../../medeq/__version__.py') as f:
    exec(f.read(), about)

release = about["__version__"]
version = about["__version__"]


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'numpydoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The root toctree document
master_doc = 'index'  # NOTE: will be changed to `root_doc` in sphinx 4

autosummary_generate = True

numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {'optional', 'type_without_description', 'BadException'}

# Run docstring validation as part of build process
numpydoc_validation_checks = {"all", "SS06", "PR01", "GL01", "SA01", "EX01",
                              "YD01", "ES01", "GL08", "SS01", "RT01"}

autodoc_default_options = {
    'show-inheritance': False,
    'members': True,
    'inherited-members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    # 'special-members': '__init__',
}

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The reST default role (used for this markup: `text`) to use for all
# documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "pydata_sphinx_theme"
html_logo = "_static/logomin.png"

html_theme_options = {
    # The URL the GitHub icon points to
    "github_url": "https://github.com/uob-positron-imaging-centre/MED",

    # Show previous and next buttons on each page
    # "show_prev_next": False,

    # Show the navbar and search at the top
    "navbar_end": ["search-field.html", "navbar-icon-links.html"],

    # Show `edit this page` button
    "use_edit_page_button": True,

    "logo": {
        "image_light": "logomin.png",
        "image_dark": "logomin.png",
    }

    # Links to external sources
    # "icon_links": [
    #     {
    #         "name": "PyPI",
    #         "url": "https://pypi.org/project/pept",
    #         "icon": "fas fa-cubes",
    #     },
    #     {
    #         "name": "Anaconda",
    #         "url": "https://anaconda.org/conda-forge/pept",
    #         "icon": "fas fa-arrow-circle-down",
    #     },
    # ],
}

html_context = {
    "github_user": "uob-positron-imaging-centre",
    "github_repo": "MED",
    "github_version": "master",
}

html_sidebars = {
    # Hide sidebar
    # "**": [],
}

html_title = "%s v%s Manual" % (project, version)
html_last_updated_fmt = '%b %d, %Y'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
# html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'project-templatedoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('index', 'MedEQ.tex', u'MedEQ Documentation',
     u'MedEQ maintainers', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/devdocs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}
