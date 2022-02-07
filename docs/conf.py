import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

project = 'se'
copyright = '2021, Giorgos Stamatelatos'
author = 'Giorgos Stamatelatos'

version = '1.0'
release = version

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx'
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': True
}

autoclass_content = 'both'
mathjax_path = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
html_show_sphinx = False
html_theme = "furo"
html_title = "se " + release
modindex_common_prefix = ["se."]
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

html_logo = "logo.svg"
html_theme_options = {
    "sidebar_hide_name": True
}
