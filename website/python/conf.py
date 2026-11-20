"""Generate the public Python API from this checkout, without probing hardware."""
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
project = "virtio-accel Python SDK"
author = "MicroPerceptron"
release = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
version = release
extensions = ["sphinx.ext.autodoc", "sphinx.ext.doctest", "sphinx.ext.napoleon"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "imported-members": True,
                           "show-inheritance": True, "special-members": "__call__"}
# Keep class headings short and show constructors separately from methods.
autodoc_class_signature = "separated"
pygments_style = "monokai"
copyright = "MicroPerceptron"
html_theme = "alabaster"
html_title = f"Python SDK {release} API"
html_theme_options = {"description": "Generated from the current main branch.",
                      "extra_nav_links": {"Python guide": "../index.html",
                                          "Project home": "../../index.html"}}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
exclude_patterns = ["_build"]
