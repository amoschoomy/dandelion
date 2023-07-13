#!/usr/bin/env python
try:
    from setuptools_scm import get_version

    __version__ = get_version(
        root="../..",
        relative_to=__file__,
        git_describe_command="git describe --dirty --tags --long --match v*.*.*",
    ).split("+")[0]
except LookupError:
    from .version import __version__
__author__ = "Zewen Kelvin Tuong"
__email__ = "kt16@sanger.ac.uk"
__classifiers__ = [
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: R",
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
]

__all__ = [
    "__author__",
    "__email__",
    "__classifiers__",
    "__version__",
]
