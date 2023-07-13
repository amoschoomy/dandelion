#!/usr/bin/env python
from typing import List, Tuple

modules = [
    "dandelion",
    "pandas",
    "numpy",
    "matplotlib",
    "networkx",
    "scipy",
    "distance",
    "polyleven",
]


# borrowed from scanpy's logging module
def _versions_dependencies(dependencies: List[str]) -> Tuple[str, str]:
    """Version dependencies.

    Parameters
    ----------
    dependencies : List[str]
        list of dependencies.

    Yields
    ------
    Tuple[str, str]
        yields dependency name and version.
    """
    for mod in dependencies:
        mod_name, dist_name = mod if isinstance(mod, tuple) else (mod, mod)
        try:
            imp = __import__(mod_name)
            yield dist_name, imp.__version__
        except (ImportError, AttributeError):
            pass


def print_versions(dependencies: List[str] = modules):
    """
    Versions that are essential for dandelion's operation.

    Parameters
    ----------
    dependencies : List[str], optional
        list of dependencies.
    """
    print(
        " ".join(
            f"{mod}=={ver}" for mod, ver in _versions_dependencies(dependencies)
        )
    )


def print_header(dependencies: List[str] = modules):
    """
    Versions that are essential for dandelion's operation.

    Parameters
    ----------
    dependencies : List[str], optional
        list of dependencies.
    """
    print(
        " ".join(
            f"{mod}=={ver}" for mod, ver in _versions_dependencies(dependencies)
        )
    )
