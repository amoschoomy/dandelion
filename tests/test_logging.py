#!/usr/bin/env python
import dandelion as ddl

from pathlib import Path
from subprocess import run


def test_badge():
    """test_badge"""
    p = run(
        ["python", str(Path("dandelion") / "logging" / "_badge.py")],
        capture_output=True,
        encoding="utf8",
    )
    assert p.returncode == 0
    # assert p.stderr == "" # because palantir tries to import some fonts and it complains.
    assert p.stdout != ""
    assert p.stdout.strip("\n") == ddl.__version__


def test_logging():
    """test_logging"""
    ddl.logging.print_header()
    ddl.logging.print_versions()


def test_metadata():
    """test_metadata"""
    assert ddl.__email__ is not None
    assert ddl.__author__ is not None
    assert ddl.__classifiers__ is not None
