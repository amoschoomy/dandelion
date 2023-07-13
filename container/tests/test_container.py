#!/opt/conda/envs/sc-dandelion-container/bin/python
import os
import pandas as pd
import dandelion as ddl
from pathlib import Path
from unittest.mock import patch
from subprocess import run

SHARE_PATH = Path("/share")
TEST_PATH = Path("/test")
PREPROC_PATH = SHARE_PATH / "dandelion_preprocess.py"
CLONO_PATH = SHARE_PATH / "changeo_clonotypes.py"


def test_callscript():
    """Test script to run preprocessing."""
    p = run(
        ["python", str(PREPROC_PATH), "-h"],
        ["python", str(PREPROC_PATH), "-h"],
        capture_output=True,
        encoding="utf8",
    )
    assert p.returncode == 0
    assert p.stdout != ""


def test_container():
    """Test script to run container."""
    os.system(
        f"cd /tests; python {str(PREPROC_PATH)} --meta test.csv --file_prefix filtered;"
    )
    dat = pd.read_csv(
        TEST_PATH
        / "sample_test_10x"
        / "dandelion"
        / "filtered_contig_dandelion.tsv",
        sep="\t",
    )
    assert not dat["c_call"].empty
    assert not dat["v_call_genotyped"].empty
    assert not dat["mu_count"].empty
    assert not dat["mu_freq"].empty
    vdj = None
    try:
        vdj = ddl.Dandelion(dat)
    except:
        pass
    assert vdj is not None


@patch("matplotlib.pyplot.show")
def test_threshold(mock_show):
    """Test script to run container."""
    os.system(
        f"cd /tests; python {str(PREPROC_PATH)} --h5ddl sample_test_10x/demo-vdj.h5ddl;"
    )
    dat = ddl.read_h5ddl(
        TEST_PATH / "sample_test_10x" / "demo-vdj_changeo.h5ddl"
    )
    assert not dat.data.changeo_clone_id.empty
