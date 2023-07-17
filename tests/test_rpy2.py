#!/usr/bin/env python
import pandas as pd
import dandelion as ddl
import sys
import pytest


@pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_importrpy2():
    """test_importrpy2"""

    from rpy2.robjects.packages import importr

    sh = importr("shazam")

    assert sh.__module__ == "rpy2.robjects.packages"


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated")
@pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_mutation(create_testfolder, airr_reannotated):
    """test_mutation"""
    f = create_testfolder / "test.tsv"
    airr_reannotated.to_csv(f, sep="\t", index=False)
    ddl.pp.quantify_mutations(f)
    out = pd.read_csv(f, sep="\t")
    vdj = ddl.Dandelion(out)
    assert not vdj.data.mu_count.empty
    ddl.pp.quantify_mutations(f, frequency=True)
    assert not vdj.data.mu_freq.empty


@pytest.mark.usefixtures("create_testfolder", "database_paths")
@pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_create_germlines(create_testfolder, database_paths):
    """test create germlines"""
    f = create_testfolder / "test.tsv"
    out = pd.read_csv(f, sep="\t")
    vdj = ddl.Dandelion(out)
    ddl.pp.create_germlines(vdj, germline=database_paths["germline"])
    assert not vdj.data.germline_alignment_d_mask.empty


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_manual_threshold_and_define_clones(create_testfolder):
    """test threshold"""
    f = create_testfolder / "test.tsv"
    out = pd.read_csv(f, sep="\t")
    vdj = ddl.Dandelion(out)
    vdj.threshold = 0.1
    ddl.tl.define_clones(vdj)
    assert not vdj.data.clone_id.empty
    ddl.tl.define_clones(vdj, key_added="changeo_clone")
    assert not vdj.data.changeo_clone.empty


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
def test_define_clones_outdir(create_testfolder):
    """test threshold"""
    f = create_testfolder / "test.tsv"
    out = pd.read_csv(f, sep="\t")
    vdj = ddl.Dandelion(out)
    vdj.threshold = 0.1
    out_path = (
        create_testfolder / "test" / "test"
    )  # this path shouldn't exist initially
    ddl.tl.define_clones(vdj, out_dir=out_path)
    assert len(list(out_path.iterdir())) == 3
    assert len(list((out_path / "tmp").iterdir())) == 2
