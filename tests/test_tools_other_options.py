#!/usr/bin/env python
import dandelion as ddl
import pytest

from pathlib import Path


@pytest.mark.usefixtures("airr_generic")
def test_find_clones_other_options(airr_generic):
    """Test find clones."""
    vdj = ddl.pp.check_contigs(airr_generic, productive_only=False)
    with pytest.raises(ValueError):
        vdj = ddl.tl.find_clones(vdj, recalculate_length=False)
    vdj.data["junction_aa_length"] = 10
    with pytest.raises(ValueError):
        ddl.tl.find_clones(vdj, recalculate_length=False)
    vdj = vdj[vdj.data.junction != ""]  # remove empty junctions
    ddl.tl.find_clones(vdj, recalculate_length=False)
    assert not vdj.data.clone_id.empty
    assert not vdj.metadata.clone_id.empty


@pytest.mark.usefixtures("create_testfolder", "airr_generic")
def test_find_clones_file(create_testfolder, airr_generic):
    """Test find clones from file."""
    in_file = create_testfolder / "test_airr.tsv"
    out_file = create_testfolder / "test_airr_clone.tsv"
    airr_generic.to_csv(in_file, sep="\t", index=False)
    ddl.tl.find_clones(in_file)
    assert Path(out_file) in list(create_testfolder.iterdir())


@pytest.mark.usefixtures("airr_generic")
def test_find_clones_after_network(airr_generic):
    """Test find clones."""
    vdj = ddl.pp.check_contigs(airr_generic)
    ddl.tl.find_clones(vdj)
    ddl.tl.generate_network(vdj, key="junction_aa", layout_method="mod_fr")
    vdj2 = vdj.copy()
    vdj2.threshold = 0.01
    vdj2.germline = {"dummy": "something"}
    ddl.tl.find_clones(vdj2)
    assert not vdj2.data.clone_id.empty
    assert not vdj2.metadata.clone_id.empty
    ddl.tl.find_clones(vdj2, key_added="cloned_idx")
    assert not vdj2.data.cloned_idx.empty
    assert not vdj2.metadata.cloned_idx.empty
