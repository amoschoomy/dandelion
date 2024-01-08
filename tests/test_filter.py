#!/usr/bin/env python
import json
import os
import dandelion as ddl
import pytest


@pytest.mark.usefixtures("create_testfolder", "dummy_adata_cr6", "json_10x_cr6")
def test_filtercontigs(create_testfolder, dummy_adata_cr6, json_10x_cr6):
    """test_filtercontigs"""
    json_file = create_testfolder / "test_all_contig_annotations.json"
    out_file = create_testfolder / "test_filtered.tsv"
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    vdj2, adata = ddl.pp.filter_contigs(vdj, dummy_adata_cr6)
    assert vdj.data.shape[0] == 26
    assert vdj2.data.shape[0] == 17
    assert adata.obs.shape[0] == 10
    vdj2, adata = ddl.pp.filter_contigs(
        vdj, dummy_adata_cr6, productive_only=False
    )
    assert vdj.data.shape[0] == 26
    assert vdj2.data.shape[0] == 26
    assert adata.obs.shape[0] == 10
    vdj2, adata = ddl.pp.filter_contigs(
        vdj, dummy_adata_cr6, productive_only=True, simple=True
    )
    assert vdj.data.shape[0] == 26
    assert vdj2.data.shape[0] == 26
    assert adata.obs.shape[0] == 10
    vdj2, adata = ddl.pp.filter_contigs(
        vdj, dummy_adata_cr6, productive_only=False, simple=True
    )
    assert vdj.data.shape[0] == 26
    assert vdj2.data.shape[0] == 26
    assert adata.obs.shape[0] == 10
    vdj2, adata = ddl.pp.filter_contigs(
        vdj, dummy_adata_cr6, productive_only=True, filter_extra_vj_chains=False
    )
    assert vdj.data.shape[0] == 26
    assert vdj2.data.shape[0] == 17
    assert adata.obs.shape[0] == 10
    vdj2, adata = ddl.pp.filter_contigs(
        vdj,
        dummy_adata_cr6,
        filter_extra_vj_chains=False,
        keep_highest_umi=False,
    )
    assert vdj.data.shape[0] == 26
    assert vdj2.data.shape[0] == 17
    assert adata.obs.shape[0] == 10
    vdj2, adata = ddl.pp.filter_contigs(vdj, dummy_adata_cr6, filter_rna=True)
    assert vdj.data.shape[0] == 26
    assert vdj2.data.shape[0] == 17
    assert adata.obs.shape[0] == 10
    vdj2, adata = ddl.pp.filter_contigs(
        vdj,
        dummy_adata_cr6,
        productive_only=False,
        filter_poorqualitycontig=True,
    )
    assert vdj.data.shape[0] == 26
    assert vdj2.data.shape[0] == 26
    assert adata.obs.shape[0] == 10
    vdj2, adata = ddl.pp.filter_contigs(vdj, dummy_adata_cr6, save=out_file)
    assert os.path.exists(out_file)


@pytest.mark.usefixtures("create_testfolder")
def test_filtercontigs_no_adata(create_testfolder):
    """test_filtercontigs_no_adata"""
    # json_file = create_testfolder / "test_all_contig_annotations.json"
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    vdj2 = ddl.pp.filter_contigs(vdj)
    assert vdj.data.shape[0] == 26
    assert vdj2.data.shape[0] == 17


@pytest.mark.usefixtures("airr_generic")
def test_generic_filter(airr_generic):
    """test data loading and filtering"""
    tmp = ddl.Dandelion(airr_generic)
    assert tmp.metadata.shape[0] == 45
    assert tmp.data.shape[0] == airr_generic.shape[0]

    tmp2 = ddl.pp.filter_contigs(tmp)
    assert tmp2.metadata.shape[0] == 18
    assert tmp2.data.shape[0] != tmp.data.shape[0]
    assert tmp2.data.shape[0] == 42

    tmp2 = ddl.pp.filter_contigs(tmp, filter_extra_vj_chains=True)
    assert tmp2.metadata.shape[0] == 16
    assert tmp2.data.shape[0] != tmp.data.shape[0]
    assert tmp2.data.shape[0] == 35

    tmp2 = ddl.pp.filter_contigs(
        tmp, filter_extra_vdj_chains=False, filter_extra_vj_chains=True
    )
    assert tmp2.metadata.shape[0] == 22
    assert tmp2.data.shape[0] != tmp.data.shape[0]
    assert tmp2.data.shape[0] == 50

    tmp2 = ddl.pp.filter_contigs(
        tmp, filter_extra_vdj_chains=False, filter_extra_vj_chains=False
    )
    assert tmp2.metadata.shape[0] == 24
    assert tmp2.data.shape[0] != tmp.data.shape[0]
    assert tmp2.data.shape[0] == 57

    tmp2 = ddl.pp.filter_contigs(
        tmp,
        filter_extra_vdj_chains=False,
        filter_extra_vj_chains=False,
        productive_only=False,
    )
    assert tmp2.metadata.shape[0] == 25
    assert tmp2.data.shape[0] != tmp.data.shape[0]
    assert tmp2.data.shape[0] == 67

    tmp2 = ddl.pp.filter_contigs(tmp, productive_only=False)
    assert tmp2.metadata.shape[0] == 19
    assert tmp2.data.shape[0] != tmp.data.shape[0]
    assert tmp2.data.shape[0] == 51

    tmp2 = ddl.pp.filter_contigs(tmp, library_type="ig")
    assert tmp2.metadata.shape[0] == 16
    assert tmp2.data.shape[0] != tmp.data.shape[0]
    assert tmp2.data.shape[0] == 32

    tmp2 = ddl.pp.filter_contigs(tmp, library_type="tr-ab")
    assert tmp2.metadata.shape[0] == 8
    assert tmp2.data.shape[0] != tmp.data.shape[0]
    assert tmp2.data.shape[0] == 15

    tmp2 = ddl.pp.filter_contigs(tmp, library_type="tr-gd")
    assert tmp2.metadata.shape[0] == 4
    assert tmp2.data.shape[0] != tmp.data.shape[0]
    assert tmp2.data.shape[0] == 4


@pytest.mark.usefixtures("airr_generic")
def test_generic_check(airr_generic):
    """test data loading and filtering"""
    tmp = ddl.Dandelion(airr_generic)
    assert tmp.metadata.shape[0] == 45
    assert tmp.data.shape[0] == airr_generic.shape[0]

    tmp2 = ddl.pp.check_contigs(tmp, productive_only=False)
    assert tmp2.metadata.shape[0] == 44
    assert tmp2.data.shape[0] == tmp.data.shape[0]

    tmp2 = ddl.pp.check_contigs(tmp, productive_only=False, library_type="ig")
    assert tmp2.metadata.shape[0] == 25
    assert tmp2.data.shape[0] != tmp.data.shape[0]
    assert tmp2.data.shape[0] == 68

    tmp2 = ddl.pp.check_contigs(tmp)
    assert tmp2.metadata.shape[0] == 42
    assert tmp2.data.shape[0] == tmp.data.shape[0]

    ddl.tl.find_clones(tmp2, identity={"tr-ab": 1})
    assert "clone_id" in tmp2.data
    assert "clone_id" in tmp2.metadata
    assert not tmp2.metadata.clone_id.empty

    ddl.tl.generate_network(tmp2, key="junction_aa", compute_layout=False)
    assert tmp2.graph is not None


@pytest.mark.usefixtures("airr_generic")
def test_check_remove_extra(airr_generic):
    """test remove extra option"""
    tmp = ddl.Dandelion(airr_generic)
    assert tmp.metadata.shape[0] == 45
    assert tmp.data.shape[0] == airr_generic.shape[0]
    tmp2 = ddl.pp.check_contigs(tmp)
    tmp3 = ddl.pp.check_contigs(tmp, filter_extra=True)
    assert tmp3.metadata.shape[0] == 42
    assert tmp2.data.shape[0] == tmp.data.shape[0]
    assert tmp3.data.shape[0] != tmp2.data.shape[0]
    assert tmp2.data.extra.value_counts()["T"] == 15
