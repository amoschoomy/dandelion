#!/usr/bin/env python
import os
import json
import pytest
import dandelion as ddl
import pandas as pd


@pytest.mark.usefixtures("create_testfolder", "airr_10x")
def test_write_airr(create_testfolder, airr_10x):
    """test_write_airr"""
    out_file = create_testfolder / "test_airr_rearrangements.tsv"
    airr_10x.to_csv(out_file, sep="\t", index=False)
    assert len(list(create_testfolder.iterdir())) == 1


@pytest.mark.usefixtures("create_testfolder")
def test_loaddata(create_testfolder):
    """test_loaddata"""
    file1 = create_testfolder / "test_airr_rearrangements.tsv"
    file2 = create_testfolder / "test_airr_rearrangements2.tsv"
    dat = ddl.utl.load_data(file1)
    assert isinstance(dat, pd.DataFrame)
    with pytest.raises(FileNotFoundError):
        dat2 = ddl.utl.load_data(file2)
    with pytest.raises(FileNotFoundError):
        dat2 = ddl.utl.load_data("something.tsv")
    dat2 = pd.read_csv(file1, sep="\t")
    dat2.drop("sequence_id", inplace=True, axis=1)
    with pytest.raises(KeyError):
        dat2 = ddl.utl.load_data(dat2)


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated")
def test_write_annotated(create_testfolder, airr_reannotated):
    """test_write_annotated"""
    out_file = create_testfolder / "test_airr_reannotated.tsv"
    airr_reannotated.to_csv(out_file, sep="\t", index=False)
    assert not airr_reannotated.np1_length.empty
    assert not airr_reannotated.np2_length.empty
    assert not airr_reannotated.junction_length.empty


@pytest.mark.usefixtures("create_testfolder")
def test_readwrite_h5ddl(create_testfolder):
    """test_readwrite_h5"""
    out_file1 = create_testfolder / "test_airr_reannotated.tsv"
    out_file2 = create_testfolder / "test_airr_reannotated.h5ddl"
    vdj = ddl.Dandelion(out_file1)
    assert not vdj.data.np1_length.empty
    assert not vdj.data.np2_length.empty
    assert not vdj.data.junction_length.empty
    vdj.write_h5ddl(out_file2)
    vdj2 = ddl.read_h5ddl(out_file2)
    assert not vdj2.data.np1_length.empty
    assert not vdj2.data.np2_length.empty
    assert not vdj2.data.junction_length.empty
    vdj.write_h5ddl(out_file2, complib="blosc:lz4")
    vdj2 = ddl.read_h5ddl(out_file2)
    assert not vdj2.data.np1_length.empty
    assert not vdj2.data.np2_length.empty
    assert not vdj2.data.junction_length.empty
    vdj.write_h5ddl(out_file2, compression="blosc:lz4")
    vdj2 = ddl.read_h5ddl(out_file2)
    assert not vdj2.data.np1_length.empty
    assert not vdj2.data.np2_length.empty
    assert not vdj2.data.junction_length.empty
    with pytest.raises(ValueError):
        vdj.write_h5ddl(out_file2, complib="blosc:lz4", compression="blosc:lz4")


@pytest.mark.usefixtures("create_testfolder")
def test_readwrite_pkl(create_testfolder):
    """test_readwrite_pkl"""
    out_file1 = create_testfolder / "test_airr_reannotated.tsv"
    out_file2 = create_testfolder / "test_airr_reannotated.pkl"
    out_file3 = create_testfolder / "test_airr_reannotated.pkl.gz"
    out_file4 = create_testfolder / "test_airr_reannotated.pkl.pbz2"
    vdj = ddl.Dandelion(out_file1)
    assert not vdj.data.np1_length.empty
    assert not vdj.data.np2_length.empty
    assert not vdj.data.junction_length.empty
    vdj.write_pkl(out_file2)
    vdj3 = ddl.read_pkl(out_file2)
    assert not vdj3.data.np1_length.empty
    assert not vdj3.data.np2_length.empty
    assert not vdj3.data.junction_length.empty
    vdj.write_pkl(out_file3)
    vdj4 = ddl.read_pkl(out_file3)
    assert not vdj4.data.np1_length.empty
    assert not vdj4.data.np2_length.empty
    assert not vdj4.data.junction_length.empty
    vdj.write_pkl(out_file4)
    vdj5 = ddl.read_pkl(out_file4)
    assert not vdj5.data.np1_length.empty
    assert not vdj5.data.np2_length.empty
    assert not vdj5.data.junction_length.empty


@pytest.mark.usefixtures("create_testfolder")
def test_readwrite10xairr(create_testfolder):
    """test_readwrite10xairr"""
    airr_file = create_testfolder / "test_airr_rearrangements.tsv"
    airr_file2 = create_testfolder / "test_airr_rearrangements2.tsv"
    vdj = ddl.read_10x_airr(airr_file)
    assert vdj.data.shape[0] == 9
    assert vdj.metadata.shape[0] == 5
    vdj.write_airr(airr_file2)
    vdj2 = ddl.read_10x_airr(airr_file2)
    assert vdj2.data.shape[0] == 9
    assert vdj2.metadata.shape[0] == 5
    os.remove(airr_file2)


@pytest.mark.usefixtures("create_testfolder", "json_10x_cr6")
def test_read10xvdj_json(create_testfolder, json_10x_cr6):
    """test_read10xvdj_json"""
    json_file = create_testfolder / "test_all_contig_annotations.json"
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = ddl.read_10x_vdj(json_file)
    assert vdj.data.shape[0] == 26
    assert vdj.metadata.shape[0] == 10
    os.remove(json_file)


@pytest.mark.usefixtures(
    "create_testfolder", "json_10x_cr6", "annotation_10x_cr6", "fasta_10x_cr6"
)
def test_read10xvdj_cr6(
    create_testfolder, json_10x_cr6, annotation_10x_cr6, fasta_10x_cr6
):
    """test_read10xvdj_cr6"""
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    json_file = create_testfolder / "test_all_contig_annotations.json"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x_cr6.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(annot_file)
    assert vdj.data.shape[0] == 26
    assert vdj.metadata.shape[0] == 10
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = ddl.read_10x_vdj(annot_file)
    assert vdj.data.shape[0] == 26
    assert vdj.metadata.shape[0] == 10
    assert not vdj.data.sequence.empty
    os.remove(json_file)
    ddl.utl.write_fasta(fasta_dict=fasta_10x_cr6, out_fasta=fasta_file)
    vdj = ddl.read_10x_vdj(annot_file)
    assert vdj.data.shape[0] == 26
    assert vdj.metadata.shape[0] == 10
    assert not vdj.data.sequence.empty
    os.remove(fasta_file)


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "fasta_10x")
def test_read10xvdj(create_testfolder, annotation_10x, fasta_10x):
    """test_read10xvdj"""
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(annot_file)
    assert vdj.data.shape[0] == 9
    assert vdj.metadata.shape[0] == 5
    ddl.utl.write_fasta(fasta_dict=fasta_10x, out_fasta=fasta_file)
    vdj = ddl.read_10x_vdj(annot_file)
    assert vdj.data.shape[0] == 9
    assert vdj.metadata.shape[0] == 5
    assert not vdj.data.sequence.empty
    os.remove(fasta_file)


@pytest.mark.usefixtures(
    "create_testfolder", "json_10x_cr6", "annotation_10x_cr6", "fasta_10x_cr6"
)
def test_read10xvdj_cr6_folder(
    create_testfolder, json_10x_cr6, annotation_10x_cr6, fasta_10x_cr6
):
    """test_read10xvdj_cr6_folder"""
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    json_file = create_testfolder / "test_all_contig_annotations.json"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x_cr6.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(create_testfolder)
    assert vdj.data.shape[0] == 26
    assert vdj.metadata.shape[0] == 10
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    assert vdj.data.shape[0] == 26
    assert vdj.metadata.shape[0] == 10
    assert not vdj.data.sequence.empty
    os.remove(json_file)
    ddl.utl.write_fasta(fasta_dict=fasta_10x_cr6, out_fasta=fasta_file)
    vdj = ddl.read_10x_vdj(create_testfolder)
    assert vdj.data.shape[0] == 26
    assert vdj.metadata.shape[0] == 10
    assert not vdj.data.sequence.empty
    os.remove(fasta_file)


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "fasta_10x")
def test_read10xvdj_folder(create_testfolder, annotation_10x, fasta_10x):
    """test_read10xvdj_folder"""
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(create_testfolder)
    assert vdj.data.shape[0] == 9
    assert vdj.metadata.shape[0] == 5
    ddl.utl.write_fasta(fasta_dict=fasta_10x, out_fasta=fasta_file)
    vdj = ddl.read_10x_vdj(create_testfolder)
    assert vdj.data.shape[0] == 9
    assert vdj.metadata.shape[0] == 5
    assert not vdj.data.sequence.empty
    os.remove(fasta_file)


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "fasta_10x")
def test_to_scirpy(create_testfolder, annotation_10x, fasta_10x):
    """test_to_scirpy"""
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(create_testfolder)
    assert vdj.data.shape[0] == 9
    assert vdj.metadata.shape[0] == 5
    adata = ddl.to_scirpy(vdj)
    assert adata.obs.shape[0] == 5
    ddl.utl.write_fasta(fasta_dict=fasta_10x, out_fasta=fasta_file)
    vdj = ddl.read_10x_vdj(create_testfolder)
    assert vdj.data.shape[0] == 9
    assert vdj.metadata.shape[0] == 5
    assert not vdj.data.sequence.empty
    adata = ddl.to_scirpy(vdj)
    assert adata.obs.shape[0] == 5
    os.remove(fasta_file)
    vdjx = ddl.from_scirpy(adata)
    assert vdjx.data.shape[0] == 9


@pytest.mark.usefixtures(
    "create_testfolder", "annotation_10x_cr6", "json_10x_cr6"
)
def test_tofro_scirpy_cr6(create_testfolder, annotation_10x_cr6, json_10x_cr6):
    """test_tofro_scirpy_cr6"""
    json_file = create_testfolder / "test_all_contig_annotations.json"
    annot_file = create_testfolder / "test_all_contig_annotations.csv"
    annotation_10x_cr6.to_csv(annot_file, index=False)
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    assert vdj.data.shape[0] == 26
    assert vdj.metadata.shape[0] == 10
    adata = ddl.to_scirpy(vdj)
    assert adata.obs.shape[0] == 10
    vdjx = ddl.from_scirpy(adata)
    assert vdjx.data.shape[0] == 26


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "json_10x_cr6")
def test_tofro_scirpy_cr6_transfer(
    create_testfolder, annotation_10x_cr6, json_10x_cr6
):
    """test_tofro_scirpy_cr6_transfer"""
    json_file = create_testfolder / "test_all_contig_annotations.json"
    annot_file = create_testfolder / "test_all_contig_annotations.csv"
    annotation_10x_cr6.to_csv(annot_file, index=False)
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    assert vdj.data.shape[0] == 26
    assert vdj.metadata.shape[0] == 10
    adata = ddl.to_scirpy(vdj, transfer=True)
    assert adata.obs.shape[0] == 10
    vdjx = ddl.from_scirpy(adata)
    assert vdjx.data.shape[0] == 26


@pytest.mark.usefixtures("airr_generic")
def test_librarytype(airr_generic):
    """test library type"""
    tmp = ddl.Dandelion(airr_generic)
    assert tmp.data.shape[0] == 114
    assert tmp.metadata.shape[0] == 45

    tmp = ddl.Dandelion(airr_generic, library_type="ig")
    assert tmp.data.shape[0] == 68
    assert tmp.metadata.shape[0] == 25

    tmp = ddl.Dandelion(airr_generic, library_type="tr-ab")
    assert tmp.data.shape[0] == 29
    assert tmp.metadata.shape[0] == 20

    tmp = ddl.Dandelion(airr_generic, library_type="tr-gd")
    assert tmp.data.shape[0] == 17
    assert tmp.metadata.shape[0] == 15


def test_convert_obsm_airr_to_data(create_testfolder):
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    mdata = ddl.utl.to_scirpy(vdj)

    result = ddl.utl.from_ak(mdata["airr"].obsm["airr"])

    assert result.shape == vdj.data.shape
    assert result.shape[0] == 26


def test_convert_data_to_obsm_airr(create_testfolder):
    vdj = ddl.read_10x_vdj(create_testfolder, filename_prefix="test_all")
    anndata = ddl.utl.to_scirpy(vdj, to_mudata=False)
    obsm_airr, obs = ddl.utl.to_ak(vdj.data)
    assert len(anndata.obsm["airr"]) == len(obsm_airr)
    assert anndata.obsm["airr"].type.show() == obsm_airr.type.show()
    assert anndata.obs.shape == obs.shape


@pytest.mark.usefixtures("create_testfolder", "annotation_10x", "fasta_10x")
def test_to_scirpy_v2(create_testfolder, annotation_10x, fasta_10x):
    """test_to_scirpy"""
    fasta_file = create_testfolder / "test_filtered_contig.fasta"
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(create_testfolder)
    assert vdj.data.shape[0] == 35
    assert vdj.metadata.shape[0] == 15
    adata = ddl.utl.to_scirpy(vdj)
    assert adata.obs.shape[0] == 15
    ddl.utl.write_fasta(fasta_dict=fasta_10x, out_fasta=fasta_file)
    vdj = ddl.read_10x_vdj(create_testfolder)
    assert vdj.data.shape[0] == 35
    assert vdj.metadata.shape[0] == 15
    assert not vdj.data.sequence.empty
    adata = ddl.utl.to_scirpy(vdj)
    assert adata.obs.shape[0] == 15
    mdata = ddl.utl.to_scirpy(vdj, to_mudata=True)
    assert mdata.mod["airr"].shape[0] == 15
    os.remove(fasta_file)
    vdjx = ddl.from_scirpy(adata)
    assert vdjx.data.shape[0] == 35
    vdjx = ddl.from_scirpy(mdata)
    assert vdjx.data.shape[0] == 35
