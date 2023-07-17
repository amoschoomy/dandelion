#!/usr/bin/env python
import dandelion as ddl
import pytest


@pytest.mark.usefixtures("create_testfolder")
def test_loadtravdv(airr_travdv):
    """test_loadtravdv"""
    temp = ddl.utilities._utilities.check_travdv(airr_travdv)
    assert temp.shape[0] == 6
    assert all([i == "TRA" for i in airr_travdv["locus"]])
    assert all([i == "TRD" for i in temp["locus"]])


@pytest.mark.usefixtures("airr_travdv")
def test_loadtravdv2(airr_travdv):
    """test_loadtravdv2"""
    vdj = ddl.Dandelion(airr_travdv)
    assert vdj.data.shape[0] == 6
    assert all([i == "TRD" for i in vdj.data["locus"]])


@pytest.mark.usefixtures("create_testfolder", "fasta_10x_travdv")
def test_write_fasta_tr(create_testfolder, fasta_10x_travdv):
    """testwrite_fasta_tr"""
    out_fasta = create_testfolder / "filtered_contig.fasta"
    ddl.utl.write_fasta(fasta_dict=fasta_10x_travdv, out_fasta=out_fasta)
    assert len(list(create_testfolder.iterdir())) == 1


@pytest.mark.usefixtures("create_testfolder", "annotation_10x_travdv")
def test_write_annotation_tr(create_testfolder, annotation_10x_travdv):
    """test write annot"""
    out_file = create_testfolder / "filtered_contig_annotations.csv"
    annotation_10x_travdv.to_csv(out_file, index=False)
    assert len(list(create_testfolder.iterdir())) == 2


@pytest.mark.usefixtures("create_testfolder")
def test_formatfasta(create_testfolder):
    """test format fasta"""
    ddl.pp.format_fastas(create_testfolder)
    assert len(list((create_testfolder / "dandelion").iterdir())) == 2


@pytest.mark.usefixtures("create_testfolder", "database_paths")
def test_reannotategenes(create_testfolder, database_paths):
    """test reannotate"""
    ddl.pp.reannotate_genes(
        create_testfolder,
        igblast_db=database_paths["igblast_db"],
        germline=database_paths["germline"],
        loci="tr",
    )
    assert len(list((create_testfolder / "dandelion" / "tmp").iterdir())) == 9
    assert len(list((create_testfolder / "dandelion").iterdir())) == 2


@pytest.mark.usefixtures("create_testfolder")
def test_loadtravdv_reannotated(create_testfolder):
    """test check tradv"""
    vdj = ddl.Dandelion(
        create_testfolder / "dandelion" / "filtered_contig_dandelion.tsv"
    )
    assert vdj.data.shape[0] == 23
    assert len([i for i in vdj.data["locus"] if i == "TRD"]) == 0


@pytest.mark.usefixtures("create_testfolder", "dummy_adata_travdv")
def test_travdv_filter(create_testfolder, dummy_adata_travdv):
    """test check tradv filter"""
    vdj = ddl.Dandelion(
        create_testfolder / "dandelion" / "filtered_contig_dandelion.tsv"
    )
    assert vdj.data.shape[0] == 23
    assert len([i for i in vdj.data["locus"] if i == "TRD"]) == 0
    vdj2, adata = ddl.pp.filter_contigs(vdj, dummy_adata_travdv)
    assert vdj2.data.shape[0] == 8
