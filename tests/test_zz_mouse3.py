#!/usr/bin/env python
import pandas as pd
import dandelion as ddl
import pytest
from pathlib import Path
import sys


@pytest.mark.usefixtures("create_testfolder", "fasta_10x_mouse")
def test_write_fasta(create_testfolder, fasta_10x_mouse):
    """test write fasta"""
    out_fasta = create_testfolder / "filtered_contig.fasta"
    ddl.utl.write_fasta(fasta_dict=fasta_10x_mouse, out_fasta=out_fasta)
    assert len(list(create_testfolder.iterdir())) == 1


@pytest.mark.usefixtures("create_testfolder", "annotation_10x_mouse")
def test_write_annotation(create_testfolder, annotation_10x_mouse):
    """test write annot"""
    out_file = create_testfolder / "filtered_contig_annotations.csv"
    annotation_10x_mouse.to_csv(out_file, index=False)
    assert len(list(create_testfolder.iterdir())) == 2


@pytest.mark.usefixtures("create_testfolder")
def test_formatfasta(create_testfolder):
    """test format fasta"""
    ddl.pp.format_fastas(create_testfolder)
    assert len(list((create_testfolder / "dandelion").iterdir())) == 2


@pytest.mark.usefixtures("create_testfolder", "database_paths_mouse")
def test_reannotategenes_strict(create_testfolder, database_paths_mouse):
    """test reannotate"""
    ddl.pp.format_fastas(create_testfolder)
    ddl.pp.reannotate_genes(
        create_testfolder,
        igblast_db=database_paths_mouse["igblast_db"],
        germline=database_paths_mouse["germline"],
        flavour="strict",
        reassign_dj=True,
        org="mouse",
    )
    assert len(list((create_testfolder / "dandelion" / "tmp").iterdir())) == 6


@pytest.mark.usefixtures("create_testfolder", "database_paths_mouse")
def test_reassignalleles(create_testfolder, database_paths_mouse):
    """test reassign"""
    ddl.pp.reassign_alleles(
        str(create_testfolder),
        combined_folder="test_mouse",
        germline=database_paths_mouse["germline"],
        org="mouse",
        novel=True,
        plot=False,
    )
    assert len(list((create_testfolder / "dandelion" / "tmp").iterdir())) == 9


@pytest.mark.usefixtures("database_paths_mouse")
def test_updateblastdb(database_paths_mouse):
    """test update blast"""
    ddl.utl.makeblastdb(database_paths_mouse["blastdb_fasta"])


@pytest.mark.usefixtures(
    "create_testfolder", "database_paths_mouse", "balbc_ighg_primers"
)
def test_assignsisotypes(
    create_testfolder, database_paths_mouse, balbc_ighg_primers
):
    """test assign isotype"""
    ddl.pp.assign_isotypes(
        create_testfolder,
        blastdb=database_paths_mouse["blastdb_fasta"],
        correction_dict=balbc_ighg_primers,
        plot=False,
    )
    assert len(list((create_testfolder / "dandelion").iterdir())) == 2


@pytest.mark.usefixtures(
    "create_testfolder", "processed_files", "dummy_adata_mouse"
)
def test_filtercontigs(create_testfolder, processed_files, dummy_adata_mouse):
    """test filter contigs"""
    f = create_testfolder / "dandelion" / processed_files["filtered"]
    dat = pd.read_csv(f, sep="\t")
    vdj, adata = ddl.pp.filter_contigs(dat, dummy_adata_mouse)
    f1 = create_testfolder / "test.h5ddl"
    f2 = create_testfolder / "test.h5ad"
    vdj.write_h5ddl(f1)
    adata.write_h5ad(f2)
    # assert dat.shape[0] == 1278
    # assert vdj.data.shape[0] == 948
    # assert vdj.metadata.shape[0] == 444
    # assert adata.n_obs == 547


@pytest.mark.skipif(sys.platform == "darwin", reason="macos CI stalls.")
@pytest.mark.usefixtures("create_testfolder")
def test_generate_network_sfdp(create_testfolder):
    """test generate network sfdp"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    with pytest.raises(ValueError):
        ddl.tl.generate_network(vdj, compute_layout=False)
    ddl.tl.find_clones(vdj)
    ddl.tl.generate_network(vdj, layout_method="sfdp")
    assert vdj.layout is not None
