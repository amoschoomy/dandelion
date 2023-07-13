#!/usr/bin/env python
import pytest
import dandelion as ddl


@pytest.mark.usefixtures("create_testfolder", "fasta_10x_mouse")
def test_write_fasta(create_testfolder, fasta_10x_mouse):
    """test_write_fasta"""
    out_fasta = create_testfolder / "filtered_contig.fasta"
    fh = open(out_fasta, "w")
    fh.close()
    out = ""
    for l in fasta_10x_mouse:
        out = ">" + l + "\n" + fasta_10x_mouse[l] + "\n"
        ddl.utl.Write_output(out, out_fasta)
    assert len(list(create_testfolder.iterdir())) == 1


@pytest.mark.usefixtures("create_testfolder", "annotation_10x_mouse")
def test_write_annotation(create_testfolder, annotation_10x_mouse):
    """test_write_annotation"""
    out_file = create_testfolder / "filtered_contig_annotations.csv"
    annotation_10x_mouse.to_csv(out_file, index=False)
    assert len(list(create_testfolder.iterdir())) == 2


@pytest.mark.usefixtures("create_testfolder")
def test_formatfasta(create_testfolder):
    """test_formatfasta"""
    ddl.pp.format_fastas(str(create_testfolder))
    assert len(list((create_testfolder / "dandelion").iterdir())) == 2


@pytest.mark.usefixtures("database_paths_mouse")
def test_updateblastdb(database_paths_mouse):
    """test update blast"""
    ddl.utl.makeblastdb(database_paths_mouse["blastdb_fasta"])


@pytest.mark.usefixtures("create_testfolder", "database_paths_mouse")
def test_reannotategenes_original(create_testfolder, database_paths_mouse):
    """test_reannotategenes_original"""
    ddl.pp.format_fastas(str(create_testfolder))
    ddl.pp.reannotate_genes(
        str(create_testfolder),
        igblast_db=database_paths_mouse["igblast_db"],
        germline=database_paths_mouse["germline"],
        flavour="original",
        org="mouse",
    )
    assert len(list((create_testfolder / "dandelion" / "tmp").iterdir())) == 4


@pytest.mark.usefixtures("create_testfolder", "database_paths_mouse")
def test_reannotategenes_other(create_testfolder, database_paths_mouse):
    """test_reannotategenes_other"""
    ddl.pp.format_fastas(str(create_testfolder))
    ddl.pp.reannotate_genes(
        str(create_testfolder),
        igblast_db=database_paths_mouse["igblast_db"],
        germline=database_paths_mouse["germline"],
        extended=False,
        org="mouse",
    )
    assert len(list((create_testfolder / "dandelion" / "tmp").iterdir())) == 6
