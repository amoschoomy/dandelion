#!/usr/bin/env python
# @Author: kt16
import os
import re
import scipy.stats

import numpy as np
import pandas as pd
import scanpy as sc

from anndata import AnnData
from datetime import timedelta
from scanpy import logging as logg
from sklearn import mixture
from subprocess import run
from time import time
from typing import Union, Optional


from ...utilities._utilities import *


def assigngenes_igblast(
    fasta: str,
    igblast_db: Optional[str] = None,
    org: Literal["human", "mouse"] = "human",
    loci: Literal["ig", "tr"] = "ig",
):
    """
    Reannotate with IgBLASTn.

    Parameters
    ----------
    fasta : str
        fasta file for reannotation.
    igblast_db : Optional[str], optional
        path to igblast database.
    org : Literal["human", "mouse"], optional
        organism for germline sequences.
    loci : Literal["ig", "tr"], optional
        `ig` or `tr` mode for running igblastn.

    Raises
    ------
    KeyError
        if $IGDATA environmental variable is not set.
    """
    env = os.environ.copy()
    if igblast_db is None:
        try:
            igdb = env["IGDATA"]
        except KeyError:
            raise KeyError(
                (
                    "Environmental variable IGDATA must be set. Otherwise,"
                    + " please provide path to igblast database"
                )
            )
    else:
        env["IGDATA"] = igblast_db
        igdb = env["IGDATA"]

    outfolder = os.path.abspath(os.path.dirname(fasta)) + "/tmp"
    informat_dict = {"blast": "_igblast.fmt7", "airr": "_igblast.tsv"}
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    for fileformat in ["blast", "airr"]:
        outfile = (
            os.path.basename(fasta).split(".fasta")[0]
            + informat_dict[fileformat]
        )
        cmd = [
            "AssignGenes.py",
            "igblast",
            "-s",
            fasta,
            "-b",
            igdb,
            "--organism",
            org,
            "--loci",
            loci,
            "--format",
            fileformat,
            "-o",
            "{}/{}".format(outfolder, outfile),
        ]

        logg.info("Running command: %s\n" % (" ".join(cmd)))
        run(cmd, env=env)  # logs are printed to terminal


def makedb_igblast(
    fasta: str,
    igblast_output: Optional[str] = None,
    germline: Optional[str] = None,
    org: Literal["human", "mouse"] = "human",
    extended: bool = True,
):
    """
    Parse IgBLAST output to airr format.

    Parameters
    ----------
    fasta : str
        fasta file use for reannotation.
    igblast_output : Optional[str], optional
        igblast output file.
    germline : Optional[str], optional
        path to germline database.
    org : Literal["human", "mouse"], optional
        organism of germline sequences.
    extended : bool, optional
        whether or not to parse extended 10x annotations.

    Raises
    ------
    KeyError
        if $GERMLINE environmental variable is not set.
    """
    env = os.environ.copy()
    if germline is None:
        try:
            gml = env["GERMLINE"]
        except KeyError:
            raise KeyError(
                (
                    "Environmental variable GERMLINE must be set. Otherwise,"
                    + " please provide path to folder containing germline"
                    + " fasta files."
                )
            )
        gml = gml + "imgt/" + org + "/vdj/"
    else:
        env["GERMLINE"] = germline
        gml = germline

    if igblast_output is None:
        indir = os.path.dirname(fasta) + "/tmp"
        infile = os.path.basename(fasta).split(".fasta")[0] + "_igblast.fmt7"
        igbo = "{}/{}".format(indir, infile)
    else:
        igbo = igblast_output

    cellranger_annotation = "{}/{}".format(
        os.path.dirname(fasta),
        os.path.basename(fasta).replace(".fasta", "_annotations.csv"),
    )

    if extended:
        cmd1 = [
            "MakeDb.py",
            "igblast",
            "-i",
            igbo,
            "-s",
            fasta,
            "-r",
            gml,
            "--10x",
            cellranger_annotation,
            "--extended",
        ]
        cmd2 = [
            "MakeDb.py",
            "igblast",
            "-i",
            igbo,
            "-s",
            fasta,
            "-r",
            gml,
            "--10x",
            cellranger_annotation,
            "--extended",
            "--failed",
        ]
    else:
        cmd1 = [
            "MakeDb.py",
            "igblast",
            "-i",
            igbo,
            "-s",
            fasta,
            "-r",
            gml,
            "--10x",
            cellranger_annotation,
        ]
        cmd2 = [
            "MakeDb.py",
            "igblast",
            "-i",
            igbo,
            "-s",
            fasta,
            "-r",
            gml,
            "--10x",
            cellranger_annotation,
            "--failed",
        ]

    logg.info("Running command: %s\n" % (" ".join(cmd1)))
    run(cmd1, env=env)  # logs are printed to terminal
    logg.info("Running command: %s\n" % (" ".join(cmd2)))
    run(cmd2, env=env)  # logs are printed to terminal


def parsedb_heavy(db_file: str):
    """
    Parse AIRR table (heavy chain contigs only).

    Parameters
    ----------
    db_file : str
        path to AIRR table.
    """
    outname = os.path.basename(db_file).split(".tsv")[0] + "_heavy"

    cmd = [
        "ParseDb.py",
        "select",
        "-d",
        db_file,
        "-f",
        "locus",
        "-u",
        "IGH",
        "--logic",
        "all",
        "--regex",
        "--outname",
        outname,
    ]

    logg.info("Running command: %s\n" % (" ".join(cmd)))
    run(cmd)  # logs are printed to terminal


def parsedb_light(db_file: str):
    """
    Parse AIRR table (light chain contigs only).

    Parameters
    ----------
    db_file : str
        path to AIRR table.

    """
    outname = os.path.basename(db_file).split(".tsv")[0] + "_light"

    cmd = [
        "ParseDb.py",
        "select",
        "-d",
        db_file,
        "-f",
        "locus",
        "-u",
        "IG[LK]",
        "--logic",
        "all",
        "--regex",
        "--outname",
        outname,
    ]

    logg.info("Running command: %s\n" % (" ".join(cmd)))
    run(cmd)  # logs are printed to terminal


def creategermlines(
    db_file: str,
    germtypes: Optional[str] = None,
    germline: Optional[str] = None,
    org: Literal["human", "mouse"] = "human",
    genotype_fasta: Optional[str] = None,
    v_field: Optional[Literal["v_call", "v_call_genotyped"]] = None,
    cloned: bool = False,
    mode: Optional[Literal["heavy", "light"]] = None,
):
    """
    Wrapper for CreateGermlines.py for reconstructing germline sequences.

    Parameters
    ----------
    db_file : str
        path to AIRR table.
    germtypes : Optional[str], optional
        germline type for reconstuction.
    germline : Optional[str], optional
        location to germline fasta files.
    org : Literal["human", "mouse"], optional
        organism for germline sequences.
    genotype_fasta : Optional[str], optional
        location to corrected v germine fasta file.
    v_field : Optional[Literal["v_call", "v_call_genotyped"]], optional
        name of column for v segment to perform reconstruction.
    cloned : bool, optional
        whether or not to run with cloned option.
    mode : Optional[Literal["heavy", "light"]], optional
        whether to run on heavy or light mode. If left as None, heavy and
        light will be run together.
    verbose : bool, optional
        whether or not to print the command used in terminal.

    Raises
    ------
    KeyError
        if $GERMLINE environmental variable is not set.
    """
    env = os.environ.copy()
    if germline is None:
        try:
            gml = env["GERMLINE"]
        except KeyError:
            raise KeyError(
                (
                    "Environmental variable GERMLINE must be set."
                    + " Otherwise, please provide path to folder"
                    + " containing germline fasta files."
                )
            )
        gml = gml + "imgt/" + org + "/vdj/"
    else:
        env["GERMLINE"] = germline
        gml = germline

    if germtypes is None:
        germ_type = "dmask"
    else:
        germ_type = germtypes

    if cloned:
        if mode == "heavy":
            print(
                (
                    "            Reconstructing heavy chain {}".format(
                        germ_type
                    )
                    + " germline sequences with {} for each clone.".format(
                        v_field
                    )
                )
            )
            if genotype_fasta is None:
                if germline is None:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "--cloned",
                        "-r",
                        gml + "/imgt_" + org + "_IGHV.fasta",
                        gml + "/imgt_" + org + "_IGHD.fasta",
                        gml + "/imgt_" + org + "_IGHJ.fasta",
                        "--vf",
                        v_field,
                    ]
                else:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "--cloned",
                        "-r",
                        gml,
                        "--vf",
                        v_field,
                    ]
            else:
                if germline is None:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "--cloned",
                        "-r",
                        genotype_fasta,
                        gml + "/imgt_" + org + "_IGHD.fasta",
                        gml + "/imgt_" + org + "_IGHJ.fasta",
                        "--vf",
                        v_field,
                    ]
                else:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "--cloned",
                        "-r",
                        genotype_fasta,
                        gml,
                        "--vf",
                        v_field,
                    ]
        elif mode == "light":
            print(
                (
                    "            Reconstructing light chain {}".format(
                        germ_type
                    )
                    + " germline sequences with {} for each clone.".format(
                        v_field
                    )
                )
            )
            if germline is None:
                cmd = [
                    "CreateGermlines.py",
                    "-d",
                    db_file,
                    "-g",
                    germ_type,
                    "--cloned",
                    "-r",
                    gml + "/imgt_" + org + "_IGKV.fasta",
                    gml + "/imgt_" + org + "_IGKJ.fasta",
                    gml + "/imgt_" + org + "_IGLV.fasta",
                    gml + "/imgt_" + org + "_IGLJ.fasta",
                    "--vf",
                    v_field,
                ]
            else:
                cmd = [
                    "CreateGermlines.py",
                    "-d",
                    db_file,
                    "-g",
                    germ_type,
                    "--cloned",
                    "-r",
                    gml,
                    "--vf",
                    v_field,
                ]
        elif mode is None:
            print(
                (
                    "            Reconstructing {}".format(germ_type)
                    + " germline sequences with {} for each clone.".format(
                        v_field
                    )
                )
            )
            if genotype_fasta is None:
                if germline is None:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "--cloned",
                        "-r",
                        gml + "/imgt_" + org + "_IGHV.fasta",
                        gml + "/imgt_" + org + "_IGHD.fasta",
                        gml + "/imgt_" + org + "_IGHJ.fasta",
                        gml + "/imgt_" + org + "_IGKV.fasta",
                        gml + "/imgt_" + org + "_IGKJ.fasta",
                        gml + "/imgt_" + org + "_IGLV.fasta",
                        gml + "/imgt_" + org + "_IGLJ.fasta",
                        "--vf",
                        v_field,
                    ]
                else:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "--cloned",
                        "-r",
                        gml,
                        "--vf",
                        v_field,
                    ]
            else:
                if germline is None:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "--cloned",
                        "-r",
                        genotype_fasta,
                        gml + "/imgt_" + org + "_IGHD.fasta",
                        gml + "/imgt_" + org + "_IGHJ.fasta",
                        gml + "/imgt_" + org + "_IGKV.fasta",
                        gml + "/imgt_" + org + "_IGKJ.fasta",
                        gml + "/imgt_" + org + "_IGLV.fasta",
                        gml + "/imgt_" + org + "_IGLJ.fasta",
                        "--vf",
                        v_field,
                    ]
                else:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "--cloned",
                        "-r",
                        genotype_fasta,
                        gml,
                        "--vf",
                        v_field,
                    ]
    else:
        if mode == "heavy":
            print(
                (
                    "            Reconstructing heavy chain {}".format(
                        germ_type
                    )
                    + " germline sequences with {}.".format(v_field)
                )
            )
            if genotype_fasta is None:
                if germline is None:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "-r",
                        gml + "/imgt_" + org + "_IGHV.fasta",
                        gml + "/imgt_" + org + "_IGHD.fasta",
                        gml + "/imgt_" + org + "_IGHJ.fasta",
                        "--vf",
                        v_field,
                    ]
                else:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "-r",
                        gml,
                        "--vf",
                        v_field,
                    ]
            else:
                if germline is None:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "-r",
                        genotype_fasta,
                        gml + "/imgt_" + org + "_IGHD.fasta",
                        gml + "/imgt_" + org + "_IGHJ.fasta",
                        "--vf",
                        v_field,
                    ]
                else:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "-r",
                        genotype_fasta,
                        gml,
                        "--vf",
                        v_field,
                    ]
        elif mode == "light":
            print(
                (
                    "            Reconstructing light chain {}".format(
                        germ_type
                    )
                    + " germline sequences with {}.".format(v_field)
                )
            )
            if germline is None:
                cmd = [
                    "CreateGermlines.py",
                    "-d",
                    db_file,
                    "-g",
                    germ_type,
                    "-r",
                    gml + "/imgt_" + org + "_IGKV.fasta",
                    gml + "/imgt_" + org + "_IGKJ.fasta",
                    gml + "/imgt_" + org + "_IGLV.fasta",
                    gml + "/imgt_" + org + "_IGLJ.fasta",
                    "--vf",
                    v_field,
                ]
            else:
                cmd = [
                    "CreateGermlines.py",
                    "-d",
                    db_file,
                    "-g",
                    germ_type,
                    "-r",
                    gml,
                    "--vf",
                    v_field,
                ]
        elif mode is None:
            print(
                (
                    "            Reconstructing {}".format(germ_type)
                    + " germline sequences with {} for each clone.".format(
                        v_field
                    )
                )
            )
            if genotype_fasta is None:
                if germline is None:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "-r",
                        gml + "/imgt_" + org + "_IGHV.fasta",
                        gml + "/imgt_" + org + "_IGHD.fasta",
                        gml + "/imgt_" + org + "_IGHJ.fasta",
                        gml + "/imgt_" + org + "_IGKV.fasta",
                        gml + "/imgt_" + org + "_IGKJ.fasta",
                        gml + "/imgt_" + org + "_IGLV.fasta",
                        gml + "/imgt_" + org + "_IGLJ.fasta",
                        "--vf",
                        v_field,
                    ]
                else:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "-r",
                        gml,
                        "--vf",
                        v_field,
                    ]
            else:
                if germline is None:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "-r",
                        genotype_fasta,
                        gml + "/imgt_" + org + "_IGHD.fasta",
                        gml + "/imgt_" + org + "_IGHJ.fasta",
                        gml + "/imgt_" + org + "_IGKV.fasta",
                        gml + "/imgt_" + org + "_IGKJ.fasta",
                        gml + "/imgt_" + org + "_IGLV.fasta",
                        gml + "/imgt_" + org + "_IGLJ.fasta",
                        "--vf",
                        v_field,
                    ]
                else:
                    cmd = [
                        "CreateGermlines.py",
                        "-d",
                        db_file,
                        "-g",
                        germ_type,
                        "-r",
                        genotype_fasta,
                        gml,
                        "--vf",
                        v_field,
                    ]

    logg.info("Running command: %s\n" % (" ".join(cmd)))
    run(cmd, env=env)  # logs are printed to terminal


def tigger_genotype(
    data: str,
    v_germline: Optional[str] = None,
    outdir: Optional[str] = None,
    org: Literal["human", "mouse"] = "human",
    fileformat: Literal["airr", "changeo"] = "airr",
    novel_: Literal["YES", "NO"] = "YES",
):
    """
    Reassign alleles with TIgGER in R.

    Parameters
    ----------
    data : str
        vdj tabulated data, in Change-O (TAB) or AIRR (TSV) format.
    v_germline : Optional[str], optional
        fasta file containing IMGT-gapped V segment reference germlines.
    outdir : Optional[str], optional
        output directory. Will be created if it does not exist.
        Defaults to the current working directory.
    org : Literal["human", "mouse"], optional
        organism for germline sequences.
    fileformat : Literal["airr", "changeo"], optional
        format for running tigger. Default is 'airr'. Also accepts 'changeo'.
    novel_ : Literal["YES", "NO"], optional
        whether or not to run novel allele discovery.

    Raises
    ------
    KeyError
        if `GERMLINE` environmental variable is not set.
    """
    start_time = time()
    env = os.environ.copy()
    if v_germline is None:
        try:
            gml = env["GERMLINE"]
        except:
            raise KeyError(
                (
                    "Environmental variable GERMLINE is not set. Please provide"
                    + " either the path to the folder containing the germline"
                    + " IGHV fasta file, or direct path to the germline IGHV"
                    + " fasta file."
                )
            )
        gml = gml + "imgt/" + org + "/vdj/imgt_" + org + "_IGHV.fasta"
    else:
        if os.path.isdir(v_germline):
            gml = v_germline.rstrip("/") + "imgt_" + org + "_IGHV.fasta"
            if not os.path.isfile(gml):
                raise KeyError(
                    (
                        "Input for germline is incorrect. Please rename IGHV"
                        + " germline file to '{}'.".format(gml)
                        + " Otherwise, please provide path to folder containing"
                        + " the germline IGHV fasta file, or direct path to the"
                        + " germline IGHV fasta file."
                    )
                )
        else:
            if not v_germline.endswith(".fasta"):
                raise KeyError(
                    (
                        "Input for germline is incorrect {}.".format(v_germline)
                        + " Please provide path to folder containing the germline"
                        + " IGHV fasta file, or direct path to the germline IGHV"
                        + " fasta file."
                    )
                )
            if (os.path.isfile(v_germline)) & ("ighv" in v_germline.lower()):
                gml = v_germline

    if outdir is not None:
        out_dir = outdir + "/"
    else:
        out_dir = os.path.dirname(data)

    cmd = [
        "tigger-genotype.R",
        "-d",
        data,
        "-r",
        gml,
        "-n",
        os.path.basename(data).split(".tsv")[0],
        "-N",
        novel_,
        "-o",
        out_dir,
        "-f",
        fileformat,
    ]

    print("      Reassigning alleles")
    logg.info("Running command: %s\n" % (" ".join(cmd)))
    run(cmd, env=env)  # logs are printed to terminal
    elapsed_time_secs = time() - start_time
    msg = (
        "tigger-genotype execution took: %s"
        % timedelta(seconds=round(elapsed_time_secs))
        + " secs (Wall clock time)\n"
    )
    logg.info(msg)


def recipe_scanpy_qc(
    adata: AnnData,
    layer: Optional[str] = None,
    mito_startswith: str = "MT-",
    max_genes: int = 2500,
    min_genes: int = 200,
    mito_cutoff: Optional[int] = 5,
    pval_cutoff: float = 0.1,
    min_counts: Optional[int] = None,
    max_counts: Optional[int] = None,
    blacklist: Optional[List[str]] = None,
    vdj_pattern: str = "^TR[AB][VDJ]|^IG[HKL][VDJC]",
):
    """
    Recipe for running a standard scanpy QC workflow.

    Parameters
    ----------
    adata : AnnData
        annotated data matrix of shape n_obs × n_vars. Rows correspond to cells
        and columns to genes.
    layer : Optional[str], optional
        name of layer to run scrublet on if supplied.
    mito_startswith : str, optional
        string pattern used for searching mitochondrial genes.
    max_genes : int, optional
        maximum number of genes expressed required for a cell to pass filtering
    min_genes : int, optional
        minimum number of genes expressed required for a cell to pass filtering
    mito_cutoff : Optional[int], optional
        maximum percentage mitochondrial content allowed for a cell to pass filtering.
    pval_cutoff : float, optional
        maximum Benjamini-Hochberg corrected p value from doublet detection
        protocol allowed for a cell to pass filtering. Default is 0.05.
    min_counts : Optional[int], optional
        minimum number of counts required for a cell to pass filtering.
    max_counts : Optional[int], optional
        maximum number of counts required for a cell to pass filtering.
    blacklist : Optional[List[str]], optional
        if provided, will exclude these genes from highly variable genes list.
    vdj_pattern : str, optional
        string pattern for search VDJ genes to exclude from highly variable genes.

    Raises
    ------
    ImportError
        if `scrublet` not installed.
    """
    _adata = adata.copy()
    # run scrublet
    try:
        import scrublet as scr
    except ImportError:
        raise ImportError("Please install scrublet with pip install scrublet.")

    if layer is None:
        scrub = scr.Scrublet(_adata.X)
    else:
        scrub = scr.Scrublet(_adata.layers[layer])
    doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)
    _adata.obs["scrublet_score"] = doublet_scores
    # overcluster prep. run basic scanpy pipeline
    sc.pp.filter_cells(_adata, min_genes=0)
    _adata.var["mt"] = _adata.var_names.str.startswith(mito_startswith)
    sc.pp.calculate_qc_metrics(
        _adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    if mito_cutoff is None:
        # use a model-based method to determine the cut off
        # for mitochondrial content
        gmm = mixture.GaussianMixture(
            n_components=2, max_iter=1000, covariance_type="full"
        )
        X = _adata.obs[["pct_counts_mt", "n_genes_by_counts"]]
        _adata.obs["gmm_pct_count_clusters"] = gmm.fit(X).predict(X)
        # use a simple metric to workout which cluster
        # is the one that contains lower mito content?
        A1 = (
            _adata[_adata.obs["gmm_pct_count_clusters"] == 0]
            .obs["pct_counts_mt"]
            .mean()
        )
        B1 = (
            _adata[_adata.obs["gmm_pct_count_clusters"] == 1]
            .obs["pct_counts_mt"]
            .mean()
        )
        A2 = (
            _adata[_adata.obs["gmm_pct_count_clusters"] == 0]
            .obs["n_genes_by_counts"]
            .mean()
        )
        B2 = (
            _adata[_adata.obs["gmm_pct_count_clusters"] == 1]
            .obs["n_genes_by_counts"]
            .mean()
        )
        if (A1 > B1) and (A2 < B2):
            keepdict = {0: False, 1: True}
        else:
            keepdict = {1: False, 0: True}
        _adata.obs["gmm_pct_count_clusters_keep"] = [
            keepdict[x] for x in _adata.obs["gmm_pct_count_clusters"]
        ]
    sc.pp.normalize_total(_adata, target_sum=1e4)
    sc.pp.log1p(_adata)
    sc.pp.highly_variable_genes(
        _adata, min_mean=0.0125, max_mean=3, min_disp=0.5
    )
    for i in _adata.var.index:
        if vdj_pattern is not None:
            if re.search(vdj_pattern, i):
                _adata.var.at[i, "highly_variable"] = False
        if blacklist is not None:
            if i in blacklist:
                _adata.var.at[i, "highly_variable"] = False
    _adata = _adata[:, _adata.var["highly_variable"]].copy()
    sc.pp.scale(_adata, max_value=10)
    sc.tl.pca(_adata, svd_solver="arpack")
    sc.pp.neighbors(_adata, n_neighbors=10, n_pcs=50)
    # overclustering proper - do basic clustering first,
    # then cluster each cluster
    sc.tl.leiden(_adata)
    for clus in list(np.unique(_adata.obs["leiden"]))[0]:
        sc.tl.leiden(
            _adata, restrict_to=("leiden", [clus]), key_added="leiden_R"
        )
    # weird how the new anndata/scanpy is forcing this
    for clus in list(np.unique(_adata.obs["leiden"]))[1:]:
        sc.tl.leiden(
            _adata, restrict_to=("leiden_R", [clus]), key_added="leiden_R"
        )
    # compute the cluster scores - the median of Scrublet scores per
    # overclustered cluster
    for clus in np.unique(_adata.obs["leiden_R"]):
        _adata.obs.loc[
            _adata.obs["leiden_R"] == clus, "scrublet_cluster_score"
        ] = np.median(
            _adata.obs.loc[_adata.obs["leiden_R"] == clus, "scrublet_score"]
        )
    # now compute doublet p-values. figure out the median and mad
    # (from above-median values) for the distribution
    med = np.median(_adata.obs["scrublet_cluster_score"])
    mask = _adata.obs["scrublet_cluster_score"] > med
    mad = np.median(_adata.obs["scrublet_cluster_score"][mask] - med)
    # 1 sided test for catching outliers
    pvals = 1 - scipy.stats.norm.cdf(
        _adata.obs["scrublet_cluster_score"], loc=med, scale=1.4826 * mad
    )
    _adata.obs["scrublet_score_bh_pval"] = bh(pvals)
    # threshold the p-values to get doublet calls.
    _adata.obs["is_doublet"] = (
        _adata.obs["scrublet_score_bh_pval"] < pval_cutoff
    )
    if mito_cutoff is not None:
        if min_counts is None and max_counts is None:
            _adata.obs["filter_rna"] = (
                (
                    pd.Series(
                        [
                            ((n < min_genes) or (n > max_genes))
                            for n in _adata.obs["n_genes_by_counts"]
                        ],
                        index=_adata.obs.index,
                    )
                )
                | (_adata.obs["pct_counts_mt"] >= mito_cutoff)
                | (_adata.obs.is_doublet)
            )
        else:
            if min_counts is not None:
                if max_counts is not None:
                    _adata.obs["filter_rna"] = (
                        (
                            pd.Series(
                                [
                                    ((n < min_genes) or (n > max_genes))
                                    for n in _adata.obs["n_genes_by_counts"]
                                ],
                                index=_adata.obs.index,
                            )
                        )
                        | (
                            pd.Series(
                                [
                                    min_counts < n > max_counts
                                    for n in _adata.obs["total_counts"]
                                ],
                                index=_adata.obs.index,
                            )
                        )
                        | (_adata.obs["pct_counts_mt"] >= mito_cutoff)
                        | (_adata.obs.is_doublet)
                    )
                else:
                    _adata.obs["filter_rna"] = (
                        (
                            pd.Series(
                                [
                                    ((n < min_genes) or (n > max_genes))
                                    for n in _adata.obs["n_genes_by_counts"]
                                ],
                                index=_adata.obs.index,
                            )
                        )
                        | (
                            pd.Series(
                                [
                                    n < min_counts
                                    for n in _adata.obs["total_counts"]
                                ],
                                index=_adata.obs.index,
                            )
                        )
                        | (_adata.obs["pct_counts_mt"] >= mito_cutoff)
                        | (_adata.obs.is_doublet)
                    )
            else:
                if max_counts is not None:
                    _adata.obs["filter_rna"] = (
                        (
                            pd.Series(
                                [
                                    ((n < min_genes) or (n > max_genes))
                                    for n in _adata.obs["n_genes_by_counts"]
                                ],
                                index=_adata.obs.index,
                            )
                        )
                        | (
                            pd.Series(
                                [
                                    n > max_counts
                                    for n in _adata.obs["total_counts"]
                                ],
                                index=_adata.obs.index,
                            )
                        )
                        | (_adata.obs["pct_counts_mt"] >= mito_cutoff)
                        | (_adata.obs.is_doublet)
                    )
    else:
        if min_counts is None and max_counts is None:
            _adata.obs["filter_rna"] = (
                (
                    pd.Series(
                        [
                            ((n < min_genes) or (n > max_genes))
                            for n in _adata.obs["n_genes_by_counts"]
                        ],
                        index=_adata.obs.index,
                    )
                )
                | ~(_adata.obs.gmm_pct_count_clusters_keep)
                | (_adata.obs.is_doublet)
            )
        else:
            if min_counts is not None:
                if max_counts is not None:
                    _adata.obs["filter_rna"] = (
                        (
                            pd.Series(
                                [
                                    ((n < min_genes) or (n > max_genes))
                                    for n in _adata.obs["n_genes_by_counts"]
                                ],
                                index=_adata.obs.index,
                            )
                        )
                        | (
                            pd.Series(
                                [
                                    min_counts < n > max_counts
                                    for n in _adata.obs["total_counts"]
                                ],
                                index=_adata.obs.index,
                            )
                        )
                        | ~(_adata.obs.gmm_pct_count_clusters_keep)
                        | (_adata.obs.is_doublet)
                    )
                else:
                    _adata.obs["filter_rna"] = (
                        (
                            pd.Series(
                                [
                                    ((n < min_genes) or (n > max_genes))
                                    for n in _adata.obs["n_genes_by_counts"]
                                ],
                                index=_adata.obs.index,
                            )
                        )
                        | (
                            pd.Series(
                                [
                                    n < min_counts
                                    for n in _adata.obs["total_counts"]
                                ],
                                index=_adata.obs.index,
                            )
                        )
                        | ~(_adata.obs.gmm_pct_count_clusters_keep)
                        | (_adata.obs.is_doublet)
                    )
            else:
                if max_counts is not None:
                    _adata.obs["filter_rna"] = (
                        (
                            pd.Series(
                                [
                                    ((n < min_genes) or (n > max_genes))
                                    for n in _adata.obs["n_genes_by_counts"]
                                ],
                                index=_adata.obs.index,
                            )
                        )
                        | (
                            pd.Series(
                                [
                                    n > max_counts
                                    for n in _adata.obs["total_counts"]
                                ],
                                index=_adata.obs.index,
                            )
                        )
                        | ~(_adata.obs.gmm_pct_count_clusters_keep)
                        | (_adata.obs.is_doublet)
                    )
    bool_dict = {True: "True", False: "False"}

    _adata.obs["is_doublet"] = [bool_dict[x] for x in _adata.obs["is_doublet"]]
    _adata.obs["filter_rna"] = [bool_dict[x] for x in _adata.obs["filter_rna"]]

    # removing columns that probably don't need anymore
    if mito_cutoff is not None:
        _adata.obs = _adata.obs.drop(
            [
                "leiden",
                "leiden_R",
                "scrublet_cluster_score",
                "scrublet_score_bh_pval",
            ],
            axis=1,
        )
    else:
        _adata.obs = _adata.obs.drop(
            [
                "leiden",
                "leiden_R",
                "scrublet_cluster_score",
                "scrublet_score_bh_pval",
                "gmm_pct_count_clusters",
            ],
            axis=1,
        )
    adata.obs = _adata.obs.copy()
