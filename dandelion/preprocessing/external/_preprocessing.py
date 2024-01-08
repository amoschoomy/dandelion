#!/usr/bin/env python
import os
import re
import scipy.stats

import numpy as np
import pandas as pd
import scanpy as sc

from anndata import AnnData
from pathlib import Path
from scanpy import logging as logg
from sklearn import mixture
from subprocess import run
from typing import Optional, Union, List

from ...utilities._utilities import *


def assigngenes_igblast(
    fasta: Union[str, Path],
    igblast_db: Optional[Union[str, Path]] = None,
    org: Literal["human", "mouse"] = "human",
    loci: Literal["ig", "tr"] = "ig",
    additional_args: List[str] = [],
):
    """
    Reannotate with IgBLASTn.

    Parameters
    ----------
    fasta : Union[str, Path]
        path to fasta file for reannotation.
    igblast_db : Optional[Union[str, Path]], optional
        path to igblast database.
    org : Literal["human", "mouse"], optional
        organism for germline sequences.
    loci : Literal["ig", "tr"], optional
        `ig` or `tr` mode for running igblastn.
    additional_args : List[str], optional
        Additional arguments to pass to `AssignGenes.py`.
    """
    env, igdb, fasta = set_igblast_env(igblast_db=igblast_db, input_file=fasta)
    outfolder = fasta.parent / "tmp"
    outfolder.mkdir(parents=True, exist_ok=True)

    informat_dict = {"blast": "_igblast.fmt7", "airr": "_igblast.tsv"}

    for fileformat in ["blast", "airr"]:
        outfile = fasta.stem + informat_dict[fileformat]
        cmd = [
            "AssignGenes.py",
            "igblast",
            "-s",
            str(fasta),
            "-b",
            str(igdb),
            "--organism",
            org,
            "--loci",
            loci,
            "--format",
            fileformat,
            "-o",
            str(outfolder / outfile),
        ]
        cmd += additional_args
        logg.info("Running command: %s\n" % (" ".join(cmd)))
        run(cmd, env=env)  # logs are printed to terminal


def makedb_igblast(
    fasta: Union[str, Path],
    igblast_output: Optional[Union[str, Path]] = None,
    germline: Optional[str] = None,
    org: Literal["human", "mouse"] = "human",
    db: Literal["imgt", "ogrdb"] = "imgt",
    extended: bool = True,
    additional_args: List[str] = [],
):
    """
    Parse IgBLAST output to AIRR format.

    Parameters
    ----------
    fasta : Union[str, Path]
        path to fasta file used for reannotation.
    igblast_output : Optional[Union[str, Path]], optional
        path to igblast output file.
    germline : Optional[str], optional
        path to germline database.
    org : Literal["human", "mouse"], optional
        organism of germline sequences.
    db : Literal["imgt", "ogrdb"], optional
        `imgt` or `ogrdb` reference database for running igblastn.
    extended : bool, optional
        whether or not to parse extended 10x annotations.
    additional_args: List[str], optional
        Additional arguments to pass to `MakeDb.py`.
    """
    env, gml, fasta = set_germline_env(
        germline=germline,
        org=org,
        input_file=fasta,
        db=db,
    )
    if igblast_output is None:
        indir = fasta.parent / "tmp"
        infile = fasta.stem + "_igblast.fmt7"
        igbo = indir / infile
    else:
        igbo = Path(igblast_output)

    cellranger_annotation = fasta.parent / (fasta.stem + "_annotations.csv")

    cmd = [
        "MakeDb.py",
        "igblast",
        "-i",
        str(igbo),
        "-s",
        str(fasta),
        "-r",
        str(gml),
        "--10x",
        str(cellranger_annotation),
    ]
    if extended:
        cmd = cmd + ["--extended"]
    for add_cmd in [[], ["--failed"]]:
        cmd = cmd + add_cmd + additional_args
        logg.info("Running command: %s\n" % (" ".join(cmd)))
        run(cmd, env=env)  # logs are printed to terminal


def parsedb_heavy(airr_file: Union[str, Path]):
    """
    Parse AIRR tsv file (heavy chain contigs only).

    Parameters
    ----------
    airr_file : Union[str, Path]
        path to AIRR tsv file.
    """
    outname = Path(airr_file).stem + "_heavy"
    cmd = [
        "ParseDb.py",
        "select",
        "-d",
        str(airr_file),
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


def parsedb_light(airr_file: Union[str, Path]):
    """
    Parse AIRR tsv file (light chain contigs only).

    Parameters
    ----------
    airr_file : Union[str, Path]
        path to AIRR tsv file.
    """
    outname = Path(airr_file).stem + "_light"
    cmd = [
        "ParseDb.py",
        "select",
        "-d",
        str(airr_file),
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
    airr_file: Union[str, Path],
    germline: Optional[List[str]] = None,
    org: Literal["human", "mouse"] = "human",
    genotyped_fasta: Optional[str] = None,
    mode: Optional[Literal["heavy", "light"]] = None,
    db: Literal["imgt", "ogrdb"] = "imgt",
    strain: Optional[
        Literal[
            "c57bl6",
            "balbc",
            "129S1_SvImJ",
            "AKR_J",
            "A_J",
            "BALB_c_ByJ",
            "BALB_c",
            "C3H_HeJ",
            "C57BL_6J",
            "C57BL_6",
            "CAST_EiJ",
            "CBA_J",
            "DBA_1J",
            "DBA_2J",
            "LEWES_EiJ",
            "MRL_MpJ",
            "MSM_MsJ",
            "NOD_ShiLtJ",
            "NOR_LtJ",
            "NZB_BlNJ",
            "PWD_PhJ",
            "SJL_J",
        ]
    ] = None,
    additional_args: List[str] = [],
):
    """
    Wrapper for CreateGermlines.py for reconstructing germline sequences.

    Parameters
    ----------
    airr_file : Union[str, Path]
        path to AIRR tsv file.
    germline : Optional[List[str]], optional
        location to germline fasta files as a list.
    org : Literal["human", "mouse"], optional
        organism for germline sequences.
    genotyped_fasta : Optional[str], optional
        location to V genotyped fasta file.
    mode : Optional[Literal["heavy", "light"]], optional
        whether to run on heavy or light mode. If left as None, heavy and
        light will be run together.
    db : Literal["imgt", "ogrdb"], optional
        `imgt` or `ogrdb` reference database.
    strain : Optional[Literal["c57bl6", "balbc", "129S1_SvImJ", "AKR_J", "A_J", "BALB_c_ByJ", "BALB_c", "C3H_HeJ", "C57BL_6J", "C57BL_6", "CAST_EiJ", "CBA_J", "DBA_1J", "DBA_2J", "LEWES_EiJ", "MRL_MpJ", "MSM_MsJ", "NOD_ShiLtJ", "NOR_LtJ", "NZB_BlNJ", "PWD_PhJ", "SJL_J"]], optional
        strain of mouse to use for germline sequences. Only for `db="ogrdb"`. Note that only "c57bl6", "balbc", "CAST_EiJ", "LEWES_EiJ", "MSM_MsJ", "NOD_ShiLt_J" and "PWD_PhJ" contains both heavy chain and light chain germline sequences as a set.
        The rest will not allow igblastn and MakeDB.py to generate a successful airr table (check the failed file). "c57bl6" and "balbc" are merged databases of "C57BL_6" with "C57BL_6J" and "BALB_c" with "BALB_c_ByJ" respectively. None defaults to all combined.
    additional_args : List[str], optional
        Additional arguments to pass to `CreateGermlines.py`.
    """
    env, gml, airr_file = set_germline_env(
        germline=germline,
        org=org,
        input_file=airr_file,
        db=db,
    )
    _strain = "_" + strain if strain is not None else ""
    if germline is None:
        if mode == "heavy":
            if genotyped_fasta is None:
                gml_ref = [
                    str(gml / ("imgt_" + org + _strain + "_IGHV.fasta")),
                ]
            else:
                gml_ref = [
                    str(genotyped_fasta),
                ]
            gml_ref += [
                str(gml / ("imgt_" + org + _strain + "_IGHJ.fasta")),
            ]
            _strainD = "" if strain not in NO_DS else _strain
            gml_ref += [
                str(gml / ("imgt_" + org + _strainD + "_IGHD.fasta")),
            ]
        elif mode == "light":
            gml_ref = [
                str(gml / ("imgt_" + org + _strain + "_IGKV.fasta")),
                str(gml / ("imgt_" + org + _strain + "_IGKJ.fasta")),
                str(gml / ("imgt_" + org + _strain + "_IGLV.fasta")),
                str(gml / ("imgt_" + org + _strain + "_IGLJ.fasta")),
            ]
        elif mode is None:
            if genotyped_fasta is None:
                gml_ref = [
                    str(gml / ("imgt_" + org + _strain + "_IGHV.fasta")),
                ]
            else:
                gml_ref = [
                    str(genotyped_fasta),
                ]
            gml_ref += [
                str(gml / ("imgt_" + org + _strain + "_IGHJ.fasta")),
            ]
            _strainD = "" if strain not in NO_DS else _strain
            gml_ref += [
                str(gml / ("imgt_" + org + _strainD + "_IGHD.fasta")),
            ]
            gml_ref += [
                str(gml / ("imgt_" + org + _strain + "_IGKV.fasta")),
                str(gml / ("imgt_" + org + _strain + "_IGKJ.fasta")),
                str(gml / ("imgt_" + org + _strain + "_IGLV.fasta")),
                str(gml / ("imgt_" + org + _strain + "_IGLJ.fasta")),
            ]
    else:
        if not isinstance(germline, list):
            germline = [str(germline)]
        gml_ref = germline
    cmd = [
        "CreateGermlines.py",
        "-d",
        str(airr_file),
        "-r",
    ]
    cmd = cmd + gml_ref + additional_args

    logg.info("Running command: %s\n" % (" ".join(cmd)))
    run(cmd, env=env)  # logs are printed to terminal


def tigger_genotype(
    airr_file: Union[str, Path],
    v_germline: Optional[Union[str, Path]] = None,
    outdir: Optional[Union[str, Path]] = None,
    org: Literal["human", "mouse"] = "human",
    fileformat: Literal["airr", "changeo"] = "airr",
    novel_: Literal["YES", "NO"] = "YES",
    db: Literal["imgt", "ogrdb"] = "imgt",
    strain: Optional[
        Literal[
            "c57bl6",
            "balbc",
            "129S1_SvImJ",
            "AKR_J",
            "A_J",
            "BALB_c_ByJ",
            "BALB_c",
            "C3H_HeJ",
            "C57BL_6J",
            "C57BL_6",
            "CAST_EiJ",
            "CBA_J",
            "DBA_1J",
            "DBA_2J",
            "LEWES_EiJ",
            "MRL_MpJ",
            "MSM_MsJ",
            "NOD_ShiLtJ",
            "NOR_LtJ",
            "NZB_BlNJ",
            "PWD_PhJ",
            "SJL_J",
        ]
    ] = None,
    additional_args: List[str] = [],
):
    """
    Reassign alleles with TIgGER in R.

    Parameters
    ----------
    airr_file : Union[str, Path]
        path to AIRR tsv file.
    v_germline : Optional[Union[str, Path]], optional
        fasta file containing IMGT-gapped V segment reference germlines.
    outdir : Optional[Union[str, Path]], optional
        output directory. Will be created if it does not exist.
        Defaults to the current working directory.
    org : Literal["human", "mouse"], optional
        organism for germline sequences.
    fileformat : Literal["airr", "changeo"], optional
        format for running tigger. Default is 'airr'. Also accepts 'changeo'.
    novel_ : Literal["YES", "NO"], optional
        whether or not to run novel allele discovery.
    db : Literal["imgt", "ogrdb"], optional
        `imgt` or `ogrdb` reference database.
    strain : Optional[Literal["c57bl6", "balbc", "129S1_SvImJ", "AKR_J", "A_J", "BALB_c_ByJ", "BALB_c", "C3H_HeJ", "C57BL_6J", "C57BL_6", "CAST_EiJ", "CBA_J", "DBA_1J", "DBA_2J", "LEWES_EiJ", "MRL_MpJ", "MSM_MsJ", "NOD_ShiLtJ", "NOR_LtJ", "NZB_BlNJ", "PWD_PhJ", "SJL_J"]], optional
        strain of mouse to use for germline sequences. Only for `db="ogrdb"`. Note that only "c57bl6", "balbc", "CAST_EiJ", "LEWES_EiJ", "MSM_MsJ", "NOD_ShiLt_J" and "PWD_PhJ" contains both heavy chain and light chain germline sequences as a set.
        The rest will not allow igblastn and MakeDB.py to generate a successful airr table (check the failed file). "c57bl6" and "balbc" are merged databases of "C57BL_6" with "C57BL_6J" and "BALB_c" with "BALB_c_ByJ" respectively. None defaults to all combined.
    additional_args : List[str], optional
        Additional arguments to pass to `tigger-genotype.R`.
    """
    env, gml, airr_file = set_germline_env(
        germline=v_germline,
        org=org,
        input_file=airr_file,
        db=db,
    )
    _strain = "_" + strain if strain is not None else ""
    if v_germline is None:
        v_gml = gml / (db + "_" + org + _strain + "_IGHV.fasta")
    else:
        v_gml = Path(v_germline)
    if outdir is not None:
        out_dir = Path(outdir)
    else:
        out_dir = airr_file.parent
    cmd = [
        "tigger-genotype.R",
        "-d",
        str(airr_file),
        "-r",
        str(v_gml),
        "-n",
        airr_file.stem,
        "-N",
        novel_,
        "-o",
        str(out_dir),
        "-f",
        fileformat,
    ]
    cmd = cmd + additional_args

    print("      Reassigning alleles")
    logg.info("Running command: %s\n" % (" ".join(cmd)))
    run(cmd, env=env)  # logs are printed to terminal


def recipe_scanpy_qc(
    adata: AnnData,
    layer: Optional[str] = None,
    mito_startswith: str = "MT-",
    max_genes: int = 2500,
    min_genes: int = 200,
    mito_cutoff: Optional[int] = 5,
    run_scrublet: bool = True,
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
    run_scrublet : bool, optional
        whether or not to run scrublet for doublet detection.
    pval_cutoff : float, optional
        maximum Benjamini-Hochberg corrected p value from doublet detection
        protocol allowed for a cell to pass filtering. Default is 0.1.
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
    # run basic scanpy pipeline
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
    if run_scrublet:
        # run scrublet
        try:
            import scrublet as scr
        except ImportError:
            raise ImportError(
                "Please install scrublet with pip install scrublet."
            )
        if layer is None:
            scrub = scr.Scrublet(_adata.X)
        else:
            scrub = scr.Scrublet(_adata.layers[layer])
        doublet_scores, _ = scrub.scrub_doublets(verbose=False)
        _adata.obs["scrublet_score"] = doublet_scores
        # overcluster prep.
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
    else:
        _adata.obs["is_doublet"] = False
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
    if not run_scrublet:
        _adata.obs = _adata.obs.drop(["is_doublet"], axis=1)
    adata.obs = _adata.obs.copy()
