#!/usr/bin/env python
import anndata as ad
import functools
import numpy as np
import os
import pandas as pd
import re
import tempfile

from anndata import AnnData
from Bio import Align
from operator import countOf
from pathlib import Path
from plotnine import (
    ggplot,
    geom_bar,
    geom_col,
    ggtitle,
    scale_fill_manual,
    coord_flip,
    options,
    element_blank,
    aes,
    xlab,
    ylab,
    facet_wrap,
    facet_grid,
    theme_classic,
    theme,
    annotate,
    theme_bw,
    geom_histogram,
    geom_vline,
    save_as_pdf_pages,
)
from scanpy import logging as logg
from subprocess import run
from time import sleep
from tqdm import tqdm
from typing import Union, List, Tuple, Optional

from dandelion.preprocessing.external._preprocessing import (
    assigngenes_igblast,
    makedb_igblast,
    parsedb_heavy,
    parsedb_light,
    tigger_genotype,
    creategermlines,
)
from dandelion.utilities._core import *
from dandelion.utilities._io import *
from dandelion.utilities._utilities import *
from dandelion.tools._tools import transfer


def format_fasta(
    fasta: Union[str, Path],
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    sep: Optional[str] = None,
    remove_trailing_hyphen_number: bool = True,
    high_confidence_filtering: bool = False,
    out_dir: Optional[Union[str, Path]] = None,
    filename_prefix: Optional[str] = None,
):
    """
    Add prefix to the headers/contig ids in input fasta and annotation file.

    Parameters
    ----------
    fasta : Union[str, Path]
        path to fasta file.
    prefix : Optional[str], optional
        prefix to append to the headers/contig ids.
    suffix : Optional[str], optional
        suffix to append to the headers/contig ids.
    sep : Optional[str], optional
        separator after prefix or before suffix to append to the headers/contig ids.
    remove_trailing_hyphen_number : bool, optional
        whether or not to remove the trailing hyphen number e.g. '-1' from the
        cell/contig barcodes.
    high_confidence_filtering : bool, optional
        whether ot not to filter to only `high confidence` contigs.
    out_dir : Optional[str], optional
        path to output location. `None` defaults to 'dandelion'.
    filename_prefix : Optional[str], optional
        prefix of file name preceding '_contig'. `None` defaults to 'filtered'.

    Raises
    ------
    FileNotFoundError
        if path to fasta file is unknown.
    """
    filename_pre = "filtered" if filename_prefix is None else filename_prefix

    file_path = check_filepath(
        fasta,
        filename_prefix=filename_pre,
        ends_with=".fasta",
        within_dandelion=False,
    )

    if file_path is None:
        raise FileNotFoundError(
            "Path to fasta file is unknown. Please "
            + "specify path to fasta file or folder containing fasta file. "
            + "Starting folder should only contain 1 fasta file."
        )
    fh = open(file_path, "r")
    seqs = {}
    if sep is None:
        separator = "_"
    else:
        separator = str(sep)
    for header, sequence in fasta_iterator(fh):
        if prefix is None and suffix is None:
            seqs[header] = sequence
        elif prefix is not None:
            if suffix is not None:
                if remove_trailing_hyphen_number:
                    newheader = (
                        str(prefix)
                        + separator
                        + str(header).split("_contig")[0].split("-")[0]
                        + separator
                        + str(suffix)
                        + "_contig"
                        + str(header).split("_contig")[1]
                    )
                else:
                    newheader = (
                        str(prefix)
                        + separator
                        + str(header).split("_contig")[0]
                        + separator
                        + str(suffix)
                        + "_contig"
                        + str(header).split("_contig")[1]
                    )
            else:
                if remove_trailing_hyphen_number:
                    newheader = (
                        str(prefix)
                        + separator
                        + str(header).split("_contig")[0].split("-")[0]
                        + "_contig"
                        + str(header).split("_contig")[1]
                    )
                else:
                    newheader = str(prefix) + separator + str(header)
            seqs[newheader] = sequence
        else:
            if suffix is not None:
                if remove_trailing_hyphen_number:
                    newheader = (
                        str(header).split("_contig")[0].split("-")[0]
                        + separator
                        + str(suffix)
                        + "_contig"
                        + str(header).split("_contig")[1]
                    )
                else:
                    newheader = (
                        str(header).split("_contig")[0]
                        + separator
                        + str(suffix)
                        + "_contig"
                        + str(header).split("_contig")[1]
                    )
            else:
                newheader = str(header)
            seqs[newheader] = sequence
    fh.close()
    base_dir = file_path.parent if file_path.is_file() else Path.cwd()
    out_dir = base_dir / "dandelion" if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # format the barcode and contig_id in the corresponding annotation file too
    anno = check_filepath(
        fasta,
        filename_prefix=filename_pre,
        ends_with="_annotations.csv",
        within_dandelion=False,
    )
    data = pd.read_csv(anno, dtype="object")
    if prefix is not None:
        if suffix is not None:
            if remove_trailing_hyphen_number:
                data["contig_id"] = [
                    str(prefix)
                    + separator
                    + str(c).split("_contig")[0].split("-")[0]
                    + separator
                    + str(suffix)
                    + "_contig"
                    + str(c).split("_contig")[1]
                    for c in data["contig_id"]
                ]
                data["barcode"] = [
                    str(prefix)
                    + separator
                    + str(b).split("-")[0]
                    + separator
                    + str(suffix)
                    for b in data["barcode"]
                ]
            else:
                data["contig_id"] = [
                    str(prefix)
                    + separator
                    + str(c).split("_contig")[0]
                    + separator
                    + str(suffix)
                    + "_contig"
                    + str(c).split("_contig")[1]
                    for c in data["contig_id"]
                ]
                data["barcode"] = [
                    str(prefix) + separator + str(b) + separator + str(suffix)
                    for b in data["barcode"]
                ]
        else:
            if remove_trailing_hyphen_number:
                data["contig_id"] = [
                    str(prefix)
                    + separator
                    + str(c).split("_contig")[0].split("-")[0]
                    + "_contig"
                    + str(c).split("_contig")[1]
                    for c in data["contig_id"]
                ]
                data["barcode"] = [
                    str(prefix) + separator + str(b).split("-")[0]
                    for b in data["barcode"]
                ]
            else:
                data["contig_id"] = [
                    str(prefix) + separator + str(c) for c in data["contig_id"]
                ]
                data["barcode"] = [
                    str(prefix) + separator + str(b) for b in data["barcode"]
                ]
    else:
        if suffix is not None:
            if remove_trailing_hyphen_number:
                data["contig_id"] = [
                    str(c).split("_contig")[0].split("-")[0]
                    + separator
                    + str(suffix)
                    + "_contig"
                    + str(c).split("_contig")[1]
                    for c in data["contig_id"]
                ]
                data["barcode"] = [
                    str(b).split("-")[0] + separator + str(suffix)
                    for b in data["barcode"]
                ]
            else:
                data["contig_id"] = [
                    str(c).split("_contig")[0]
                    + separator
                    + str(suffix)
                    + "_contig"
                    + str(c).split("_contig")[1]
                    for c in data["contig_id"]
                ]
                data["barcode"] = [
                    str(b) + separator + str(suffix) for b in data["barcode"]
                ]
        else:
            data["contig_id"] = [str(c) for c in data["contig_id"]]
            data["barcode"] = [str(b) for b in data["barcode"]]
    anno = check_filepath(
        fasta,
        filename_prefix=filename_pre,
        ends_with="_annotations.csv",
        within_dandelion=False,
    )
    out_anno = out_dir / (file_path.stem + "_annotations.csv")
    out_fasta = out_dir / file_path.name
    fh1 = open(out_fasta, "w")
    fh1.close()
    if high_confidence_filtering:
        hiconf_contigs = [
            x
            for x, y in zip(data["contig_id"], data["high_confidence"])
            if y in TRUES
        ]
        seqs = {hiconf: seqs[hiconf] for hiconf in hiconf_contigs}
        data = data[data["contig_id"].isin(hiconf_contigs)]
    write_fasta(fasta_dict=seqs, out_fasta=out_fasta)
    data.to_csv(out_anno, index=False)


def format_fastas(
    fastas: List[Union[str, Path]],
    prefix: Optional[List[str]] = None,
    suffix: Optional[List[str]] = None,
    sep: Optional[str] = None,
    remove_trailing_hyphen_number: bool = True,
    high_confidence_filtering: bool = False,
    out_dir: Optional[Union[str, Path]] = None,
    filename_prefix: Optional[Union[List[str], str]] = None,
):
    """
    Add prefix to the headers/contig ids in input fasta and annotation file.

    Parameters
    ----------
    fastas : List[Union[str, Path]]
        list of paths to fasta files.
    prefix : Optional[List[str]], optional
        list of prefixes to append to headers/contig ids in each fasta file.
    suffix : Optional[List[str]], optional
        list of suffixes to append to headers/contig ids in each fasta file.
    sep : Optional[str], optional
        separator after prefix or before suffix to append to the headers/contig
        ids.
    remove_trailing_hyphen_number : bool, optional
        whether or not to remove the trailing hyphen number e.g. '-1' from the
        cell/contig barcodes.
    high_confidence_filtering : bool, optional
        whether ot not to filter to only `high confidence` contigs.
    out_dir : Optional[Union[str, Path]], optional
        path to out put location.
    filename_prefix : Optional[Union[List[str], str]], optional
        list of prefixes of file names preceding '_contig'. `None` defaults to
        'filtered'.
    """
    fastas = [fastas] if not isinstance(fastas, list) else fastas
    if not isinstance(filename_prefix, list):
        filename_prefix = [filename_prefix]
        if len(filename_prefix) == 1:
            if len(fastas) > 1:
                filename_prefix = filename_prefix * len(fastas)
    if prefix is not None:
        if not isinstance(prefix, list):
            prefix = [prefix]
        prefix_dict = dict(zip(fastas, prefix))
    if suffix is not None:
        if not isinstance(suffix, list):
            suffix = [suffix]
        suffix_dict = dict(zip(fastas, suffix))

    for i in tqdm(
        range(0, len(fastas)),
        desc="Formatting fasta(s) ",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        if prefix is None and suffix is None:
            format_fasta(
                fastas[i],
                prefix=None,
                suffix=None,
                sep=None,
                remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                high_confidence_filtering=high_confidence_filtering,
                out_dir=out_dir,
                filename_prefix=filename_prefix[i],
            )
        elif prefix is not None:
            if suffix is not None:
                format_fasta(
                    fastas[i],
                    prefix=prefix_dict[fastas[i]],
                    suffix=suffix_dict[fastas[i]],
                    sep=sep,
                    remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    high_confidence_filtering=high_confidence_filtering,
                    out_dir=out_dir,
                    filename_prefix=filename_prefix[i],
                )
            else:
                format_fasta(
                    fastas[i],
                    prefix=prefix_dict[fastas[i]],
                    suffix=None,
                    sep=sep,
                    remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    high_confidence_filtering=high_confidence_filtering,
                    out_dir=out_dir,
                    filename_prefix=filename_prefix[i],
                )
        else:
            if suffix is not None:
                format_fasta(
                    fastas[i],
                    prefix=None,
                    suffix=suffix_dict[fastas[i]],
                    sep=sep,
                    remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    high_confidence_filtering=high_confidence_filtering,
                    out_dir=out_dir,
                    filename_prefix=filename_prefix[i],
                )
            else:
                format_fasta(
                    fastas[i],
                    prefix=None,
                    suffix=None,
                    sep=None,
                    remove_trailing_hyphen_number=remove_trailing_hyphen_number,
                    high_confidence_filtering=high_confidence_filtering,
                    out_dir=out_dir,
                    filename_prefix=filename_prefix[i],
                )


def assign_isotype(
    fasta: Union[str, Path],
    org: Literal["human", "mouse"] = "human",
    evalue: float = 1e-4,
    correct_c_call: bool = True,
    correction_dict: Optional[Dict[str, Dict[str, str]]] = None,
    plot: bool = True,
    save_plot: bool = False,
    show_plot: bool = True,
    figsize: Tuple[Union[int, float], Union[int, float]] = (4, 4),
    blastdb: Optional[Union[str, Path]] = None,
    filename_prefix: Optional[str] = None,
    additional_args: List[str] = [],
):
    """
    Annotate contigs with constant region call using blastn.

    Parameters
    ----------
    fasta : Union[str, Path]
        path to fasta file.
    org : Literal["human", "mouse"], optional
        organism of reference folder.
    evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets.
    correct_c_call : bool, optional
        whether or not to adjust the c_calls after blast based on provided
        primers specified in `primer_dict` option.
    correction_dict : Optional[Union[Dict[str, str]]], optional
        a nested dictionary contain isotype/c_genes as keys and primer
        sequences as records to use for correcting annotated c_calls. Defaults
        to a curated dictionary for human sequences if left as none.
    plot : bool, optional
        whether or not to plot reassignment summary metrics.
    save_plot : bool, optional
        whether or not to save plot.
    show_plot : bool, optional
        whether or not to show plot.
    figsize : Tuple[Union[int, float], Union[int, float]], optional
        size of figure.
    blastdb : Optional[Union[str, Path]], optional
        path to blast database. Defaults to `$BLASTDB` environmental variable.
    filename_prefix : Optional[str], optional
        prefix of file name preceding '_contig'. `None` defaults to 'filtered'.
    additional_args : List[str], optional
        additional arguments to pass to `blastn`.
    Raises
    ------
    FileNotFoundError
        if path to fasta file is unknown.
    """
    aligner = Align.PairwiseAligner()

    def two_gene_correction(
        self: pd.DataFrame, i: str, dictionary: Dict[str, str]
    ):
        """Pairwise alignment for two genes.

        Parameters
        ----------
        i : str
            index name.
        dictionary : Dict[str, str]
            dictionary holding gene name as key and sequence as value.
        """
        key1, key2 = dictionary.keys()
        seq = self.loc[i, "c_sequence_alignment"].replace("-", "")
        alignments1 = aligner.align(dictionary[key1], seq)
        alignments2 = aligner.align(dictionary[key2], seq)
        score1 = alignments1.score
        score2 = alignments2.score
        if score1 == score2:
            self.at[i, "c_call"] = str(key1) + "," + str(key2)
        if score1 > score2:
            self.at[i, "c_call"] = str(key1)
        if score1 < score2:
            self.at[i, "c_call"] = str(key2)

    def three_gene_correction(
        self: pd.DataFrame, i: str, dictionary: Dict[str, str]
    ):
        """Pairwise alignment for three genes.

        Parameters
        ----------
        i : str
            index name.
        dictionary : Dict[str, str]
            dictionary holding gene name as key and sequence as value.
        """
        key1, key2, key3 = dictionary.keys()
        seq = self.loc[i, "c_sequence_alignment"].replace("-", "")
        alignments1 = aligner.align(dictionary[key1], seq)
        alignments2 = aligner.align(dictionary[key2], seq)
        alignments3 = aligner.align(dictionary[key3], seq)
        score1 = alignments1.score
        score2 = alignments2.score
        score3 = alignments3.score
        if score1 == score2 == score3:
            self.at[i, "c_call"] = str(key1) + "," + str(key2) + "," + str(key3)
        elif score1 > score2 and score1 > score3:
            self.at[i, "c_call"] = str(key1)
        elif score2 > score1 and score2 > score3:
            self.at[i, "c_call"] = str(key2)
        elif score3 > score1 and score3 > score2:
            self.at[i, "c_call"] = str(key3)
        elif score1 == score2 and score1 > score3:
            self.at[i, "c_call"] = str(key1) + "," + str(key2)
        elif score1 > score2 and score1 == score3:
            self.at[i, "c_call"] = str(key1) + "," + str(key3)
        elif score2 > score1 and score2 == score3:
            self.at[i, "c_call"] = str(key2) + "," + str(key3)

    def four_gene_correction(
        self: pd.DataFrame, i: str, dictionary: Dict[str, str]
    ):
        """Pairwise alignment for four genes.

        Parameters
        ----------
        i : str
            index name.
        dictionary : Dict[str, str]
            dictionary holding gene name as key and sequence as value.
        """
        key1, key2, key3, key4 = dictionary.keys()
        seq = self.loc[i, "c_sequence_alignment"].replace("-", "")
        alignments1 = aligner.align(dictionary[key1], seq)
        alignments2 = aligner.align(dictionary[key2], seq)
        alignments3 = aligner.align(dictionary[key3], seq)
        alignments4 = aligner.align(dictionary[key4], seq)
        score1 = alignments1.score
        score2 = alignments2.score
        score3 = alignments3.score
        score4 = alignments4.score
        if score1 == score2 == score3 == score4:
            self.at[i, "c_call"] = (
                str(key1) + "," + str(key2) + "," + str(key3) + "," + str(key4)
            )
        elif score1 > score2 and score1 > score3 and score1 > score4:
            self.at[i, "c_call"] = str(key1)
        elif score2 > score1 and score2 > score3 and score2 > score4:
            self.at[i, "c_call"] = str(key2)
        elif score3 > score1 and score3 > score2 and score3 > score4:
            self.at[i, "c_call"] = str(key3)
        elif score4 > score1 and score4 > score2 and score4 > score3:
            self.at[i, "c_call"] = str(key4)
        elif score1 == score2 and score1 > score3 and score1 > score4:
            self.at[i, "c_call"] = str(key1) + "," + str(key2)
        elif score1 > score2 and score1 == score3 and score1 > score4:
            self.at[i, "c_call"] = str(key1) + "," + str(key3)
        elif score1 > score2 and score1 > score3 and score1 == score4:
            self.at[i, "c_call"] = str(key1) + "," + str(key4)
        elif score2 == score3 and score2 > score1 and score2 > score4:
            self.at[i, "c_call"] = str(key1) + "," + str(key3)
        elif score2 == score4 and score2 > score1 and score2 > score3:
            self.at[i, "c_call"] = str(key2) + "," + str(key4)
        elif score3 == score4 and score3 > score1 and score3 > score2:
            self.at[i, "c_call"] = str(key3) + "," + str(key4)
        elif score1 == score2 == score3 and score1 > score4:
            self.at[i, "c_call"] = str(key1) + "," + str(key2) + "," + str(key3)
        elif score1 == score2 == score4 and score1 > score3:
            self.at[i, "c_call"] = str(key1) + "," + str(key2) + "," + str(key4)
        elif score1 == score3 == score4 and score1 > score2:
            self.at[i, "c_call"] = str(key1) + "," + str(key3) + "," + str(key4)
        elif score2 == score3 == score4 and score2 > score1:
            self.at[i, "c_call"] = str(key2) + "," + str(key3) + "," + str(key4)

    def _correct_c_call(
        data: pd.DataFrame,
        primers_dict: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> pd.DataFrame:
        """Pairwise alignment for c genes.

        Parameters
        ----------
        data : pd.DataFrame
            Input data Frame.
        primers_dict : Optional[Dict[str, Dict[str, str]]], optional
            Gene:Sequence dictionary to do pairwise alignment with.

        Returns
        -------
        pd.DataFrame
            Output data frame with c_call adjusted.
        """
        dat = data.copy()
        if primers_dict is None:
            primer_dict = {
                "IGHG": {
                    "IGHG1": "GCCTCCACCAAGGGCCCATCGGTCTTCCCCCTGGCACCCTCCTCCAAGAGCACCTCTGGGGGCACAGCGGCCCTGGGC",
                    "IGHG2": "GCCTCCACCAAGGGCCCATCGGTCTTCCCCCTGGCGCCCTGCTCCAGGAGCACCTCCGAGAGCACAGCGGCCCTGGGC",
                    "IGHG3": "GCTTCCACCAAGGGCCCATCGGTCTTCCCCCTGGCGCCCTGCTCCAGGAGCACCTCTGGGGGCACAGCGGCCCTGGGC",
                    "IGHG4": "GCTTCCACCAAGGGCCCATCCGTCTTCCCCCTGGCGCCCTGCTCCAGGAGCACCTCCGAGAGCACAGCCGCCCTGGGC",
                },
                "IGHA": {
                    "IGHA1": "GCATCCCCGACCAGCCCCAAGGTCTTCCCGCTGAGCCTCTGCAGCACCCAGCCAGATGGGAACGTGGTCATCGCCTGC",
                    "IGHA2": "GCATCCCCGACCAGCCCCAAGGTCTTCCCGCTGAGCCTCGACAGCACCCCCCAAGATGGGAACGTGGTCGTCGCATGC",
                },
                "IGLC7": {
                    "IGLC": "GTCAGCCCAAGGCTGCCCCCTCGGTCACTCTGTTCCCGCCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGTG"
                    "TGTCTCATAA",
                    "IGLC7": "GTCAGCCCAAGGCTGCCCCCTCGGTCACTCTGTTCCCACCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGT"
                    "GTGTCTCGTAA",
                },
                "IGLC3": {
                    "IGLC": "GTCAGCCCAAGGCTGCCCCCTCGGTCACTCTGTTCCCGCCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGTG"
                    "TGTCTCATAA",
                    "IGLC3": "GTCAGCCCAAGGCTGCCCCCTCGGTCACTCTGTTCCCACCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGT"
                    "GTGTCTCATAA",
                },
                "IGLC6": {
                    "IGLC": "TCGGTCACTCTGTTCCCGCCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGTGTGTCTCA",
                    "IGLC6": "TCGGTCACTCTGTTCCCGCCCTCCTCTGAGGAGCTTCAAGCCAACAAGGCCACACTGGTGTGCCTGA",
                },
            }
        else:
            primer_dict = primers_dict

        for i in dat.index:
            if (dat.loc[i, "c_call"] == dat.loc[i, "c_call"]) & (
                dat.loc[i, "c_call"] is not None
            ):
                for k in primer_dict:
                    if k in dat.loc[i, "c_call"]:
                        if len(primer_dict[k]) == 2:
                            two_gene_correction(dat, i, primer_dict[k])
                        elif len(primer_dict[k]) == 3:
                            three_gene_correction(dat, i, primer_dict[k])
                        elif len(primer_dict[k]) == 4:
                            four_gene_correction(dat, i, primer_dict[k])
        return dat

    # main function from here
    # format_dict = {
    #     "changeo": "_igblast_db-pass",
    #     "blast": "_igblast_db-pass",
    #     "airr": "_igblast_gap",
    # }

    filePath = check_filepath(
        fasta, filename_prefix=filename_prefix, ends_with=".fasta"
    )
    if filePath is None:
        raise FileNotFoundError(
            (
                "Path to fasta file is unknown. Please specify path to "
                + "fasta file or folder containing fasta file."
            )
        )

    blast_out = run_blastn(
        fasta=filePath,
        database=blastdb,
        org=org,
        loci="ig",
        call="c",
        max_hsps=1,
        evalue=evalue,
        outfmt=(
            "6 qseqid sseqid pident length mismatch gapopen "
            + "qstart qend sstart send evalue bitscore qseq sseq"
        ),
        dust="no",
        additional_args=additional_args,
    )
    blast_out.drop_duplicates(subset="sequence_id", keep="first", inplace=True)

    _10xfile = check_filepath(
        fasta,
        filename_prefix=filename_prefix,
        ends_with="_annotations.csv",
    )
    _airrfile = check_filepath(
        fasta,
        filename_prefix=filename_prefix,
        ends_with="_igblast.tsv",
        sub_dir="tmp",
    )
    _processedfile = check_filepath(
        fasta,
        filename_prefix=filename_prefix,
        ends_with="_igblast_db-pass_genotyped.tsv",
        sub_dir="tmp",
    )
    if _processedfile is None:
        _processedfile = check_filepath(
            fasta,
            filename_prefix=filename_prefix,
            ends_with="_igblast_db-pass.tsv",
            sub_dir="tmp",
        )
        out_ex = "_igblast_db-pass.tsv"
    else:
        out_ex = "_igblast_db-pass_genotyped.tsv"
    dat = load_data(_processedfile)
    logg.info("Loading 10X annotations \n")
    if _10xfile is not None:
        dat_10x = read_10x_vdj(_10xfile)
        res_10x = pd.DataFrame(dat_10x.data["c_call"])
    else:  # pragma: no cover
        res_10x = pd.DataFrame(dat["c_call"])
        res_10x["c_call"] = "None"
    logg.info("Preparing new calls \n")
    for col in [
        "c_call",
        "c_sequence_alignment",
        "c_germline_alignment",
        "c_sequence_start",
        "c_sequence_end",
        "c_score",
        "c_identity",
    ]:
        dat[col] = pd.Series(blast_out[col])
    res_blast = pd.DataFrame(dat["c_call"])
    res_blast = res_blast.fillna(value="None")
    res_10x_sum = pd.DataFrame(
        res_10x["c_call"].value_counts(normalize=True) * 100
    )
    res_10x_sum["group"] = "10X"
    res_10x_sum.columns = ["counts", "group"]
    res_10x_sum.index = res_10x_sum.index.set_names(["c_call"])
    res_10x_sum.reset_index(drop=False, inplace=True)
    res_blast_sum = pd.DataFrame(
        res_blast["c_call"].value_counts(normalize=True) * 100
    )
    res_blast_sum["group"] = "blast"
    res_blast_sum.columns = ["counts", "group"]
    res_blast_sum.index = res_blast_sum.index.set_names(["c_call"])
    res_blast_sum.reset_index(drop=False, inplace=True)
    if (
        correct_c_call
    ):  # TODO: figure out if i need to set up a None correction?
        logg.info("Correcting C calls \n")
        dat = _correct_c_call(dat, primers_dict=correction_dict)
        res_corrected = pd.DataFrame(dat["c_call"])
        res_corrected = res_corrected.fillna(value="None")
        res_corrected_sum = pd.DataFrame(
            res_corrected["c_call"].value_counts(normalize=True) * 100
        )
        res_corrected_sum["group"] = "corrected"
        res_corrected_sum.columns = ["counts", "group"]
        res_corrected_sum.index = res_corrected_sum.index.set_names(["c_call"])
        res_corrected_sum.reset_index(drop=False, inplace=True)
        res = pd.concat([res_10x_sum, res_blast_sum, res_corrected_sum])
    else:  # pragma: no cover
        res = pd.concat([res_10x_sum, res_blast_sum])

    res = res.reset_index(drop=True)
    res["c_call"] = res["c_call"].fillna(value="None")
    res["c_call"] = [re.sub("[*][0-9][0-9]", "", c) for c in res["c_call"]]
    res["c_call"] = res["c_call"].astype("category")
    res["c_call"] = res["c_call"].cat.reorder_categories(
        sorted(list(set(res["c_call"])), reverse=True)
    )

    logg.info("Finishing up \n")
    dat["c_call_10x"] = pd.Series(res_10x["c_call"])
    # some minor adjustment to the final output table
    airr_output = load_data(_airrfile)
    cols_to_merge = [
        "junction_aa_length",
        "fwr1_aa",
        "fwr2_aa",
        "fwr3_aa",
        "fwr4_aa",
        "cdr1_aa",
        "cdr2_aa",
        "cdr3_aa",
        "sequence_alignment_aa",
        "v_sequence_alignment_aa",
        "d_sequence_alignment_aa",
        "j_sequence_alignment_aa",
    ]
    for x in cols_to_merge:
        dat[x] = pd.Series(airr_output[x])

    # remove allellic calls
    dat["c_call"] = dat["c_call"].fillna(value="")
    dat["c_call"] = [re.sub("[*][0-9][0-9]", "", c) for c in dat["c_call"]]

    write_airr(dat, _processedfile)
    if plot:
        options.figure_size = figsize
        if correct_c_call:
            p = (
                ggplot(res, aes(x="c_call", y="counts", fill="group"))
                + coord_flip()
                + theme_classic()
                + xlab("c_call")
                + ylab("% c calls")
                + geom_col(stat="identity", position="dodge")
                + scale_fill_manual(values=("#79706e", "#86bcb6", "#F28e2b"))
                + theme(legend_title=element_blank())
            )
        else:
            p = (
                ggplot(res, aes(x="c_call", y="counts", fill="group"))
                + coord_flip()
                + theme_classic()
                + xlab("c_call")
                + ylab("% c calls")
                + geom_col(stat="identity", position="dodge")
                + scale_fill_manual(values=("#79706e", "#86bcb6"))
                + theme(legend_title=element_blank())
            )
        if save_plot:
            _file3 = filePath.parent / "assign_isotype.pdf"
            save_as_pdf_pages([p], filename=_file3)
            if show_plot:  # pragma: no cover
                print(p)
        else:  # pragma: no cover
            if show_plot:  # pragma: no cover
                print(p)
    # move and rename
    move_to_tmp(fasta, filename_prefix)
    make_all(fasta, filename_prefix, loci="ig")
    rename_dandelion(fasta, filename_prefix, ends_with=out_ex, sub_dir="tmp")
    update_j_multimap(fasta, filename_prefix)


def assign_isotypes(
    fastas: List[Union[str, Path]],
    org: Literal["human", "mouse"] = "human",
    evalue: float = 1e4,
    correct_c_call: bool = True,
    correction_dict: Optional[Dict[str, Dict[str, str]]] = None,
    plot: bool = True,
    save_plot: bool = False,
    show_plot: bool = True,
    figsize: Tuple[Union[int, float], Union[int, float]] = (4, 4),
    blastdb: Optional[Union[str, Path]] = None,
    filename_prefix: Optional[Union[List, str]] = None,
    additional_args: List[str] = [],
):
    """
    Annotate contigs with constant region call using blastn.

    Parameters
    ----------
    fastas : List[str]
        list of paths to fasta files.
    org : Literal["human", "mouse"], optional
        organism of reference folder.
    evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets.
    correct_c_call : bool, optional
        whether or not to adjust the c_calls after blast based on provided primers specified in `primer_dict` option.
    correction_dict : Optional[Dict[str, Dict[str, str]]], optional
        a nested dictionary contain isotype/c_genes as keys and primer sequences as records to use for correcting
        annotated c_calls. Defaults to a curated dictionary for human sequences if left as none.
    plot : bool, optional
        whether or not to plot reassignment summary metrics.
    save_plot : bool, optional
        whether or not to save plots.
    show_plot : bool, optional
        whether or not to show plots.
    figsize : Tuple[Union[int, float], Union[int, float]], optional
        size of figure.
    blastdb : Optional[Union[str, Path]], optional
        path to blast database. Defaults to `$BLASTDB` environmental variable.
    filename_prefix : Optional[Union[List, str]], optional
        list of prefixes of file names preceding '_contig'. `None` defaults to 'filtered'.
    additional_args : List[str], optional
        additional arguments to pass to `blastn`.
    """
    if type(fastas) is not list:
        fastas = [fastas]
    if type(filename_prefix) is not list:
        filename_prefix = [filename_prefix]
    if all(t is None for t in filename_prefix):
        filename_prefix = [None for f in fastas]

    logg.info("Assign isotypes \n")

    for i in range(0, len(fastas)):
        assign_isotype(
            fastas[i],
            org=org,
            evalue=evalue,
            correct_c_call=correct_c_call,
            correction_dict=correction_dict,
            plot=plot,
            save_plot=save_plot,
            show_plot=show_plot,
            figsize=figsize,
            blastdb=blastdb,
            filename_prefix=filename_prefix[i],
            additional_args=additional_args,
        )


def reannotate_genes(
    data: List[str],
    igblast_db: Optional[str] = None,
    germline: Optional[str] = None,
    org: Literal["human", "mouse"] = "human",
    loci: Literal["ig", "tr"] = "ig",
    extended: bool = True,
    filename_prefix: Optional[Union[List[str], str]] = None,
    flavour: Literal["strict", "original"] = "strict",
    min_j_match: int = 7,
    min_d_match: int = 9,
    v_evalue: float = 1e-4,
    d_evalue: float = 1e-3,
    j_evalue: float = 1e-4,
    reassign_dj: bool = True,
    overwrite: bool = True,
    dust: Optional[Union[Literal["yes", "no"], str]] = "no",
    additional_args: Dict[str, List[str]] = {
        "assigngenes": [],
        "makedb": [],
        "igblastn": [],
        "blastn_j": [],
        "blastn_d": [],
    },
):
    """
    Reannotate cellranger fasta files with igblastn and parses to airr format.

    Parameters
    ----------
    data : List[str]
        list of fasta file locations, or folder name containing fasta files.
        if provided as a single string, it will first be converted to a list;
        this allows for the function to be run on single/multiple samples.
    igblast_db : Optional[str], optional
        path to igblast database folder. Defaults to `IGDATA` environmental
        variable.
    germline : Optional[str], optional
        path to germline database folder. Defaults to `GERMLINE` environmental
        variable.
    org : Literal["human", "mouse"], optional
        organism of germline database.
    loci : Literal["ig", "tr"], optional
        mode for igblastn. 'ig' for BCRs, 'tr' for TCRs.
    extended : bool, optional
        whether or not to transfer additional 10X annotations to output file.
    filename_prefix : Optional[Union[List[str], str]], optional
        list of prefixes of file names preceding '_contig'. `None` defaults
        to 'filtered'.
    flavour : Literal["strict", "original"], optional
        Either 'strict' or 'original'. Determines how igblastn should
        be run. Running in 'strict' flavour will add the additional the
        evalue and min_d_match options to the run.
    min_j_match : int, optional
        Minimum D gene nucleotide matches. This controls the threshold for
        D gene detection. You can set the minimal number of required
        consecutive nucleotide matches between the query sequence and the D
        genes based on your own criteria. Note that the matches do not include
        overlapping matches at V-D or D-J junctions.
    min_d_match : int, optional
        Minimum D gene nucleotide matches. This controls the threshold for
        D gene detection. You can set the minimal number of required
        consecutive nucleotide matches between the query sequence and the D
        genes based on your own criteria. Note that the matches do not include
        overlapping matches at V-D or D-J junctions.
    v_evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets. for v gene.
    d_evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets. for d gene.
    j_evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets. for j gene.
    reassign_dj : bool, optional
        whether or not to perform a targetted blastn reassignment for D and J genes.
    overwrite : bool, optional
        whether or not to overwrite the assignment if flavour = 'strict'.
    dust : Optional[Union[Literal["yes", "no"], str]], optional
        dustmasker options. Filter query sequence with DUST
        Format: 'yes', or 'no' to disable. Accepts str.
        If None, defaults to `20 64 1`.
    additional_args : Dict[str, List[str]], optional
        additional arguments to pass to `AssignGenes.py`, `MakeDb.py`, `igblastn` and `blastn`.
        This accepts a dictionary with keys as the name of the sub-function (`assigngenes`, `makedb`,
        `igblastn`, `blastn_j` and `blastn_d`) and the records as lists of arguments to pass to the
        relevant scripts/tools.

    Raises
    ------
    FileNotFoundError
        if path to fasta file is unknown.
    """
    if type(data) is not list:
        data = [data]
    if type(filename_prefix) is not list:
        filename_prefix = [filename_prefix]
    if all(t is None for t in filename_prefix):
        filename_prefix = [None for d in data]

    filePath = None
    for i in tqdm(
        range(0, len(data)),
        desc="Assigning genes ",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        filePath = check_filepath(
            data[i], filename_prefix=filename_prefix[i], ends_with=".fasta"
        )
        if filePath is None:
            if filename_prefix[i] is not None:
                raise FileNotFoundError(
                    "Path to fasta file with filename prefix `{}_contig` is unknown. ".format(
                        filename_prefix[i]
                    )
                    + "Please specify path to fasta file or folder containing fasta file."
                )
            else:
                raise FileNotFoundError(
                    "Path to fasta file is unknown. "
                    + "Please specify path to fasta file or folder containing fasta file."
                )

        logg.info(f"Processing {str(filePath)} \n")

        if flavour == "original":
            assigngenes_igblast(
                filePath,
                igblast_db=igblast_db,
                org=org,
                loci=loci,
                additional_args=additional_args["assigngenes"],
            )
        elif flavour == "strict":
            run_igblastn(
                filePath,
                igblast_db=igblast_db,
                org=org,
                loci=loci,
                evalue=v_evalue,
                min_d_match=min_d_match,
                additional_args=additional_args["igblastn"],
            )
        makedb_igblast(
            filePath,
            org=org,
            germline=germline,
            extended=extended,
            additional_args=additional_args["makedb"],
        )
        # block this for now, until I figure out if it's
        # worth it
        if flavour == "strict":
            if reassign_dj:
                assign_DJ(
                    fasta=filePath,
                    org=org,
                    loci=loci,
                    call="j",
                    database=igblast_db,
                    evalue=j_evalue,
                    filename_prefix=filename_prefix,
                    dust=dust,
                    word_size=min_j_match,
                    overwrite=overwrite,
                    additional_args=additional_args["blastn_j"],
                )
                assign_DJ(
                    fasta=filePath,
                    org=org,
                    loci=loci,
                    call="d",
                    database=igblast_db,
                    evalue=d_evalue,
                    filename_prefix=filename_prefix,
                    dust=dust,
                    word_size=min_d_match,
                    overwrite=overwrite,
                    additional_args=additional_args["blastn_d"],
                )
                ensure_columns_transferred(
                    fasta=filePath,
                    filename_prefix=filename_prefix,
                )

    if loci == "tr":
        change_file_location(data, filename_prefix)
        if flavour == "strict":
            mask_dj(data, filename_prefix, d_evalue, j_evalue)
        move_to_tmp(data, filename_prefix)
        make_all(data, filename_prefix, loci=loci)
        rename_dandelion(
            data, filename_prefix, ends_with="_igblast_db-pass.tsv"
        )
        update_j_multimap(data, filename_prefix)


def return_pass_fail_filepaths(
    fasta: Union[str, Path],
    filename_prefix: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """Return necessary file paths for internal use only.

    Parameters
    ----------
    fasta : Union[str, Path]
        path to fasta file.
    filename_prefix : Optional[str], optional
        prefix of file name preceding '_contig'. `None` defaults to 'filtered'.

    Returns
    -------
    Tuple[Path, Path, Path]
        file paths for downstream functions.

    Raises
    ------
    FileNotFoundError
        if path to fasta file is unknown.
    """
    file_path = check_filepath(
        fasta, filename_prefix=filename_prefix, ends_with=".fasta"
    )
    if file_path is None:
        raise FileNotFoundError(
            (
                "Path to fasta file is unknown. Please specify "
                + "path to fasta file or folder containing fasta file."
            )
        )
    # read the original object
    pass_path = (
        file_path.parent / "tmp" / (file_path.stem + "_igblast_db-pass.tsv")
    )
    fail_path = (
        file_path.parent / "tmp" / (file_path.stem + "_igblast_db-fail.tsv")
    )
    return file_path, pass_path, fail_path


def ensure_columns_transferred(
    fasta: str,
    filename_prefix: Optional[str] = None,
):
    """Ensure the additional columns are successfully populated.

    Parameters
    ----------
    fasta : str
        path to fasta file.
    filename_prefix : Optional[str], optional
        prefix of file name preceding '_contig'. `None` defaults to 'filtered'.
    """
    filePath, passfile, failfile = return_pass_fail_filepaths(
        fasta, filename_prefix=filename_prefix
    )
    addcols = [
        "_support_igblastn",
        "_score_igblastn",
        "_call_igblastn",
        "_call_blastn",
        "_identity_blastn",
        "_alignment_length_blastn",
        "_number_of_mismatches_blastn",
        "_number_of_gap_openings_blastn",
        "_sequence_start_blastn",
        "_sequence_end_blastn",
        "_germline_start_blastn",
        "_germline_end_blastn",
        "_support_blastn",
        "_score_blastn",
        "_sequence_alignment_blastn",
        "_germline_alignment_blastn",
        "_source",
    ]
    if passfile.is_file():
        db_pass = load_data(passfile)
    else:
        db_pass = None
    if failfile.is_file():
        db_fail = load_data(failfile)
    else:
        db_fail = None
    if db_pass is not None:
        for call in ["d", "j"]:
            for col in addcols:
                add_col = call + col
                if add_col not in db_pass:
                    db_pass[add_col] = ""
        db_pass = sanitize_data(db_pass)
        db_pass.to_csv(passfile, sep="\t", index=False)
    if db_fail is not None:
        for call in ["d", "j"]:
            for col in addcols:
                add_col = call + col
                if add_col not in db_fail:
                    db_fail[add_col] = ""
        db_fail = sanitize_data(db_fail)
        db_fail.to_csv(failfile, sep="\t", index=False)


def reassign_alleles(
    data: List[str],
    combined_folder: str,
    v_germline: Optional[str] = None,
    germline: Optional[str] = None,
    org: Literal["human", "mouse"] = "human",
    novel: bool = True,
    plot: bool = True,
    save_plot: bool = False,
    show_plot: bool = True,
    figsize: Tuple[Union[int, float], Union[int, float]] = (4, 3),
    sample_id_dictionary: Optional[Dict[str, str]] = None,
    filename_prefix: Optional[Union[List[str], str]] = None,
    additional_args: Dict[str, List[str]] = {
        "tigger": [],
        "creategermlines": [],
    },
):
    """
    Correct allele calls based on a personalized genotype using tigger.

    It uses a subject-specific genotype to correct correct preliminary allele
    assignments of a set of sequences derived from a single subject.

    Parameters
    ----------
    data : List[str]
        list of data folders containing the .tsv files. if provided as a single
        string, it will first be converted to a list; this allows for the
        function to be run on single/multiple samples.
    combined_folder : str
        name of folder for concatenated data file and genotyped files.
    v_germline : Optional[str], optional
        path to heavy chain v germline fasta. Defaults to IGHV fasta in
        `$GERMLINE` environmental variable.
    germline : Optional[str], optional
        path to germline database folder. `None` defaults to `GERMLINE` environmental
        variable.
    org : Literal["human", "mouse"], optional
        organism of germline database.
    novel : bool, optional
        whether or not to run novel allele discovery during tigger-genotyping.
    plot : bool, optional
        whether or not to plot reassignment summary metrics.
    save_plot : bool, optional
        whether or not to save plot.
    show_plot : bool, optional
        whether or not to show plot.
    figsize : Tuple[Union[int, float], Union[int, float]], optional
        size of figure.
    sample_id_dictionary : Optional[Dict[str, str]], optional
        dictionary for creating a sample_id column in the concatenated file.
    filename_prefix : Optional[Union[List[str], str]], optional
        list of prefixes of file names preceding '_contig'. `None` defaults to
        'filtered'.
    additional_args : Dict[str, List[str]], optional
        additional arguments to pass to `tigger-genotype.R` and `CreateGermlines.py`.
        This accepts a dictionary with keys as the name of the sub-function (`tigger` or `creategermlines`)
        and the records as lists of arguments to pass to the relevant scripts/tools.

    Raises
    ------
    FileNotFoundError
        if reannotated file is not found.
    """
    fileformat = "blast"
    if type(data) is not list:
        data = [data]
    if type(filename_prefix) is not list:
        filename_prefix = [filename_prefix]
    if all(t is None for t in filename_prefix):
        filename_prefix = [None for d in data]

    informat_dict = {
        "changeo": "_igblast_db-pass.tsv",
        "blast": "_igblast_db-pass.tsv",
        "airr": "_igblast_gap.tsv",
    }
    germpass_dict = {
        "changeo": "_igblast_db-pass_germ-pass.tsv",
        "blast": "_igblast_db-pass_germ-pass.tsv",
        "airr": "_igblast_gap_germ-pass.tsv",
    }
    fileformat_dict = {
        "changeo": "_igblast_db-pass_genotyped.tsv",
        "blast": "_igblast_db-pass_genotyped.tsv",
        "airr": "_igblast_gap_genotyped.tsv",
    }
    fileformat_passed_dict = {
        "changeo": "_igblast_db-pass_genotyped_germ-pass.tsv",
        "blast": "_igblast_db-pass_genotyped_germ-pass.tsv",
        "airr": "_igblast_gap_genotyped_germ-pass.tsv",
    }
    inferred_fileformat_dict = {
        "changeo": "_igblast_db-pass_inferredGenotype.txt",
        "blast": "_igblast_db-pass_inferredGenotype.txt",
        "airr": "_igblast_gap_inferredGenotype.txt",
    }
    germline_dict = {
        "changeo": "_igblast_db-pass_genotype.fasta",
        "blast": "_igblast_db-pass_genotype.fasta",
        "airr": "_igblast_gap_genotype.fasta",
    }
    fform_dict = {"blast": "airr", "airr": "airr", "changeo": "changeo"}

    filepathlist_heavy = []
    filepathlist_light = []
    filePath = None
    sampleNames_dict = {}
    filePath_dict = {}
    for i in tqdm(
        range(0, len(data)),
        desc="Processing data file(s) ",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        filePath = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with=informat_dict[fileformat],
            sub_dir="tmp",
        )
        if filePath is None:
            raise FileNotFoundError(
                "Path to .tsv file for {} is unknown. ".format(data[i])
                + "Please specify path to reannotated .tsv file or folder "
                + "containing reannotated .tsv file."
            )

        filePath_heavy = filePath.parent / (
            filePath.stem + "_heavy_parse-select.tsv"
        )
        filePath_light = filePath.parent / (
            filePath.stem + "_light_parse-select.tsv"
        )

        if sample_id_dictionary is not None:
            sampleNames_dict[filePath] = sample_id_dictionary[data[i]]
        else:
            sampleNames_dict[filePath] = str(data[i])

        filePath_dict[str(data[i])] = filePath

        # splitting up to heavy chain and light chain files
        parsedb_heavy(filePath)
        parsedb_light(filePath)

        # add to counter
        filepathlist_heavy.append(filePath_heavy)
        filepathlist_light.append(filePath_light)

    # make output directory
    out_dir = Path(combined_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    # concatenate
    if len(filepathlist_heavy) > 1:
        logg.info("Concatenating objects")
        try:
            cmd1 = " ".join(
                [
                    'awk "FNR==1 && NR!=1 { while (/^sequence_id/) getline; } 1 {print}"'
                ]
                + [f for f in filepathlist_heavy]
                + [">"]
                + [
                    str(
                        out_dir
                        / (out_dir.stem + "_heavy" + informat_dict[fileformat])
                    )
                ]
            )
            cmd2 = " ".join(
                [
                    'awk "FNR==1 && NR!=1 { while (/^sequence_id/) getline; } 1 {print}"'
                ]
                + [f for f in filepathlist_light]
                + [">"]
                + [
                    str(
                        out_dir
                        / (out_dir.stem + "_light" + informat_dict[fileformat])
                    )
                ]
            )
            os.system(cmd1)
            os.system(cmd2)
        except:  # pragma: no cover
            fh = open(
                out_dir / (out_dir.stem + "_heavy" + informat_dict[fileformat]),
                "w",
            )
            fh.close()
            with open(
                out_dir / (out_dir.stem + "_heavy" + informat_dict[fileformat]),
                "a",
            ) as out_file:
                for filenum, filename in enumerate(filepathlist_heavy):
                    with open(filename, "r") as in_file:
                        for line_num, line in enumerate(in_file):
                            if (line_num == 0) and (filenum > 0):
                                continue
                            out_file.write(line)
            fh = open(
                out_dir / (out_dir.stem + "_light" + informat_dict[fileformat]),
                "w",
            )
            fh.close()
            with open(
                out_dir / (out_dir.stem + "_light" + informat_dict[fileformat]),
                "a",
            ) as out_file:
                for filenum, filename in enumerate(filepathlist_light):
                    with open(filename, "r") as in_file:
                        skip_next_line = False
                        for line_num, line in enumerate(in_file):
                            if (line_num == 0) and (filenum > 0):
                                continue
                            out_file.write(line)
    else:
        shutil.copyfile(
            Path(filepathlist_heavy[0]),
            out_dir / (out_dir.stem + "_heavy" + informat_dict[fileformat]),
        )
        shutil.copyfile(
            Path(filepathlist_light[0]),
            out_dir / (out_dir.stem + "_light" + informat_dict[fileformat]),
        )

    novel_dict = {True: "YES", False: "NO"}
    if novel:
        try:
            logg.info(
                "      Running tigger-genotype with novel allele discovery."
            )
            tigger_genotype(
                airr_file=str(
                    out_dir
                    / (out_dir.stem + "_heavy" + informat_dict[fileformat])
                ),
                v_germline=v_germline,
                org=org,
                fileformat=fform_dict[fileformat],
                novel_=novel_dict[novel],
                additional_args=additional_args["tigger"],
            )
            creategermlines(
                airr_file=str(
                    out_dir
                    / (out_dir.stem + "_heavy" + fileformat_dict[fileformat])
                ),
                germline=germline,
                org=org,
                genotyped_fasta=str(
                    out_dir
                    / (out_dir.stem + "_heavy" + germline_dict[fileformat])
                ),
                mode="heavy",
                additional_args=["--vf", "v_call_genotyped"]
                + additional_args["creategermlines"],
            )
            _ = load_data(
                out_dir
                / (out_dir.stem + "_heavy" + fileformat_passed_dict[fileformat])
            )
        except:
            try:
                logg.info("      Novel allele discovery execution halted.")
                logg.info(
                    "      Attempting to run tigger-genotype without novel allele discovery."
                )
                tigger_genotype(
                    airr_file=str(
                        out_dir
                        / (out_dir.stem + "_heavy" + informat_dict[fileformat])
                    ),
                    v_germline=v_germline,
                    org=org,
                    fileformat=fform_dict[fileformat],
                    novel_=novel_dict[False],
                    additional_args=additional_args["tigger"],
                )
                creategermlines(
                    airr_file=str(
                        out_dir
                        / (
                            out_dir.stem
                            + "_heavy"
                            + fileformat_dict[fileformat]
                        )
                    ),
                    germline=germline,
                    org=org,
                    genotyped_fasta=str(
                        out_dir
                        / (out_dir.stem + "_heavy" + germline_dict[fileformat])
                    ),
                    mode="heavy",
                    additional_args=["--vf", "v_call_genotyped"]
                    + additional_args["creategermlines"],
                )
                _ = load_data(
                    out_dir
                    / (
                        out_dir.stem
                        + "_heavy"
                        + fileformat_passed_dict[fileformat]
                    )
                )
            except:
                logg.info(
                    "     Insufficient contigs for running tigger-genotype. Defaulting to original heavy chain v_calls."
                )
                tigger_failed = ""
    else:
        try:
            logg.info(
                "      Running tigger-genotype without novel allele discovery."
            )
            tigger_genotype(
                airr_file=str(
                    out_dir
                    / (out_dir.stem + "_heavy" + informat_dict[fileformat])
                ),
                v_germline=v_germline,
                org=org,
                fileformat=fform_dict[fileformat],
                novel_=novel_dict[False],
                additional_args=additional_args["tigger"],
            )
            creategermlines(
                airr_file=str(
                    out_dir
                    / (out_dir.stem + "_heavy" + fileformat_dict[fileformat])
                ),
                germline=germline,
                org=org,
                genotyped_fasta=str(
                    out_dir
                    / (out_dir.stem + "_heavy" + germline_dict[fileformat])
                ),
                mode="heavy",
                additional_args=["--vf", "v_call_genotyped"]
                + additional_args["creategermlines"],
            )
            _ = load_data(
                str(
                    out_dir
                    / (
                        out_dir.stem
                        + "_heavy"
                        + fileformat_passed_dict[fileformat]
                    )
                )
            )
        except:
            logg.info(
                "      Insufficient contigs for running tigger-genotype. Defaulting to original heavy chain v_calls."
            )
            tigger_failed = ""

    if "tigger_failed" in locals():
        creategermlines(
            airr_file=str(
                out_dir / (out_dir.stem + "_heavy" + informat_dict[fileformat])
            ),
            germline=germline,
            org=org,
            genotyped_fasta=None,
            mode="heavy",
            additional_args=["--vf", "v_call"]
            + additional_args["creategermlines"],
        )
    creategermlines(
        airr_file=str(
            out_dir / (out_dir.stem + "_light" + informat_dict[fileformat])
        ),
        germline=germline,
        org=org,
        genotyped_fasta=None,
        mode="light",
        additional_args=["--vf", "v_call"] + additional_args["creategermlines"],
    )
    if "tigger_failed" in locals():
        logg.info(
            "      For convenience, entries for heavy chain in `v_call` are copied to `v_call_genotyped`."
        )
        heavy = load_data(
            out_dir / (out_dir.stem + "_heavy" + germpass_dict[fileformat])
        )
        heavy["v_call_genotyped"] = heavy["v_call"]
    else:
        heavy = load_data(
            out_dir
            / (out_dir.stem + "_heavy" + fileformat_passed_dict[fileformat])
        )

    logg.info(
        "      For convenience, entries for light chain `v_call` are copied to `v_call_genotyped`."
    )
    light = load_data(
        out_dir / (out_dir.stem + "_light" + germpass_dict[fileformat])
    )
    light["v_call_genotyped"] = light["v_call"]

    sampledict = {}
    heavy["sample_id"], light["sample_id"] = None, None
    for file in sampleNames_dict.keys():
        dat_f = load_data(file)
        dat_f["sample_id"] = sampleNames_dict[file]
        heavy["sample_id"].update(dat_f["sample_id"])
        light["sample_id"].update(dat_f["sample_id"])

    dat_ = pd.concat([heavy, light])
    if "cell_id" in dat_.columns:
        dat_.sort_values(by="cell_id", inplace=True)
    else:
        dat_.sort_values(by="sequence_id", inplace=True)

    if plot:
        if "tigger_failed" not in locals():
            logg.info("Returning summary plot")
            inferred_genotype = out_dir / (
                out_dir.stem + "_heavy" + inferred_fileformat_dict[fileformat]
            )
            inf_geno = pd.read_csv(inferred_genotype, sep="\t", dtype="object")
            s2 = set(inf_geno["gene"])
            results = []
            try:
                for samp in list(set(heavy["sample_id"])):
                    res_x = heavy[(heavy["sample_id"] == samp)]
                    V_ = [
                        re.sub("[*][0-9][0-9]", "", v) for v in res_x["v_call"]
                    ]
                    V_g = [
                        re.sub("[*][0-9][0-9]", "", v)
                        for v in res_x["v_call_genotyped"]
                    ]
                    s1 = set(
                        list(
                            ",".join(
                                [",".join(list(set(v.split(",")))) for v in V_]
                            ).split(",")
                        )
                    )
                    setdiff = s1 - s2
                    ambiguous = (
                        ["," in i for i in V_].count(True) / len(V_) * 100,
                        ["," in i for i in V_g].count(True) / len(V_g) * 100,
                    )
                    not_in_genotype = (
                        [i in setdiff for i in V_].count(True) / len(V_) * 100,
                        [i in setdiff for i in V_g].count(True)
                        / len(V_g)
                        * 100,
                    )
                    stats = pd.DataFrame(
                        [ambiguous, not_in_genotype],
                        columns=["ambiguous", "not_in_genotype"],
                        index=["before", "after"],
                    ).T
                    stats.index.set_names(["vgroup"], inplace=True)
                    stats.reset_index(drop=False, inplace=True)
                    stats["sample_id"] = samp
                    # stats['donor'] = str(combined_folder)
                    results.append(stats)
                results = pd.concat(results)
                ambiguous_table = results[results["vgroup"] == "ambiguous"]
                not_in_genotype_table = results[
                    results["vgroup"] == "not_in_genotype"
                ]
                ambiguous_table.reset_index(inplace=True, drop=True)
                not_in_genotype_table.reset_index(inplace=True, drop=True)
                # melting the dataframe
                ambiguous_table_before = ambiguous_table.drop("after", axis=1)
                ambiguous_table_before.rename(
                    columns={"before": "var"}, inplace=True
                )
                ambiguous_table_before["var_group"] = "before"
                ambiguous_table_after = ambiguous_table.drop("before", axis=1)
                ambiguous_table_after.rename(
                    columns={"after": "var"}, inplace=True
                )
                ambiguous_table_after["var_group"] = "after"
                ambiguous_table = pd.concat(
                    [ambiguous_table_before, ambiguous_table_after]
                )
                not_in_genotype_table_before = not_in_genotype_table.drop(
                    "after", axis=1
                )
                not_in_genotype_table_before.rename(
                    columns={"before": "var"}, inplace=True
                )
                not_in_genotype_table_before["var_group"] = "before"
                not_in_genotype_table_after = not_in_genotype_table.drop(
                    "before", axis=1
                )
                not_in_genotype_table_after.rename(
                    columns={"after": "var"}, inplace=True
                )
                not_in_genotype_table_after["var_group"] = "after"
                not_in_genotype_table = pd.concat(
                    [not_in_genotype_table_before, not_in_genotype_table_after]
                )
                ambiguous_table["var_group"] = ambiguous_table[
                    "var_group"
                ].astype("category")
                not_in_genotype_table["var_group"] = not_in_genotype_table[
                    "var_group"
                ].astype("category")
                ambiguous_table["var_group"] = ambiguous_table[
                    "var_group"
                ].cat.reorder_categories(["before", "after"])
                not_in_genotype_table["var_group"] = not_in_genotype_table[
                    "var_group"
                ].cat.reorder_categories(["before", "after"])

                options.figure_size = figsize
                final_table = pd.concat(
                    [ambiguous_table, not_in_genotype_table]
                )
                p = (
                    ggplot(
                        final_table,
                        aes(x="sample_id", y="var", fill="var_group"),
                    )
                    + coord_flip()
                    + theme_classic()
                    + xlab("sample_id")
                    + ylab("% allele calls")
                    + ggtitle("Genotype reassignment with TIgGER")
                    + geom_bar(stat="identity")
                    + facet_grid("~" + str("vgroup"), scales="free_y")
                    + scale_fill_manual(values=("#86bcb6", "#F28e2b"))
                    + theme(legend_title=element_blank())
                )
                if save_plot:
                    savefile = str(
                        out_dir / (out_dir.stem + "_reassign_alleles.pdf")
                    )
                    save_as_pdf_pages([p], filename=savefile)
                    if show_plot:
                        print(p)
                else:
                    if show_plot:
                        print(p)
            except:
                logg.info("Error in plotting encountered. Skipping.")
                pass
        else:
            pass
    sleep(0.5)
    # if split_write_out:
    if "tigger_failed" in locals():
        logg.info(
            "Although tigger-genotype was not run successfully, file will still be saved with `_genotyped.tsv`"
            "extension for convenience."
        )
    for s in tqdm(
        data,
        desc="Writing out to individual folders ",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        if sample_id_dictionary is not None:
            out_file = dat_[dat_["sample_id"] == sample_id_dictionary[s]]
        else:
            out_file = dat_[dat_["sample_id"] == s]
        outfilepath = filePath_dict[s]
        write_airr(
            out_file, outfilepath.parent / (outfilepath.stem + "_genotyped.tsv")
        )


def create_germlines(
    vdj_data: Union[Dandelion, pd.DataFrame, str],
    germline: Optional[str] = None,
    org: Literal["human", "mouse"] = "human",
    genotyped_fasta: Optional[str] = None,
    additional_args: List[str] = [],
    save: Optional[str] = None,
) -> Dandelion:
    """
    Run CreateGermlines.py to reconstruct the germline V(D)J sequence.

    Parameters
    ----------
    vdj_data : Union[Dandelion, pd.DataFrame, str]
        `Dandelion` object, pandas `DataFrame` in changeo/airr format, or file path to changeo/airr
        file after clones have been determined.
    germline : Optional[str], optional
        path to germline database folder. `None` defaults to  environmental variable.
    org : Literal["human", "mouse"], optional
        organism of germline database.
    genotyped_fasta : Optional[str], optional
        location to corrected v genotyped fasta file.
    additional_args : List[str], optional
        additional arguments to pass to `CreateGermlines.py.`
    save : Optional[str], optional
        if provided, saves to specified file path.

    Returns
    -------
    Dandelion
        Dandelion object with `.germlines` slot populated.
    """
    start = logg.info("Reconstructing germline sequences")
    if not isinstance(vdj_data, Dandelion):
        tmpfile = (
            Path(vdj_data)
            if os.path.isfile(vdj_data)
            else Path(tempfile.TemporaryDirectory().name) / "tmp.tsv"
        )
        if isinstance(vdj_data, pd.DataFrame):
            write_airr(data=vdj_data.germline, save=tmpfile)
        creategermlines(
            airr_file=tmpfile,
            germline=germline,
            org=org,
            genotyped_fasta=genotyped_fasta,
            additional_args=additional_args,
        )
    else:
        tmpfile = Path(tempfile.TemporaryDirectory().name) / "tmp.tsv"
        vdj_data.write_airr(filename=tmpfile)
        if len(vdj_data.germline) > 0:
            tmpgmlfile = Path(tempfile.TemporaryDirectory().name) / "germ.fasta"
            write_fasta(fasta_dict=vdj_data.germline, out_fasta=tmpgmlfile)
            creategermlines(
                airr_file=tmpfile,
                germline=tmpgmlfile,
                org=org,
                additional_args=additional_args,
            )
        else:
            creategermlines(
                airr_file=tmpfile,
                germline=germline,
                org=org,
                genotyped_fasta=genotyped_fasta,
                additional_args=additional_args,
            )
    # return as Dandelion object
    germpass_outfile = tmpfile.parent / (tmpfile.stem + "_germ-pass.tsv")
    if isinstance(vdj_data, Dandelion):
        vdj_data.__init__(
            data=germpass_outfile,
            metadata=vdj_data.metadata,
            germline=vdj_data.germline,
            layout=vdj_data.layout,
            graph=vdj_data.graph,
            initialize=True,
        )
        out_vdj = vdj_data.copy()
    else:
        out_vdj = Dandelion(germpass_outfile)
        out_vdj.store_germline_reference(
            corrected=genotyped_fasta, germline=germline, org=org
        )
    if save is not None:
        shutil.move(germpass_outfile, save)
    logg.info(
        " finished",
        time=start,
        deep=("Returning Dandelion object: \n"),
    )
    return out_vdj


def filter_contigs(
    data: Union[Dandelion, pd.DataFrame, str],
    adata: Optional[AnnData] = None,
    filter_contig: bool = True,
    library_type: Optional[Literal["ig", "tr-ab", "tr-gd"]] = None,
    filter_rna: bool = False,
    filter_poorqualitycontig: bool = False,
    keep_highest_umi: bool = True,
    umi_foldchange_cutoff: int = 2,
    filter_extra_vdj_chains: bool = True,
    filter_extra_vj_chains: bool = False,
    filter_missing: bool = True,
    productive_only: bool = True,
    simple: bool = False,
    save: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    **kwargs,
) -> Tuple[Dandelion, AnnData]:
    """
    Filter doublets and poor quality cells and corresponding contigs based on provided V(D)J `DataFrame` and `AnnData`.

    Depends on a `AnnData`.obs slot populated with 'filter_rna' column. If the aligned sequence is an exact match
    between contigs, the contigs will be merged into the one with the highest umi count, adding the summing the
    umi count of the duplicated contigs to duplicate_count column. After this check, if there are still multiple
    contigs, cells with multiple contigs are filtered unless `keep_highest_umi` is False, where by the umi counts for
    each contig will then be compared and only the highest is retained. The contig with the highest umi that is
    > umi_foldchange_cutoff (default is empirically set at 2) will be retained. For productive heavy/long chains,
    if there are multiple contigs that survive the umi testing, then all contigs will be filtered. The default behaviour
    is to also filter cells with multiple light/short chains but this may sometimes be a true biological occurrence;
    toggling filter_extra_vj_chains to False will rescue the mutltiplet light chains. Lastly, contigs with no
    corresponding cell barcode in the AnnData object is filtered if filter_missing is True. However, this may be useful
    to toggle to False if more contigs are preferred to be kept or for integrating with bulk reperotire seq data.

    Parameters
    ----------
    data : Union[Dandelion, pd.DataFrame, str]
        V(D)J airr/changeo data to filter. Can be pandas `DataFrame` object or file path as string.
    adata : Optional[AnnData], optional
        AnnData object to filter. If not provided, will assume to keep all cells in the airr table.
    filter_contig : bool, optional
        If True, V(D)J `DataFrame` object returned will be filtered.
    library_type : Optional[Literal["ig", "tr-ab", "tr-gd"]], optional
        if specified, it will first filter based on the expected type of contigs:
            `ig`:
                IGH, IGK, IGL
            `tr-ab`:
                TRA, TRB
            `tr-gd`:
                TRG, TRD
        The rationale is that the choice of the library type should mean that the primers used would most likely
        amplify those related sequences and if there's any unexpected contigs, then they shouldn't be analysed.
    filter_rna : bool, optional
        If True, `AnnData` object returned will be filtered based on potential V(D)J doublets.
    filter_poorqualitycontig : bool, optional
        If True, barcodes marked with poor quality contigs will be filtered.
    keep_highest_umi : bool, optional
        If True, rescues IGH contigs with highest umi counts with a requirement that it passes the
        `umi_foldchange_cutoff` option. In addition, the sum of the all the heavy chain contigs must be greater than 3
        umi or all contigs will be filtered.
    umi_foldchange_cutoff : int, optional
        related to minimum fold change required to rescue heavy chain contigs/barcode otherwise they will be marked as
        doublets.
    filter_extra_vdj_chains : bool, optional
        cells with multiple heavy chains will be marked to filter.
        Exception is with TRD chains where allelic inclusion has been reported.
    filter_extra_vj_chains : bool, optional
        cells with multiple light chains will be marked to filter.
    filter_missing : bool, optional
        cells in V(D)J data not found in `AnnData` object will be marked to filter.
    productive_only : bool, optional
        whether or not to retain only productive contigs.
    simple : bool, optional
        simple filtering mode where only checks for potential gene assignment mismatches.
    save : Optional[Union[str, Path]], optional
        Only used if a pandas data frame or dandelion object is provided. Specifying will save the formatted vdj table.
    verbose : bool, optional
        whether to print progress.
    **kwargs
        additional kwargs passed to `Dandelion.Dandelion`.

    Returns
    -------
    Tuple[Dandelion, AnnData]
        filtered dandelion V(D)J object and `AnnData` object.

    Raises
    ------
    IndexError
        if no contigs passed filtering.
    ValueError
        if save file name is not suitable.
    """
    start = logg.info("Filtering contigs")
    if isinstance(data, Dandelion):
        dat_ = load_data(data.data)
    else:
        dat_ = load_data(data)

    if library_type is not None:
        acceptable = lib_type(library_type)
    else:
        if isinstance(data, Dandelion):
            if data.library_type is not None:
                acceptable = lib_type(data.library_type)
            else:
                acceptable = None
        else:
            acceptable = None

    if not simple:
        if productive_only:
            dat = dat_[dat_["productive"].isin(TRUES)].copy()
        else:
            dat = dat_.copy()
    else:
        dat = dat_.copy()

    if acceptable is not None:
        dat = dat[dat.locus.isin(acceptable)].copy()

    barcode = list(set(dat["cell_id"]))

    if adata is not None:
        adata_provided = True
        adata_ = adata.copy()
        if "filter_rna" not in adata_.obs:
            adata_.obs["filter_rna"] = "False"
        contig_check = pd.DataFrame(index=adata_.obs_names)
        bc_ = {}
        for b in barcode:
            bc_.update({b: "True"})
        contig_check["has_contig"] = pd.Series(bc_)
        contig_check.replace(np.nan, "No_contig", inplace=True)
        adata_.obs["has_contig"] = pd.Series(contig_check["has_contig"])
    else:
        adata_provided = False
        obs = pd.DataFrame(index=barcode)
        adata_ = ad.AnnData(obs=obs)
        adata_.obs["filter_rna"] = "False"
        adata_.obs["has_contig"] = "True"

    if not simple:
        tofilter = FilterContigs(
            dat,
            keep_highest_umi,
            umi_foldchange_cutoff,
            filter_poorqualitycontig,
            filter_extra_vdj_chains,
            filter_extra_vj_chains,
            verbose,
        )
    else:
        tofilter = FilterContigsLite(dat, verbose)

    poor_qual = tofilter.poor_qual.copy()
    h_doublet = tofilter.h_doublet.copy()
    l_doublet = tofilter.l_doublet.copy()
    drop_contig = tofilter.drop_contig.copy()
    umi_adjustment = tofilter.umi_adjustment.copy()

    if len(umi_adjustment) > 0:
        dat["duplicate_count"].update(umi_adjustment)

    poorqual = {c: "False" for c in adata_.obs_names}
    hdoublet = {c: "False" for c in adata_.obs_names}
    ldoublet = {c: "False" for c in adata_.obs_names}

    poorqual_ = {x: "True" for x in poor_qual}
    hdoublet_ = {x: "True" for x in h_doublet}
    ldoublet_ = {x: "True" for x in l_doublet}

    poorqual.update(poorqual_)
    hdoublet.update(hdoublet_)
    ldoublet.update(ldoublet_)

    adata_.obs["filter_contig_quality"] = pd.Series(poorqual)
    adata_.obs["filter_contig_VDJ"] = pd.Series(hdoublet)
    adata_.obs["filter_contig_VJ"] = pd.Series(ldoublet)

    drop_contig = list(set(flatten(drop_contig)))

    filter_ids = []
    if filter_contig:
        logg.info("Finishing up filtering")
        if filter_poorqualitycontig:
            filter_ids = poor_qual
        else:
            filter_ids = []

        if filter_extra_vdj_chains:
            filter_ids = filter_ids + h_doublet

        if filter_extra_vj_chains:
            filter_ids = filter_ids + l_doublet

        filter_ids = list(set(filter_ids))

        filter_ids = filter_ids + list(
            adata_[adata_.obs["filter_rna"].isin(TRUES)].obs_names
        )
        filter_ids = list(set(filter_ids))

        if filter_missing:
            dat = dat[dat["cell_id"].isin(adata_.obs_names)].copy()

        _dat = dat[~(dat["cell_id"].isin(filter_ids))].copy()
        _dat = _dat[~(_dat["sequence_id"].isin(drop_contig))].copy()

        # final check
        barcodes_final = list(set(_dat["cell_id"]))
        check_dat_barcodes = list(
            set(_dat[_dat["locus"].isin(HEAVYLONG)]["cell_id"])
        )
        filter_ids2 = list(set(barcodes_final) - set(check_dat_barcodes))
        _dat = _dat[~(_dat["cell_id"].isin(filter_ids2))].copy()

        if _dat.shape[0] == 0:
            raise IndexError(
                "No contigs passed filtering. Are you sure that the cell barcodes are matching?"
            )

        if os.path.isfile(str(data)):
            data_path = Path(data)
            write_airr(
                _dat,
                data_path.parent / (data_path.stem + "_filtered.tsv"),
            )
        else:
            if save is not None:
                if str(save).endswith(".tsv"):
                    write_airr(_dat, save)
                else:
                    raise ValueError(
                        "{} not suitable. Please provide a file name that ends with .tsv".format(
                            str(save)
                        )
                    )
    else:
        _dat = dat.copy()

    if filter_contig:
        barcode1 = list(set(dat["cell_id"]))

    barcode2 = list(set(_dat["cell_id"]))

    if filter_contig:
        failed = list(set(barcode1) ^ set(barcode2))

    logg.info("Initializing Dandelion object")
    out_dat = Dandelion(data=_dat, **kwargs)
    if isinstance(data, Dandelion):
        out_dat.germline = data.germline

    if adata_provided:
        bc_2 = {b: "True" for b in barcode2}
        if filter_contig:
            failed2 = {b: "False" for b in failed}
            bc_2.update(failed2)
        contig_check["contig_QC_pass"] = pd.Series(bc_2)
        contig_check.replace(np.nan, "No_contig", inplace=True)
        adata_.obs["contig_QC_pass"] = pd.Series(contig_check["contig_QC_pass"])
        adata_.obs["filter_contig"] = adata_.obs_names.isin(filter_ids)
        if filter_rna:
            # not saving the scanpy object because there's no need to at the moment
            out_adata = adata_[adata_.obs["filter_contig"].isin(FALSES)].copy()
        else:
            out_adata = adata_.copy()
        transfer(out_adata, out_dat, overwrite=True)
        logg.info(
            " finished",
            time=start,
            deep=("Returning Dandelion and AnnData objects: \n"),
        )
        return (out_dat, out_adata)
    else:
        logg.info(
            " finished",
            time=start,
            deep=("Returning Dandelion object: \n"),
        )
        return out_dat


def quantify_mutations(
    data: Union[Dandelion, str],
    split_locus: bool = False,
    sequence_column: Optional[str] = None,
    germline_column: Optional[str] = None,
    region_definition: Optional[str] = None,
    mutation_definition: Optional[str] = None,
    frequency: bool = False,
    combine: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Run basic mutation load analysis.

    Implemented in `shazam` https://shazam.readthedocs.io/en/stable/vignettes/Mutation-Vignette.

    Parameters
    ----------
    data : Union[Dandelion, str]
        `Dandelion` object, file path to AIRR file.
    split_locus : bool, optional
        whether to return the results for heavy chain and light chain separately.
    sequence_column : Optional[str], optional
        passed to shazam's `observedMutations`. https://shazam.readthedocs.io/en/stable/topics/observedMutations
    germline_column : Optional[str], optional
        passed to shazam's `observedMutations`. https://shazam.readthedocs.io/en/stable/topics/observedMutations
    region_definition : Optional[str], optional
        passed to shazam's `observedMutations`. https://shazam.readthedocs.io/en/stable/topics/IMGT_SCHEMES/
    mutation_definition : Optional[str], optional
        passed to shazam's `observedMutations`. https://shazam.readthedocs.io/en/stable/topics/MUTATION_SCHEMES/
    frequency : bool, optional
        whether to return the results a frequency or counts.
    combine : bool, optional
        whether to return the results for replacement and silent mutations separately.
    **kwargs
        passed to shazam::observedMutations.

    Returns
    -------
    pd.DataFrame
        pandas DataFrame holding mutation information.
    """
    start = logg.info("Quantifying mutations")
    try:
        import rpy2
        from rpy2.robjects.packages import importr
        from rpy2.rinterface import NULL
        from rpy2.robjects import pandas2ri
    except:
        raise (
            ImportError(
                "Unable to initialise R instance. Please run this separately through R with Shazam's tutorial."
            )
        )

    sh = importr("shazam")
    base = importr("base")
    if isinstance(data, Dandelion):
        dat = load_data(data.data)
    else:
        dat = load_data(data)

    pandas2ri.activate()
    warnings.filterwarnings("ignore")

    dat = sanitize_data(dat)

    if "ambiguous" in dat:
        dat_ = dat[dat["ambiguous"] == "F"].copy()
    else:
        dat_ = dat.copy()

    if sequence_column is None:
        seq_ = "sequence_alignment"
    else:
        seq_ = sequence_column

    if germline_column is None:
        germline_ = "germline_alignment_d_mask"
    else:
        germline_ = germline_column

    if region_definition is None:
        reg_d = NULL
    else:
        reg_d = base.get(region_definition)

    if mutation_definition is None:
        mut_d = NULL
    else:
        mut_d = base.get(mutation_definition)

    if split_locus is False:
        dat_ = dat_.where(dat_.isna(), dat_.astype(str))
        try:
            dat_r = pandas2ri.py2rpy(dat_)
        except:
            dat_ = dat_.astype(str)
            dat_r = pandas2ri.py2rpy(dat_)

        results = sh.observedMutations(
            dat_r,
            sequenceColumn=seq_,
            germlineColumn=germline_,
            regionDefinition=reg_d,
            mutationDefinition=mut_d,
            frequency=frequency,
            combine=combine,
            **kwargs,
        )
        if rpy2.__version__ >= "3.4.5":
            from rpy2.robjects.conversion import localconverter

            with localconverter(
                rpy2.robjects.default_converter + pandas2ri.converter
            ):
                pd_df = rpy2.robjects.conversion.rpy2py(results)
        else:
            # pd_df = pandas2ri.rpy2py_data frame(results)
            pd_df = results.copy()
    else:
        dat_h = dat_[dat_["locus"] == "IGH"]
        dat_l = dat_[dat_["locus"].isin(["IGK", "IGL"])]

        dat_h = dat_h.where(dat_h.isna(), dat_h.astype(str))
        try:
            dat_h_r = pandas2ri.py2rpy(dat_h)
        except:
            dat_h = dat_h.astype(str)
            dat_h_r = pandas2ri.py2rpy(dat_h)

        dat_l = dat_l.where(dat_l.isna(), dat_l.astype(str))
        try:
            dat_l_r = pandas2ri.py2rpy(dat_l)
        except:
            dat_l = dat_l.astype(str)
            dat_l_r = pandas2ri.py2rpy(dat_l)

        results_h = sh.observedMutations(
            dat_h_r,
            sequenceColumn=seq_,
            germlineColumn=germline_,
            regionDefinition=reg_d,
            mutationDefinition=mut_d,
            frequency=frequency,
            combine=combine,
            **kwargs,
        )
        results_l = sh.observedMutations(
            dat_l_r,
            sequenceColumn=seq_,
            germlineColumn=germline_,
            regionDefinition=reg_d,
            mutationDefinition=mut_d,
            frequency=frequency,
            combine=combine,
            **kwargs,
        )
        if rpy2.__version__ >= "3.4.5":
            from rpy2.robjects.conversion import localconverter

            with localconverter(
                rpy2.robjects.default_converter + pandas2ri.converter
            ):
                results_h = rpy2.robjects.conversion.rpy2py(results_h)
                results_l = rpy2.robjects.conversion.rpy2py(results_l)
        pd_df = pd.concat([results_h, results_l])

    pd_df.set_index("sequence_id", inplace=True, drop=False)
    # this doesn't actually catch overwritten columns
    cols_to_return = pd_df.columns.difference(dat.columns)
    if len(cols_to_return) < 1:
        cols_to_return = list(
            filter(re.compile("mu_.*").match, [c for c in pd_df.columns])
        )
    else:
        cols_to_return = cols_to_return

    res = {}
    if isinstance(data, Dandelion):
        for x in cols_to_return:
            res[x] = list(pd_df[x])
            # TODO: str will make it work for the back and forth conversion with rpy2. but maybe can use a better option
            dat_[x] = [str(r) for r in res[x]]
            data.data[x] = pd.Series(dat_[x])
        if split_locus is False:
            metadata_ = data.data[["cell_id"] + list(cols_to_return)]
        else:
            metadata_ = data.data[["locus", "cell_id"] + list(cols_to_return)]

        for x in cols_to_return:
            metadata_[x] = metadata_[x].astype(float)

        if split_locus is False:
            metadata_ = metadata_.groupby("cell_id").sum()
        else:
            metadata_ = metadata_.groupby(["locus", "cell_id"]).sum()
            metadatas = []
            for x in list(set(data.data["locus"])):
                tmp = metadata_.iloc[
                    metadata_.index.isin([x], level="locus"), :
                ]
                tmp.index = tmp.index.droplevel()
                tmp.columns = [c + "_" + str(x) for c in tmp.columns]
                metadatas.append(tmp)
            metadata_ = functools.reduce(
                lambda x, y: pd.merge(
                    x, y, left_index=True, right_index=True, how="outer"
                ),
                metadatas,
            )

        metadata_.index.name = None
        data.data = sanitize_data(data.data)
        if data.metadata is None:
            data.metadata = metadata_
        else:
            for x in metadata_.columns:
                data.metadata[x] = pd.Series(metadata_[x])
        logg.info(
            " finished",
            time=start,
            deep=(
                "Updated Dandelion object: \n"
                "   'data', contig-indexed clone table\n"
                "   'metadata', cell-indexed clone table\n"
            ),
        )
    else:
        for x in cols_to_return:
            res[x] = list(pd_df[x])
            # TODO: str will make it work for the back and forth conversion with rpy2. but maybe can use a better option
            dat[x] = [str(r) for r in res[x]]
        # dat = sanitize_data(dat)
        if isinstance(data, pd.DataFrame):
            logg.info(" finished", time=start, deep=("Returning DataFrame\n"))
            return dat
        elif os.path.isfile(data):
            logg.info(
                " finished",
                time=start,
                deep=("saving DataFrame at {}\n".format(str(data))),
            )
            write_airr(dat, data)


def calculate_threshold(
    data: Union[Dandelion, pd.DataFrame, str],
    mode: Literal["single-cell", "heavy"] = "single-cell",
    manual_threshold: Optional[float] = None,
    VJthenLen: bool = False,
    onlyHeavy: bool = False,
    model: Optional[
        Literal[
            "ham",
            "aa",
            "hh_s1f",
            "hh_s5f",
            "mk_rs1nf",
            "hs1f_compat",
            "m1n_compat",
        ]
    ] = None,
    normalize_method: Optional[Literal["len"]] = None,
    threshold_method: Optional[Literal["gmm", "density"]] = None,
    edge: Optional[float] = None,
    cross: Optional[List[float]] = None,
    subsample: Optional[int] = None,
    threshold_model: Optional[
        Literal["norm-norm", "norm-gamma", "gamma-norm", "gamma-gamma"]
    ] = None,
    cutoff: Optional[Literal["optimal", "intersect", "user"]] = None,
    sensitivity: Optional[float] = None,
    specificity: Optional[float] = None,
    plot: bool = True,
    plot_group: Optional[str] = None,
    figsize: Tuple[Union[int, float], Union[int, float]] = (4.5, 2.5),
    save_plot: Optional[str] = None,
    ncpu: int = 1,
    **kwargs,
) -> Dandelion:
    """
    Calculating nearest neighbor distances for tuning clonal assignment with `shazam`.

    https://shazam.readthedocs.io/en/stable/vignettes/DistToNearest-Vignette/

    Runs the following:

    distToNearest
        Get non-zero distance of every heavy chain (IGH) sequence (as defined by sequenceColumn) to its nearest sequence
        in a partition of heavy chains sharing the same V gene, J gene, and junction length (VJL), or in a partition of
        single cells with heavy chains sharing the same heavy chain VJL combination, or of single cells with heavy and
        light chains sharing the same heavy chain VJL and light chain VJL combinations.
    findThreshold
        automtically determines an optimal threshold for clonal assignment of Ig sequences using a vector of nearest
        neighbor distances. It provides two alternative methods using either a Gamma/Gaussian Mixture Model fit
        (threshold_method="gmm") or kernel density fit (threshold_method="density").

    Parameters
    ----------
    data : Union[Dandelion, pd.DataFrame, str]
        input `Danelion`, AIRR data as pandas DataFrame or path to file.
    mode : Literal["single-cell", "heavy"], optional
        accepts one of "heavy" or "single-cell".
        Refer to https://shazam.readthedocs.io/en/stable/vignettes/DistToNearest-Vignette.
    manual_threshold : Optional[float], optional
        value to manually plot in histogram.
    VJthenLen : bool, optional
        logical value specifying whether to perform partitioning as a 2-stage process.
        If True, partitions are made first based on V and J gene, and then further split
        based on junction lengths corresponding to sequenceColumn.
        If False, perform partition as a 1-stage process during which V gene, J gene, and junction length
        are used to create partitions simultaneously.
        Defaults to False.
    onlyHeavy : bool, optional
        use only the IGH (BCR) or TRB/TRD (TCR) sequences for grouping. Only applicable to single-cell mode.
        See groupGenes for further details.
    model : Optional[Literal["ham", "aa", "hh_s1f", "hh_s5f", "mk_rs1nf", "hs1f_compat", "m1n_compat", ]], optional
        underlying SHM model, which must be one of "ham","aa","hh_s1f","hh_s5f","mk_rs1nf","hs1f_compat","m1n_compat".
    normalize_method : Optional[Literal["len"]], optional
        method of normalization. The default is "len", which divides the distance by the length of the sequence group.
        If "none" then no normalization if performed.
    threshold_method : Optional[Literal["gmm", "density"]], optional
        string defining the method to use for determining the optimal threshold. One of "gmm" or "density".
    edge : Optional[float], optional
        upper range as a fraction of the data density to rule initialization of Gaussian fit parameters.
        Default value is 0.9 (or 90). Applies only when threshold_method="density".
    cross : Optional[List[float]], optional
        supplementary nearest neighbor distance vector output from distToNearest for initialization of the Gaussian fit
        parameters. Applies only when method="gmm".
    subsample : Optional[int], optional
        maximum number of distances to subsample to before threshold detection.
    threshold_model : Optional[Literal["norm-norm", "norm-gamma", "gamma-norm", "gamma-gamma"]], optional
        allows the user to choose among four possible combinations of fitting curves: "norm-norm", "norm-gamma",
        "gamma-norm", and "gamma-gamma". Applies only when method="gmm".
    cutoff : Optional[Literal["optimal", "intersect", "user"]], optional
        method to use for threshold selection: the optimal threshold "optimal", the intersection point of the two fitted
        curves "intersect", or a value defined by user for one of the sensitivity or specificity "user". Applies only
        when method="gmm".
    sensitivity : Optional[float], optional
        sensitivity required. Applies only when method="gmm" and cutoff="user".
    specificity : Optional[float], optional
        specificity required. Applies only when method="gmm" and cutoff="user".
    plot : bool, optional
        whether or not to return plot.
    plot_group : Optional[str], optional
        determines the fill color and facets.
    figsize : Tuple[Union[int, float], Union[int, float]], optional
        size of plot.
    save_plot : Optional[str], optional
        if specified, plot will be save with this path.
    ncpu : int, optional
        number of cpus to run `distToNearest`. defaults to 1.
    **kwargs
        passed to shazam's `distToNearest <https://shazam.readthedocs.io/en/stable/topics/distToNearest/>`__.

    Returns
    -------
    Dandelion
        Dandelion object with `.threshold` slot filled.

    Raises
    ------
    ValueError
        if automatic thresholding failed.
    """
    start = logg.info("Calculating threshold")
    try:
        import rpy2
        from rpy2.robjects.packages import importr
        from rpy2.rinterface import NULL
        from rpy2.robjects import pandas2ri, FloatVector
    except:
        raise (
            ImportError(
                "Unable to initialise R instance. Please run this separately through R with Shazam's tutorial."
            )
        )

    if isinstance(data, Dandelion):
        dat = load_data(data.data)
    elif isinstance(data, pd.DataFrame) or os.path.isfile(str(data)):
        dat = load_data(data)
        warnings.filterwarnings("ignore")

    sh = importr("shazam")
    pandas2ri.activate()
    if "v_call_genotyped" in dat.columns:
        v_call = "v_call_genotyped"
    else:
        v_call = "v_call"
    if model is None:
        model_ = "ham"
    else:
        model_ = model
    if normalize_method is None:
        norm_ = "len"
    else:
        norm_ = normalize_method
    if threshold_method is None:
        threshold_method_ = "density"
    else:
        threshold_method_ = threshold_method
    if subsample is None:
        subsample_ = NULL
    else:
        subsample_ = subsample

    if mode == "heavy":
        dat_h = dat[dat["locus"].isin(["IGH", "TRB", "TRD"])].copy()
        try:
            dat_h_r = pandas2ri.py2rpy(dat_h)
        except:
            dat_h = dat_h.astype(str)
            dat_h_r = pandas2ri.py2rpy(dat_h)

        dist_ham = sh.distToNearest(
            dat_h_r, vCallColumn=v_call, model=model_, normalize=norm_, **kwargs
        )
    elif mode == "single-cell":
        try:
            dat_r = pandas2ri.py2rpy(dat)
        except:
            dat = dat.astype(str)
            dat_r = pandas2ri.py2rpy(dat)
        try:
            dist_ham = sh.distToNearest(
                dat_r,
                cellIdColumn="cell_id",
                locusColumn="locus",
                VJthenLen=VJthenLen,
                vCallColumn=v_call,
                onlyHeavy=onlyHeavy,
                normalize=norm_,
                model=model_,
                nproc=ncpu,
                **kwargs,
            )
        except:
            logg.info(
                "Rerun this after filtering. For now, switching to heavy mode."
            )
            dat_h = dat[dat["locus"].isin(["IGH", "TRB", "TRD"])].copy()
            try:
                dat_h_r = pandas2ri.py2rpy(dat_h)
            except:
                dat_h = dat_h.astype(str)
                dat_h_r = pandas2ri.py2rpy(dat_h)

            dist_ham = sh.distToNearest(
                dat_h_r,
                vCallColumn=v_call,
                model=model_,
                normalize=norm_,
                nproc=ncpu,
                **kwargs,
            )
    if rpy2.__version__ >= "3.4.5":
        from rpy2.robjects.conversion import localconverter

        with localconverter(
            rpy2.robjects.default_converter + pandas2ri.converter
        ):
            dist_ham = rpy2.robjects.conversion.rpy2py(dist_ham)
    # Find threshold using density method
    dist = np.array(dist_ham["dist_nearest"])
    if threshold_method_ == "density":
        if edge is None:
            edge_ = 0.9
        else:
            edge_ = edge
        dist_threshold = sh.findThreshold(
            FloatVector(dist[~np.isnan(dist)]),
            method=threshold_method_,
            subsample=subsample_,
            edge=edge_,
        )
        threshold = np.array(dist_threshold.slots["threshold"])[0]
        if np.isnan(threshold):
            logg.info(
                "      Threshold method 'density' did not return with any values. Switching to method = 'gmm'."
            )
            threshold_method_ = "gmm"
            if threshold_model is None:
                threshold_model_ = "gamma-gamma"
            else:
                threshold_model_ = threshold_model
            if cross is None:
                cross_ = NULL
            else:
                cross_ = cross
            if cutoff is None:
                cutoff_ = "optimal"
            else:
                cutoff_ = cutoff
            if sensitivity is None:
                sen_ = NULL
            else:
                sen_ = sensitivity
            if specificity is None:
                spc_ = NULL
            else:
                spc_ = specificity
            dist_threshold = sh.findThreshold(
                FloatVector(dist[~np.isnan(dist)]),
                method=threshold_method_,
                model=threshold_model_,
                cross=cross_,
                subsample=subsample_,
                cutoff=cutoff_,
                sen=sen_,
                spc=spc_,
            )
            if rpy2.__version__ >= "3.4.5":
                from rpy2.robjects.conversion import localconverter

                with localconverter(
                    rpy2.robjects.default_converter + pandas2ri.converter
                ):
                    dist_threshold = rpy2.robjects.conversion.rpy2py(
                        dist_threshold
                    )

            threshold = np.array(dist_threshold.slots["threshold"])[0]
    else:
        if threshold_model is None:
            threshold_model_ = "gamma-gamma"
        else:
            threshold_model_ = threshold_model
        if cross is None:
            cross_ = NULL
        else:
            cross_ = cross
        if cutoff is None:
            cutoff_ = "optimal"
        else:
            cutoff_ = cutoff
        if sensitivity is None:
            sen_ = NULL
        else:
            sen_ = sensitivity
        if specificity is None:
            spc_ = NULL
        else:
            spc_ = specificity
        dist_threshold = sh.findThreshold(
            FloatVector(dist[~np.isnan(dist)]),
            method=threshold_method_,
            model=threshold_model_,
            cross=cross_,
            subsample=subsample_,
            cutoff=cutoff_,
            sen=sen_,
            spc=spc_,
        )
        if rpy2.__version__ >= "3.4.5":
            from rpy2.robjects.conversion import localconverter

            with localconverter(
                rpy2.robjects.default_converter + pandas2ri.converter
            ):
                dist_threshold = rpy2.robjects.conversion.rpy2py(dist_threshold)
        threshold = np.array(dist_threshold.slots["threshold"])[0]
    if np.isnan(threshold):
        raise ValueError(
            "Automatic thresholding failed. Please visually inspect the resulting distribution fits"
            + " and choose a threshold value manually."
        )
    # dist_ham = pandas2ri.rpy2py_data frame(dist_ham)

    if manual_threshold is None:
        tr = threshold
    else:
        tr = manual_threshold

    if plot:
        options.figure_size = figsize
        if plot_group is None:
            plot_group = "sample_id"
        else:
            plot_group = plot_group

        p = (
            ggplot(dist_ham, aes("dist_nearest", fill=str(plot_group)))
            + theme_bw()
            + xlab("Grouped Hamming distance")
            + ylab("Count")
            + geom_histogram(binwidth=0.01)
            + geom_vline(
                xintercept=tr, linetype="dashed", color="blue", size=0.5
            )
            + annotate(
                "text",
                x=tr + 0.02,
                y=10,
                label="Threshold:\n" + str(np.around(tr, decimals=2)),
                size=8,
                color="Blue",
            )
            + facet_wrap("~" + str(plot_group), scales="free_y")
            + theme(legend_position="none")
        )
        if save_plot is not None:
            save_as_pdf_pages([p], filename=save_plot)
        print(p)
    else:
        logg.info(
            "Automatic Threshold : "
            + str(np.around(threshold, decimals=2))
            + "\n method = "
            + str(threshold_method_)
        )
    if isinstance(data, Dandelion):
        data.threshold = tr
        logg.info(
            " finished",
            time=start,
            deep=(
                "Updated Dandelion object: \n"
                "   'threshold', threshold value for tuning clonal assignment\n"
            ),
        )
    else:
        output = Dandelion(dat)
        output.threshold = tr
        return output


class FilterContigs:
    """
    `FilterContigs` class object.

    Main class object to run filter_contigs.

    Attributes
    ----------
    Cell : dandelion.utilities._utilities.Tree
        nested dictionary of cells.
    drop_contig : List[str]
        list of `sequence_id`s to drop.
    h_doublet : List[str]
        list of `sequence_id`s that are VDJ 'multiplets'.
    l_doublet : List[str]
        list of `sequence_id`s that are VJ 'multiplets'.
    poor_qual : List[str]
        list of `sequence_id`s that are VDJ 'multiplets'.
    umi_adjustment : Dict[str, int]
        dictionary of `sequence_id`s with adjusted umi value.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        keep_highest_umi: bool,
        umi_foldchange_cutoff: Union[int, float],
        filter_poorqualitycontig: bool,
        filter_extra_vdj_chains: bool,
        filter_extra_vj_chains: bool,
        verbose: bool,
    ):
        """Init method for FilterContigs.

        Parameters
        ----------
        data : pd.DataFrame
            AIRR data frame in Dandelion.data.
        keep_highest_umi : bool
            whether or not to keep highest UMI contig.
        umi_foldchange_cutoff : Union[int, float]
            fold-change cut off for decision.
        filter_poorqualitycontig : bool
            whether or not to flter poor quality contigs.
        filter_extra_vdj_chains : bool
            whether or not to flter extra VDJ chains.
        filter_extra_vj_chains : bool
            whether or not to flter extra VJ chains.
        verbose : bool
            whether or not to print progress.
        """
        self.Cell = Tree()
        self.poor_qual = []
        self.h_doublet = []
        self.l_doublet = []
        self.drop_contig = []
        self.umi_adjustment = {}
        if "v_call_genotyped" in data.columns:
            v_dict = dict(zip(data["sequence_id"], data["v_call_genotyped"]))
        else:
            v_dict = dict(zip(data["sequence_id"], data["v_call"]))
        d_dict = dict(zip(data["sequence_id"], data["d_call"]))
        j_dict = dict(zip(data["sequence_id"], data["j_call"]))
        c_dict = dict(zip(data["sequence_id"], data["c_call"]))
        l_dict = dict(zip(data["sequence_id"], data["locus"]))
        for contig, row in tqdm(
            data.iterrows(),
            desc="Preparing data",
        ):
            cell = row["cell_id"]
            if row["locus"] in HEAVYLONG:
                if row["productive"] in TRUES:
                    self.Cell[cell]["VDJ"]["P"][contig].update(row)
                elif row["productive"] in FALSES:
                    self.Cell[cell]["VDJ"]["NP"][contig].update(row)
            elif row["locus"] in LIGHTSHORT:
                if row["productive"] in TRUES:
                    self.Cell[cell]["VJ"]["P"][contig].update(row)
                elif row["productive"] in FALSES:
                    self.Cell[cell]["VJ"]["NP"][contig].update(row)
        for cell in tqdm(
            self.Cell,
            desc="Scanning for poor quality/ambiguous contigs",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            disable=not verbose,
        ):
            if len(self.Cell[cell]["VDJ"]["P"]) > 0:
                data1 = pd.DataFrame(
                    [
                        self.Cell[cell]["VDJ"]["P"][x]
                        for x in self.Cell[cell]["VDJ"]["P"]
                        if isinstance(self.Cell[cell]["VDJ"]["P"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VDJ"]["P"][x]["sequence_id"]
                        for x in self.Cell[cell]["VDJ"]["P"]
                        if isinstance(self.Cell[cell]["VDJ"]["P"][x], dict)
                    ],
                )
                h_p = list(data1["sequence_id"])
                h_umi_p = [
                    int(x) for x in pd.to_numeric(data1["duplicate_count"])
                ]
                h_ccall_p = list(data1["c_call"])
                h_locus_p = list(data1["locus"])
                if len(h_p) > 1:
                    if "sequence_alignment" in data1:
                        (
                            data1,
                            h_p,
                            h_umi_p,
                            h_ccall_p,
                            umi_adjust_h,
                            drop_h,
                        ) = check_update_same_seq(data1)
                        if len(umi_adjust_h) > 0:
                            self.umi_adjustment.update(umi_adjust_h)
                        if len(drop_h) > 0:
                            for d_h in drop_h:
                                self.drop_contig.append(d_h)
                    if len(h_p) > 1:
                        highest_umi_h = max(h_umi_p)
                        highest_umi_idx = [
                            i
                            for i, j in enumerate(h_umi_p)
                            if j == highest_umi_h
                        ]
                        keep_index_h = highest_umi_idx[0]
                        keep_hc_contig = h_p[keep_index_h]
                        umi_test = [
                            int(highest_umi_h) / x < umi_foldchange_cutoff
                            for x in h_umi_p[:keep_index_h]
                            + h_umi_p[keep_index_h:]
                        ]
                        sum_umi = sum(h_umi_p)
                        if "IGHD" in h_ccall_p:
                            if all(x in ["IGHM", "IGHD"] for x in h_ccall_p):
                                h_ccall_p_igm_count = dict(
                                    data1[data1["c_call"] == "IGHM"][
                                        "duplicate_count"
                                    ]
                                )
                                h_ccall_p_igd_count = dict(
                                    data1[data1["c_call"] == "IGHD"][
                                        "duplicate_count"
                                    ]
                                )

                                if len(h_ccall_p_igm_count) > 1:
                                    if filter_extra_vdj_chains:
                                        max_igm_count = max(
                                            h_ccall_p_igm_count.values()
                                        )
                                        max_id_keys = [
                                            k
                                            for k, v in h_ccall_p_igm_count.items()
                                            if v == max_igm_count
                                        ]
                                        if len(max_id_keys) == 1:
                                            drop_keys = [
                                                k
                                                for k, v in h_ccall_p_igm_count.items()
                                                if v < max_igm_count
                                            ]
                                            for dk in drop_keys:
                                                self.drop_contig.append(dk)
                                        else:
                                            self.h_doublet.append(cell)
                                if len(h_ccall_p_igd_count) > 1:
                                    if filter_extra_vdj_chains:
                                        max_igd_count = max(
                                            h_ccall_p_igd_count.values()
                                        )
                                        max_id_keys = [
                                            k
                                            for k, v in h_ccall_p_igd_count.items()
                                            if v == max_igd_count
                                        ]
                                        if len(max_id_keys) == 1:
                                            drop_keys = [
                                                k
                                                for k, v in h_ccall_p_igd_count.items()
                                                if v < max_igd_count
                                            ]
                                            for dk in drop_keys:
                                                self.drop_contig.append(dk)
                                        else:
                                            self.h_doublet.append(cell)
                            else:
                                if len(highest_umi_idx) > 1:
                                    if filter_extra_vdj_chains:
                                        self.h_doublet.append(cell)
                                if sum_umi < 4:
                                    if filter_extra_vdj_chains:
                                        self.h_doublet.append(cell)
                                if any(umi_test):
                                    if filter_extra_vdj_chains:
                                        self.h_doublet.append(cell)
                                if len(highest_umi_idx) == 1:
                                    other_umi_idx = [
                                        i
                                        for i, j in enumerate(h_umi_p)
                                        if j != highest_umi_h
                                    ]
                                    umi_test_ = [
                                        highest_umi_h / x
                                        >= umi_foldchange_cutoff
                                        for x in h_umi_p[:keep_index_h]
                                        + h_umi_p[keep_index_h:]
                                    ]
                                    umi_test_dict = dict(
                                        zip(other_umi_idx, umi_test_)
                                    )
                                    for otherindex in umi_test_dict:
                                        if umi_test_dict[otherindex]:
                                            if keep_highest_umi:
                                                self.drop_contig.append(
                                                    h_p[otherindex]
                                                )
                                    # refresh
                                    data1 = pd.DataFrame(
                                        [data1.loc[keep_hc_contig]]
                                    )
                                    h_p = list(data1["sequence_id"])
                        elif all(x in ["TRB", "TRD"] for x in h_locus_p):
                            if len(list(set(h_locus_p))) == 2:
                                h_locus_p_trb_count = dict(
                                    data1[data1["locus"] == "TRB"][
                                        "duplicate_count"
                                    ]
                                )
                                h_locus_p_trd_count = dict(
                                    data1[data1["locus"] == "TRD"][
                                        "duplicate_count"
                                    ]
                                )

                                if len(h_locus_p_trb_count) > 1:
                                    if filter_extra_vdj_chains:
                                        max_trb_count = max(
                                            h_locus_p_trb_count.values()
                                        )
                                        max_id_keys = [
                                            k
                                            for k, v in h_locus_p_trb_count.items()
                                            if v == max_trb_count
                                        ]
                                        if len(max_id_keys) == 1:
                                            drop_keys = [
                                                k
                                                for k, v in h_locus_p_trb_count.items()
                                                if v < max_trb_count
                                            ]
                                            for dk in drop_keys:
                                                self.drop_contig.append(dk)
                                        else:
                                            self.h_doublet.append(cell)
                                if len(h_locus_p_trd_count) > 1:
                                    if filter_extra_vdj_chains:
                                        max_trd_count = max(
                                            h_locus_p_trd_count.values()
                                        )
                                        max_id_keys = [
                                            k
                                            for k, v in h_locus_p_trd_count.items()
                                            if v == max_trd_count
                                        ]
                                        if len(max_id_keys) == 1:
                                            drop_keys = [
                                                k
                                                for k, v in h_locus_p_trd_count.items()
                                                if v < max_trd_count
                                            ]
                                            for dk in drop_keys:
                                                self.drop_contig.append(dk)
                                        else:
                                            self.h_doublet.append(cell)
                            else:
                                if len(highest_umi_idx) > 1:
                                    if filter_extra_vdj_chains:
                                        self.h_doublet.append(cell)
                                if sum_umi < 4:
                                    if filter_extra_vdj_chains:
                                        self.h_doublet.append(cell)
                                if any(umi_test):
                                    if filter_extra_vdj_chains:
                                        self.h_doublet.append(cell)
                                if len(highest_umi_idx) == 1:
                                    other_umi_idx = [
                                        i
                                        for i, j in enumerate(h_umi_p)
                                        if j != highest_umi_h
                                    ]
                                    umi_test_ = [
                                        highest_umi_h / x
                                        >= umi_foldchange_cutoff
                                        for x in h_umi_p[:keep_index_h]
                                        + h_umi_p[keep_index_h:]
                                    ]
                                    umi_test_dict = dict(
                                        zip(other_umi_idx, umi_test_)
                                    )
                                    for otherindex in umi_test_dict:
                                        if umi_test_dict[otherindex]:
                                            if keep_highest_umi:
                                                self.drop_contig.append(
                                                    h_p[otherindex]
                                                )
                                    # refresh
                                    data1 = pd.DataFrame(
                                        [data1.loc[keep_hc_contig]]
                                    )
                                    h_p = list(data1["sequence_id"])
                        else:
                            if len(highest_umi_idx) > 1:
                                if filter_extra_vdj_chains:
                                    self.h_doublet.append(cell)
                            if sum_umi < 4:
                                if filter_extra_vdj_chains:
                                    self.h_doublet.append(cell)
                            if any(umi_test):
                                if filter_extra_vdj_chains:
                                    self.h_doublet.append(cell)
                            if len(highest_umi_idx) == 1:
                                other_umi_idx = [
                                    i
                                    for i, j in enumerate(h_umi_p)
                                    if j != highest_umi_h
                                ]
                                umi_test_ = [
                                    highest_umi_h / x >= umi_foldchange_cutoff
                                    for x in h_umi_p[:keep_index_h]
                                    + h_umi_p[keep_index_h:]
                                ]
                                umi_test_dict = dict(
                                    zip(other_umi_idx, umi_test_)
                                )
                                for otherindex in umi_test_dict:
                                    if umi_test_dict[otherindex]:
                                        if keep_highest_umi:
                                            self.drop_contig.append(
                                                h_p[otherindex]
                                            )
                                # refresh
                                data1 = pd.DataFrame(
                                    [data1.loc[keep_hc_contig]]
                                )
                                h_p = list(data1["sequence_id"])
            if len(self.Cell[cell]["VDJ"]["NP"]) > 0:
                data2 = pd.DataFrame(
                    [
                        self.Cell[cell]["VDJ"]["NP"][x]
                        for x in self.Cell[cell]["VDJ"]["NP"]
                        if isinstance(self.Cell[cell]["VDJ"]["NP"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VDJ"]["NP"][x]["sequence_id"]
                        for x in self.Cell[cell]["VDJ"]["NP"]
                        if isinstance(self.Cell[cell]["VDJ"]["NP"][x], dict)
                    ],
                )
                h_np = list(data2["sequence_id"])
                h_umi_np = [
                    int(x) for x in pd.to_numeric(data2["duplicate_count"])
                ]
                if len(h_np) > 1:
                    highest_umi_h = max(h_umi_np)
                    highest_umi_idx = [
                        i for i, j in enumerate(h_umi_np) if j == highest_umi_h
                    ]
                    if len(highest_umi_idx) == 1:
                        keep_index_h = highest_umi_idx[0]
                        keep_hc_contig = h_np[keep_index_h]
                        other_umi_idx = [
                            i
                            for i, j in enumerate(h_umi_np)
                            if j != highest_umi_h
                        ]
                        umi_test_ = [
                            highest_umi_h / x >= umi_foldchange_cutoff
                            for x in h_umi_np[:keep_index_h]
                            + h_umi_np[keep_index_h:]
                        ]
                        umi_test_dict = dict(zip(other_umi_idx, umi_test_))
                        for otherindex in umi_test_dict:
                            if umi_test_dict[otherindex]:
                                self.drop_contig.append(h_np[otherindex])
                        # refresh
                        data2 = pd.DataFrame([data2.loc[keep_hc_contig]])
                        h_np = list(data2["sequence_id"])
                        h_umi_np = [
                            int(x)
                            for x in pd.to_numeric(data2["duplicate_count"])
                        ]
            if len(self.Cell[cell]["VJ"]["P"]) > 0:
                data3 = pd.DataFrame(
                    [
                        self.Cell[cell]["VJ"]["P"][x]
                        for x in self.Cell[cell]["VJ"]["P"]
                        if isinstance(self.Cell[cell]["VJ"]["P"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VJ"]["P"][x]["sequence_id"]
                        for x in self.Cell[cell]["VJ"]["P"]
                        if isinstance(self.Cell[cell]["VJ"]["P"][x], dict)
                    ],
                )
                l_p = list(data3["sequence_id"])
                l_umi_p = [
                    int(x) for x in pd.to_numeric(data3["duplicate_count"])
                ]
                if len(l_p) > 1:
                    if "sequence_alignment" in data3:
                        (
                            data3,
                            l_p,
                            l_umi_p,
                            l_ccall_p,
                            umi_adjust_l,
                            drop_l,
                        ) = check_update_same_seq(data3)
                        if len(umi_adjust_l) > 0:
                            self.umi_adjustment.update(umi_adjust_l)
                        if len(drop_l) > 0:
                            for d_l in drop_l:
                                self.drop_contig.append(d_l)
                    if len(l_p) > 1:
                        highest_umi_l = max(l_umi_p)
                        highest_umi_l_idx = [
                            i
                            for i, j in enumerate(l_umi_p)
                            if j == highest_umi_l
                        ]
                        keep_index_l = highest_umi_l_idx[0]
                        keep_lc_contig = l_p[keep_index_l]
                        umi_test = [
                            highest_umi_l / x < umi_foldchange_cutoff
                            for x in l_umi_p[:keep_index_l]
                            + l_umi_p[keep_index_l:]
                        ]
                        sum_umi = sum(l_umi_p)
                        if len(highest_umi_l_idx) > 1:
                            if filter_extra_vj_chains:
                                self.l_doublet.append(cell)
                        if sum_umi < 4:
                            if filter_extra_vj_chains:
                                self.l_doublet.append(cell)
                        if any(umi_test):
                            if filter_extra_vj_chains:
                                self.l_doublet.append(cell)
                        if len(highest_umi_l_idx) == 1:
                            other_umi_idx_l = [
                                i
                                for i, j in enumerate(l_umi_p)
                                if j != highest_umi_l
                            ]
                            umi_test_l = [
                                highest_umi_l / x >= umi_foldchange_cutoff
                                for x in l_umi_p[:keep_index_l]
                                + l_umi_p[keep_index_l:]
                            ]
                            umi_test_dict_l = dict(
                                zip(other_umi_idx_l, umi_test_l)
                            )
                            for otherindex in umi_test_dict_l:
                                if umi_test_dict_l[otherindex]:
                                    if keep_highest_umi:
                                        self.drop_contig.append(l_p[otherindex])
                                        # refresh
                            data3 = pd.DataFrame([data3.loc[keep_lc_contig]])
                            l_p = list(data3["sequence_id"])
            if len(self.Cell[cell]["VJ"]["NP"]) > 0:
                data4 = pd.DataFrame(
                    [
                        self.Cell[cell]["VJ"]["NP"][x]
                        for x in self.Cell[cell]["VJ"]["NP"]
                        if isinstance(self.Cell[cell]["VJ"]["NP"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VJ"]["NP"][x]["sequence_id"]
                        for x in self.Cell[cell]["VJ"]["NP"]
                        if isinstance(self.Cell[cell]["VJ"]["NP"][x], dict)
                    ],
                )
                l_np = list(data4["sequence_id"])
                l_umi_np = [
                    int(x) for x in pd.to_numeric(data4["duplicate_count"])
                ]
                if len(l_np) > 1:
                    highest_umi_l = max(l_umi_np)
                    highest_umi_l_idx = [
                        i for i, j in enumerate(l_umi_np) if j == highest_umi_l
                    ]
                    keep_index_l = highest_umi_l_idx[0]
                    keep_lc_contig = l_np[keep_index_l]
                    other_umi_idx_l = [
                        i for i, j in enumerate(l_umi_np) if j != highest_umi_l
                    ]
                    umi_test_l = [
                        highest_umi_l / x >= umi_foldchange_cutoff
                        for x in l_umi_np[:keep_index_l]
                        + l_umi_np[keep_index_l:]
                    ]
                    if len(highest_umi_l_idx) == 1:
                        umi_test_dict_l = dict(zip(other_umi_idx_l, umi_test_l))
                        for otherindex in umi_test_dict_l:
                            if umi_test_dict_l[otherindex]:
                                if keep_highest_umi:
                                    self.drop_contig.append(l_np[otherindex])
                        data4 = pd.DataFrame([data4.loc[keep_lc_contig]])
                        l_np = list(data4["sequence_id"])

            if "h_p" not in locals():
                h_p = []
            if "l_p" not in locals():
                l_p = []
            if "h_np" not in locals():
                h_np = []
            if "l_np" not in locals():
                l_np = []

            # marking doublets defined by VJ chains
            if (len(h_p) == 1) & (len(l_p) > 1):
                if filter_extra_vj_chains:
                    self.l_doublet.append(cell)

            # ok check here for bad combinations
            if len(h_p) > 0:
                loci_h = [l_dict[hx] for hx in h_p]
            else:
                loci_h = []
            if len(l_p) > 0:
                loci_l = [l_dict[lx] for lx in l_p]
            else:
                loci_l = []

            loci_ = list(set(loci_h + loci_l))

            if len(loci_) > 0:
                if all(lc in ["IGK", "IGL", "TRG", "TRA"] for lc in loci_):
                    if len(loci_) >= 2:
                        if filter_extra_vj_chains:
                            self.drop_contig.append(l_p)
                elif all(lc in ["TRA", "TRD"] for lc in loci_):
                    if len(loci_) == 2:
                        self.drop_contig.append(l_p)
                        self.drop_contig.append(h_p)
                elif all(lc in ["TRB", "TRG"] for lc in loci_):
                    if len(loci_) == 2:
                        self.drop_contig.append(l_p)
                        self.drop_contig.append(h_p)
                elif all(lc in ["TRB", "IGK", "IGL"] for lc in loci_):
                    if len(loci_) >= 2:
                        self.drop_contig.append(l_p)
                        self.drop_contig.append(h_p)
                elif all(lc in ["IGH", "TRA", "TRG"] for lc in loci_):
                    if len(loci_) >= 2:
                        self.drop_contig.append(l_p)
                        self.drop_contig.append(h_p)

            # marking poor bcr quality, defined as those with only VJ chains, those
            # that were have conflicting assignment of locus and V(D)J v-, d-, j- and c- calls,
            # and also those that are missing j calls (to catch non-productive).
            if len(h_p) < 1:
                if filter_poorqualitycontig:
                    self.poor_qual.append(cell)
                self.drop_contig.append(l_p)
            if len(h_p) == 1:
                v = v_dict[h_p[0]]
                j = j_dict[h_p[0]]
                d = d_dict[h_p[0]]
                c = c_dict[h_p[0]]
                if present(v):
                    if not re.search("IGH|TR[BD]|TRAV.*/DV", v):
                        if filter_poorqualitycontig:
                            self.poor_qual.append(cell)
                        self.drop_contig.append(l_p)
                        self.drop_contig.append(h_p)
                if present(d):
                    if not re.search("IGH|TR[BD]", d):
                        if filter_poorqualitycontig:
                            self.poor_qual.append(cell)
                        self.drop_contig.append(l_p)
                        self.drop_contig.append(h_p)
                if present(j):
                    if not re.search("IGH|TR[BD]", j):
                        if filter_poorqualitycontig:
                            self.poor_qual.append(cell)
                        self.drop_contig.append(l_p)
                        self.drop_contig.append(h_p)
                if present(c):
                    if not re.search("IGH|TR[BD]", c):
                        if filter_poorqualitycontig:
                            self.poor_qual.append(cell)
                        self.drop_contig.append(l_p)
                        self.drop_contig.append(h_p)

                if present(j):
                    if present(v):
                        if not_same_call(v, j, "IGH"):
                            if filter_poorqualitycontig:
                                self.poor_qual.append(cell)
                            self.drop_contig.append(l_p)
                            self.drop_contig.append(h_p)
                        elif not_same_call(v, j, "TRB"):
                            if filter_poorqualitycontig:
                                self.poor_qual.append(cell)
                            self.drop_contig.append(l_p)
                            self.drop_contig.append(h_p)
                        elif not_same_call(v, j, "TRD"):
                            if not re.search("TRAV.*/DV", v):
                                if filter_poorqualitycontig:
                                    self.poor_qual.append(cell)
                                self.drop_contig.append(l_p)
                                self.drop_contig.append(h_p)

                    if present(d):
                        if not_same_call(d, j, "IGH"):
                            if filter_poorqualitycontig:
                                self.poor_qual.append(cell)
                            self.drop_contig.append(l_p)
                            self.drop_contig.append(h_p)
                        elif not_same_call(d, j, "TRB"):
                            if filter_poorqualitycontig:
                                self.poor_qual.append(cell)
                            self.drop_contig.append(l_p)
                            self.drop_contig.append(h_p)
                        elif not_same_call(d, j, "TRD"):
                            if filter_poorqualitycontig:
                                self.poor_qual.append(cell)
                            self.drop_contig.append(l_p)
                            self.drop_contig.append(h_p)
                else:
                    if filter_poorqualitycontig:
                        self.poor_qual.append(cell)
                    self.drop_contig.append(l_p)
                    self.drop_contig.append(h_p)

            if len(h_p) > 1:
                for hx in h_p:
                    v = v_dict[hx]
                    d = d_dict[hx]
                    j = j_dict[hx]
                    c = c_dict[hx]
                    if present(v):
                        if not re.search("IGH|TR[BD]|TRAV.*/DV", v):
                            if filter_poorqualitycontig:
                                self.poor_qual.append(cell)
                            self.drop_contig.append(hx)
                    if present(d):
                        if not re.search("IGH|TR[BD]", d):
                            if filter_poorqualitycontig:
                                self.poor_qual.append(cell)
                            self.drop_contig.append(hx)
                    if present(j):
                        if not re.search("IGH|TR[BD]", j):
                            if filter_poorqualitycontig:
                                self.poor_qual.append(cell)
                            self.drop_contig.append(hx)
                    if present(c):
                        if not re.search("IGH|TR[BD]", c):
                            if filter_poorqualitycontig:
                                self.poor_qual.append(cell)
                            self.drop_contig.append(hx)
                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGH"):
                                if filter_poorqualitycontig:
                                    self.poor_qual.append(cell)
                                self.drop_contig.append(hx)
                            elif not_same_call(v, j, "TRB"):
                                if filter_poorqualitycontig:
                                    self.poor_qual.append(cell)
                                self.drop_contig.append(hx)
                            elif not_same_call(v, j, "TRD"):
                                if not re.search("TRAV.*/DV", v):
                                    if filter_poorqualitycontig:
                                        self.poor_qual.append(cell)
                                    self.drop_contig.append(hx)
                        if present(d):
                            if not_same_call(d, j, "IGH"):
                                if filter_poorqualitycontig:
                                    self.poor_qual.append(cell)
                                self.drop_contig.append(hx)
                            elif not_same_call(d, j, "TRB"):
                                if filter_poorqualitycontig:
                                    self.poor_qual.append(cell)
                                self.drop_contig.append(hx)
                            elif not_same_call(d, j, "TRD"):
                                if filter_poorqualitycontig:
                                    self.poor_qual.append(cell)
                                self.drop_contig.append(hx)
                    else:
                        if filter_poorqualitycontig:
                            self.poor_qual.append(cell)
                        self.drop_contig.append(hx)

            if len(h_np) > 0:
                for hx in h_np:
                    v = v_dict[hx]
                    d = d_dict[hx]
                    j = j_dict[hx]
                    c = c_dict[hx]
                    if present(v):
                        if not re.search("IGH|TR[BD]|TRAV.*/DV", v):
                            self.drop_contig.append(hx)
                    if present(d):
                        if not re.search("IGH|TR[BD]", d):
                            self.drop_contig.append(hx)
                    if present(j):
                        if not re.search("IGH|TR[BD]", j):
                            self.drop_contig.append(hx)
                    if present(c):
                        if not re.search("IGH|TR[BD]", c):
                            self.drop_contig.append(hx)

                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGH"):
                                self.drop_contig.append(hx)
                            elif not_same_call(v, j, "TRB"):
                                self.drop_contig.append(hx)
                            elif not_same_call(v, j, "TRD"):
                                if not re.search("TRAV.*/DV", v):
                                    self.drop_contig.append(hx)
                        if present(d):
                            if not_same_call(d, j, "IGH"):
                                self.drop_contig.append(hx)
                            elif not_same_call(d, j, "TRB"):
                                self.drop_contig.append(hx)
                            elif not_same_call(d, j, "TRD"):
                                self.drop_contig.append(hx)
                    else:
                        self.drop_contig.append(hx)
            if len(l_p) > 0:
                for lx in l_p:
                    v = v_dict[lx]
                    j = j_dict[lx]
                    c = c_dict[lx]
                    if present(v):
                        if re.search("IGH|TRB", v):
                            if filter_poorqualitycontig:
                                self.poor_qual.append(cell)
                            self.drop_contig.append(lx)
                    if present(j):
                        if re.search("IGH|TRB", j):
                            if filter_poorqualitycontig:
                                self.poor_qual.append(cell)
                            self.drop_contig.append(lx)
                    if present(c):
                        if re.search("IGH|TRB", c):
                            if filter_poorqualitycontig:
                                self.poor_qual.append(cell)
                            self.drop_contig.append(lx)

                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGK"):
                                if filter_poorqualitycontig:
                                    self.poor_qual.append(cell)
                                self.drop_contig.append(lx)
                            elif not_same_call(v, j, "IGL"):
                                if filter_poorqualitycontig:
                                    self.poor_qual.append(cell)
                                self.drop_contig.append(lx)
                            elif not_same_call(v, j, "TRA"):
                                if not re.search("TR[AD]", v):
                                    if not re.search("TRA", j):
                                        if filter_poorqualitycontig:
                                            self.poor_qual.append(cell)
                                        self.drop_contig.append(lx)
                            elif not_same_call(v, j, "TRG"):
                                if filter_poorqualitycontig:
                                    self.poor_qual.append(cell)
                                self.drop_contig.append(lx)
                    else:
                        if filter_poorqualitycontig:
                            self.poor_qual.append(cell)
                        self.drop_contig.append(lx)

            if len(l_np) > 0:
                for lx in l_np:
                    v = v_dict[lx]
                    j = j_dict[lx]
                    c = c_dict[lx]
                    if present(v):
                        if re.search("IGH|TRB", v):
                            self.drop_contig.append(lx)
                    if present(j):
                        if re.search("IGH|TRB", j):
                            self.drop_contig.append(lx)
                    if present(c):
                        if re.search("IGH|TRB", c):
                            self.drop_contig.append(lx)

                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGK"):
                                self.drop_contig.append(lx)
                            elif not_same_call(v, j, "IGL"):
                                self.drop_contig.append(lx)
                            elif not_same_call(v, j, "TRA"):
                                if not re.search("TR[AD]", v):
                                    if not re.search("TRA", j):
                                        self.drop_contig.append(lx)
                            elif not_same_call(v, j, "TRG"):
                                self.drop_contig.append(lx)
                    else:
                        self.drop_contig.append(lx)


class FilterContigsLite:
    """
    `FilterContigsLite` class object.

    Main class object to run filter_contigs, lite mode.

    Attributes
    ----------
    Cell : dandelion.utilities._utilities.Tree
        nested dictionary of cells.
    drop_contig : List[str]
        list of `sequence_id`s to drop.
    h_doublet : List[str]
        list of `sequence_id`s that are VDJ 'multiplets'.
    l_doublet : List[str]
        list of `sequence_id`s that are VJ 'multiplets'.
    poor_qual : List[str]
        list of `sequence_id`s that are VDJ 'multiplets'.
    umi_adjustment : Dict[str, int]
        dictionary of `sequence_id`s with adjusted umi value.
    """

    def __init__(self, data: pd.DataFrame, verbose: bool):
        """Init method for FilterContigsLite.

        Parameters
        ----------
        data : pd.DataFrame
            AIRR data frame in Dandelion.data.
        verbose : bool
            whether or not to print progress.
        """
        self.Cell = Tree()
        self.poor_qual = []
        self.h_doublet = []
        self.l_doublet = []
        self.drop_contig = []
        self.umi_adjustment = {}
        if "v_call_genotyped" in data.columns:
            v_dict = dict(zip(data["sequence_id"], data["v_call_genotyped"]))
        else:
            v_dict = dict(zip(data["sequence_id"], data["v_call"]))
        d_dict = dict(zip(data["sequence_id"], data["d_call"]))
        j_dict = dict(zip(data["sequence_id"], data["j_call"]))
        c_dict = dict(zip(data["sequence_id"], data["c_call"]))
        for contig, row in tqdm(
            data.iterrows(),
            desc="Preparing data",
        ):
            cell = row["cell_id"]
            if row["locus"] in HEAVYLONG:
                if row["productive"] in TRUES:
                    self.Cell[cell]["VDJ"]["P"][contig].update(row)
                elif row["productive"] in FALSES:
                    self.Cell[cell]["VDJ"]["NP"][contig].update(row)
            elif row["locus"] in LIGHTSHORT:
                if row["productive"] in TRUES:
                    self.Cell[cell]["VJ"]["P"][contig].update(row)
                elif row["productive"] in FALSES:
                    self.Cell[cell]["VJ"]["NP"][contig].update(row)
        for cell in tqdm(
            self.Cell,
            desc="Scanning for poor quality/ambiguous contigs",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            disable=not verbose,
        ):
            if len(self.Cell[cell]["VDJ"]["P"]) > 0:
                data1 = pd.DataFrame(
                    [
                        self.Cell[cell]["VDJ"]["P"][x]
                        for x in self.Cell[cell]["VDJ"]["P"]
                        if isinstance(self.Cell[cell]["VDJ"]["P"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VDJ"]["P"][x]["sequence_id"]
                        for x in self.Cell[cell]["VDJ"]["P"]
                        if isinstance(self.Cell[cell]["VDJ"]["P"][x], dict)
                    ],
                )
                h_p = list(data1["sequence_id"])
                h_umi_p = [
                    int(x) for x in pd.to_numeric(data1["duplicate_count"])
                ]
                h_ccall_p = list(data1["c_call"])
                if len(h_p) > 1:
                    if "sequence_alignment" in data1:
                        h_seq_p = list(data1["sequence_alignment"])
                        if len(set(h_seq_p)) == 1:
                            if len(set(h_ccall_p)) == 1:
                                highest_umi_h = max(h_umi_p)
                                highest_umi_h_idx = [
                                    i
                                    for i, j in enumerate(h_umi_p)
                                    if j == highest_umi_h
                                ]
                                keep_index_h = highest_umi_h_idx[0]
                                self.drop_contig.append(
                                    h_p[:keep_index_h] + h_p[keep_index_h:]
                                )
                                keep_hc_contig = h_p[keep_index_h]
                                data1[keep_hc_contig, "duplicate_count"] = int(
                                    np.sum(
                                        h_umi_p[:keep_index_h]
                                        + h_umi_p[keep_index_h:]
                                    )
                                )
                                self.umi_adjustment.update(
                                    {
                                        keep_hc_contig: int(
                                            np.sum(
                                                h_umi_p[:keep_index_h]
                                                + h_umi_p[keep_index_h:]
                                            )
                                        )
                                    }
                                )
                                # refresh
                                data1 = pd.DataFrame(
                                    [data1.loc[keep_hc_contig]]
                                )
                                h_p = list(data1["sequence_id"])
                                h_umi_p = [
                                    int(x)
                                    for x in pd.to_numeric(
                                        data1["duplicate_count"]
                                    )
                                ]
            if len(self.Cell[cell]["VDJ"]["NP"]) > 0:
                data2 = pd.DataFrame(
                    [
                        self.Cell[cell]["VDJ"]["NP"][x]
                        for x in self.Cell[cell]["VDJ"]["NP"]
                        if isinstance(self.Cell[cell]["VDJ"]["NP"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VDJ"]["NP"][x]["sequence_id"]
                        for x in self.Cell[cell]["VDJ"]["NP"]
                        if isinstance(self.Cell[cell]["VDJ"]["NP"][x], dict)
                    ],
                )
                h_np = list(data2["sequence_id"])
                h_umi_np = [
                    int(x) for x in pd.to_numeric(data2["duplicate_count"])
                ]
            if len(self.Cell[cell]["VJ"]["P"]) > 0:
                data3 = pd.DataFrame(
                    [
                        self.Cell[cell]["VJ"]["P"][x]
                        for x in self.Cell[cell]["VJ"]["P"]
                        if isinstance(self.Cell[cell]["VJ"]["P"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VJ"]["P"][x]["sequence_id"]
                        for x in self.Cell[cell]["VJ"]["P"]
                        if isinstance(self.Cell[cell]["VJ"]["P"][x], dict)
                    ],
                )
                l_p = list(data3["sequence_id"])
                l_umi_p = [
                    int(x) for x in pd.to_numeric(data3["duplicate_count"])
                ]
                if len(l_p) > 1:
                    if "sequence_alignment" in data3:
                        l_seq_p = list(data3["sequence_alignment"])
                        if len(list(set(l_seq_p))) == 1:
                            highest_umi_l = max(l_umi_p)
                            highest_umi_l_idx = [
                                i
                                for i, j in enumerate(l_umi_p)
                                if j == highest_umi_l
                            ]
                            keep_index_l = highest_umi_l_idx[0]
                            self.drop_contig.append(
                                l_p[:keep_index_l] + l_p[keep_index_l:]
                            )
                            keep_lc_contig = l_p[keep_index_l]
                            data3.at[keep_lc_contig, "duplicate_count"] = int(
                                np.sum(
                                    l_umi_p[:keep_index_l]
                                    + l_umi_p[keep_index_l:]
                                )
                            )
                            self.umi_adjustment.update(
                                {
                                    keep_lc_contig: int(
                                        np.sum(
                                            l_umi_p[:keep_index_l]
                                            + l_umi_p[keep_index_l:]
                                        )
                                    )
                                }
                            )
                            # refresh
                            data3 = pd.DataFrame([data3.loc[keep_lc_contig]])
                            l_p = list(data3["sequence_id"])
                            l_umi_p = [
                                int(x)
                                for x in pd.to_numeric(data3["duplicate_count"])
                            ]
            if len(self.Cell[cell]["VJ"]["NP"]) > 0:
                data4 = pd.DataFrame(
                    [
                        self.Cell[cell]["VJ"]["NP"][x]
                        for x in self.Cell[cell]["VJ"]["NP"]
                        if isinstance(self.Cell[cell]["VJ"]["NP"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VJ"]["NP"][x]["sequence_id"]
                        for x in self.Cell[cell]["VJ"]["NP"]
                        if isinstance(self.Cell[cell]["VJ"]["NP"][x], dict)
                    ],
                )
                l_np = list(data4["sequence_id"])
                l_umi_np = [
                    int(x) for x in pd.to_numeric(data4["duplicate_count"])
                ]

            if "h_p" not in locals():
                h_p = []
            if "l_p" not in locals():
                l_p = []
            if "h_np" not in locals():
                h_np = []
            if "l_np" not in locals():
                l_np = []

            if len(h_p) > 0:
                for hx in h_p:
                    v = v_dict[hx]
                    d = d_dict[hx]
                    j = j_dict[hx]
                    c = c_dict[hx]
                    if present(v):
                        if not re.search("IGH|TR[BD]|TRAV.*/DV", v):
                            self.drop_contig.append(hx)
                    if present(d):
                        if not re.search("IGH|TR[BD]", d):
                            self.drop_contig.append(hx)
                    if present(j):
                        if not re.search("IGH|TR[BD]", j):
                            self.drop_contig.append(hx)
                    if present(c):
                        if not re.search("IGH|TR[BD]", c):
                            self.drop_contig.append(hx)
                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGH"):
                                self.drop_contig.append(hx)
                            elif not_same_call(v, j, "TRB"):
                                self.drop_contig.append(hx)
                            elif not_same_call(v, j, "TRD"):
                                if not re.search("TRAV.*/DV", v):
                                    self.drop_contig.append(hx)
                        if present(d):
                            if not_same_call(d, j, "IGH"):
                                self.drop_contig.append(hx)
                            elif not_same_call(d, j, "TRB"):
                                self.drop_contig.append(hx)
                            elif not_same_call(d, j, "TRD"):
                                self.drop_contig.append(hx)
                    else:
                        self.drop_contig.append(hx)

            if len(h_np) > 0:
                for hx in h_np:
                    v = v_dict[hx]
                    d = d_dict[hx]
                    j = j_dict[hx]
                    c = c_dict[hx]
                    if present(v):
                        if not re.search("IGH|TR[BD]|TRAV.*/DV", v):
                            self.drop_contig.append(hx)
                    if present(d):
                        if not re.search("IGH|TR[BD]", d):
                            self.drop_contig.append(hx)
                    if present(j):
                        if not re.search("IGH|TR[BD]", j):
                            self.drop_contig.append(hx)
                    if present(c):
                        if not re.search("IGH|TR[BD]", c):
                            self.drop_contig.append(hx)

                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGH"):
                                self.drop_contig.append(hx)
                            elif not_same_call(v, j, "TRB"):
                                self.drop_contig.append(hx)
                            elif not_same_call(v, j, "TRD"):
                                if not re.search("TRAV.*/DV", v):
                                    self.drop_contig.append(hx)
                        if present(d):
                            if not_same_call(d, j, "IGH"):
                                self.drop_contig.append(hx)
                            elif not_same_call(d, j, "TRB"):
                                self.drop_contig.append(hx)
                            elif not_same_call(d, j, "TRD"):
                                self.drop_contig.append(hx)
                    else:
                        self.drop_contig.append(hx)
            if len(l_p) > 0:
                for lx in l_p:
                    v = v_dict[lx]
                    j = j_dict[lx]
                    c = c_dict[lx]
                    if present(v):
                        if re.search("IGH|TRB", v):
                            self.drop_contig.append(lx)
                    if present(j):
                        if re.search("IGH|TRB", j):
                            self.drop_contig.append(lx)
                    if present(c):
                        if re.search("IGH|TRB", c):
                            self.drop_contig.append(lx)

                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGK"):
                                self.drop_contig.append(lx)
                            elif not_same_call(v, j, "IGL"):
                                self.drop_contig.append(lx)
                            elif not_same_call(v, j, "TRA"):
                                if not re.search("TR[AD]", v):
                                    if not re.search("TRA", j):
                                        self.drop_contig.append(lx)
                            elif not_same_call(v, j, "TRG"):
                                self.drop_contig.append(lx)
                    else:
                        self.drop_contig.append(lx)

            if len(l_np) > 0:
                for lx in l_np:
                    v = v_dict[lx]
                    j = j_dict[lx]
                    c = c_dict[lx]
                    if present(v):
                        if re.search("IGH|TRB", v):
                            self.drop_contig.append(lx)
                    if present(j):
                        if re.search("IGH|TRB", j):
                            self.drop_contig.append(lx)
                    if present(c):
                        if re.search("IGH|TRB", c):
                            self.drop_contig.append(lx)

                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGK"):
                                self.drop_contig.append(lx)
                            elif not_same_call(v, j, "IGL"):
                                self.drop_contig.append(lx)
                            elif not_same_call(v, j, "TRA"):
                                if not re.search("TR[AD]", v):
                                    if not re.search("TRA", j):
                                        self.drop_contig.append(lx)
                            elif not_same_call(v, j, "TRG"):
                                self.drop_contig.append(lx)
                    else:
                        self.drop_contig.append(lx)


def run_igblastn(
    fasta: Union[str, Path],
    igblast_db: Optional[Union[str, Path]] = None,
    org: Literal["human", "mouse"] = "human",
    loci: Literal["ig", "tr"] = "ig",
    evalue: float = 1e-4,
    min_d_match: int = 9,
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
    evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets.
    min_d_match : int, optional
        minimum D nucleotide match.
    additional_args: List[str], optional
        additional arguments to pass to `igblastn`.
    """
    env, igdb, fasta = set_igblast_env(igblast_db=igblast_db, input_file=fasta)
    outfolder = fasta.parent / "tmp"
    outfolder.mkdir(parents=True, exist_ok=True)
    informat_dict = {"blast": "_igblast.fmt7", "airr": "_igblast.tsv"}

    loci_type = {"ig": "Ig", "tr": "TCR"}
    outformat = {"blast": "7 std qseq sseq btop", "airr": "19"}

    dbpath = igdb / "database"
    imgt_org_loci = "imgt_" + org + "_" + loci + "_"
    vpath = dbpath / (imgt_org_loci + "v")
    dpath = dbpath / (imgt_org_loci + "d")
    jpath = dbpath / (imgt_org_loci + "j")
    cpath = dbpath / (imgt_org_loci + "c")
    auxpath = igdb / "optional_file" / (org + "_gl.aux")

    for fileformat in ["blast", "airr"]:
        outfile = str(fasta.stem + informat_dict[fileformat])
        if loci == "tr":
            cmd = [
                "igblastn",
                "-germline_db_V",
                str(vpath),
                "-germline_db_D",
                str(dpath),
                "-germline_db_J",
                str(jpath),
                "-auxiliary_data",
                str(auxpath),
                "-domain_system",
                "imgt",
                "-ig_seqtype",
                loci_type[loci],
                "-organism",
                org,
                "-outfmt",
                outformat[fileformat],
                "-query",
                str(fasta),
                "-out",
                str(outfolder / outfile),
                "-evalue",
                str(evalue),
                "-min_D_match",
                str(min_d_match),
                "-D_penalty",
                str(-4),
                "-c_region_db",
                str(cpath),
            ]
        else:
            cmd = [
                "igblastn",
                "-germline_db_V",
                str(vpath),
                "-germline_db_D",
                str(dpath),
                "-germline_db_J",
                str(jpath),
                "-auxiliary_data",
                str(auxpath),
                "-domain_system",
                "imgt",
                "-ig_seqtype",
                loci_type[loci],
                "-organism",
                org,
                "-outfmt",
                outformat[fileformat],
                "-query",
                str(fasta),
                "-out",
                str(outfolder / outfile),
                "-evalue",
                str(evalue),
                "-min_D_match",
                str(min_d_match),
                "-c_region_db",
                str(cpath),
            ]
        cmd = cmd + additional_args
        logg.info("Running command: %s\n" % (" ".join(cmd)))
        run(cmd, env=env)  # logs are printed to terminal


def assign_DJ(
    fasta: Union[str, Path],
    org: Literal["human", "mouse"] = "human",
    loci: Literal["ig", "tr"] = "tr",
    call: Literal["d", "j"] = "j",
    database: Optional[str] = None,
    evalue: float = 1e-4,
    max_hsps: int = 10,
    dust: Optional[Union[Literal["yes", "no"], str]] = None,
    word_size: Optional[int] = None,
    outfmt: str = (
        "6 qseqid sseqid pident length mismatch gapopen "
        + "qstart qend sstart send evalue bitscore qseq sseq"
    ),
    filename_prefix: Optional[str] = None,
    overwrite: bool = False,
    additional_args: List[str] = [],
):
    """
    Annotate contigs with constant region call using blastn.

    Parameters
    ----------
    fasta : Union[str, Path]
        path to fasta file.
    org : Literal["human", "mouse"], optional
        organism of reference folder.
    loci : Literal["ig", "tr"], optional
        locus. 'ig' or 'tr',
    call : Literal["d", "j"], optional
        Either 'd' of 'j' gene.
    database : Optional[str], optional
        path to database.
        Defaults to `IGDATA` environmental variable if v/d/j_call.
        Defaults to `BLASTDB` environmental variable if c_call.
    evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets.
    max_hsps : int, optional
        Maximum number of HSPs (alignments) to keep for any single query-subject pair.
        The HSPs shown will be the best as judged by expect value. This number should
        be an integer that is one or greater. Setting it to one will show only the best
        HSP for every query-subject pair. Only affects the output file in the tmp folder.
    dust : Optional[Union[Literal["yes", "no"], str]], optional
        dustmasker options. Filter query sequence with DUST
        Format: 'yes', or 'no' to disable. Accepts str.
        If None, defaults to `20 64 1`.
    word_size : Optional[int], optional
        Word size for wordfinder algorithm (length of best perfect match).
        Must be >=4. `None` defaults to 4.
    outfmt : str, optional
        specification of output format for blast.
    filename_prefix : Optional[str], optional
        prefix of file name preceding '_contig'. `None` defaults to 'filtered'.
    overwrite : bool, optional
        whether or not to overwrite the assignments.
    additional_args: List[str], optional
        additional arguments to pass to `blastn`.
    """
    # main function from here
    file_path, passfile, failfile = return_pass_fail_filepaths(
        fasta, filename_prefix=filename_prefix
    )

    # run blast
    blast_out = run_blastn(
        fasta=file_path,
        database=database,
        org=org,
        loci=loci,
        call=call,
        max_hsps=max_hsps,
        evalue=evalue,
        outfmt=outfmt,
        dust=dust,
        word_size=word_size,
        additional_args=additional_args,
    )

    transfer_assignment(
        passfile=passfile,
        failfile=failfile,
        blast_result=blast_out.drop_duplicates(
            subset="sequence_id", keep="first"
        ),
        eval_threshold=evalue,
        call=call,
        overwrite=overwrite,
    )


def run_blastn(
    fasta: Union[str, Path],
    database: Optional[str],
    org: Literal["human", "mouse"] = "human",
    loci: Literal["ig", "tr"] = "ig",
    call: Literal["v", "d", "j", "c"] = "c",
    max_hsps: int = 10,
    evalue: float = 1e-4,
    outfmt: str = (
        "6 qseqid sseqid pident length mismatch gapopen "
        + "qstart qend sstart send evalue bitscore qseq sseq"
    ),
    dust: Optional[Union[Literal["yes", "no"], str]] = None,
    word_size: Optional[int] = None,
    additional_args: List[str] = [],
) -> pd.DataFrame:
    """
    Annotate contigs using blastn.

    Parameters
    ----------
    fasta : Union[str, Path]
        path to fasta file.
    database : Optional[str]
        path to database.
        Defaults to `IGDATA` environmental variable if v/d/j_call.
        Defaults to `BLASTDB` environmental variable if c_call.
    org : Literal["human", "mouse"], optional
        organism of reference folder.
    loci : Literal["ig", "tr"], optional
        locus. 'ig' or 'tr',
    call : Literal["v", "d", "j", "c"], optional
        Either 'v', 'd', 'j' or 'c' gene.
    max_hsps : int, optional
        Maximum number of HSPs (alignments) to keep for any single query-subject pair.
        The HSPs shown will be the best as judged by expect value. This number should
        be an integer that is one or greater. Setting it to one will show only the best
        HSP for every query-subject pair. Only affects the output file in the tmp folder.
    evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets.
    outfmt : str, optional
        blastn output format.
    dust : Optional[Union[Literal["yes", "no"], str]], optional
        dustmasker options. Filter query sequence with DUST
        Format: 'yes', or 'no' to disable. Accepts str.
        If None, defaults to `20 64 1`.
    word_size : Optional[int], optional
        Word size for wordfinder algorithm (length of best perfect match).
        Must be >=4. `None` defaults to 4.
    additional_args: List[str], optional
        additional arguments to pass to `blastn`.

    Returns
    -------
    pd.DataFrame
        reannotated information after blastn.Annotate contigs using blastn.

    Parameters
    ----------
    fasta : Union[str, Path]
        path to fasta file.
    database : Optional[str]
        path to database.
        Defaults to `IGDATA` environmental variable if v/d/j_call.
        Defaults to `BLASTDB` environmental variable if c_call.
    org : Literal["human", "mouse"], optional
        organism of reference folder.
    loci : Literal["ig", "tr"], optional
        locus. 'ig' or 'tr',
    call : Literal["v", "d", "j", "c"], optional
        Either 'v', 'd', 'j' or 'c' gene.
    max_hsps : int, optional
        Maximum number of HSPs (alignments) to keep for any single query-subject pair.
        The HSPs shown will be the best as judged by expect value. This number should
        be an integer that is one or greater. Setting it to one will show only the best
        HSP for every query-subject pair. Only affects the output file in the tmp folder.
    evalue : float, optional
        This is the statistical significance threshold for reporting matches
        against database sequences. Lower EXPECT thresholds are more stringent
        and report only high similarity matches. Choose higher EXPECT value
        (for example 1 or more) if you expect a low identity between your query
        sequence and the targets.
    outfmt : str, optional
        blastn output format.
    dust : Optional[Union[Literal["yes", "no"], str]], optional
        dustmasker options. Filter query sequence with DUST
        Format: 'yes', or 'no' to disable. Accepts str.
        If None, defaults to `20 64 1`.
    word_size : Optional[int], optional
        Word size for wordfinder algorithm (length of best perfect match).
        Must be >=4. `None` defaults to 4.
    additional_args: List[str], optional
        additional arguments to pass to `blastn`.

    Returns
    -------
    pd.DataFrame
        reannotated information after blastn.
    """
    if call != "c":
        env, bdb, fasta = set_igblast_env(igblast_db=database, input_file=fasta)
        bdb = bdb / "database" / ("imgt_" + org + "_" + loci + "_" + call)
    else:
        env, bdb, fasta = set_blast_env(blast_db=database, input_file=fasta)
        if database is None:
            bdb = bdb / org / (org + "_BCR_C.fasta")
        else:
            if not bdb.stem.endswith("_" + loci + "_" + call):
                bdb = (
                    bdb / "database" / ("imgt_" + org + "_" + loci + "_" + call)
                )
    cmd = [
        "blastn",
        "-db",
        str(bdb),
        "-evalue",
        str(evalue),
        "-max_hsps",
        str(max_hsps),
        "-outfmt",
        outfmt,
        "-query",
        str(fasta),
    ]
    if dust is not None:
        cmd = cmd + ["-dust", str(dust)]
    if word_size is not None:
        cmd = cmd + ["-word_size", str(word_size)]
    cmd = cmd + additional_args
    blast_out = fasta.parent / "tmp" / (fasta.stem + "_" + call + "_blast.tsv")
    logg.info("Running command: %s\n" % (" ".join(cmd)))
    with open(blast_out, "w") as out:
        run(cmd, stdout=out, env=env)
    try:
        dat = pd.read_csv(blast_out, sep="\t", header=None)
        dat.columns = [
            "sequence_id",
            call + "_call",
            call + "_identity",
            call + "_alignment_length",
            call + "_number_of_mismatches",
            call + "_number_of_gap_openings",
            call + "_sequence_start",
            call + "_sequence_end",
            call + "_germline_start",
            call + "_germline_end",
            call + "_support",
            call + "_score",
            call + "_sequence_alignment",
            call + "_germline_alignment",
        ]
    except pd.errors.EmptyDataError:
        dat = pd.DataFrame(
            columns=[
                "sequence_id",
                call + "_call",
                call + "_identity",
                call + "_alignment_length",
                call + "_number_of_mismatches",
                call + "_number_of_gap_openings",
                call + "_sequence_start",
                call + "_sequence_end",
                call + "_germline_start",
                call + "_germline_end",
                call + "_support",
                call + "_score",
                call + "_sequence_alignment",
                call + "_germline_alignment",
            ]
        )
    write_blastn(data=dat, save=blast_out)
    dat = load_data(dat)
    return dat


def transfer_assignment(
    passfile: str,
    failfile: str,
    blast_result: pd.DataFrame,
    eval_threshold: float,
    call: Literal["v", "d", "j", "c"] = "c",
    overwrite: bool = False,
):
    """Update gene calls with blastn results.

    Parameters
    ----------
    passfile : str
        path to db-pass.tsv file.
    failfile : str
        path to db-fail.tsv file.
    blast_result : pd.DataFrame
        path to blastn results file.
    eval_threshold : float
        e-value threshold to filter.
    call : Literal["v", "d", "j", "c"], optional
        which gene call.
    overwrite : bool, optional
        whether or not to overwrite.
    """
    if os.path.isfile(passfile):
        db_pass = load_data(passfile)
    else:
        db_pass = None
    if os.path.isfile(failfile):
        db_fail = load_data(failfile)
        # should be pretty safe to fill this in
        db_fail["vj_in_frame"].fillna(value="F", inplace=True)
        db_fail["productive"].fillna(value="F", inplace=True)
        db_fail["c_call"].fillna(value="", inplace=True)
        db_fail["v_call"].fillna(value="", inplace=True)
        db_fail["d_call"].fillna(value="", inplace=True)
        db_fail["j_call"].fillna(value="", inplace=True)
        db_fail["locus"].fillna(value="", inplace=True)
        for i, r in db_fail.iterrows():
            if not present(r.locus):
                calls = list(
                    set(
                        [r.v_call[:3], r.d_call[:3], r.j_call[:3], r.c_call[:3]]
                    )
                )
                locus = "".join([c for c in calls if present(c)])
                if len(locus) == 3:
                    db_fail.at[i, "locus"] = locus
    else:
        db_fail = None
    if blast_result.shape[0] < 1:
        blast_result = None

    if blast_result is not None:
        if db_pass is not None:
            if call + "_support" in db_pass:
                db_pass_evalues = dict(db_pass[call + "_support"])
            if call + "_score" in db_pass:
                db_pass_scores = dict(db_pass[call + "_score"])
            db_pass[call + "_call"].fillna(value="", inplace=True)
            db_pass_call = dict(db_pass[call + "_call"])
            if call + "_support" in db_pass:
                db_pass[call + "_support_igblastn"] = pd.Series(db_pass_evalues)
            if call + "_score" in db_pass:
                db_pass[call + "_score_igblastn"] = pd.Series(db_pass_scores)
            db_pass[call + "_call_igblastn"] = pd.Series(db_pass_call)
            db_pass[call + "_call_igblastn"].fillna(value="", inplace=True)
            for col in blast_result:
                if col not in ["sequence_id", "cell_id"]:
                    db_pass[col + "_blastn"] = pd.Series(blast_result[col])
                    if col in [
                        call + "_call",
                        call + "_sequence_alignment",
                        call + "_germline_alignment",
                    ]:
                        db_pass[col + "_blastn"].fillna(value="", inplace=True)
            db_pass[call + "_source"] = ""
            if overwrite:
                for i in db_pass["sequence_id"]:
                    vend = db_pass.loc[i, "v_sequence_end"]
                    if not present(vend):
                        vend_ = 0
                    else:
                        vend_ = vend
                    jstart = db_pass.loc[i, "j_sequence_start"]
                    if not present(jstart):
                        jstart_ = 1000
                    else:
                        jstart_ = jstart
                    callstart = db_pass.loc[i, call + "_sequence_start_blastn"]
                    callend = db_pass.loc[i, call + "_sequence_end_blastn"]
                    if (callstart >= vend_) and (callend <= jstart_):
                        if call + "_support_igblastn" in db_pass:
                            eval1 = db_pass.loc[i, call + "_support_igblastn"]
                        else:
                            eval1 = 1
                        eval2 = db_pass.loc[i, call + "_support_blastn"]
                        if (
                            db_pass.loc[i, call + "_call_igblastn"]
                            != db_pass.loc[i, call + "_call_blastn"]
                        ):
                            if call + "_call_10x" in db_pass:
                                if (
                                    re.sub(
                                        "[*][0-9][0-9]",
                                        "",
                                        db_pass.loc[i, call + "_call_blastn"],
                                    )
                                    != db_pass.loc[i, call + "_call_10x"]
                                ):
                                    if present(eval1):
                                        if eval1 > eval2:
                                            db_pass.at[
                                                i, call + "_call"
                                            ] = db_pass.at[
                                                i, call + "_call_blastn"
                                            ]
                                            db_pass.at[
                                                i, call + "_sequence_start"
                                            ] = db_pass.at[
                                                i,
                                                call + "_sequence_start_blastn",
                                            ]
                                            db_pass.at[
                                                i, call + "_sequence_end"
                                            ] = db_pass.at[
                                                i, call + "_sequence_end_blastn"
                                            ]
                                            db_pass.at[
                                                i, call + "_germline_start"
                                            ] = db_pass.at[
                                                i,
                                                call + "_germline_start_blastn",
                                            ]
                                            db_pass.at[
                                                i, call + "_germline_end"
                                            ] = db_pass.at[
                                                i, call + "_germline_end_blastn"
                                            ]
                                            db_pass.at[
                                                i, call + "_source"
                                            ] = "blastn"
                                    else:
                                        if present(eval2):
                                            db_pass.at[
                                                i, call + "_call"
                                            ] = db_pass.at[
                                                i, call + "_call_blastn"
                                            ]
                                            db_pass.at[
                                                i, call + "_sequence_start"
                                            ] = db_pass.at[
                                                i,
                                                call + "_sequence_start_blastn",
                                            ]
                                            db_pass.at[
                                                i, call + "_sequence_end"
                                            ] = db_pass.at[
                                                i, call + "_sequence_end_blastn"
                                            ]
                                            db_pass.at[
                                                i, call + "_germline_start"
                                            ] = db_pass.at[
                                                i,
                                                call + "_germline_start_blastn",
                                            ]
                                            db_pass.at[
                                                i, call + "_germline_end"
                                            ] = db_pass.at[
                                                i, call + "_germline_end_blastn"
                                            ]
                                            db_pass.at[
                                                i, call + "_source"
                                            ] = "blastn"
                                else:
                                    db_pass.at[i, call + "_source"] = "10x"
                                    db_pass.at[i, call + "_call"] = db_pass.at[
                                        i, call + "_call_blastn"
                                    ]
                                    if present(db_pass.loc[i, "junction_10x"]):
                                        if present(db_pass.loc[i, "junction"]):
                                            if (
                                                db_pass.loc[i, "junction"]
                                                != db_pass.loc[
                                                    i, "junction_10x"
                                                ]
                                            ):
                                                db_pass.at[
                                                    i, "junction"
                                                ] = db_pass.at[
                                                    i, "junction_10x"
                                                ]
                                                db_pass.at[
                                                    i, "junction_aa"
                                                ] = db_pass.at[
                                                    i, "junction_10x_aa"
                                                ]
                        else:
                            if present(eval1):
                                if eval1 > eval2:
                                    db_pass.at[i, call + "_call"] = db_pass.at[
                                        i, call + "_call_blastn"
                                    ]
                                    db_pass.at[
                                        i, call + "_sequence_start"
                                    ] = db_pass.at[
                                        i, call + "_sequence_start_blastn"
                                    ]
                                    db_pass.at[
                                        i, call + "_sequence_end"
                                    ] = db_pass.at[
                                        i, call + "_sequence_end_blastn"
                                    ]
                                    db_pass.at[
                                        i, call + "_germline_start"
                                    ] = db_pass.at[
                                        i, call + "_germline_start_blastn"
                                    ]
                                    db_pass.at[
                                        i, call + "_germline_end"
                                    ] = db_pass.at[
                                        i, call + "_germline_end_blastn"
                                    ]
                                    db_pass.at[i, call + "_source"] = "blastn"
                            else:
                                if present(eval2):
                                    db_pass.at[i, call + "_call"] = db_pass.at[
                                        i, call + "_call_blastn"
                                    ]
                                    db_pass.at[
                                        i, call + "_sequence_start"
                                    ] = db_pass.at[
                                        i, call + "_sequence_start_blastn"
                                    ]
                                    db_pass.at[
                                        i, call + "_sequence_end"
                                    ] = db_pass.at[
                                        i, call + "_sequence_end_blastn"
                                    ]
                                    db_pass.at[
                                        i, call + "_germline_start"
                                    ] = db_pass.at[
                                        i, call + "_germline_start_blastn"
                                    ]
                                    db_pass.at[
                                        i, call + "_germline_end"
                                    ] = db_pass.at[
                                        i, call + "_germline_end_blastn"
                                    ]
                                    db_pass.at[i, call + "_source"] = "blastn"

                vend = db_pass["v_sequence_end"]
                dstart = db_pass["d_sequence_start"]
                dend = db_pass["d_sequence_end"]
                jstart = db_pass["j_sequence_start"]

                np1 = [
                    str(int(n)) if n >= 0 else ""
                    for n in [
                        (d - v) - 1
                        if pd.notnull(v) and pd.notnull(d)
                        else np.nan
                        for v, d in zip(vend, dstart)
                    ]
                ]
                np2 = [
                    str(int(n)) if n >= 0 else ""
                    for n in [
                        (j - d) - 1
                        if pd.notnull(j) and pd.notnull(d)
                        else np.nan
                        for d, j in zip(dend, jstart)
                    ]
                ]

                db_pass["np1_length"] = np1
                db_pass["np2_length"] = np2

                for i in db_pass["sequence_id"]:
                    if not present(db_pass.loc[i, "np1_length"]):
                        vend = db_pass.loc[i, "v_sequence_end"]
                        if present(vend):
                            jstart = db_pass.loc[i, "j_sequence_start"]
                            if present(jstart):
                                np1l = (jstart - vend) - 1
                                if np1l >= 0:
                                    db_pass.loc[i, "np1_length"] = np1l
            # fill in blanks
            db_pass = sanitize_data(db_pass)
            db_pass.to_csv(passfile, sep="\t", index=False)

        if db_fail is not None:
            if call + "_support" in db_fail:
                db_fail_evalues = dict(db_fail[call + "_support"])
            if call + "_score" in db_fail:
                db_fail_scores = dict(db_fail[call + "_score"])
            db_fail[call + "_call"].fillna(value="", inplace=True)
            db_fail_call = dict(db_fail[call + "_call"])
            if call + "_support" in db_fail:
                db_fail[call + "_support_igblastn"] = pd.Series(db_fail_evalues)
            if call + "_score" in db_fail:
                db_fail[call + "_score_igblastn"] = pd.Series(db_fail_scores)
            db_fail[call + "_call_igblastn"] = pd.Series(db_fail_call)
            db_fail[call + "_call_igblastn"].fillna(value="", inplace=True)
            for col in blast_result:
                if col not in ["sequence_id", "cell_id"]:
                    db_fail[col + "_blastn"] = pd.Series(blast_result[col])
                    if col in [
                        call + "_call",
                        call + "_sequence_alignment",
                        call + "_germline_alignment",
                    ]:
                        db_fail[col + "_blastn"].fillna(value="", inplace=True)
            db_fail[call + "_source"] = ""
            if overwrite:
                for i in db_fail["sequence_id"]:
                    vend = db_fail.loc[i, "v_sequence_end"]
                    if not present(vend):
                        vend_ = 0
                    else:
                        vend_ = vend
                    jstart = db_fail.loc[i, "j_sequence_start"]
                    if not present(jstart):
                        jstart_ = 1000
                    else:
                        jstart_ = jstart
                    callstart = db_fail.loc[i, call + "_sequence_start_blastn"]
                    callend = db_fail.loc[i, call + "_sequence_end_blastn"]
                    if (callstart >= vend_) and (callend <= jstart_):
                        if call + "_support_igblastn" in db_fail:
                            eval1 = db_fail.loc[i, call + "_support_igblastn"]
                        else:
                            eval1 = 1
                        eval2 = db_fail.loc[i, call + "_support_blastn"]
                        if (
                            db_fail.loc[i, call + "_call_igblastn"]
                            != db_fail.loc[i, call + "_call_blastn"]
                        ):
                            if call + "_call_10x" in db_fail:
                                if (
                                    re.sub(
                                        "[*][0-9][0-9]",
                                        "",
                                        db_fail.loc[i, call + "_call_blastn"],
                                    )
                                    != db_fail.loc[i, call + "_call_10x"]
                                ):
                                    if present(eval1):
                                        if eval1 > eval2:
                                            db_fail.at[
                                                i, call + "_call"
                                            ] = db_fail.at[
                                                i, call + "_call_blastn"
                                            ]
                                            db_fail.at[
                                                i, call + "_sequence_start"
                                            ] = db_fail.at[
                                                i,
                                                call + "_sequence_start_blastn",
                                            ]
                                            db_fail.at[
                                                i, call + "_sequence_end"
                                            ] = db_fail.at[
                                                i, call + "_sequence_end_blastn"
                                            ]
                                            db_fail.at[
                                                i, call + "_germline_start"
                                            ] = db_fail.at[
                                                i,
                                                call + "_germline_start_blastn",
                                            ]
                                            db_fail.at[
                                                i, call + "_germline_end"
                                            ] = db_fail.at[
                                                i, call + "_germline_end_blastn"
                                            ]
                                            db_fail.at[
                                                i, call + "_source"
                                            ] = "blastn"
                                    else:
                                        if present(eval2):
                                            db_fail.at[
                                                i, call + "_call"
                                            ] = db_fail.at[
                                                i, call + "_call_blastn"
                                            ]
                                            db_fail.at[
                                                i, call + "_sequence_start"
                                            ] = db_fail.at[
                                                i,
                                                call + "_sequence_start_blastn",
                                            ]
                                            db_fail.at[
                                                i, call + "_sequence_end"
                                            ] = db_fail.at[
                                                i, call + "_sequence_end_blastn"
                                            ]
                                            db_fail.at[
                                                i, call + "_germline_start"
                                            ] = db_fail.at[
                                                i,
                                                call + "_germline_start_blastn",
                                            ]
                                            db_fail.at[
                                                i, call + "_germline_end"
                                            ] = db_fail.at[
                                                i, call + "_germline_end_blastn"
                                            ]
                                            db_fail.at[
                                                i, call + "_source"
                                            ] = "blastn"
                                else:
                                    db_fail.at[i, call + "_source"] = "10x"
                                    db_fail.at[i, call + "_call"] = db_fail.at[
                                        i, call + "_call_blastn"
                                    ]
                                    if present(db_fail.loc[i, "junction_10x"]):
                                        if present(db_fail.loc[i, "junction"]):
                                            if (
                                                db_fail.loc[i, "junction"]
                                                != db_fail.loc[
                                                    i, "junction_10x"
                                                ]
                                            ):
                                                db_fail.at[
                                                    i, "junction"
                                                ] = db_fail.at[
                                                    i, "junction_10x"
                                                ]
                                                db_fail.at[
                                                    i, "junction_aa"
                                                ] = db_fail.at[
                                                    i, "junction_10x_aa"
                                                ]
                        else:
                            if present(eval1):
                                if eval1 > eval2:
                                    db_fail.at[i, call + "_call"] = db_fail.at[
                                        i, call + "_call_blastn"
                                    ]
                                    db_fail.at[
                                        i, call + "_sequence_start"
                                    ] = db_fail.at[
                                        i, call + "_sequence_start_blastn"
                                    ]
                                    db_fail.at[
                                        i, call + "_sequence_end"
                                    ] = db_fail.at[
                                        i, call + "_sequence_end_blastn"
                                    ]
                                    db_fail.at[
                                        i, call + "_germline_start"
                                    ] = db_fail.at[
                                        i, call + "_germline_start_blastn"
                                    ]
                                    db_fail.at[
                                        i, call + "_germline_end"
                                    ] = db_fail.at[
                                        i, call + "_germline_end_blastn"
                                    ]
                                    db_fail.at[i, call + "_source"] = "blastn"
                            else:
                                if present(eval2):
                                    db_fail.at[i, call + "_call"] = db_fail.at[
                                        i, call + "_call_blastn"
                                    ]
                                    db_fail.at[
                                        i, call + "_sequence_start"
                                    ] = db_fail.at[
                                        i, call + "_sequence_start_blastn"
                                    ]
                                    db_fail.at[
                                        i, call + "_sequence_end"
                                    ] = db_fail.at[
                                        i, call + "_sequence_end_blastn"
                                    ]
                                    db_fail.at[
                                        i, call + "_germline_start"
                                    ] = db_fail.at[
                                        i, call + "_germline_start_blastn"
                                    ]
                                    db_fail.at[
                                        i, call + "_germline_end"
                                    ] = db_fail.at[
                                        i, call + "_germline_end_blastn"
                                    ]
                                    db_fail.at[i, call + "_source"] = "blastn"

                vend = db_fail["v_sequence_end"]
                dstart = db_fail["d_sequence_start"]
                dend = db_fail["d_sequence_end"]
                jstart = db_fail["j_sequence_start"]

                np1 = [
                    str(int(n)) if n >= 0 else ""
                    for n in [
                        (d - v) - 1
                        if pd.notnull(v) and pd.notnull(d)
                        else np.nan
                        for v, d in zip(vend, dstart)
                    ]
                ]
                np2 = [
                    str(int(n)) if n >= 0 else ""
                    for n in [
                        (j - d) - 1
                        if pd.notnull(j) and pd.notnull(d)
                        else np.nan
                        for d, j in zip(dend, jstart)
                    ]
                ]
                db_fail["np1_length"] = np1
                db_fail["np2_length"] = np2

                # rescue the d blanks
                for i in db_fail["sequence_id"]:
                    if not present(db_fail.loc[i, "np1_length"]):
                        vend = db_fail.loc[i, "v_sequence_end"]
                        if present(vend):
                            jstart = db_fail.loc[i, "j_sequence_start"]
                            if present(jstart):
                                np1l = (jstart - vend) - 1
                                if np1l >= 0:
                                    db_fail.loc[i, "np1_length"] = np1l

            # fill in blanks
            db_fail = sanitize_data(db_fail)
            db_fail.to_csv(failfile, sep="\t", index=False)


def check_contigs(
    data: Union[Dandelion, pd.DataFrame, str],
    adata: Optional[AnnData] = None,
    productive_only: bool = True,
    library_type: Optional[Literal["ig", "tr-ab", "tr-gd"]] = None,
    umi_foldchange_cutoff: int = 2,
    filter_missing: bool = True,
    save: Optional[str] = None,
    verbose: bool = True,
    **kwargs,
) -> Tuple[Dandelion, AnnData]:
    """
    Check contigs for whether they can be considered as ambiguous or not.

    Returns an `ambiguous` column with boolean T/F in the data. If the `sequence_alignment` is an exact match between
    contigs, the contigs will be merged into the one with the highest umi count, summing the umi/duplicate count. After
    this check, if there are still multiple contigs, cells with multiple contigs checked for whether there is a clear
    dominance in terms of UMI count resulting in two scenarios: 1) if true, all other contigs will be flagged as
    ambiguous; 2) if false, all contigs will be flagged as ambiguous. This is repeated for each cell, for their
    productive and non-productive VDJ and VJ contigs separately. Dominance is assessed by whether or not the umi counts
    demonstrate a > umi_foldchange_cutoff. There are some exceptions: 1) IgM and IgD are allowed to co-exist in the same
    B cell if no other isotypes are detected; 2) TRD and TRB contigs are allowed in the same cell because rearrangement
    of TRB and TRD loci happens at the same time during development and TRD variable region genes exhibits allelic
    inclusion. Thus this can potentially result in some situations where T cells expressing productive TRA-TRB chains
    can also express productive TRD chains.

    Default behvaiour is to only consider productive contigs and remove all non-productive before checking, toggled by
    `productive_only` argument.

    If library_type is provided, it will remove all contigs that do not belong to the related loci. The rationale is
    that the choice of the library type should mean that the primers used would most likely amplify those related
    sequences and if there's any unexpected loci, they likely represent artifacts and shouldn't be analysed.

    If an `adata` object is provided, contigs with no corresponding cell barcode in the `AnnData` object is filtered in
    the output if filter_missing is True.

    Parameters
    ----------
    data : Union[Dandelion, pd.DataFrame, str]
        V(D)J AIRR data to check. Can be `Dandelion`, pandas `DataFrame` and file path to AIRR `.tsv` file.
    adata : Optional[AnnData], optional
        AnnData object to filter. If not provided, it will assume to keep all cells in the airr table and just return a
        Dandelion object.
    productive_only : bool, optional
        whether or not to retain only productive contigs.
    library_type : Optional[Literal["ig", "tr-ab", "tr-gd"]], optional
        if specified, it will first filter based on the expected type of contigs:
            `ig`:
                IGH, IGK, IGL
            `tr-ab`:
                TRA, TRB
            `tr-gd`:
                TRG, TRD
    umi_foldchange_cutoff : int, optional
        related to minimum fold change required to rescue heavy chain contigs/barcode otherwise they will be marked as
        doublets.
    filter_missing : bool, optional
        cells in V(D)J data not found in `AnnData` object will removed from the dandelion object.
    save : Optional[str], optional
        Only used if a pandas data frame or dandelion object is provided. Specifying will save the formatted vdj table
        with a `_checked.tsv` suffix extension.
    verbose : bool, optional
        whether to print progress when marking contigs.
    **kwargs
        additional kwargs passed to `dandelion.utilities._core.Dandelion`.

    Returns
    -------
    Tuple[Dandelion, AnnData]
        checked dandelion V(D)J object and `AnnData` object.

    Raises
    ------
    IndexError
        if no contigs passed filtering.
    ValueError
        if save file name is not suitable.
    """
    start = logg.info("Filtering contigs")
    if isinstance(data, Dandelion):
        dat_ = load_data(data.data)
    else:
        dat_ = load_data(data)

    if library_type is not None:
        acceptable = lib_type(library_type)
    else:
        if isinstance(data, Dandelion):
            if data.library_type is not None:
                acceptable = lib_type(data.library_type)
            else:
                acceptable = None
        else:
            acceptable = None

    if productive_only:
        dat = dat_[dat_["productive"].isin(TRUES)].copy()
    else:
        dat = dat_.copy()

    if acceptable is not None:
        dat = dat[dat.locus.isin(acceptable)].copy()

    barcode = list(set(dat.cell_id))

    if adata is not None:
        adata_provided = True
        adata_ = adata.copy()
        contig_check = pd.DataFrame(index=adata_.obs_names)
        bc_ = {}
        for b in barcode:
            bc_.update({b: "True"})
        contig_check["has_contig"] = pd.Series(bc_)
        contig_check.replace(np.nan, "No_contig", inplace=True)
        adata_.obs["has_contig"] = pd.Series(contig_check["has_contig"])
    else:
        adata_provided = False
        obs = pd.DataFrame(index=barcode)
        adata_ = ad.AnnData(obs=obs)
        adata_.obs["has_contig"] = "True"
    contig_status = MarkAmbiguousContigs(dat, umi_foldchange_cutoff, verbose)

    ambigous = contig_status.ambiguous_contigs.copy()
    umi_adjustment = contig_status.umi_adjustment.copy()
    if len(umi_adjustment) > 0:
        dat["duplicate_count"].update(umi_adjustment)

    ambi = {c: "F" for c in dat_.sequence_id}
    ambiguous_ = {x: "T" for x in ambigous}
    ambi.update(ambiguous_)
    dat["ambiguous"] = pd.Series(ambi)

    if filter_missing:
        dat = dat[dat["cell_id"].isin(adata_.obs_names)].copy()

    if dat.shape[0] == 0:
        raise IndexError(
            "No contigs passed filtering. Are you sure that the cell barcodes are matching?"
        )
    if os.path.isfile(str(data)):
        data_path = Path(data)
        write_airr(
            dat, data_path.parent / "{}_checked.tsv".format(data_path.stem)
        )
    else:
        if save is not None:
            if save.endswith(".tsv"):
                write_airr(dat, str(save))
            else:
                raise ValueError(
                    "{} not suitable. Please provide a file name that ends with .tsv".format(
                        str(save)
                    )
                )

    if productive_only:
        dat_["duplicate_count"].update(dat["duplicate_count"])
        dat_["ambiguous"] = dat["ambiguous"]
        dat_["ambiguous"].fillna("T", inplace=True)
        dat = dat_.copy()

    logg.info("Initializing Dandelion object")
    out_dat = Dandelion(data=dat, **kwargs)
    if isinstance(data, Dandelion):
        out_dat.germline = data.germline
        out_dat.threshold = data.threshold
    if adata_provided:
        transfer(adata_, out_dat, overwrite=True)
        logg.info(
            " finished",
            time=start,
            deep=("Returning Dandelion and AnnData objects: \n"),
        )
        return (out_dat, adata_)
    else:
        logg.info(
            " finished",
            time=start,
            deep=("Returning Dandelion object: \n"),
        )
        return out_dat


class MarkAmbiguousContigs:
    """
    `MarkAmbiguousContigs` class object.

    New main class object to run filter_contigs.

    Attributes
    ----------
    ambiguous_contigs : List[str]
        list of `sequence_id`s that are ambiguous.
    Cell : dandelion.utilities._utilities.Tree
        nested dictionary of cells.
    umi_adjustment : Dict[str, int]
        dictionary of `sequence_id`s with adjusted umi value.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        umi_foldchange_cutoff: Union[int, float],
        verbose: bool,
    ):
        """Init method for MarkAmbiguousContigs.

        Parameters
        ----------
        data : pd.DataFrame
            AIRR data frame in Dandelion.data.
        umi_foldchange_cutoff : Union[int, float]
            fold-change cut off for decision.
        verbose : bool
            whether or not to print progress.
        """
        self.Cell = Tree()
        self.ambiguous_contigs = []
        self.umi_adjustment = {}
        if "v_call_genotyped" in data.columns:
            v_dict = dict(zip(data["sequence_id"], data["v_call_genotyped"]))
        else:
            v_dict = dict(zip(data["sequence_id"], data["v_call"]))
        d_dict = dict(zip(data["sequence_id"], data["d_call"]))
        j_dict = dict(zip(data["sequence_id"], data["j_call"]))
        c_dict = dict(zip(data["sequence_id"], data["c_call"]))
        l_dict = dict(zip(data["sequence_id"], data["locus"]))
        for contig, row in tqdm(
            data.iterrows(),
            desc="Preparing data",
        ):
            cell = row["cell_id"]
            if row["locus"] in HEAVYLONG:
                if row["productive"] in TRUES:
                    self.Cell[cell]["VDJ"]["P"][contig].update(row)
                elif row["productive"] in FALSES:
                    self.Cell[cell]["VDJ"]["NP"][contig].update(row)
            elif row["locus"] in LIGHTSHORT:
                if row["productive"] in TRUES:
                    self.Cell[cell]["VJ"]["P"][contig].update(row)
                elif row["productive"] in FALSES:
                    self.Cell[cell]["VJ"]["NP"][contig].update(row)

        for cell in tqdm(
            self.Cell,
            desc="Scanning for poor quality/ambiguous contigs",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            disable=not verbose,
        ):
            if len(self.Cell[cell]["VDJ"]["P"]) > 0:
                # VDJ productive
                data1 = pd.DataFrame(
                    [
                        self.Cell[cell]["VDJ"]["P"][x]
                        for x in self.Cell[cell]["VDJ"]["P"]
                        if isinstance(self.Cell[cell]["VDJ"]["P"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VDJ"]["P"][x]["sequence_id"]
                        for x in self.Cell[cell]["VDJ"]["P"]
                        if isinstance(self.Cell[cell]["VDJ"]["P"][x], dict)
                    ],
                )
                vdj_p = list(data1["sequence_id"])
                vdj_umi_p = [
                    int(x) for x in pd.to_numeric(data1["duplicate_count"])
                ]
                vdj_ccall_p = list(data1["c_call"])
                vdj_locus_p = list(data1["locus"])
                if len(vdj_p) > 1:
                    if "sequence_alignment" in data1:
                        (
                            data1,
                            vdj_p,
                            vdj_umi_p,
                            vdj_ccall_p,
                            umi_adjust_vdj,
                            ambi_cont_vdj,
                        ) = check_update_same_seq(data1)
                        if len(umi_adjust_vdj) > 0:
                            self.umi_adjustment.update(umi_adjust_vdj)
                        if len(ambi_cont_vdj) > 0:
                            for avdj in ambi_cont_vdj:
                                self.ambiguous_contigs.append(avdj)
                    if len(vdj_p) > 1:
                        if "IGHD" in vdj_ccall_p:
                            if all(x in ["IGHM", "IGHD"] for x in vdj_ccall_p):
                                if len(list(set(vdj_ccall_p))) == 2:
                                    vdj_ccall_p_igm_count = dict(
                                        data1[data1["c_call"] == "IGHM"][
                                            "duplicate_count"
                                        ]
                                    )
                                    vdj_ccall_p_igd_count = dict(
                                        data1[data1["c_call"] == "IGHD"][
                                            "duplicate_count"
                                        ]
                                    )

                                if len(vdj_ccall_p_igm_count) > 1:
                                    (
                                        keep_igm,
                                        extra_igm,
                                        ambiguous_igm,
                                    ) = check_productive_vdj(
                                        vdj_ccall_p_igm_count,
                                        umi_foldchange_cutoff,
                                    )
                                else:
                                    keep_igm, extra_igm, ambiguous_igm = (
                                        [],
                                        [],
                                        [],
                                    )

                                if len(vdj_ccall_p_igd_count) > 1:
                                    (
                                        keep_igd,
                                        extra_igd,
                                        ambiguous_igd,
                                    ) = check_productive_vdj(
                                        vdj_ccall_p_igd_count,
                                        umi_foldchange_cutoff,
                                    )
                                else:
                                    keep_igd, extra_igd, ambiguous_igd = (
                                        [],
                                        [],
                                        [],
                                    )

                                vdj_p = keep_igm + keep_igd
                                extra_vdj = extra_igm + extra_igd
                                ambiguous_vdj = ambiguous_igm + ambiguous_igd
                            else:
                                vdj_ccall_p_count = dict(
                                    data1["duplicate_count"]
                                )
                                if len(vdj_ccall_p_count) > 1:
                                    (
                                        vdj_p,
                                        extra_vdj,
                                        ambiguous_vdj,
                                    ) = check_productive_vdj(
                                        vdj_ccall_p_count, umi_foldchange_cutoff
                                    )
                                else:
                                    vdj_p, extra_vdj, ambiguous_vdj = [], [], []
                        elif all(x in ["TRB", "TRD"] for x in vdj_locus_p):
                            if len(list(set(vdj_locus_p))) == 2:
                                vdj_locus_p_trb_count = dict(
                                    data1[data1["locus"] == "TRB"][
                                        "duplicate_count"
                                    ]
                                )
                                vdj_locus_p_trd_count = dict(
                                    data1[data1["locus"] == "TRD"][
                                        "duplicate_count"
                                    ]
                                )

                                if len(vdj_locus_p_trb_count) > 1:
                                    (
                                        keep_trb,
                                        extra_trb,
                                        ambiguous_trb,
                                    ) = check_productive_vdj(
                                        vdj_locus_p_trb_count,
                                        umi_foldchange_cutoff,
                                    )
                                else:
                                    keep_trb, extra_trb, ambiguous_trb = (
                                        [],
                                        [],
                                        [],
                                    )

                                if len(vdj_locus_p_trd_count) > 1:
                                    (
                                        keep_trd,
                                        extra_trd,
                                        ambiguous_trd,
                                    ) = check_productive_vdj(
                                        vdj_locus_p_trd_count,
                                        umi_foldchange_cutoff,
                                    )
                                else:
                                    keep_trd, extra_trd, ambiguous_trd = (
                                        [],
                                        [],
                                        [],
                                    )

                                vdj_p = keep_trb + keep_trd
                                extra_vdj = extra_trb + extra_trd
                                ambiguous_vdj = ambiguous_trb + ambiguous_trd
                            else:
                                vdj_ccall_p_count = dict(
                                    data1["duplicate_count"]
                                )
                                if len(vdj_ccall_p_count) > 1:
                                    (
                                        vdj_p,
                                        extra_vdj,
                                        ambiguous_vdj,
                                    ) = check_productive_vdj(
                                        vdj_ccall_p_count, umi_foldchange_cutoff
                                    )
                                else:
                                    vdj_p, extra_vdj, ambiguous_vdj = [], [], []
                        else:
                            vdj_ccall_p_count = dict(data1["duplicate_count"])
                            if len(vdj_ccall_p_count) > 1:
                                (
                                    vdj_p,
                                    extra_vdj,
                                    ambiguous_vdj,
                                ) = check_productive_vdj(
                                    vdj_ccall_p_count, umi_foldchange_cutoff
                                )
                            else:
                                vdj_p, extra_vdj, ambiguous_vdj = [], [], []
                    else:
                        vdj_p, extra_vdj, ambiguous_vdj = [], [], []
                else:
                    vdj_p, extra_vdj, ambiguous_vdj = [], [], []

                if len(ambiguous_vdj) > 0:
                    for a in ambiguous_vdj:
                        self.ambiguous_contigs.append(a)

            # VDJ non-productive
            if len(self.Cell[cell]["VDJ"]["NP"]) > 0:
                data2 = pd.DataFrame(
                    [
                        self.Cell[cell]["VDJ"]["NP"][x]
                        for x in self.Cell[cell]["VDJ"]["NP"]
                        if isinstance(self.Cell[cell]["VDJ"]["NP"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VDJ"]["NP"][x]["sequence_id"]
                        for x in self.Cell[cell]["VDJ"]["NP"]
                        if isinstance(self.Cell[cell]["VDJ"]["NP"][x], dict)
                    ],
                )
                vdj_np = list(data2["sequence_id"])
                (
                    data2,
                    vdj_np,
                    _,
                    _,
                    umi_adjust_vdjnp,
                    ambi_cont_vdjnp,
                ) = check_update_same_seq(data2)
                if len(umi_adjust_vdjnp) > 0:
                    self.umi_adjustment.update(umi_adjust_vdjnp)
                if len(ambi_cont_vdjnp) > 0:
                    for avdj in ambi_cont_vdjnp:
                        self.ambiguous_contigs.append(avdj)

            # VJ productive
            if len(self.Cell[cell]["VJ"]["P"]) > 0:
                data3 = pd.DataFrame(
                    [
                        self.Cell[cell]["VJ"]["P"][x]
                        for x in self.Cell[cell]["VJ"]["P"]
                        if isinstance(self.Cell[cell]["VJ"]["P"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VJ"]["P"][x]["sequence_id"]
                        for x in self.Cell[cell]["VJ"]["P"]
                        if isinstance(self.Cell[cell]["VJ"]["P"][x], dict)
                    ],
                )
                vj_p = list(data3["sequence_id"])
                vj_umi_p = [
                    int(x) for x in pd.to_numeric(data3["duplicate_count"])
                ]
                if len(vj_p) > 1:
                    if "sequence_alignment" in data3:
                        (
                            data3,
                            vj_p,
                            vj_umi_p,
                            vj_ccall_p,
                            umi_adjust_vj,
                            ambi_cont_vj,
                        ) = check_update_same_seq(data3)
                        if len(umi_adjust_vj) > 0:
                            self.umi_adjustment.update(umi_adjust_vj)
                        if len(ambi_cont_vj) > 0:
                            for avj in ambi_cont_vj:
                                self.ambiguous_contigs.append(avj)
                    if len(vj_p) > 1:
                        vj_ccall_p_count = dict(data3["duplicate_count"])
                        # maximum keep 2?
                        vj_p, extra_vj, ambiguous_vj = check_productive_vj(
                            vj_ccall_p_count
                        )
                    else:
                        vj_p, extra_vj, ambiguous_vj = [], [], []
                else:
                    vj_p, extra_vj, ambiguous_vj = [], [], []

                if len(ambiguous_vj) > 0:
                    for a in ambiguous_vj:
                        self.ambiguous_contigs.append(a)

            # VJ non-productive
            if len(self.Cell[cell]["VJ"]["NP"]) > 0:
                data4 = pd.DataFrame(
                    [
                        self.Cell[cell]["VJ"]["NP"][x]
                        for x in self.Cell[cell]["VJ"]["NP"]
                        if isinstance(self.Cell[cell]["VJ"]["NP"][x], dict)
                    ],
                    index=[
                        self.Cell[cell]["VJ"]["NP"][x]["sequence_id"]
                        for x in self.Cell[cell]["VJ"]["NP"]
                        if isinstance(self.Cell[cell]["VJ"]["NP"][x], dict)
                    ],
                )
                (
                    data4,
                    vj_np,
                    _,
                    _,
                    umi_adjust_vjnp,
                    ambi_cont_vjnp,
                ) = check_update_same_seq(data4)
                if len(umi_adjust_vjnp) > 0:
                    self.umi_adjustment.update(umi_adjust_vjnp)
                if len(ambi_cont_vjnp) > 0:
                    for avj in ambi_cont_vjnp:
                        self.ambiguous_contigs.append(avj)

            if "vdj_p" not in locals():
                vdj_p = []
            if "vj_p" not in locals():
                vj_p = []
            if "vdj_np" not in locals():
                vdj_np = []
            if "vj_np" not in locals():
                vj_np = []
            if "extra_vdj" not in locals():
                extra_vdj = []
            if "extra_vj" not in locals():
                extra_vj = []

            # check here for bad combinations
            # marking poor bcr quality, defined as those with conflicting assignment of
            # locus and V(D)J v-, d-, j- and c- calls, and also those that are missing
            # j calls.
            if len(vdj_p) > 0:
                for vdjx in vdj_p:
                    v = v_dict[vdjx]
                    d = d_dict[vdjx]
                    j = j_dict[vdjx]
                    c = c_dict[vdjx]
                    if present(v):
                        if not re.search("IGH|TR[BD]|TRAV.*/DV", v):
                            self.ambiguous_contigs.append(vdjx)
                    if present(d):
                        if not re.search("IGH|TR[BD]", d):
                            self.ambiguous_contigs.append(vdjx)
                    if present(j):
                        if not re.search("IGH|TR[BD]", j):
                            self.ambiguous_contigs.append(vdjx)
                    if present(c):
                        if not re.search("IGH|TR[BD]", c):
                            self.ambiguous_contigs.append(vdjx)
                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGH"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(v, j, "TRB"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(v, j, "TRD"):
                                if not re.search("TRAV.*/DV", v):
                                    self.ambiguous_contigs.append(vdjx)
                        if present(d):
                            if not_same_call(d, j, "IGH"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(d, j, "TRB"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(d, j, "TRD"):
                                self.ambiguous_contigs.append(vdjx)
                    else:
                        self.ambiguous_contigs.append(vdjx)

            if len(vdj_np) > 0:
                for vdjx in vdj_np:
                    v = v_dict[vdjx]
                    d = d_dict[vdjx]
                    j = j_dict[vdjx]
                    c = c_dict[vdjx]
                    if present(v):
                        if not re.search("IGH|TR[BD]|TRAV.*/DV", v):
                            self.ambiguous_contigs.append(vdjx)
                    if present(d):
                        if not re.search("IGH|TR[BD]", d):
                            self.ambiguous_contigs.append(vdjx)
                    if present(j):
                        if not re.search("IGH|TR[BD]", j):
                            self.ambiguous_contigs.append(vdjx)
                    if present(c):
                        if not re.search("IGH|TR[BD]", c):
                            self.ambiguous_contigs.append(vdjx)
                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGH"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(v, j, "TRB"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(v, j, "TRD"):
                                if not re.search("TRAV.*/DV", v):
                                    self.ambiguous_contigs.append(vdjx)
                        if present(d):
                            if not_same_call(d, j, "IGH"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(d, j, "TRB"):
                                self.ambiguous_contigs.append(vdjx)
                            elif not_same_call(d, j, "TRD"):
                                self.ambiguous_contigs.append(vdjx)
                    else:
                        self.ambiguous_contigs.append(vdjx)
            if len(vj_p) > 0:
                for vjx in vj_p:
                    v = v_dict[vjx]
                    j = j_dict[vjx]
                    c = c_dict[vjx]
                    if present(v):
                        if re.search("IGH|TRB", v):
                            self.ambiguous_contigs.append(vjx)
                    if present(j):
                        if re.search("IGH|TRB", j):
                            self.ambiguous_contigs.append(vjx)
                    if present(c):
                        if re.search("IGH|TRB", c):
                            self.ambiguous_contigs.append(vjx)

                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGK"):
                                self.ambiguous_contigs.append(vjx)
                            elif not_same_call(v, j, "IGL"):
                                self.ambiguous_contigs.append(vjx)
                            elif not_same_call(v, j, "TRA"):
                                if not re.search("TR[AD]", v):
                                    if not re.search("TRA", j):
                                        self.ambiguous_contigs.append(vjx)
                            elif not_same_call(v, j, "TRG"):
                                self.ambiguous_contigs.append(vjx)
                    else:
                        self.ambiguous_contigs.append(vjx)

            if len(vj_np) > 0:
                for vjx in vj_np:
                    v = v_dict[vjx]
                    j = j_dict[vjx]
                    c = c_dict[vjx]
                    if present(v):
                        if re.search("IGH|TRB", v):
                            self.ambiguous_contigs.append(vjx)
                    if present(j):
                        if re.search("IGH|TRB", j):
                            self.ambiguous_contigs.append(vjx)
                    if present(c):
                        if re.search("IGH|TRB", c):
                            self.ambiguous_contigs.append(vjx)

                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGK"):
                                self.ambiguous_contigs.append(vjx)
                            elif not_same_call(v, j, "IGL"):
                                self.ambiguous_contigs.append(vjx)
                            elif not_same_call(v, j, "TRA"):
                                if not re.search("TR[AD]", v):
                                    if not re.search("TRA", j):
                                        self.ambiguous_contigs.append(vjx)
                            elif not_same_call(v, j, "TRG"):
                                self.ambiguous_contigs.append(vjx)
                    else:
                        self.ambiguous_contigs.append(vjx)

            if len(extra_vdj) > 0:
                for evdj in extra_vdj:
                    v = v_dict[evdj]
                    d = d_dict[evdj]
                    j = j_dict[evdj]
                    c = c_dict[evdj]
                    if present(v):
                        if not re.search("IGH|TR[BD]|TRAV.*/DV", v):
                            self.ambiguous_contigs.append(evdj)
                    if present(d):
                        if not re.search("IGH|TR[BD]", d):
                            self.ambiguous_contigs.append(evdj)
                    if present(j):
                        if not re.search("IGH|TR[BD]", j):
                            self.ambiguous_contigs.append(evdj)
                    if present(c):
                        if not re.search("IGH|TR[BD]", c):
                            self.ambiguous_contigs.append(evdj)
                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGH"):
                                self.ambiguous_contigs.append(evdj)
                            elif not_same_call(v, j, "TRB"):
                                self.ambiguous_contigs.append(evdj)
                            elif not_same_call(v, j, "TRD"):
                                if not re.search("TRAV.*/DV", v):
                                    self.ambiguous_contigs.append(evdj)
                        if present(d):
                            if not_same_call(d, j, "IGH"):
                                self.ambiguous_contigs.append(evdj)
                            elif not_same_call(d, j, "TRB"):
                                self.ambiguous_contigs.append(evdj)
                            elif not_same_call(d, j, "TRD"):
                                self.ambiguous_contigs.append(evdj)
                    else:
                        self.ambiguous_contigs.append(evdj)

            if len(extra_vj) > 0:
                for evj in extra_vj:
                    v = v_dict[evj]
                    j = j_dict[evj]
                    c = c_dict[evj]
                    if present(v):
                        if re.search("IGH|TRB", v):
                            self.ambiguous_contigs.append(evj)
                    if present(j):
                        if re.search("IGH|TRB", j):
                            self.ambiguous_contigs.append(evj)
                    if present(c):
                        if re.search("IGH|TRB", c):
                            self.ambiguous_contigs.append(evj)
                    if present(j):
                        if present(v):
                            if not_same_call(v, j, "IGK"):
                                self.ambiguous_contigs.append(evj)
                            elif not_same_call(v, j, "IGL"):
                                self.ambiguous_contigs.append(evj)
                            elif not_same_call(v, j, "TRA"):
                                if not re.search("TR[AD]", v):
                                    if not re.search("TRA", j):
                                        self.ambiguous_contigs.append(evj)
                            elif not_same_call(v, j, "TRG"):
                                self.ambiguous_contigs.append(evj)


def check_productive_vdj(
    vdj_contigs: Dict[str, int], umi_foldchange_cutoff: Union[int, float]
) -> Tuple[List[str], List[str], List[str]]:
    """Keep top productive because of allelic exclusion."""
    keep_contigs, extra_contigs, ambiguous_contigs = [], [], []
    counts = vdj_contigs.values()
    max_count = max(counts)
    max_id_keys = [k for k, v in vdj_contigs.items() if v == max_count]
    if len(max_id_keys) == 1:
        other_counts = {
            k: v for k, v in vdj_contigs.items() if k != max_id_keys[0]
        }
        umi_test = {
            i: ((max_count / j) < umi_foldchange_cutoff)
            for i, j in other_counts.items()
        }
        if any(umi_test.values()):
            for dk in vdj_contigs.keys():
                ambiguous_contigs.append(dk)
        elif max_count >= 3:
            drop_keys = [k for k, v in vdj_contigs.items() if v < max_count]
            for dk in drop_keys:
                extra_contigs.append(dk)
            for kk in max_id_keys:
                keep_contigs.append(kk)
        else:
            for dk in vdj_contigs.keys():
                ambiguous_contigs.append(dk)
    else:
        for dk in vdj_contigs.keys():
            ambiguous_contigs.append(dk)
    return keep_contigs, extra_contigs, ambiguous_contigs


def check_productive_vj(
    vj_contigs: Dict[str, int]
) -> Tuple[List[str], List[str], List[str]]:
    """Function to keep top two productive vj chains because of allelic inclusions.

    Parameters
    ----------
    vj_contigs : Dict[str, int]
        dictionary of contigs with umi count.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        lists of contigs to keep, are extra or are ambiguous.
    """
    keep_contigs, extra_contigs, ambiguous_contigs = [], [], []
    counts = vj_contigs.values()
    max_counts = max(counts)
    if len(vj_contigs) > 2:
        if max(counts) >= 3:
            set_counts = set(counts)
            set_counts.remove(max_counts)
            if len(set_counts) > 0:
                max_id_keys = [
                    k for k, v in vj_contigs.items() if v >= max(set_counts)
                ]
                if len(max_id_keys) > 2:
                    for dk in vj_contigs.keys():
                        ambiguous_contigs.append(dk)
                else:
                    drop_keys = [
                        k for k, v in vj_contigs.items() if v < max(set_counts)
                    ]
                    for dk in drop_keys:
                        extra_contigs.append(dk)
                    for kk in max_id_keys:
                        keep_contigs.append(kk)
            else:
                for k in vj_contigs.keys():
                    keep_contigs.append(k)
        else:
            for dk in vj_contigs.keys():
                ambiguous_contigs.append(dk)
    else:
        for k in vj_contigs.keys():
            keep_contigs.append(k)
    return keep_contigs, extra_contigs, ambiguous_contigs


def check_update_same_seq(
    data: pd.DataFrame,
) -> Tuple[
    pd.DataFrame, List[str], List[int], List[str], Dict[str, int], List[str]
]:
    """Check if sequences are the same.

    Parameters
    ----------
    data : pd.DataFrame
        AIRR data frame in Dandelion.data.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[int], List[str], Dict[str, int], List[str]]
        updated  AIRR data frame, lists of contigs to keep, their umi counts, their c_calls,
        adjusted umi counts, and list of ambiguous contigs.
    """
    keep_id = []
    keep_ccall = []
    umi_adjust = {}
    ambi_cont = []
    sequencecol = (
        "sequence_alignment" if "sequence_alignment" in data else "sequence"
    )
    if sequencecol in data:
        seq_ = list(data[sequencecol])
        seq_2 = [s for s in seq_ if present(s)]
        if len(set(seq_2)) < len(seq_2):
            _seq = {
                k: r for k, r in dict(data[sequencecol]).items() if present(r)
            }
            _count = {
                k: r for k, r in dict(data.duplicate_count).items() if k in _seq
            }
            rep_seq = [
                seq
                for seq in set(_seq.values())
                if countOf(_seq.values(), seq) > 1
            ]
            keep_seqs = [
                seq
                for seq in set(_seq.values())
                if countOf(_seq.values(), seq) == 1
            ]
            keep_seqs_ids = [i for i, seq in _seq.items() if seq in keep_seqs]
            if len(rep_seq) > 0:
                for rep in rep_seq:
                    dup_keys = [k for k, v in _seq.items() if v == rep]
                    keep_index_vj = dup_keys[0]
                    keep_index_count = int(_count[keep_index_vj])
                    sum_rep_counts = np.sum([_count[k] for k in dup_keys[1:]])
                    umi_adjust.update(
                        {
                            keep_index_vj: int(
                                np.sum(
                                    [
                                        sum_rep_counts,
                                        keep_index_count,
                                    ],
                                )
                            )
                        }
                    )
                    for dk in dup_keys[1:]:
                        ambi_cont.append(dk)
                    keep_seqs_ids.append(keep_index_vj)
                    data.duplicate_count.update(
                        {keep_index_vj: keep_index_count}
                    )
                # refresh
                empty_seqs_ids = [
                    k
                    for k, r in dict(data[sequencecol]).items()
                    if not present(r)
                ]
                if len(empty_seqs_ids) > 0:
                    keep_seqs_ids = keep_seqs_ids + empty_seqs_ids
                data = data.loc[keep_seqs_ids]
        keep_id = list(data.sequence_id)
        keep_umi = [int(x) for x in pd.to_numeric(data.duplicate_count)]
        keep_ccall = list(data.c_call)

    return (data, keep_id, keep_umi, keep_ccall, umi_adjust, ambi_cont)


def choose_segments(
    starts: pd.Series, ends: pd.Series, scores: pd.Series
) -> List[str]:
    """Choose left most segment

    Parameters
    ----------
    starts : pd.Series
        nucleotide start positions.
    ends : pd.Series
        nucleotide end positions.
    scores : pd.Series
        alignment scores.

    Returns
    -------
    List[str]
        list of chosen segments.
    """
    starts = np.array(starts)
    ends = np.array(ends)
    scores = np.array(scores)
    ind = np.arange(len(starts))
    chosen = []
    while len(ind) > 0:
        best = np.argmax(scores)
        chosen.append(ind[best])
        overlap = (starts <= ends[best]) & (ends >= starts[best])
        ind = ind[~overlap]
        starts = starts[~overlap]
        ends = ends[~overlap]
        scores = scores[~overlap]
    return chosen


def multimapper(filename: str) -> pd.DataFrame:
    """Select the left more segment as the final call

    Parameters
    ----------
    filename : str
        path to multimapper file.

    Returns
    -------
    pd.DataFrame
        Mapped multimapper data frame.
    """
    df = pd.read_csv(filename, delimiter="\t")
    df_new = df.loc[
        df["j_support"] < 1e-3, :
    ]  # maybe not needing to filter if j_support has already been filtered
    mapped = pd.DataFrame(
        index=list(set(df_new["sequence_id"])),
        columns=[
            "multimappers",
            "multiplicity",
            "sequence_start_multimappers",
            "sequence_end_multimappers",
            "support_multimappers",
        ],
    )

    for j in range(mapped.shape[0]):
        id = mapped.index[j]
        tmp = df_new.loc[
            df_new["sequence_id"] == id,
            ["j_sequence_start", "j_sequence_end", "j_support", "j_call"],
        ]

        starts = tmp["j_sequence_start"]
        ends = tmp["j_sequence_end"]
        scores = -tmp["j_support"]
        chosen_ind = choose_segments(starts, ends, scores)
        tmp = tmp.iloc[chosen_ind, :]
        tmp = tmp.sort_values(by=["j_sequence_start"], ascending=True)

        mapped["multimappers"][j] = ";".join(tmp["j_call"])
        mapped["multiplicity"][j] = tmp.shape[0]
        mapped["sequence_start_multimappers"][j] = ";".join(
            tmp["j_sequence_start"].astype(str)
        )
        mapped["sequence_end_multimappers"][j] = ";".join(
            tmp["j_sequence_end"].astype(str)
        )
        mapped["support_multimappers"][j] = ";".join(
            tmp["j_support"].astype(str)
        )

    return mapped


def update_j_multimap(data: List[str], filename_prefix: List[str]):
    """Update j multimapper call.

    Parameters
    ----------
    data : List[str]
        input folders.
    filename_prefix : List[str]
        prefixes to append to front of files.
    """
    if not isinstance(data, list):
        data = [data]
    if not isinstance(filename_prefix, list):
        filename_prefix = [filename_prefix]
    for i in range(0, len(data)):
        filePath0 = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_j_blast.tsv",
            sub_dir="tmp",
        )
        filePath1 = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_igblast_db-pass.tsv",
            sub_dir="tmp",
        )
        filePath1g = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_igblast_db-pass_genotyped.tsv",
            sub_dir="tmp",
        )
        filePath2 = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_igblast_db-all.tsv",
            sub_dir="tmp",
        )
        filePath3 = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_igblast_db-fail.tsv",
            sub_dir="tmp",
        )
        filePath4 = check_filepath(
            data[i],
            filename_prefix=filename_prefix[i],
            ends_with="_dandelion.tsv",
        )

        jmm_transfer_cols = [
            "multimappers",
            "multiplicity",
            "sequence_start_multimappers",
            "sequence_end_multimappers",
            "support_multimappers",
        ]
        check_multimapper(filePath0, filePath2)
        if filePath0 is not None:
            jmulti = multimapper(filePath0)
            if filePath1 is not None:
                dbpass = load_data(filePath1)
                for col in jmm_transfer_cols:
                    dbpass["j_call_" + col] = ""
                    dbpass["j_call_" + col].update(jmulti[col])
                write_airr(dbpass, filePath1)
            if filePath1g is not None:
                dbpassg = load_data(filePath1g)
                for col in jmm_transfer_cols:
                    dbpassg["j_call_" + col] = ""
                    dbpassg["j_call_" + col].update(jmulti[col])
                write_airr(dbpassg, filePath1g)
            if filePath2 is not None:
                dbfail = load_data(filePath2)
                for col in jmm_transfer_cols:
                    dbfail["j_call_" + col] = ""
                    dbfail["j_call_" + col].update(jmulti[col])
                for i in dbfail.index:
                    if not present(dbfail.loc[i, "v_call"]):
                        jmmappers = dbfail.at[i, "j_call_multimappers"].split(
                            ";"
                        )
                        jmmappersstart = dbfail.at[
                            i, "j_call_sequence_start_multimappers"
                        ].split(";")
                        jmmappersend = dbfail.at[
                            i, "j_call_sequence_end_multimappers"
                        ].split(";")
                        jmmapperssupport = dbfail.at[
                            i, "j_call_support_multimappers"
                        ].split(";")
                        if len(jmmappers) > 1:
                            dbfail.at[i, "j_call"] = jmmappers[0]
                            dbfail.at[i, "j_sequence_start"] = jmmappersstart[0]
                            dbfail.at[i, "j_sequence_end"] = jmmappersend[0]
                            dbfail.at[i, "j_support"] = jmmapperssupport[0]
                write_airr(dbfail, filePath2)
            if filePath3 is not None:
                dball = load_data(filePath3)
                for col in jmm_transfer_cols:
                    dball["j_call_" + col] = ""
                    dball["j_call_" + col].update(jmulti[col])
                for i in dball.index:
                    if not present(dball.loc[i, "v_call"]):
                        jmmappers = dball.at[i, "j_call_multimappers"].split(
                            ";"
                        )
                        jmmappersstart = dball.at[
                            i, "j_call_sequence_start_multimappers"
                        ].split(";")
                        jmmappersend = dball.at[
                            i, "j_call_sequence_end_multimappers"
                        ].split(";")
                        jmmapperssupport = dball.at[
                            i, "j_call_support_multimappers"
                        ].split(";")
                        if len(jmmappers) > 1:
                            dball.at[i, "j_call"] = jmmappers[0]
                            dball.at[i, "j_sequence_start"] = jmmappersstart[0]
                            dball.at[i, "j_sequence_end"] = jmmappersend[0]
                            dball.at[i, "j_support"] = jmmapperssupport[0]
                write_airr(dball, filePath3)
            if filePath4 is not None:
                dandy = load_data(filePath4)
                for col in jmm_transfer_cols:
                    dandy["j_call_" + col] = ""
                    dandy["j_call_" + col].update(jmulti[col])
                write_airr(dandy, filePath4)


def check_multimapper(
    filename1: str,
    filename2: str,
) -> pd.DataFrame:
    """Select the left more segment as the final call
    Parameters
    ----------
    filename1 : str
        path to multimapper file.
    filename2 : str
        path to reference file containing all information.
    Returns
    -------
    pd.DataFrame
        Mapped multimapper data frame.
    """
    if filename1 is not None:
        if filename2 is not None:
            df = pd.read_csv(filename1, sep="\t")
            df_new = df[
                df["j_support"] < 1e-3
            ]  # maybe not needing to filter if j_support has already been filtered
            df_ref = load_data(filename2)
            mapped = list(set(df_new["sequence_id"]))
            keep = []
            for j in mapped:
                tmp = df_new[df_new["sequence_id"] == j][
                    [
                        "j_sequence_start",
                        "j_sequence_end",
                        "j_support",
                        "j_call",
                    ]
                ]
                if j in df_ref.index:
                    vend = df_ref.loc[j, "v_sequence_end"]
                    vend_ = 0 if not present(vend) else vend
                    for i in tmp.index:
                        callstart = tmp.loc[i, "j_sequence_start"]
                        if callstart >= vend_:
                            keep.append(i)
            keepdf = df_new.loc[keep]
            keepdf.to_csv(filename1, sep="\t", index=False)
