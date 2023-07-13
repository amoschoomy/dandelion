#!/usr/bin/env python
import math
import os
import re
import sys

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

from anndata import AnnData
from changeo.Gene import getGene
from collections import defaultdict, Counter
from distance import hamming
from itertools import combinations
from scanpy import logging as logg
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from subprocess import run
from time import sleep
from tqdm import tqdm
from typing import Union, List, Optional

from dandelion.tools._network import *
from dandelion.utilities._core import *
from dandelion.utilities._io import *
from dandelion.utilities._utilities import *


def find_clones(
    vdj_data: Union[Dandelion, pd.DataFrame],
    identity: Union[Dict, float] = 0.85,
    key: Optional[str] = None,
    by_alleles: bool = False,
    key_added: Optional[str] = None,
    recalculate_length: bool = True,
    collapse_label: bool = False,
    verbose: bool = True,
) -> Dandelion:
    """
    Find clones based on VDJ chain and VJ chain CDR3 junction hamming distance.

    Parameters
    ----------
    vdj_data : Union[Dandelion, pd.DataFrame]
        `Dandelion` object, pandas `DataFrame` in changeo/airr format, or file path to changeo/airr file
        after clones have been determined.
    identity : Union[Dict, float], optional
        junction similarity parameter. Default 0.85. If provided as a dictionary, please use the following
        keys:'ig', 'tr-ab', 'tr-gd'.
    key : Optional[str], optional
        column name for performing clone clustering. `None` defaults to 'junction_aa'.
    by_alleles : bool, optional
        whether or not to collapse alleles to genes. `None` defaults to False.
    key_added : Optional[str], optional
        If specified, this will be the column name for clones. `None` defaults to 'clone_id'
    recalculate_length : bool, optional
        whether or not to re-calculate junction length, rather than rely on parsed assignment (which occasionally is
        wrong). Default is True
    collapse_label : bool, optional
        whether or not to return the clone_ids with the full keys for VDJ and VJ groups.
        Default (False) will expand VDJ and VJ. If True, VJ will be collapsed to a singular number.
    verbose : bool, optional
        whether or not to print progress.

    Returns
    -------
    Dandelion
        `Dandelion` object with clone_id annotated in `.data` slot and `.metadata` initialized.

    Raises
    ------
    ValueError
        if `key` not found in Dandelion.data.
    """
    start = logg.info("Finding clonotypes")
    pd.set_option("mode.chained_assignment", None)
    if isinstance(vdj_data, Dandelion):
        dat_ = load_data(vdj_data.data)
    else:
        dat_ = load_data(vdj_data)

    if key_added is None:
        clone_key = "clone_id"
    else:
        clone_key = key_added
    dat_[clone_key] = ""

    dat = dat_.copy()
    if "ambiguous" in dat_:
        dat = dat_[dat_["ambiguous"] == "F"].copy()

    locus_log = {"ig": "B", "tr-ab": "abT", "tr-gd": "gdT"}
    locus_dict1 = {"ig": ["IGH"], "tr-ab": ["TRB"], "tr-gd": ["TRD"]}
    locus_dict2 = {"ig": ["IGK", "IGL"], "tr-ab": ["TRA"], "tr-gd": ["TRG"]}
    locus_log1_dict = {"ig": "IGH", "tr-ab": "TRB", "tr-gd": "TRD"}
    locus_log2_dict = {"ig": "IGL/IGL", "tr-ab": "TRA", "tr-gd": "TRG"}
    DEFAULTIDENTITY = {"ig": 0.85, "tr-ab": 1, "tr-gd": 1}
    if key is None:
        key_ = "junction_aa"  # default
    else:
        key_ = key

    if key_ not in dat.columns:
        raise ValueError("key {} not found in input table.".format(key_))

    locuses = ["ig", "tr-ab", "tr-gd"]

    locus_dat = {}
    # quick test first
    for locus in locuses:
        locus_1 = locus_dict1[locus]
        locus_2 = locus_dict2[locus]

        dat_vj = dat[dat["locus"].isin(locus_2)].copy()
        dat_vdj = dat[dat["locus"].isin(locus_1)].copy()

        if dat_vj.shape[0] > 0:
            dump = dat_vj[~(dat_vj["cell_id"].isin(dat_vdj["cell_id"]))].copy()
            if dump.shape[0] > 0:
                dat = dat[~(dat["cell_id"].isin(dump["cell_id"]))].copy()
        dat_vdj = dat[dat["locus"].isin(locus_1)].copy()
        if dat_vdj.shape[0] == 0:
            locuses.remove(locus)

    if len(locuses) > 0:
        for locusx in locuses:
            locus_1 = locus_dict1[locusx]
            locus_2 = locus_dict2[locusx]
            if isinstance(identity, dict):
                if locusx not in identity:
                    identity.update({locusx: DEFAULTIDENTITY[locusx]})
                    warnings.warn(
                        UserWarning(
                            "Identity value for {} chains ".format(locusx)
                            + "not specified in provided dictionary. "
                            + "Defaulting to {} for {} chains.".format(
                                DEFAULTIDENTITY[locusx], locusx
                            )
                        )
                    )
                identity_ = identity[locusx]
            else:
                identity_ = identity

            dat_vj = dat[dat["locus"].isin(locus_2)].copy()
            dat_vdj = dat[dat["locus"].isin(locus_1)].copy()

            if dat_vj.shape[0] > 0:
                dump = dat_vj[
                    ~(dat_vj["cell_id"].isin(dat_vdj["cell_id"]))
                ].copy()
                if dump.shape[0] > 0:
                    dat = dat[~(dat["cell_id"].isin(dump["cell_id"]))].copy()
            dat_vdj = dat[dat["locus"].isin(locus_1)].copy()
            if dat_vdj.shape[0] > 0:
                # retrieve the J genes and J genes
                if not by_alleles:
                    if "v_call_genotyped" in dat_vdj.columns:
                        V = [
                            re.sub("[*][0-9][0-9]", "", str(v))
                            for v in dat_vdj["v_call_genotyped"]
                        ]
                    else:
                        V = [
                            re.sub("[*][0-9][0-9]", "", str(v))
                            for v in dat_vdj["v_call"]
                        ]
                    J = [
                        re.sub("[*][0-9][0-9]", "", str(j))
                        for j in dat_vdj["j_call"]
                    ]
                else:
                    if "v_call_genotyped" in dat_vdj.columns:
                        V = [str(v) for v in dat_vdj["v_call_genotyped"]]
                    else:
                        V = [str(v) for v in dat_vdj["v_call"]]
                    J = [str(j) for j in dat_vdj["j_call"]]

                # collapse the alleles to just genes
                V = [",".join(list(set(v.split(",")))) for v in V]
                J = [",".join(list(set(j.split(",")))) for j in J]

                seq = dict(zip(dat_vdj.index, dat_vdj[key_]))
                if recalculate_length:
                    seq_length = [len(str(l)) for l in dat_vdj[key_]]
                else:
                    try:
                        seq_length = [l for l in dat_vdj[key_ + "_length"]]
                    except:
                        raise ValueError(
                            "{} not found in {} input table.".format(
                                key_ + "_length", locus_log1_dict[locusx]
                            )
                        )
                seq_length_dict = dict(zip(dat_vdj.index, seq_length))

                # Create a dictionary and group sequence ids with same V and J genes
                V_J = dict(zip(dat_vdj.index, zip(V, J)))
                vj_grp = defaultdict(list)
                for key, val in sorted(V_J.items()):
                    vj_grp[val].append(key)
                # and now we split the groups based on lengths of the seqs
                vj_len_grp = Tree()
                seq_grp = Tree()
                for g in vj_grp:
                    # first, obtain what's the unique lengths
                    jlen = []
                    for contig_id in vj_grp[g]:
                        jlen.append(seq_length_dict[contig_id])
                    setjlen = list(set(jlen))
                    # then for each unique length, we add the contigs to a new tree if it matches the length
                    # and also the actual seq sequences into a another one
                    for s in setjlen:
                        for contig_id in vj_grp[g]:
                            jlen_ = seq_length_dict[contig_id]
                            if jlen_ == s:
                                vj_len_grp[g][s][contig_id].value = 1
                                seq_grp[g][s][seq[contig_id]].value = 1
                                for c in [contig_id]:
                                    vj_len_grp[g][s][c] = seq[c]
                clones = Tree()
                # for each seq group, calculate the hamming distance matrix
                for g in tqdm(
                    seq_grp,
                    desc="Finding clones based on {} cell VDJ chains ".format(
                        locus_log[locusx]
                    ),
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                    disable=not verbose,
                ):
                    for l in seq_grp[g]:
                        seq_ = list(seq_grp[g][l])
                        tdarray = np.array(seq_).reshape(-1, 1)
                        d_mat = squareform(
                            pdist(tdarray, lambda x, y: hamming(x[0], y[0]))
                        )
                        # then calculate what the acceptable threshold is for each length of sequence
                        tr = math.floor(int(l) * (1 - identity_))
                        # convert diagonal and upper triangle to zeroes
                        d_mat = np.tril(d_mat)
                        np.fill_diagonal(d_mat, 0)
                        # get the coordinates/indices of seqs to match against the threshold later
                        indices_temp = []
                        indices = []
                        indices_temp = [
                            list(x) for x in np.tril_indices_from(d_mat)
                        ]
                        indices = list(zip(indices_temp[0], indices_temp[1]))
                        # if there's more than 1 contig, remove the diagonal
                        if len(indices) > 1:
                            for pairs in indices:
                                # remove diagonals
                                if pairs[0] == pairs[1]:
                                    indices.remove(pairs)
                        indices_j = []
                        # use the coordinates/indices to retrieve the seq sequences
                        for p in range(0, len(indices)):
                            a1, b1 = indices[p]
                            indices_j.append(seq_[a1])
                            indices_j.append(seq_[b1])
                        # retain only the unique sequences
                        indices_j_f = list(set(indices_j))
                        # convert the distance matrix to coordinate (source) and distance (target) and create it
                        # as a dictionary
                        source, target = d_mat.nonzero()
                        source_target = list(
                            zip(source.tolist(), target.tolist())
                        )
                        if len(source) == 0 & len(target) == 0:
                            source_target = list([(0, 0)])
                        dist = {}
                        for st in source_target:
                            dist.update({st: d_mat[st]})

                        if d_mat.shape[0] > 1:
                            seq_tmp_dict = clustering(dist, tr, seq_)
                        else:
                            seq_tmp_dict = {seq_[0]: tuple([seq_[0]])}
                        # sort the list so that clones that are larger have a smaller number
                        clones_tmp = sorted(
                            list(set(seq_tmp_dict.values())),
                            key=len,
                            reverse=True,
                        )
                        for x in range(0, len(clones_tmp)):
                            clones[g][l][x + 1] = clones_tmp[x]

                clone_dict = {}
                # now to retrieve the contig ids that are grouped together
                cid = Tree()
                for g in clones:
                    for l in clones[g]:
                        # retrieve the clone 'numbers'
                        for c in clones[g][l]:
                            grp_seq = clones[g][l][c]
                            for key, value in vj_len_grp[g][l].items():
                                if value in grp_seq:
                                    cid[g][l][c][key].value = 1
                # rename clone ids - get dictionaries step by step
                first_key = []
                for k1 in cid.keys():
                    first_key.append(k1)
                first_key = list(set(first_key))
                first_key_dict = dict(
                    zip(first_key, range(1, len(first_key) + 1))
                )
                # and now for the middle key
                for g in cid:
                    second_key = []
                    for k2 in cid[g].keys():
                        second_key.append(k2)
                    second_key = list(set(second_key))
                    second_key_dict = dict(
                        zip(second_key, range(1, len(second_key) + 1))
                    )
                    for l in cid[g]:
                        # and now for the last key
                        third_key = []
                        for k3 in cid[g][l].keys():
                            third_key.append(k3)
                        third_key = list(set(third_key))
                        third_key_dict = dict(
                            zip(third_key, range(1, len(third_key) + 1))
                        )
                        for key, value in dict(cid[g][l]).items():
                            vL = []
                            for v in value:
                                if type(v) is int:
                                    break
                                # instead of converting to another tree, i will just make it a dictionary
                                clone_dict[v] = (
                                    str(locus_log[locusx])
                                    + "_"
                                    + str(first_key_dict[g])
                                    + "_"
                                    + str(second_key_dict[l])
                                    + "_"
                                    + str(third_key_dict[key])
                                )
                # add it to the original dataframes
                dat_vdj[clone_key] = pd.Series(clone_dict)
                dat[clone_key].update(pd.Series(dat_vdj[clone_key]))

                dat_vj_c = dat[dat["locus"].isin(locus_2)].copy()
                if dat_vj_c.shape[0] != 0:
                    if not by_alleles:
                        if "v_call_genotyped" in dat_vj_c.columns:
                            Vvj = [
                                re.sub("[*][0-9][0-9]", "", str(v))
                                for v in dat_vj_c["v_call_genotyped"]
                            ]
                        else:
                            Vvj = [
                                re.sub("[*][0-9][0-9]", "", str(v))
                                for v in dat_vj_c["v_call"]
                            ]
                        Jvj = [
                            re.sub("[*][0-9][0-9]", "", str(j))
                            for j in dat_vj_c["j_call"]
                        ]
                    else:
                        if "v_call_genotyped" in dat_vj_c.columns:
                            Vvj = [str(v) for v in dat_vj_c["v_call_genotyped"]]
                        else:
                            Vvj = [str(v) for v in dat_vj_c["v_call"]]
                        Jvj = [str(j) for j in dat_vj_c["j_call"]]
                    # collapse the alleles to just genes
                    Vvj = [",".join(list(set(v.split(",")))) for v in Vvj]
                    Jvj = [",".join(list(set(j.split(",")))) for j in Jvj]
                    seq = dict(zip(dat_vj_c.index, dat_vj_c[key_]))
                    if recalculate_length:
                        seq_length = [len(str(l)) for l in dat_vj_c[key_]]
                    else:
                        try:
                            seq_length = [
                                len(str(l)) for l in dat_vj_c[key_ + "_length"]
                            ]
                        except:
                            raise ValueError(
                                "{} not found in {} input table.".format(
                                    key_ + "_length", locus_log2_dict[locusx]
                                )
                            )
                    seq_length_dict = dict(zip(dat_vj_c.index, seq_length))
                    # Create a dictionary and group sequence ids with same V and J genes
                    V_Jvj = dict(zip(dat_vj_c.index, zip(Vvj, Jvj)))
                    vj_lightgrp = defaultdict(list)
                    for key, val in sorted(V_Jvj.items()):
                        vj_lightgrp[val].append(key)
                    # and now we split the groups based on lengths of the seqs
                    vj_len_lightgrp = Tree()
                    seq_lightgrp = Tree()
                    for g in vj_lightgrp:
                        # first, obtain what's the unique lengths
                        jlen = []
                        for contig_id in vj_lightgrp[g]:
                            jlen.append(seq_length_dict[contig_id])
                        setjlen = list(set(jlen))
                        # then for each unique length, we add the contigs to a new tree if it matches the length
                        # and also the actual seq sequences into a another one
                        for s in setjlen:
                            for contig_id in vj_lightgrp[g]:
                                jlen_ = seq_length_dict[contig_id]
                                if jlen_ == s:
                                    vj_len_lightgrp[g][s][contig_id].value = 1
                                    seq_lightgrp[g][s][seq[contig_id]].value = 1
                                    for c in [contig_id]:
                                        vj_len_lightgrp[g][s][c] = seq[c]
                    clones_light = Tree()
                    for g in seq_lightgrp:
                        for l in seq_lightgrp[g]:
                            seq_ = list(seq_lightgrp[g][l])
                            tdarray = np.array(seq_).reshape(-1, 1)
                            d_mat = squareform(
                                pdist(tdarray, lambda x, y: hamming(x[0], y[0]))
                            )
                            # then calculate what the acceptable threshold is for each length of sequence
                            tr = math.floor(int(l) * (1 - identity_))
                            d_mat = np.tril(d_mat)
                            np.fill_diagonal(d_mat, 0)
                            # convert diagonal and upper triangle to zeroes
                            indices_temp = []
                            indices = []
                            indices_temp = [
                                list(x) for x in np.tril_indices_from(d_mat)
                            ]
                            # get the coordinates/indices of seqs to match against the threshold later
                            indices = list(
                                zip(indices_temp[0], indices_temp[1])
                            )
                            # if there's more than 1 contig, remove the diagonal
                            if len(indices) > 1:
                                for pairs in indices:
                                    if pairs[0] == pairs[1]:
                                        indices.remove(pairs)
                            indices_j = []
                            # use the coordinates/indices to retrieve the seq sequences
                            for p in range(0, len(indices)):
                                a1, b1 = indices[p]
                                indices_j.append(seq_[a1])
                                indices_j.append(seq_[b1])
                            # retain only the unique sequences
                            indices_j_f = list(set(indices_j))
                            # convert the distance matrix to coordinate (source) and distance (target)
                            # and create it as a dictionary
                            source, target = d_mat.nonzero()
                            source_target = list(
                                zip(source.tolist(), target.tolist())
                            )
                            if len(source) == 0 & len(target) == 0:
                                source_target = list([(0, 0)])
                            dist = {}
                            for st in source_target:
                                dist.update({st: d_mat[st]})

                            if d_mat.shape[0] > 1:
                                seq_tmp_dict_l = clustering(dist, tr, seq_)
                            else:
                                seq_tmp_dict_l = {seq_[0]: tuple([seq_[0]])}
                            # sort the list so that clones that are larger have a smaller number
                            clones_tmp_l = sorted(
                                list(set(seq_tmp_dict_l.values())),
                                key=len,
                                reverse=True,
                            )
                            for x in range(0, len(clones_tmp_l)):
                                clones_light[g][l][x + 1] = clones_tmp_l[x]

                    clone_dict_light = {}
                    # now to retrieve the contig ids that are grouped together
                    cid_light = Tree()
                    for g in clones_light:
                        for l in clones_light[g]:
                            # retrieve the clone 'numbers'
                            for c in clones_light[g][l]:
                                grp_seq = clones_light[g][l][c]
                                for key, value in vj_len_lightgrp[g][l].items():
                                    if value in grp_seq:
                                        cid_light[g][l][c][key].value = 1
                    # rename clone ids - get dictionaries step by step
                    first_key = []
                    for k1 in cid_light.keys():
                        first_key.append(k1)
                    first_key = list(set(first_key))
                    first_key_dict = dict(
                        zip(first_key, range(1, len(first_key) + 1))
                    )
                    # and now for the middle key
                    for g in cid_light:
                        second_key = []
                        for k2 in cid_light[g].keys():
                            second_key.append(k2)
                        second_key = list(set(second_key))
                        second_key_dict = dict(
                            zip(second_key, range(1, len(second_key) + 1))
                        )
                        for l in cid_light[g]:
                            # and now for the last key
                            third_key = []
                            for k3 in cid_light[g][l].keys():
                                third_key.append(k3)
                            third_key = list(set(third_key))
                            third_key_dict = dict(
                                zip(third_key, range(1, len(third_key) + 1))
                            )
                            for key, value in dict(cid_light[g][l]).items():
                                vL = []
                                for v in value:
                                    if type(v) is int:
                                        break
                                    # instead of converting to another tree, i will just make it a dictionary
                                    clone_dict_light[v] = (
                                        str(first_key_dict[g])
                                        + "_"
                                        + str(second_key_dict[l])
                                        + "_"
                                        + str(third_key_dict[key])
                                    )
                    lclones = list(clone_dict_light.values())
                    renamed_clone_dict_light = {}
                    if collapse_label:
                        # will just update the main dat directly
                        if len(list(set(lclones))) > 1:
                            lclones_dict = dict(
                                zip(
                                    sorted(list(set(lclones))),
                                    [
                                        str(x)
                                        for x in range(
                                            1, len(list(set(lclones))) + 1
                                        )
                                    ],
                                )
                            )
                        else:
                            lclones_dict = dict(
                                zip(sorted(list(set(lclones))), str(1))
                            )
                        for key, value in clone_dict_light.items():
                            renamed_clone_dict_light[key] = lclones_dict[value]
                    else:
                        for key, value in clone_dict_light.items():
                            renamed_clone_dict_light[key] = value

                    cellclonetree = Tree()
                    seqcellclonetree = Tree()
                    for c, s, z in zip(
                        dat["cell_id"], dat["sequence_id"], dat[clone_key]
                    ):
                        seqcellclonetree[c][s].value = 1
                        if pd.notnull(z):
                            cellclonetree[c][z].value = 1

                    for c in cellclonetree:
                        cellclonetree[c] = list(cellclonetree[c])

                    fintree = Tree()
                    for c in tqdm(
                        cellclonetree,
                        desc="Refining clone assignment based on VJ chain pairing ",
                        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                        disable=not verbose,
                    ):
                        suffix = [
                            renamed_clone_dict_light[x]
                            for x in seqcellclonetree[c]
                            if x in renamed_clone_dict_light
                        ]
                        fintree[c] = []
                        if len(suffix) > 1:
                            for cl in cellclonetree[c]:
                                if present(cl):
                                    for s in suffix:
                                        fintree[c].append(cl + "_" + s)
                            # fintree[c] = fintree[c]
                        else:
                            for cl in cellclonetree[c]:
                                if present(cl):
                                    if len(suffix) > 0:
                                        fintree[c].append(
                                            cl + "_" + "".join(suffix)
                                        )
                                    else:
                                        fintree[c].append(cl)
                        fintree[c] = "|".join(fintree[c])
                    dat[clone_key] = [fintree[x] for x in dat["cell_id"]]

                dat_[clone_key].update(pd.Series(dat[clone_key]))
    # dat_[clone_key].replace('', 'unassigned')
    if os.path.isfile(str(vdj_data)):
        write_airr(
            dat_,
            "{}/{}_clone.tsv".format(
                os.path.dirname(vdj_data),
                os.path.basename(vdj_data).split(".tsv")[0],
            ),
        )

    logg.info(
        " finished",
        time=start,
        deep=(
            "Updated Dandelion object: \n"
            "   'data', contig-indexed clone table\n"
            "   'metadata', cell-indexed clone table\n"
        ),
    )
    if isinstance(vdj_data, Dandelion):
        if vdj_data.germline is not None:
            germline_ = vdj_data.germline
        else:
            germline_ = None
        if vdj_data.layout is not None:
            layout_ = vdj_data.layout
        else:
            layout_ = None
        if vdj_data.graph is not None:
            graph_ = vdj_data.graph
        else:
            graph_ = None
        if vdj_data.threshold is not None:
            threshold_ = vdj_data.threshold
        else:
            threshold_ = None
        if ("clone_id" in vdj_data.data.columns) and (key_added is None):
            # TODO: need to check the following bits if it works properly if only heavy chain tables are provided
            vdj_data.__init__(
                data=dat_,
                germline=germline_,
                layout=layout_,
                graph=graph_,
            )
            vdj_data.update_metadata(reinitialize=True)
        elif ("clone_id" in vdj_data.data.columns) and (key_added is not None):
            vdj_data.__init__(
                data=dat_,
                germline=germline_,
                layout=layout_,
                graph=graph_,
            )
            vdj_data.update_metadata(
                reinitialize=True,
                clone_key="clone_id",
                retrieve=clone_key,
                retrieve_mode="merge and unique only",
            )
        else:
            vdj_data.__init__(
                data=dat_,
                germline=germline_,
                layout=layout_,
                graph=graph_,
                clone_key=clone_key,
            )
            vdj_data.update_metadata(reinitialize=True, clone_key=clone_key)
        vdj_data.threshold = threshold_

    else:
        out = Dandelion(
            data=dat_,
            clone_key=clone_key,
            retrieve=clone_key,
            retrieve_mode="merge and unique only",
        )
        return out


def transfer(
    adata: AnnData,
    dandelion: Dandelion,
    expanded_only: bool = False,
    neighbors_key: Optional[str] = None,
    rna_key: Optional[str] = None,
    vdj_key: Optional[str] = None,
    clone_key: Optional[str] = None,
    collapse_nodes: bool = False,
    overwrite: Optional[Union[bool, List[str], str]] = None,
):
    """
    Transfer data in `Dandelion` slots to `AnnData` object, updating the `.obs`, `.uns`, `.obsm` and `.obsp`slots.

    Parameters
    ----------
    adata : AnnData
        `AnnData` object.
    dandelion : Dandelion
        `Dandelion` object.
    expanded_only : bool, optional
        Whether or not to transfer the embedding with all cells with BCR (False) or only for expanded clones (True).
    neighbors_key : Optional[str], optional
        key for 'neighbors' slot in `.uns`.
    rna_key : Optional[str], optional
        prefix for stashed RNA connectivities and distances.
    vdj_key : Optional[str], optional
        prefix for stashed VDJ connectivities and distances.
    clone_key : Optional[str], optional
        column name of clone/clonotype ids. Only used for integration with scirpy.
    collapse_nodes : bool, optional
        Whether or not to transfer a cell x cell or clone x clone connectivity matrix into `.uns`. Only used for
        integration with scirpy.
    overwrite : Optional[Union[bool, List[str], str]], optional
        Whether or not to overwrite existing anndata columns. Specifying a string indicating column name or
        list of column names will overwrite that specific column(s).
    """
    start = logg.info("Transferring network")
    # always overwrite with whatever columns are in dandelion's metadata:
    for x in dandelion.metadata.columns:
        if x not in adata.obs.columns:
            adata.obs[x] = pd.Series(dandelion.metadata[x])
        elif overwrite is True:
            adata.obs[x] = pd.Series(dandelion.metadata[x])
        if type_check(dandelion.metadata, x):
            adata.obs[x].replace(np.nan, "No_contig", inplace=True)
        if adata.obs[x].dtype == "bool":
            adata.obs[x] = [str(x) for x in adata.obs[x]]

    if (overwrite is not None) and (overwrite is not True):
        if not type(overwrite) is list:
            overwrite = [overwrite]
        for ow in overwrite:
            adata.obs[ow] = pd.Series(dandelion.metadata[ow])
            if type_check(dandelion.metadata, ow):
                adata.obs[ow].replace(np.nan, "No_contig", inplace=True)

    if dandelion.graph is not None:
        if expanded_only:
            G = dandelion.graph[1]
        else:
            G = dandelion.graph[0]
        logg.info("converting matrices")
        distances = nx.to_pandas_adjacency(
            G, dtype=np.float32, weight="weight", nonedge=np.nan
        )
        df_distances = distances.reindex(
            index=adata.obs_names, columns=adata.obs_names
        )
        connectivities = nx.to_pandas_adjacency(
            G, dtype=np.float32, weight="weight", nonedge=np.nan
        )
        # convert to connectivity
        distances[~distances.isnull()] = 1 / np.exp(
            distances[~distances.isnull()]
        )
        df_connectivities = distances.reindex(
            index=adata.obs_names, columns=adata.obs_names
        )

        df_connectivities = df_connectivities.values
        df_connectivities[np.isnan(df_connectivities)] = 0
        df_distances = df_distances.values
        df_distances[np.isnan(df_distances)] = 0

        df_connectivities_ = csr_matrix(df_connectivities, dtype=np.float32)
        df_distances = csr_matrix(df_distances, dtype=np.float32)

        logg.info("Updating anndata slots")
        if neighbors_key is None:
            neighbors_key = "neighbors"
        if neighbors_key not in adata.uns:
            skip_stash = True
            adata.uns[neighbors_key] = {}

        rna_neighbors_key = "rna_" + neighbors_key
        vdj_neighbors_key = "vdj_" + neighbors_key
        if rna_neighbors_key not in adata.uns:
            adata.uns[rna_neighbors_key] = adata.uns[neighbors_key].copy()

        if rna_key is None:
            r_connectivities_key = "rna_connectivities"
            r_distances_key = "rna_distances"
        else:
            r_connectivities_key = rna_key + "_connectivitites"
            r_distances_key = rna_key + "_distances"

        if vdj_key is None:
            b_connectivities_key = "vdj_connectivities"
            b_distances_key = "vdj_distances"
        else:
            b_connectivities_key = vdj_key + "_connectivitites"
            b_distances_key = vdj_key + "_distances"

        # stash_rna_connectivities:
        if r_connectivities_key not in adata.obsp:
            if "skip_stash" not in locals():
                try:
                    adata.obsp[r_connectivities_key] = adata.obsp[
                        "connectivities"
                    ].copy()
                    adata.obsp[r_distances_key] = adata.obsp["distances"].copy()
                except:
                    adata.obsp[r_connectivities_key] = adata.uns[neighbors_key][
                        "connectivities"
                    ]
                    adata.obsp[r_distances_key] = adata.uns[neighbors_key][
                        "distances"
                    ]

        # always overwrite the bcr slots
        adata.obsp["connectivities"] = df_connectivities_.copy()
        adata.obsp["distances"] = df_distances.copy()
        adata.obsp[b_connectivities_key] = adata.obsp["connectivities"].copy()
        adata.obsp[b_distances_key] = adata.obsp["distances"].copy()

        # create the dictionary that will enable the use of scirpy's plotting.
        if clone_key is None:
            clonekey = "clone_id"
        else:
            clonekey = clone_key

        if not collapse_nodes:
            df_connectivities_[df_connectivities_.nonzero()] = 1
            cell_indices = {
                str(i): np.array([k])
                for i, k in zip(range(0, len(adata.obs_names)), adata.obs_names)
            }
        else:
            cell_indices = Tree()
            for x, y in adata.obs[clonekey].iteritems():
                if y not in [
                    "",
                    "unassigned",
                    np.nan,
                    "NaN",
                    "NA",
                    "nan",
                    "None",
                    None,
                    "none",
                ]:
                    cell_indices[y][x].value = 1
            cell_indices = {
                str(x): np.array(list(r))
                for x, r in zip(
                    range(0, len(cell_indices)), cell_indices.values()
                )
            }
            df_connectivities_ = np.zeros(
                [len(cell_indices), len(cell_indices)]
            )
            np.fill_diagonal(df_connectivities_, 1)
            df_connectivities_ = csr_matrix(df_connectivities_)

        adata.uns[clonekey] = {
            # this is a symmetrical, pairwise, sparse distance matrix of clonotypes
            # the matrix is offset by 1, i.e. 0 = no connection, 1 = distance 0
            "distances": df_connectivities_,
            # '0' refers to the row/col index in the `distances` matrix
            # (numeric index, but needs to be strbecause of h5py)
            # np.array(["cell1", "cell2"]) points to the rows in `adata.obs`
            "cell_indices": cell_indices,
        }

    tmp = adata.obs.copy()
    if dandelion.graph is not None:
        if dandelion.layout is not None:
            if expanded_only:
                coord = pd.DataFrame.from_dict(
                    dandelion.layout[1], orient="index"
                )
            else:
                coord = pd.DataFrame.from_dict(
                    dandelion.layout[0], orient="index"
                )
            for x in coord.columns:
                tmp[x] = coord[x]

            X_vdj = np.array(tmp[[0, 1]], dtype=np.float32)
            adata.obsm["X_vdj"] = X_vdj

        logg.info(
            " finished",
            time=start,
            deep=(
                "updated `.obs` with `.metadata`\n"
                "added to `.uns['"
                + neighbors_key
                + "']` and `.uns['"
                + clonekey
                + "']`\n"
                "and `.obsp`\n"
                "   'distances', clonotype-weighted adjacency matrix\n"
                "   'connectivities', clonotype-weighted adjacency matrix"
            ),
        )
    else:
        logg.info(
            " finished", time=start, deep=("updated `.obs` with `.metadata`\n")
        )


def define_clones(
    vdj_data: Union[Dandelion, pd.DataFrame, str],
    dist: Optional[float] = None,
    action: Literal["first", "set"] = "set",
    model: Literal[
        "ham",
        "aa",
        "hh_s1f",
        "hh_s5f",
        "mk_rs1nf",
        "mk_rs5nf",
        "hs1f_compat",
        "m1n_compat",
    ] = "ham",
    norm: Literal["len", "mut", "none"] = "len",
    doublets: Literal["drop", "count"] = "drop",
    fileformat: Literal["changeo", "airr"] = "airr",
    ncpu: Optional[int] = None,
    outFilePrefix: Optional[int] = None,
    key_added: Optional[int] = None,
    additional_args: List[str] = [],
) -> Dandelion:
    """
    Find clones using changeo's `DefineClones.py <https://changeo.readthedocs.io/en/stable/tools/DefineClones.html>`__.

    Only callable for BCR data at the moment.

    Parameters
    ----------
    vdj_data : Union[Dandelion, pd.DataFrame, str]
        `Dandelion` object, pandas `DataFrame` in changeo/airr format, or file path to changeo/airr file after
        clones have been determined.
    dist : Optional[float], optional
        The distance threshold for clonal grouping. If None, the value will be retrieved from the Dandelion class
        `.threshold` slot.
    action : Literal["first", "set"], optional
        Specifies how to handle multiple V(D)J assignments for initial grouping. Default is 'set'.
        The “first” action will use only the first gene listed. The “set” action will use all gene assignments and
        construct a larger gene grouping composed of any sequences sharing an assignment or linked to another sequence
        by a common assignment (similar to single-linkage).
    model : Literal["ham", "aa", "hh_s1f", "hh_s5f", "mk_rs1nf", "mk_rs5nf", "hs1f_compat", "m1n_compat", ], optional
        Specifies which substitution model to use for calculating distance between sequences. Default is 'ham'.
        The “ham” model is nucleotide Hamming distance and “aa” is amino acid Hamming distance. The “hh_s1f” and
        “hh_s5f” models are human specific single nucleotide and 5-mer content models, respectively, from Yaari et al,
        2013. The “mk_rs1nf” and “mk_rs5nf” models are mouse specific single nucleotide and 5-mer content models,
        respectively, from Cui et al, 2016. The “m1n_compat” and “hs1f_compat” models are deprecated models provided
        backwards compatibility with the “m1n” and “hs1f” models in Change-O v0.3.3 and SHazaM v0.1.4. Both 5-mer
        models should be considered experimental.
    norm : Literal["len", "mut", "none"], optional
        Specifies how to normalize distances. Default is 'len'. 'none' (do not normalize), 'len' (normalize by length),
        or 'mut' (normalize by number of mutations between sequences).
    doublets : Literal["drop", "count"], optional
        Option to control behaviour when dealing with heavy chain 'doublets'. Default is 'drop'. 'drop' will filter out
        the doublets while 'count' will retain only the highest umi count contig.
    fileformat : Literal["changeo", "airr"], optional
        Format of V(D)J file/objects. Default is 'airr'. Also accepts 'changeo'.
    ncpu : Optional[int], optional
        Number of cpus for parallelization. Default is 1, no parallelization.
    outFilePrefix : Optional[int], optional
        If specified, the out file name will have this prefix. `None` defaults to 'dandelion_define_clones'
    key_added : Optional[int], optional
        Column name to add for define_clones.
    additional_args : List[str], optional
        Additional arguments to pass to `DefineClones.py`.

    Returns
    -------
    Dandelion
        `Dandelion` object with clone_id annotated in `.data` slot and `.metadata` initialized.

    Raises
    ------
    ValueError
        if .threshold not found in `Dandelion`.
    """
    start = logg.info("Finding clones")
    if ncpu is None:
        nproc = 1
    else:
        nproc = ncpu

    if key_added is None:
        clone_key = "clone_id"
    else:
        clone_key = key_added

    if isinstance(vdj_data, Dandelion):
        dat_ = load_data(vdj_data.data)
    else:
        dat_ = load_data(vdj_data)
    if os.path.isfile(str(vdj_data)):
        dat_ = load_data(vdj_data)

    if "ambiguous" in dat_:
        dat = dat_[dat_["ambiguous"] == "F"].copy()
    else:
        dat = dat_.copy()
    dat_h = dat[dat["locus"] == "IGH"]
    dat_l = dat[dat["locus"].isin(["IGK", "IGL"])]

    if os.path.isfile(str(vdj_data)):
        tmpFolder = "{}/tmp".format(os.path.dirname(vdj_data))
        outFolder = "{}".format(os.path.dirname(vdj_data))
    else:
        import tempfile

        tmpFolder = "{}/tmp".format(tempfile.TemporaryDirectory().name)
        outFolder = "{}".format(tempfile.TemporaryDirectory().name)

    if not os.path.exists(tmpFolder):
        os.makedirs(tmpFolder)
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    if os.path.isfile(str(vdj_data)):
        h_file1 = "{}/{}_heavy-clone.tsv".format(
            tmpFolder, os.path.basename(vdj_data).split(".tsv")[0]
        )
        h_file2 = "{}/{}_heavy-clone.tsv".format(
            outFolder, os.path.basename(vdj_data).split(".tsv")[0]
        )
        l_file = "{}/{}_light.tsv".format(
            tmpFolder, os.path.basename(vdj_data).split(".tsv")[0]
        )
        outfile = "{}/{}_clone.tsv".format(
            outFolder, os.path.basename(vdj_data).split(".tsv")[0]
        )
    else:
        if outFilePrefix is not None:
            out_FilePrefix = outFilePrefix
        else:
            out_FilePrefix = "dandelion_define_clones"
        h_file1 = "{}/{}_heavy-clone.tsv".format(tmpFolder, out_FilePrefix)
        h_file2 = "{}/{}_heavy-clone.tsv".format(outFolder, out_FilePrefix)
        l_file = "{}/{}_light.tsv".format(tmpFolder, out_FilePrefix)
        outfile = "{}/{}_clone.tsv".format(outFolder, out_FilePrefix)

    write_airr(dat_h, h_file1)
    write_airr(dat_l, l_file)

    if "v_call_genotyped" in dat.columns:
        v_field = "v_call_genotyped"
    else:
        v_field = "v_call"

    if dist is None:
        if isinstance(vdj_data, Dandelion):
            if vdj_data.threshold is not None:
                dist_ = vdj_data.threshold
            else:
                raise ValueError(
                    "Threshold value in Dandelion object is None. Please run calculate_threshold first"
                )
        else:
            raise ValueError(
                "Distance value is None. Please provide a distance value (float)"
            )
    else:
        dist_ = dist

    cmd = [
        "DefineClones.py",
        "-d",
        h_file1,
        "-o",
        h_file2,
        "--act",
        action,
        "--model",
        model,
        "--norm",
        norm,
        "--dist",
        str(dist_),
        "--nproc",
        str(nproc),
        "--vf",
        v_field,
    ]
    cmd = cmd + additional_args

    def clusterLinkage(cell_series, group_series):
        """
        Return a dictionary of {cell_id : cluster_id}.

        that identifies clusters of cells by analyzing their shared
        features (group_series) using single linkage.

        Arguments:
        cell_series (iter): iter of cell ids.
        group_series (iter): iter of group ids.

        Returns:
        dict:  dictionary of {cell_id : cluster_id}.

        """
        # assign initial clusters
        # initial_dict = {cluster1: [cell1], cluster2: [cell1]}
        initial_dict = {}
        for cell, group in zip(cell_series, group_series):
            try:
                initial_dict[group].append(cell)
            except KeyError:
                initial_dict[group] = [cell]

        # naive single linkage clustering (ON^2 best case, ON^3 worst case) ...ie for cells with multiple light chains
        # cluster_dict = {cluster1: [cell1, cell2]}, 2 cells belong in same group if they share 1 light chain
        while True:
            cluster_dict = {}
            for i, group in enumerate(initial_dict.keys()):
                cluster_dict[i] = initial_dict[group]
                for cluster in cluster_dict:
                    # if initial_dict[group] and cluster_dict[cluster] share common cells, add initial_dict[group] to
                    # cluster
                    if cluster != i and any(
                        cell in initial_dict[group]
                        for cell in cluster_dict[cluster]
                    ):
                        cluster_dict[cluster] = (
                            cluster_dict[cluster] + initial_dict[group]
                        )
                        del cluster_dict[i]
                        break
            # break if clusters stop changing, otherwise restart
            if len(cluster_dict.keys()) == len(initial_dict.keys()):
                break
            else:
                initial_dict = cluster_dict.copy()

        # invert cluster_dict for return
        assign_dict = {
            cell: k for k, v in cluster_dict.items() for cell in set(v)
        }

        return assign_dict

    def _lightCluster(heavy_file, light_file, out_file, doublets, fileformat):
        """
        Split heavy chain clones based on light chains.

        Arguments:
        heavy_file (str): heavy chain input file.
        light_file (str): light chain input file.
        out_file (str): heavy chain output file.
        doublets (str): method for handling multiple heavy chains per cell. one of 'drop' or 'count'.
        format (str): file format. one of 'changeo' or 'airr'.
        """
        # Set column names
        if fileformat == "changeo":
            cell_id = "cell_id"
            clone_id = "clone_id"
            v_call = "v_call"
            j_call = "j_call"
            junction_length = "junction_length"
            umi_count = "umicount"
        elif fileformat == "airr":
            cell_id = "cell_id"
            clone_id = "clone_id"
            v_call = "v_call"
            j_call = "j_call"
            junction_length = "junction_length"
            umi_count = "duplicate_count"
        else:
            sys.exit("Invalid format %s" % fileformat)

        # read in heavy and light DFs
        heavy_df = pd.read_csv(
            heavy_file, dtype="object", na_values=["", "None", "NA"], sep="\t"
        )
        light_df = pd.read_csv(
            light_file, dtype="object", na_values=["", "None", "NA"], sep="\t"
        )

        # column checking
        expected_heavy_columns = [
            cell_id,
            clone_id,
            v_call,
            j_call,
            junction_length,
            umi_count,
        ]
        if set(expected_heavy_columns).issubset(heavy_df.columns) is False:
            raise ValueError(
                "Missing one or more columns in heavy chain file: "
                + ", ".join(expected_heavy_columns)
            )
        expected_light_columns = [
            cell_id,
            v_call,
            j_call,
            junction_length,
            umi_count,
        ]
        if set(expected_light_columns).issubset(light_df.columns) is False:
            raise ValueError(
                "Missing one or more columns in light chain file: "
                + ", ".join(expected_light_columns)
            )

        # Fix types
        try:
            heavy_df[junction_length] = heavy_df[junction_length].astype("int")
            light_df[junction_length] = light_df[junction_length].astype("int")
        except:
            heavy_df[junction_length] = heavy_df[junction_length].replace(
                np.nan, pd.NA, inplace=True
            )
            light_df[junction_length] = light_df[junction_length].replace(
                np.nan, pd.NA, inplace=True
            )
            heavy_df[junction_length] = heavy_df[junction_length].astype(
                "Int64"
            )
            light_df[junction_length] = light_df[junction_length].astype(
                "Int64"
            )

        # filter multiple heavy chains
        if doublets == "drop":
            heavy_df = heavy_df.drop_duplicates(cell_id, keep=False)
            if heavy_df.empty is True:
                raise ValueError(
                    "Empty heavy chain data, after doublets drop. Are you combining experiments "
                    "in a single file? If so, split your data into multiple files."
                )
        elif doublets == "count":
            heavy_df[umi_count] = heavy_df[umi_count].astype("int")
            heavy_df = heavy_df.groupby(cell_id, sort=False).apply(
                lambda x: x.nlargest(1, umi_count)
            )

        # transfer clone IDs from heavy chain df to light chain df
        clone_dict = {
            v[cell_id]: v[clone_id]
            for k, v in heavy_df[[clone_id, cell_id]].T.to_dict().items()
        }
        light_df = light_df.loc[
            light_df[cell_id].apply(lambda x: x in clone_dict.keys()),
        ]
        light_df[clone_id] = light_df.apply(
            lambda row: clone_dict[row[cell_id]], axis=1
        )

        # generate a "cluster_dict" of CELL:CLONE dictionary from light df  (TODO: use receptor object V/J gene names)
        cluster_dict = clusterLinkage(
            light_df[cell_id],
            light_df.apply(
                lambda row: getGene(row[v_call])
                + ","
                + getGene(row[j_call])
                + ","
                + str(row[junction_length])
                + ","
                + row[clone_id],
                axis=1,
            ),
        )

        # add assignments to heavy_df
        heavy_df = heavy_df.loc[
            heavy_df[cell_id].apply(lambda x: x in cluster_dict.keys()), :
        ]
        heavy_df[clone_id] = (
            heavy_df[clone_id]
            + "_"
            + heavy_df.apply(
                lambda row: str(cluster_dict[row[cell_id]]), axis=1
            )
        )

        # write heavy chains
        write_airr(heavy_df, out_file)
        return (heavy_df, light_df)

    logg.info("Running command: %s\n" % (" ".join(cmd)))
    run(cmd)

    h_df, l_df = _lightCluster(
        h_file2, l_file, outfile, doublets=doublets, fileformat=fileformat
    )

    h_df = load_data(h_df)
    # create a dictionary for cell_id : clone_id from h_df
    linked_clones = dict(zip(h_df["cell_id"], h_df["clone_id"]))

    # create a clone_reference
    clone_ref = list(set(h_df["clone_id"]))
    clone_ref = [c.split("_")[1] if c is not np.nan else c for c in clone_ref]
    l_df = load_data(l_df)

    for x in l_df.index:
        if l_df.loc[x, "clone_id"] in clone_ref:
            l_df.at[x, "clone_id"] = linked_clones[l_df.loc[x, "cell_id"]]
        else:
            try:
                l_df.at[x, "clone_id"] = l_df.loc[x, "cell_id"] + "_notlinked"
            except:
                pass

    cloned_ = pd.concat([h_df, l_df])
    # transfer the new clone_id to the heavy + light file
    dat_[str(clone_key)] = pd.Series(cloned_["clone_id"])
    dat_[str(clone_key)].fillna("", inplace=True)
    if isinstance(vdj_data, Dandelion):
        if vdj_data.germline is not None:
            germline_ = vdj_data.germline
        else:
            germline_ = None
        if vdj_data.layout is not None:
            layout_ = vdj_data.layout
        else:
            layout_ = None
        if vdj_data.graph is not None:
            graph_ = vdj_data.graph
        else:
            graph_ = None
        if vdj_data.threshold is not None:
            threshold_ = vdj_data.threshold
        else:
            threshold_ = None

        if ("clone_id" in vdj_data.data.columns) and (clone_key is not None):
            vdj_data.__init__(
                data=dat_,
                germline=germline_,
                layout=layout_,
                graph=graph_,
                initialize=True,
                retrieve=clone_key,
                retrieve_mode="merge and unique only",
            )
        elif ("clone_id" not in vdj_data.data.columns) and (
            clone_key is not None
        ):
            vdj_data.__init__(
                data=dat_,
                germline=germline_,
                layout=layout_,
                graph=graph_,
                initialize=True,
                clone_key=clone_key,
                retrieve=clone_key,
                retrieve_mode="merge and unique only",
            )
        else:
            vdj_data.__init__(
                data=dat_,
                germline=germline_,
                layout=layout_,
                graph=graph_,
                initialize=True,
                clone_key=clone_key,
            )
        vdj_data.threshold = threshold_
    else:
        if ("clone_id" in dat_.columns) and (clone_key is not None):
            out = Dandelion(
                data=dat_,
                retrieve=clone_key,
                retrieve_mode="merge and unique only",
            )
        elif ("clone_id" not in dat_.columns) and (clone_key is not None):
            out = Dandelion(data=dat_, clone_key=clone_key)
        else:
            out = Dandelion(data=dat_)
        return out
    logg.info(
        " finished",
        time=start,
        deep=(
            "Updated Dandelion object: \n"
            "   'data', contig-indexed clone table\n"
            "   'metadata', cell-indexed clone table\n"
        ),
    )


def clone_size(
    vdj_data: Dandelion,
    max_size: Optional[int] = None,
    clone_key: Optional[str] = None,
    key_added: Optional[str] = None,
):
    """
    Quantify size of clones.

    Parameters
    ----------
    vdj_data : Dandelion
        `Dandelion` object
    max_size : Optional[int], optional
        The maximum size before value gets clipped. If None, the value will be returned as a numerical value.
    clone_key : Optional[str], optional
        Column name specifying the clone_id column in metadata.
    key_added : Optional[str], optional
        column name where clone size is tabulated into.
    """
    start = logg.info("Quantifying clone sizes")

    metadata_ = vdj_data.metadata.copy()

    if clone_key is None:
        clonekey = "clone_id"
    else:
        clonekey = clone_key

    tmp = metadata_[str(clonekey)].str.split("|", expand=True).stack()
    tmp = tmp.reset_index(drop=False)
    tmp.columns = ["cell_id", "tmp", str(clonekey)]

    clonesize = tmp[str(clonekey)].value_counts()
    if "None" in clonesize.index:
        clonesize.drop("None", inplace=True)

    if max_size is not None:
        clonesize_ = clonesize.astype("object")
        for i in clonesize.index:
            if clonesize.loc[i] >= max_size:
                clonesize_.at[i] = ">= " + str(max_size)
        clonesize_ = clonesize_.astype("category")
    else:
        clonesize_ = clonesize.copy()

    clonesize_dict = dict(clonesize_)
    clonesize_dict.update({"None": np.nan})

    if max_size is not None:
        if key_added is None:
            vdj_data.metadata[
                str(clonekey) + "_size_max_" + str(max_size)
            ] = pd.Series(
                dict(
                    zip(
                        metadata_.index,
                        [
                            str(y) if pd.notnull(y) else str(0)
                            for y in [
                                sorted(
                                    list(
                                        set(
                                            [
                                                clonesize_dict[c_]
                                                for c_ in c.split("|")
                                            ]
                                        )
                                    ),
                                    key=lambda x: int(x.split(">= ")[1])
                                    if type(x) is str
                                    else int(x),
                                    reverse=True,
                                )[0]
                                if "|" in c
                                else clonesize_dict[c]
                                for c in metadata_[str(clonekey)]
                            ]
                        ],
                    )
                )
            )
            vdj_data.metadata[
                str(clonekey) + "_size_max_" + str(max_size)
            ] = vdj_data.metadata[
                str(clonekey) + "_size_max_" + str(max_size)
            ].astype(
                "category"
            )
        else:
            vdj_data.metadata[key_added] = pd.Series(
                dict(
                    zip(
                        metadata_.index,
                        [
                            str(y) if pd.notnull(y) else str(0)
                            for y in [
                                sorted(
                                    list(
                                        set(
                                            [
                                                clonesize_dict[c_]
                                                for c_ in c.split("|")
                                            ]
                                        )
                                    ),
                                    key=lambda x: int(x.split(">= ")[1])
                                    if type(x) is str
                                    else int(x),
                                    reverse=True,
                                )[0]
                                if "|" in c
                                else clonesize_dict[c]
                                for c in metadata_[str(clonekey)]
                            ]
                        ],
                    )
                )
            )
            vdj_data.metadata[
                str(clonekey) + "_size_max_" + str(max_size)
            ] = vdj_data.metadata[
                str(clonekey) + "_size_max_" + str(max_size)
            ].astype(
                "category"
            )
    else:
        if key_added is None:
            vdj_data.metadata[str(clonekey) + "_size"] = pd.Series(
                dict(
                    zip(
                        metadata_.index,
                        [
                            str(y) if pd.notnull(y) else str(0)
                            for y in [
                                sorted(
                                    list(
                                        set(
                                            [
                                                clonesize_dict[c_]
                                                for c_ in c.split("|")
                                            ]
                                        )
                                    ),
                                    key=lambda x: int(x.split(">= ")[1])
                                    if type(x) is str
                                    else int(x),
                                    reverse=True,
                                )[0]
                                if "|" in c
                                else clonesize_dict[c]
                                for c in metadata_[str(clonekey)]
                            ]
                        ],
                    )
                )
            )
            try:
                vdj_data.metadata[str(clonekey) + "_size"] = [
                    float(x) for x in vdj_data.metadata[str(clonekey) + "_size"]
                ]
            except:
                pass
        else:
            vdj_data.metadata[key_added] = pd.Series(
                dict(
                    zip(
                        metadata_.index,
                        [
                            str(y) if pd.notnull(y) else str(0)
                            for y in [
                                sorted(
                                    list(
                                        set(
                                            [
                                                clonesize_dict[c_]
                                                for c_ in c.split("|")
                                            ]
                                        )
                                    ),
                                    key=lambda x: int(x.split(">= ")[1])
                                    if type(x) is str
                                    else int(x),
                                    reverse=True,
                                )[0]
                                if "|" in c
                                else clonesize_dict[c]
                                for c in metadata_[str(clonekey)]
                            ]
                        ],
                    )
                )
            )
            try:
                vdj_data.metadata[key_added] = [
                    float(x) for x in vdj_data.metadata[str(clonekey) + "_size"]
                ]
            except:
                pass
    logg.info(
        " finished",
        time=start,
        deep=(
            "Updated Dandelion object: \n"
            "   'metadata', cell-indexed clone table"
        ),
    )


def clone_overlap(
    vdj_data: Union[Dandelion, AnnData],
    groupby: str,
    min_clone_size: Optional[int] = None,
    weighted_overlap: bool = False,
    clone_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    A function to tabulate clonal overlap for input as a circos-style plot.

    Parameters
    ----------
    vdj_data : Union[Dandelion, AnnData]
        `Dandelion` or `AnnData` object.
    groupby : str
        column name in obs/metadata for collapsing to columns in the clone_id x groupby data frame.
    min_clone_size : Optional[int], optional
        minimum size of clone for plotting connections. Defaults to 2 if left as None.
    weighted_overlap : bool, optional
        if True, instead of collapsing to overlap to binary, overlap will be returned as the number of cells.
        In the future, there will be the option to use something like a jaccard index.
    clone_key : Optional[str], optional
        column name for clones. `None` defaults to 'clone_id'.

    Returns
    -------
    pd.DataFrame
        clone_id x groupby overlap :class:`pandas.core.frame.DataFrame'.

    Raises
    ------
    ValueError
        if min_clone_size is 0.
    """
    start = logg.info("Finding clones")
    if isinstance(vdj_data, Dandelion):
        data = vdj_data.metadata.copy()
    elif isinstance(vdj_data, AnnData):
        data = vdj_data.obs.copy()

    if min_clone_size is None:
        min_size = 2
    else:
        min_size = int(min_clone_size)

    if clone_key is None:
        clone_ = "clone_id"
    else:
        clone_ = clone_key

    # get rid of problematic rows that appear because of category conversion?
    allgroups = list(data[groupby].unique())
    data = data[
        ~(
            data[clone_].isin(
                [np.nan, "nan", "NaN", "No_contig", "unassigned", "None", None]
            )
        )
    ]

    # prepare a summary table
    datc_ = data[clone_].str.split("|", expand=True).stack()
    datc_ = pd.DataFrame(datc_)
    datc_.reset_index(drop=False, inplace=True)
    datc_.columns = ["cell_id", "tmp", clone_]
    datc_.drop("tmp", inplace=True, axis=1)
    datc_ = datc_[
        ~(
            datc_[clone_].isin(
                [
                    "",
                    np.nan,
                    "nan",
                    "NaN",
                    "No_contig",
                    "unassigned",
                    "None",
                    None,
                ]
            )
        )
    ]
    dictg_ = dict(data[groupby])
    datc_[groupby] = [dictg_[l] for l in datc_["cell_id"]]

    overlap = pd.crosstab(datc_[clone_], datc_[groupby])
    for x in allgroups:
        if x not in overlap:
            overlap[x] = 0

    if min_size == 0:
        raise ValueError("min_size must be greater than 0.")
    if not weighted_overlap:
        if min_size > 2:
            overlap[overlap < min_size] = 0
            overlap[overlap >= min_size] = 1
        elif min_size == 2:
            overlap[overlap >= min_size] = 1

    overlap.index.name = None
    overlap.columns.name = None

    if isinstance(vdj_data, AnnData):
        vdj_data.uns["clone_overlap"] = overlap.copy()
        logg.info(
            " finished",
            time=start,
            deep=("Updated AnnData: \n" "   'uns', clone overlap table"),
        )
    else:
        return overlap


def clustering(distance_dict, threshold, sequences_dict):
    """Clustering the sequences."""
    out_dict = {}
    # find out the unique indices in this subset
    i_unique = list(set(flatten(distance_dict)))
    # for every pair of i1,i2 is their dictance smaller than the thresholdeshold?
    i_pair_d = {
        (i1, i2): distance_dict[(i1, i2)] <= threshold
        if (i1, i2) in distance_dict
        else False
        for i1, i2 in combinations(i_unique, 2)
    }
    i_pair_d.update(
        {
            (i2, i1): distance_dict[(i2, i1)] <= threshold
            if (i2, i1) in distance_dict
            else False
            for i1, i2 in combinations(i_unique, 2)
        }
    )
    # so which indices should not be part of a clone?
    canbetogether = defaultdict(list)
    for ii1, ii2 in combinations(i_unique, 2):
        if i_pair_d[(ii1, ii2)] or i_pair_d[(ii2, ii1)]:
            if (ii1, ii2) in distance_dict:
                canbetogether[ii1].append((ii1, ii2))
                canbetogether[ii2].append((ii1, ii2))
            elif (ii2, ii1) in distance_dict:
                canbetogether[ii2].append((ii2, ii1))
                canbetogether[ii1].append((ii2, ii1))
        else:
            if (ii1, ii2) or (ii2, ii1) in distance_dict:
                canbetogether[ii1].append(())
                canbetogether[ii2].append(())
    for x in canbetogether:
        canbetogether[x] = list(
            set([y for y in canbetogether[x] if len(y) > 0])
        )
    # convert the indices to sequences
    for x in canbetogether:
        if len(canbetogether[x]) > 0:
            out_dict[sequences_dict[x]] = tuple(
                sorted(
                    set(
                        list(
                            [
                                sequences_dict[y]
                                for y in flatten(canbetogether[x])
                            ]
                        )
                        + [sequences_dict[x]]
                    )
                )
            )
        else:
            out_dict[sequences_dict[x]] = tuple([sequences_dict[x]])
    return out_dict


def productive_ratio(
    adata: AnnData,
    vdj: Dandelion,
    groupby: str,
    groups: Optional[List[str]] = None,
    locus: Literal["TRB", "TRA", "TRD", "TRG", "IGH", "IGK", "IGL"] = "TRB",
):
    """
    Compute the cell-level productive/non-productive contig ratio.

    Only the contig with the highest umi count in a cell will be used for this
    tabulation.

    Returns inplace `AnnData` with `.uns['productive_ratio']`.

    Parameters
    ----------
    adata : AnnData
        AnnData object holding the cell level metadata (`.obs`).
    vdj : Dandelion
        Dandelion object holding the repertoire data (`.data`).
    groupby : str
        Name of column in `AnnData.obs` to return the row tabulations.
    groups : Optional[List[str]], optional
        Optional list of categories to return.
    locus : Literal["TRB", "TRA", "TRD", "TRG", "IGH", "IGK", "IGL"], optional
        One of the accepted locuses to perform the tabulation
    """
    start = logg.info("Tabulating productive ratio")
    vdjx = vdj[(vdj.data.cell_id.isin(adata.obs_names))].copy()
    if "ambiguous" in vdjx.data:
        tmp = vdjx[
            (vdjx.data.locus == locus) & (vdjx.data.ambiguous == "F")
        ].copy()
    else:
        tmp = vdjx[(vdjx.data.locus == locus)].copy()

    if groups is None:
        if is_categorical(adata.obs[groupby]):
            groups = list(adata.obs[groupby].cat.categories)
        else:
            groups = list(set(adata.obs[groupby]))
    df = tmp.data.drop_duplicates(subset="cell_id")
    dict_df = dict(zip(df.cell_id, df.productive))
    res = pd.DataFrame(
        columns=["productive", "non-productive", "total"],
        index=groups,
    )

    adata.obs[locus + "_productive"] = pd.Series(dict_df)
    for i in range(res.shape[0]):
        cell = res.index[i]
        res.loc[cell, "total"] = sum(adata.obs[groupby] == cell)
        if res.loc[cell, "total"] > 0:
            res.loc[cell, "productive"] = (
                sum(
                    adata.obs.loc[
                        adata.obs[groupby] == cell, locus + "_productive"
                    ].isin(["T"])
                )
                / res.loc[cell, "total"]
                * 100
            )
            res.loc[cell, "non-productive"] = (
                sum(
                    adata.obs.loc[
                        adata.obs[groupby] == cell, locus + "_productive"
                    ].isin(
                        [
                            "F",
                        ]
                    )
                )
                / res.loc[cell, "total"]
                * 100
            )
    res[groupby] = res.index
    res["productive+non-productive"] = res["productive"] + res["non-productive"]
    out = {"results": res, "locus": locus, "groupby": groupby}
    adata.uns["productive_ratio"] = out
    logg.info(
        " finished",
        time=start,
        deep=("Updated AnnData: \n" "   'uns', productive_ratio"),
    )


def vj_usage_pca(
    adata: AnnData,
    groupby: str,
    min_size: int = 20,
    mode: Literal["B", "abT", "gdT"] = "abT",
    transfer_mapping=None,
    n_comps: int = 30,
    groups: Optional[List[str]] = None,
    allowed_chain_status: Optional[List[str]] = [
        "Single pair",
        "Extra pair",
        "Extra pair-exception",
        "Orphan VDJ-exception",
    ],
    verbose=False,
    **kwargs,
) -> AnnData:
    """
    Extract productive V/J gene usage from single cell data and compute PCA.

    Parameters
    ----------
    adata : AnnData
        AnnData object holding the cell level metadata with Dandelion VDJ info transferred.
    groupby : str
        Column name in `adata.obs` to groupby as observations for PCA.
    min_size : int, optional
        Minimum cell size numbers to keep for computing the final matrix. Defaults to 20.
    mode : Literal["B", "abT", "gdT"], optional
        Mode for extract the V/J genes.
    transfer_mapping : None, optional
        If provided, the columns will be mapped to the output AnnData from the original AnnData.
    n_comps : int, optional
        Number of principal components to compute. Defaults to 30.
    groups : Optional[List[str]], optional
        If provided, only the following groups/categories will be used for computing the PCA.
    allowed_chain_status : Optional[List[str]], optional
        If provided, only the ones in this list are kept from the `chain_status` column.
        Defaults to ["Single pair", "Extra pair", "Extra pair-exception", "Orphan VDJ-exception"].
    verbose : bool, optional
        Whether to display progress
    **kwargs
        Additional keyword arguments passed to `scanpy.pp.pca`.

    Returns
    -------
    AnnData
        AnnData object with obs as groups and V/J genes as features.
    """
    start = logg.info("Computing PCA for V/J gene usage")
    if allowed_chain_status is not None:
        adata_ = adata[
            adata.obs["chain_status"].isin(allowed_chain_status)
        ].copy()

    if groups is not None:
        adata_ = adata_[adata_.obs[group].isin(groups)].copy()

    if "v_call_genotyped_VDJ" in adata_.obs:
        v_call = "v_call_genotyped_"
    else:
        v_call = "v_call_"

    # prep data
    adata_.obs[v_call + mode + "_VJ_main"] = [
        x.split("|")[0] for x in adata_.obs[v_call + mode + "_VJ"]
    ]
    adata_.obs["j_call_" + mode + "_VJ_main"] = [
        x.split("|")[0] for x in adata_.obs["j_call_" + mode + "_VJ"]
    ]
    adata_.obs[v_call + mode + "_VDJ_main"] = [
        x.split("|")[0] for x in adata_.obs[v_call + mode + "_VDJ"]
    ]
    adata_.obs["j_call_" + mode + "_VDJ_main"] = [
        x.split("|")[0] for x in adata_.obs["j_call_" + mode + "_VDJ"]
    ]

    df1 = pd.DataFrame(
        {
            groupby: Counter(adata_.obs[groupby]).keys(),
            "cellcount": Counter(adata_.obs[groupby]).values(),
        }
    )
    new_list = df1.loc[df1["cellcount"] >= min_size, groupby]

    vj_v_list = [
        x
        for x in list(set(adata_.obs[v_call + mode + "_VJ_main"]))
        if x not in ["None", "No_contig"]
    ]
    vj_j_list = [
        x
        for x in list(set(adata_.obs["j_call_" + mode + "_VJ_main"]))
        if x not in ["None", "No_contig"]
    ]
    vdj_v_list = [
        x
        for x in list(set(adata_.obs[v_call + mode + "_VDJ_main"]))
        if x not in ["None", "No_contig"]
    ]
    vdj_j_list = [
        x
        for x in list(set(adata_.obs["j_call_" + mode + "_VDJ_main"]))
        if x not in ["None", "No_contig"]
    ]

    new_list = df1.loc[
        df1["cellcount"] > min_size, groupby
    ]  # smp_celltype of at least 20 cells
    vdj_list = vj_v_list + vj_j_list + vdj_v_list + vdj_j_list

    vdj_df = pd.DataFrame(columns=vdj_list, index=new_list)
    # this is a bit slow
    for i in tqdm(
        range(vdj_df.shape[0]),
        desc="Tabulating V/J gene usage",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        disable=not verbose,
    ):
        cell = vdj_df.index[i]
        counter1 = Counter(
            adata_.obs.loc[adata_.obs[groupby] == cell, v_call + mode + "_VJ"]
        )
        for vj_v in vj_v_list:
            vdj_df.loc[cell, vj_v] = counter1[vj_v]

        counter2 = Counter(
            adata_.obs.loc[
                adata_.obs[groupby] == cell, "j_call_" + mode + "_VJ"
            ]
        )
        for vj_j in vj_j_list:
            vdj_df.loc[cell, vj_j] = counter2[vj_j]

        counter3 = Counter(
            adata_.obs.loc[adata_.obs[groupby] == cell, v_call + mode + "_VDJ"]
        )
        for vdj_v in vdj_v_list:
            vdj_df.loc[cell, vdj_v] = counter3[vdj_v]
        counter5 = Counter(
            adata_.obs.loc[
                adata_.obs[groupby] == cell, "j_call_" + mode + "_VDJ"
            ]
        )
        for vdj_j in vdj_j_list:
            vdj_df.loc[cell, vdj_j] = counter5[vdj_j]
        # normalise
        vdj_df.loc[cell, vdj_df.columns.isin(vj_v_list)] = (
            vdj_df.loc[cell, vdj_df.columns.isin(vj_v_list)]
            / np.sum(vdj_df.loc[cell, vdj_df.columns.isin(vj_v_list)])
            * 100
        )
        vdj_df.loc[cell, vdj_df.columns.isin(vj_j_list)] = (
            vdj_df.loc[cell, vdj_df.columns.isin(vj_j_list)]
            / np.sum(vdj_df.loc[cell, vdj_df.columns.isin(vj_j_list)])
            * 100
        )
        vdj_df.loc[cell, vdj_df.columns.isin(vdj_v_list)] = (
            vdj_df.loc[cell, vdj_df.columns.isin(vdj_v_list)]
            / np.sum(vdj_df.loc[cell, vdj_df.columns.isin(vdj_v_list)])
            * 100
        )
        vdj_df.loc[cell, vdj_df.columns.isin(vdj_j_list)] = (
            vdj_df.loc[cell, vdj_df.columns.isin(vdj_j_list)]
            / np.sum(vdj_df.loc[cell, vdj_df.columns.isin(vdj_j_list)])
            * 100
        )

    df2 = pd.DataFrame(index=vdj_df.index, columns=["cell_type"])
    df2["cell_type"] = list(vdj_df.index)
    vdj_df_adata = sc.AnnData(
        X=vdj_df.values, obs=df2, var=pd.DataFrame(index=vdj_df.columns)
    )
    vdj_df_adata.obs["cell_count"] = pd.Series(
        dict(zip(df1[groupby], df1["cellcount"]))
    )
    sc.pp.pca(
        vdj_df_adata, n_comps=n_comps, use_highly_variable=False, **kwargs
    )
    if transfer_mapping is not None:
        collapsed_obs = adata_.obs.drop_duplicates(subset=groupby)
        for to in transfer_mapping:
            transfer_dict = dict(zip(collapsed_obs[groupby], collapsed_obs[to]))
            vdj_df_adata.obs[to] = [
                transfer_dict[x] for x in vdj_df_adata.obs_names
            ]
    logg.info(
        " finished",
        time=start,
        deep=("Returned AnnData: \n" "   'obsm', X_pca for V/J gene usage"),
    )
    return vdj_df_adata
