#!/usr/bin/env python
import bz2
import copy
import gzip
import h5py
import os
import re
import warnings

import _pickle as cPickle
import networkx as nx
import numpy as np
import pandas as pd

from anndata._core.index import (
    _normalize_index,
    Index,
    unpack_index,
)
from changeo.IO import readGermlines
from collections import defaultdict
from pandas.api.types import infer_dtype
from pathlib import Path
from scanpy import logging as logg
from textwrap import dedent
from tqdm import tqdm
from typing import Union, List, Dict, Optional, Tuple

from dandelion.utilities._io import *
from dandelion.utilities._utilities import *


class Dandelion:
    """
    `Dandelion` class object.

    Main class object storing input/output slots for all functions.

    Attributes
    ----------
    data : pd.DataFrame
        AIRR formatted data.
    germline : dict
        dictionary of germline gene:sequence records.
    graph : Tuple[NetworkxGraph, NetworkxGraph]
        networkx graphs for clonotype networks.
    layout : pd.DataFrame
        node positions for computed graph.
    library_type : str
        One of "tr-ab", "tr-gd", "ig".
    metadata : pd.DataFrame
        AIRR data collapsed per cell.
    n_contigs : int
        number of contigs in `.data` slot.
    n_obs : int
        number of cells in `.metadata` slot.
    querier : Query
        internal `Query` class.
    threshold : float
        threshold for `define_clones`.
    write : None
        write method.
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        metadata: Optional[pd.DataFrame] = None,
        germline: Optional[Dict] = None,
        layout: Optional[pd.DataFrame] = None,
        graph: Optional[Tuple[NetworkxGraph, NetworkxGraph]] = None,
        initialize: bool = True,
        library_type: Optional[Literal["tr-ab", "tr-gd", "ig"]] = None,
        **kwargs,
    ):
        """Init method for Dandelion.

        Parameters
        ----------
        data : Optional[pd.DataFrame], optional
            AIRR formatted data.
        metadata : Optional[pd.DataFrame], optional
            AIRR data collapsed per cell.
        germline : Optional[Dict], optional
            dictionary of germline gene:sequence records.
        layout : Optional[pd.DataFrame], optional
            node positions for computed graph.
        graph : Optional[Tuple[NetworkxGraph, NetworkxGraph]], optional
            networkx graphs for clonotype networks.
        initialize : bool, optional
            whether or not to initialize `.metadata` slot.
        library_type : Optional[Literal["tr-ab", "tr-gd", "ig"]], optional
            One of "tr-ab", "tr-gd", "ig".
        **kwargs
            passed to `Dandelion.update_metadata`.
        """
        self._data = data
        self._metadata = metadata
        self.layout = layout
        self.graph = graph
        self.threshold = None
        self.germline = {}
        self.querier = None
        self.library_type = library_type
        self.data = self._data
        self.metadata = self._metadata

        if germline is not None:
            self.germline.update(germline)

        if self.data is not None:
            if self.library_type is not None:
                acceptable = lib_type(self.library_type)
            else:
                acceptable = None

            if acceptable is not None:
                self.data = self.data[self.data.locus.isin(acceptable)].copy()

            try:
                self.data = check_travdv(self.data)
            except:
                pass
            if (
                pd.Series(["cell_id", "duplicate_count", "productive"])
                .isin(self.data.columns)
                .all()
            ):  # sort so that the productive contig with the largest umi is first
                self.data.sort_values(
                    by=["cell_id", "productive", "duplicate_count"],
                    inplace=True,
                    ascending=[True, False, False],
                )
            # self.data = sanitize_data(self.data) # this is too slow. and unnecessary at this point.
            self.n_contigs = self.data.shape[0]
            if metadata is None:
                if initialize is True:
                    self.update_metadata(**kwargs)
                try:
                    self.n_obs = self.metadata.shape[0]
                except:
                    self.n_obs = 0
            else:
                self.metadata = metadata
                self.n_obs = self.metadata.shape[0]
        else:
            self.n_contigs = 0
            self.n_obs = 0

    def _gen_repr(self, n_obs, n_contigs) -> str:
        """Report."""
        # inspire by AnnData's function
        descr = f"Dandelion class object with n_obs = {n_obs} and n_contigs = {n_contigs}"
        for attr in ["data", "metadata"]:
            try:
                keys = getattr(self, attr).keys()
            except:
                keys = []
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        if self.layout is not None:
            descr += f"\n    layout: {', '.join(['layout for '+ str(len(x)) + ' vertices' for x in (self.layout[0], self.layout[1])])}"
        if self.graph is not None:
            descr += f"\n    graph: {', '.join(['networkx graph of '+ str(len(x)) + ' vertices' for x in (self.graph[0], self.graph[1])])} "
        return descr

    def __repr__(self) -> str:
        """Report."""
        # inspire by AnnData's function
        return self._gen_repr(self.n_obs, self.n_contigs)

    def __getitem__(self, index: Index) -> "Dandelion":
        """Returns a sliced object."""
        if isinstance(index, np.ndarray):
            if len(index) == self._metadata.shape[0]:
                idx, idxtype = self._normalize_indices(
                    self._metadata.index[index]
                )
            elif len(index) == self._data.shape[0]:
                idx, idxtype = self._normalize_indices(self._data.index[index])
        else:
            idx, idxtype = self._normalize_indices(index[index].index)
        if idxtype == "metadata":
            _data = self._data[
                self._data.cell_id.isin(self._metadata.iloc[idx].index)
            ]
            _metadata = self._metadata.iloc[idx]
        elif idxtype == "data":
            _metadata = self._metadata[
                self._metadata.index.isin(self._data.iloc[idx].cell_id)
            ]
            _data = self._data.iloc[idx]
        _keep_cells = _metadata.index
        if self.layout is not None:
            _layout0 = {
                k: r for k, r in self.layout[0].items() if k in _keep_cells
            }
            _layout1 = {
                k: r for k, r in self.layout[1].items() if k in _keep_cells
            }
            _layout = (_layout0, _layout1)
        else:
            _layout = None
        if self.graph is not None:
            _g0 = self.graph[0].subgraph(_keep_cells)
            _g1 = self.graph[1].subgraph(
                [n for n in self.graph[1].nodes if n in _keep_cells]
            )
            _graph = (_g0, _g1)
        else:
            _graph = None
        return Dandelion(
            data=_data,
            metadata=_metadata,
            layout=_layout,
            graph=_graph,
        )

    @property
    def data(self) -> pd.DataFrame:
        """One-dimensional annotation of contig observations (`pd.DataFrame`)."""
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame):
        """data setter"""
        value = load_data(value)
        self._set_dim_df(value, "data")

    @property
    def data_names(self) -> pd.Index:
        """Names of observations (alias for `.data.index`)."""
        return self.data.index

    @data_names.setter
    def data_names(self, names: List[str]):
        """data names setter"""
        names = self._prep_dim_index(names, "data")
        self._set_dim_index(names, "data")

    @property
    def metadata(self) -> pd.DataFrame:
        """One-dimensional annotation of cell observations (`pd.DataFrame`)."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: pd.DataFrame):
        """metadata setter"""
        self._set_dim_df(value, "metadata")

    @property
    def metadata_names(self) -> pd.Index:
        """Names of observations (alias for `.metadata.index`)."""
        return self.metadata.index

    @metadata_names.setter
    def metadata_names(self, names: List[str]):
        """metadata names setter"""
        names = self._prep_dim_index(names, "metadata")
        self._set_dim_index(names, "metadata")

    def _normalize_indices(self, index: Index) -> Tuple[slice, str]:
        """retrieve indices"""
        return _normalize_indices(index, self.metadata_names, self.data_names)

    def _set_dim_df(self, value: pd.DataFrame, attr: str):
        """dim df setter"""
        if value is not None:
            if not isinstance(value, pd.DataFrame):
                raise ValueError(f"Can only assign pd.DataFrame to {attr}.")
            value_idx = self._prep_dim_index(value.index, attr)
            setattr(self, f"_{attr}", value)

    def _prep_dim_index(self, value, attr: str) -> pd.Index:
        """Prepares index to be uses as metadata_names or data_names for Dandelion object.
        If a pd.Index is passed, this will use a reference, otherwise a new index object is created.
        """
        if isinstance(value, pd.Index) and not isinstance(
            value.name, (str, type(None))
        ):
            raise ValueError(
                f"Dandelion expects .{attr}.index.name to be a string or None, "
                f"but you passed a name of type {type(value.name).__name__!r}"
            )
        else:
            value = pd.Index(value)
            if not isinstance(value.name, (str, type(None))):
                value.name = None
        # fmt: off
        if (
            not isinstance(value, pd.RangeIndex)
            and not infer_dtype(value) in ("string", "bytes")
        ):
            sample = list(value[: min(len(value), 5)])
            warnings.warn(dedent(
                f"""
                Dandelion expects .{attr}.index to contain strings, but got values like:
                    {sample}
                    Inferred to be: {infer_dtype(value)}
                """
                ), # noqa
                stacklevel=2,
            )
        # fmt: on
        return value

    def _set_dim_index(self, value: pd.Index, attr: str):
        """set dim index"""
        # Assumes _prep_dim_index has been run
        getattr(self, attr).index = value
        for v in getattr(self, f"{attr}m").values():
            if isinstance(v, pd.DataFrame):
                v.index = value

    def copy(self) -> "Dandelion":
        """
        Performs a deep copy of all slots in `Dandelion` class.

        Returns
        -------
        Dandelion
            a deep copy of `Dandelion` class.
        """
        return copy.deepcopy(self)

    def update_plus(
        self,
        option: Literal[
            "all",
            "sequence",
            "mutations",
            "cdr3 lengths",
            "mutations and cdr3 lengths",
        ] = "mutations and cdr3 lengths",
    ):
        """
        Retrieve additional data columns that are useful.

        Parameters
        ----------
        option : Literal["all", "sequence", "mutations", "cdr3 lengths", "mutations and cdr3 lengths", ], optional
            One of 'all', 'sequence', 'mutations', 'cdr3 lengths',
            'mutations and cdr3 lengths'
        """
        mutations_type = ["mu_count", "mu_freq"]
        mutationsdef = [
            "cdr_r",
            "cdr_s",
            "fwr_r",
            "fwr_s",
            "1_r",
            "1_s",
            "2_r",
            "2_s",
            "3_r",
            "3_s",
            "4_r",
            "4_s",
            "5_r",
            "5_s",
            "6_r",
            "6_s",
            "7_r",
            "7_s",
            "8_r",
            "8_s",
            "9_r",
            "9_s",
            "10_r",
            "10_s",
            "11_r",
            "11_s",
            "12_r",
            "12_s",
            "13_r",
            "13_s",
            "14_r",
            "14_s",
            "15_r",
            "15_s",
            "16_r",
            "16_s",
            "17_r",
            "17_s",
            "18_r",
            "18_s",
            "19_r",
            "19_s",
            "20_r",
            "20_s",
            "21_r",
            "21_s",
            "22_r",
            "22_s",
            "23_r",
            "23_s",
            "24_r",
            "24_s",
            "25_r",
            "25_s",
            "26_r",
            "26_s",
            "27_r",
            "27_s",
            "28_r",
            "28_s",
            "29_r",
            "29_s",
            "30_r",
            "30_s",
            "31_r",
            "31_s",
            "32_r",
            "32_s",
            "33_r",
            "33_s",
            "34_r",
            "34_s",
            "35_r",
            "35_s",
            "36_r",
            "36_s",
            "37_r",
            "37_s",
            "38_r",
            "38_s",
            "39_r",
            "39_s",
            "40_r",
            "40_s",
            "41_r",
            "41_s",
            "42_r",
            "42_s",
            "43_r",
            "43_s",
            "44_r",
            "44_s",
            "45_r",
            "45_s",
            "46_r",
            "46_s",
            "47_r",
            "47_s",
            "48_r",
            "48_s",
            "49_r",
            "49_s",
            "50_r",
            "50_s",
            "51_r",
            "51_s",
            "52_r",
            "52_s",
            "53_r",
            "53_s",
            "54_r",
            "54_s",
            "55_r",
            "55_s",
            "56_r",
            "56_s",
            "57_r",
            "57_s",
            "58_r",
            "58_s",
            "59_r",
            "59_s",
            "60_r",
            "60_s",
            "61_r",
            "61_s",
            "62_r",
            "62_s",
            "63_r",
            "63_s",
            "64_r",
            "64_s",
            "65_r",
            "65_s",
            "66_r",
            "66_s",
            "67_r",
            "67_s",
            "68_r",
            "68_s",
            "69_r",
            "69_s",
            "70_r",
            "70_s",
            "71_r",
            "71_s",
            "72_r",
            "72_s",
            "73_r",
            "73_s",
            "74_r",
            "74_s",
            "75_r",
            "75_s",
            "76_r",
            "76_s",
            "77_r",
            "77_s",
            "78_r",
            "78_s",
            "79_r",
            "79_s",
            "80_r",
            "80_s",
            "81_r",
            "81_s",
            "82_r",
            "82_s",
            "83_r",
            "83_s",
            "84_r",
            "84_s",
            "85_r",
            "85_s",
            "86_r",
            "86_s",
            "87_r",
            "87_s",
            "88_r",
            "88_s",
            "89_r",
            "89_s",
            "90_r",
            "90_s",
            "91_r",
            "91_s",
            "92_r",
            "92_s",
            "93_r",
            "93_s",
            "94_r",
            "94_s",
            "95_r",
            "95_s",
            "96_r",
            "96_s",
            "97_r",
            "97_s",
            "98_r",
            "98_s",
            "99_r",
            "99_s",
            "100_r",
            "100_s",
            "101_r",
            "101_s",
            "102_r",
            "102_s",
            "103_r",
            "103_s",
            "104_r",
            "104_s",
            "cdr1_r",
            "cdr1_s",
            "cdr2_r",
            "cdr2_s",
            "fwr1_r",
            "fwr1_s",
            "fwr2_r",
            "fwr2_s",
            "fwr3_r",
            "fwr3_s",
            "v_r",
            "v_s",
        ]
        mutations = [] + mutations_type
        for m in mutations_type:
            for d in mutationsdef:
                mutations.append(m + "_" + d)

        vdjlengths = [
            "junction_length",
            "junction_aa_length",
            "np1_length",
            "np2_length",
        ]
        seqinfo = [
            "sequence",
            "sequence_alignment",
            "sequence_alignment_aa",
            "junction",
            "junction_aa",
            "germline_alignment",
            "fwr1",
            "fwr1_aa",
            "fwr2",
            "fwr2_aa",
            "fwr3",
            "fwr3_aa",
            "fwr4",
            "fwr4_aa",
            "cdr1",
            "cdr1_aa",
            "cdr2",
            "cdr2_aa",
            "cdr3",
            "cdr3_aa",
            "v_sequence_alignment_aa",
            "d_sequence_alignment_aa",
            "j_sequence_alignment_aa",
        ]
        mutations = [x for x in mutations if x in self.data]
        vdjlengths = [x for x in vdjlengths if x in self.data]
        seqinfo = [x for x in seqinfo if x in self.data]

        if option == "all":
            if len(mutations) > 0:
                self.update_metadata(
                    retrieve=mutations, retrieve_mode="split and average"
                )
                self.update_metadata(
                    retrieve=mutations, retrieve_mode="average"
                )
            if len(vdjlengths) > 0:
                self.update_metadata(
                    retrieve=vdjlengths, retrieve_mode="split and average"
                )
            if len(seqinfo) > 0:
                self.update_metadata(
                    retrieve=seqinfo,
                    retrieve_mode="split and merge",
                )
        if option == "sequence":
            if len(seqinfo) > 0:
                self.update_metadata(
                    retrieve=seqinfo,
                    retrieve_mode="split and merge",
                )
        if option == "mutations":
            if len(mutations) > 0:
                self.update_metadata(
                    retrieve=mutations, retrieve_mode="split and average"
                )
                self.update_metadata(
                    retrieve=mutations, retrieve_mode="average"
                )
        if option == "cdr3 lengths":
            if len(vdjlengths) > 0:
                self.update_metadata(
                    retrieve=vdjlengths, retrieve_mode="split and average"
                )
        if option == "mutations and cdr3 lengths":
            if len(mutations) > 0:
                self.update_metadata(
                    retrieve=mutations, retrieve_mode="split and average"
                )
                self.update_metadata(
                    retrieve=mutations, retrieve_mode="average"
                )
            if len(vdjlengths) > 0:
                self.update_metadata(
                    retrieve=vdjlengths, retrieve_mode="split and average"
                )

    def store_germline_reference(
        self,
        corrected: Optional[Union[Dict[str, str], str]] = None,
        germline: Optional[str] = None,
        org: Literal["human", "mouse"] = "human",
    ):
        """
        Update germline reference with corrected sequences and store in `Dandelion` object.

        Parameters
        ----------
        corrected : Optional[Union[Dict[str, str], str]], optional
            dictionary of corrected germline sequences or file path to corrected germline sequences fasta file.
        germline : Optional[str], optional
            path to germline database folder. Defaults to `` environmental variable.
        org : Literal["human", "mouse"], optional
            organism of reference folder. Default is 'human'.

        Raises
        ------
        KeyError
            if `GERMLINE` environmental variable is not set.
        TypeError
            if incorrect germline provided.
        """
        start = logg.info("Updating germline reference")
        env = os.environ.copy()
        if germline is None:
            try:
                gml = Path(env["GERMLINE"])
            except:
                raise KeyError(
                    (
                        "Environmental variable GERMLINE must be set. Otherwise, "
                        + "please provide path to folder containing germline IGHV, IGHD, and IGHJ fasta files."
                    )
                )
            gml = gml / "imgt" / org / "vdj"
        else:
            if type(germline) is list:
                if len(germline) < 3:
                    raise TypeError(
                        (
                            "Input for germline is incorrect. Please provide path to folder containing germline IGHV, IGHD, "
                            + "and IGHJ fasta files, or individual paths to the germline IGHV, IGHD, and IGHJ fasta "
                            + "files (with .fasta extension) as a list."
                        )
                    )
                else:
                    gml = []
                    for x in germline:
                        if not x.endswith((".fasta", ".fa")):
                            raise TypeError(
                                (
                                    "Input for germline is incorrect. Please provide path to folder containing germline "
                                    + "IGHV, IGHD, and IGHJ fasta files, or individual paths to the germline IGHV, IGHD, and IGHJ fasta "
                                    + "files (with .fasta extension) as a list."
                                )
                            )
                        gml.append(x)
            elif type(germline) is not list:
                if os.path.isdir(germline):
                    germline_ = [
                        str(Path(germline, g)) for g in os.listdir(germline)
                    ]
                    if len(germline_) < 3:
                        raise TypeError(
                            (
                                "Input for germline is incorrect. Please provide path to folder containing germline IGHV, "
                                + "IGHD, and IGHJ fasta files, or individual paths to the germline IGHV, IGHD, and IGHJ "
                                + "fasta files (with .fasta extension) as a list."
                            )
                        )
                    else:
                        gml = []
                        for x in germline_:
                            if not x.endswith((".fasta", ".fa")):
                                raise TypeError(
                                    (
                                        "Input for germline is incorrect. Please provide path to folder containing germline "
                                        + "IGHV, IGHD, and IGHJ fasta files, or individual paths to the germline IGHV, IGHD, "
                                        + "and IGHJ fasta files (with .fasta extension) as a list."
                                    )
                                )
                            gml.append(x)
                elif os.path.isfile(germline) and str(germline).endswith(
                    (".fasta", ".fa")
                ):
                    gml = []
                    gml.append(germline)
                    warnings.warn(
                        "Only 1 fasta file provided to updating germline slot. Please check if this is intended.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        if type(gml) is not list:
            gml = [gml]
        
        gml = [str(g) for g in gml]

        germline_ref = readGermlines(gml)
        if corrected is not None:
            if type(corrected) is dict:
                personalized_ref_dict = corrected
            elif os.path.isfile(str(corrected)):
                personalized_ref_dict = readGermlines([corrected])
            # update with the personalized germline database
            if "personalized_ref_dict" in locals():
                germline_ref.update(personalized_ref_dict)
            else:
                raise TypeError(
                    (
                        "Input for corrected germline fasta is incorrect. Please provide path to file containing "
                        + "corrected germline fasta sequences."
                    )
                )

        self.germline.update(germline_ref)
        logg.info(
            " finished",
            time=start,
            deep=(
                "Updated Dandelion object: \n"
                "   'germline', updated germline reference\n"
            ),
        )

    def update_metadata(
        self,
        retrieve: Optional[Union[List[str], str]] = None,
        clone_key: Optional[str] = None,
        retrieve_mode: Literal[
            "split and unique only",
            "merge and unique only",
            "split and merge",
            "split and sum",
            "split and average",
            "split",
            "merge",
            "sum",
            "average",
        ] = "split and merge",
        collapse_alleles: bool = True,
        reinitialize: bool = True,
        verbose: bool = False,
        by_celltype: bool = False,
    ):
        """
        A `Dandelion` initialisation function to update and populate the `.metadata` slot.

        Parameters
        ----------
        retrieve : Optional[Union[List[str], str]], optional
            column name in `.data` slot to retrieve and update the metadata.
        clone_key : Optional[str], optional
            column name of clone id. None defaults to 'clone_id'.
        retrieve_mode : Literal["split and unique only", "merge and unique only", "split and merge", "split and sum", "split and average", "split", "merge", "sum", "average", ], optional
            one of:
                `split and unique only`
                    returns the retrieval splitted into two columns,
                    i.e. one for VDJ and one for VJ chains, separated by `|` for unique elements.
                `merge and unique only`
                    returns the retrieval merged into one column,
                    separated by `|` for unique elements.
                `split and merge`
                    returns the retrieval splitted into two columns,
                    i.e. one for VDJ and one for VJ chains, separated by `|` for every elements.
                `split`
                    returns the retrieval splitted into separate columns for each contig.
                `merge`
                    returns the retrieval merged into one columns for each contig,
                    separated by `|` for unique elements.
                `split and sum`
                    returns the retrieval sum in the VDJ and VJ columns (separately).
                `split and average`
                    returns the retrieval averaged in the VDJ and VJ columns (separately).
                `sum`
                    returns the retrieval sum into one column for all contigs.
                `average`
                    returns the retrieval averaged into one column for all contigs.
        collapse_alleles : bool, optional
            returns the V(D)J genes with allelic calls if False.
        reinitialize : bool, optional
            whether or not to reinitialize the current metadata.
            useful when updating older versions of `dandelion` to newer version.
        verbose : bool, optional
            whether to print progress.
        by_celltype : bool, optional
            whether to return the query/update by celltype.

        Raises
        ------
        KeyError
            if columns provided not found in Dandelion.data.
        ValueError
            if missing columns in Dandelion.data.
        """

        if clone_key is None:
            clonekey = "clone_id"
        else:
            clonekey = clone_key

        cols = [
            "sequence_id",
            "cell_id",
            "locus",
            "productive",
            "v_call",
            "d_call",
            "j_call",
            "c_call",
            "duplicate_count",
            "junction",
            "junction_aa",
        ]

        if "duplicate_count" not in self.data:
            raise ValueError(
                "Unable to initialize metadata due to missing keys. "
                "Please ensure either 'umi_count' or 'duplicate_count' is in the input data."
            )

        if not all([c in self.data for c in cols]):
            raise ValueError(
                "Unable to initialize metadata due to missing keys. "
                "Please ensure the input data contains all the following columns: {}".format(
                    cols
                )
            )

        if "sample_id" in self.data:
            cols = ["sample_id"] + cols

        if "v_call_genotyped" in self.data:
            cols = list(
                map(lambda x: "v_call_genotyped" if x == "v_call" else x, cols)
            )

        for c in ["sequence_id", "cell_id"]:
            cols.remove(c)

        if clonekey in self.data:
            if not all(pd.isnull(self.data[clonekey])):
                cols = [clonekey] + cols

        metadata_status = self.metadata
        if (metadata_status is None) or reinitialize:
            initialize_metadata(self, cols, clonekey, collapse_alleles)

        tmp_metadata = self.metadata.copy()

        if retrieve is not None:
            ret_dict = {}
            if type(retrieve) is str:
                retrieve = [retrieve]
            if self.querier is None:
                querier = Query(self.data)
                self.querier = querier
            else:
                if any([r not in self.querier.data for r in retrieve]):
                    querier = Query(self.data)
                    self.querier = querier
                else:
                    querier = self.querier

            if type(retrieve_mode) is str:
                retrieve_mode = [retrieve_mode]
                if len(retrieve) > len(retrieve_mode):
                    retrieve_mode = [x for x in retrieve_mode for i in retrieve]
            for ret, mode in zip(retrieve, retrieve_mode):
                ret_dict.update(
                    {
                        ret: {
                            "query": ret,
                            "retrieve_mode": mode,
                        }
                    }
                )

            vdj_gene_ret = ["v_call", "d_call", "j_call"]

            retrieve_ = defaultdict(dict)
            for k, v in ret_dict.items():
                if k in self.data.columns:
                    if by_celltype:
                        retrieve_[k] = querier.retrieve_celltype(**v)
                    else:
                        retrieve_[k] = querier.retrieve(**v)
                else:
                    raise KeyError(
                        "Cannot retrieve '%s' : Unknown column name." % k
                    )
            ret_metadata = pd.concat(retrieve_.values(), axis=1, join="inner")
            ret_metadata.dropna(axis=1, how="all", inplace=True)
            for col in ret_metadata:
                if all_missing(ret_metadata[col]):
                    ret_metadata.drop(col, axis=1, inplace=True)

            if collapse_alleles:
                for k in ret_dict.keys():
                    if k in vdj_gene_ret:
                        for c in ret_metadata:
                            if k in c:
                                ret_metadata[c] = [
                                    "|".join(
                                        [
                                            "|".join(list(set(yy.split(","))))
                                            for yy in list(
                                                set(
                                                    [
                                                        re.sub(
                                                            "[*][0-9][0-9]",
                                                            "",
                                                            tx,
                                                        )
                                                        for tx in t.split("|")
                                                    ]
                                                )
                                            )
                                        ]
                                    )
                                    for t in ret_metadata[c]
                                ]

            for r in ret_metadata:
                tmp_metadata[r] = pd.Series(ret_metadata[r])
            self.metadata = tmp_metadata.copy()

    def write_pkl(self, filename: str = "dandelion_data.pkl.pbz2", **kwargs):
        """
        Writes a `Dandelion` class to .pkl format.

        Parameters
        ----------
        filename : str, optional
            path to `.pkl` file.
        **kwargs
            passed to `_pickle`.
        """
        if isBZIP(str(filename)):
            try:
                with bz2.BZ2File(filename, "wb") as f:
                    cPickle.dump(self, f, **kwargs)
            except:
                with bz2.BZ2File(filename, "wb") as f:
                    cPickle.dump(self, f, protocol=4, **kwargs)
        elif isGZIP(str(filename)):
            try:
                with gzip.open(filename, "wb") as f:
                    cPickle.dump(self, f, **kwargs)
            except:
                with gzip.open(filename, "wb") as f:
                    cPickle.dump(self, f, protocol=4, **kwargs)
        else:
            f = open(filename, "wb")
            cPickle.dump(self, f, **kwargs)
            f.close()

    def write_airr(self, filename: str = "dandelion_airr.tsv", **kwargs):
        """
        Writes a `Dandelion` class to AIRR formatted .tsv format.

        Parameters
        ----------
        filename : str, optional
            path to `.tsv` file.
        **kwargs
            passed to `pandas.DataFrame.to_csv`.
        """
        data = sanitize_data(self.data)
        data.to_csv(filename, sep="\t", index=False, **kwargs)

    def write_h5ddl(
        self,
        filename: str = "dandelion_data.h5ddl",
        complib: Literal[
            "zlib",
            "lzo",
            "bzip2",
            "blosc",
            "blosc:blosclz",
            "blosc:lz4",
            "blosc:lz4hc",
            "blosc:snappy",
            "blosc:zlib",
            "blosc:zstd",
        ] = None,
        compression: Literal[
            "zlib",
            "lzo",
            "bzip2",
            "blosc",
            "blosc:blosclz",
            "blosc:lz4",
            "blosc:lz4hc",
            "blosc:snappy",
            "blosc:zlib",
            "blosc:zstd",
        ] = None,
        compression_level: Optional[int] = None,
        **kwargs,
    ):
        """
        Writes a `Dandelion` class to .h5ddl format.

        Parameters
        ----------
        filename : str, optional
            path to `.h5ddl` file.
        complib : Literal["zlib", "lzo", "bzip2", "blosc", "blosc:blosclz", "blosc:lz4", "blosc:lz4hc", "blosc:snappy", "blosc:zlib", "blosc:zstd", ], optional
            method for compression for data frames. see
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_hdf.html
        compression : Literal["zlib", "lzo", "bzip2", "blosc", "blosc:blosclz", "blosc:lz4", "blosc:lz4hc", "blosc:snappy", "blosc:zlib", "blosc:zstd", ], optional
            same call as complib. Just a convenience option.
        compression_level : Optional[int], optional
            Specifies a compression level for data. A value of 0 disables compression.
        **kwargs
            passed to `pd.DataFrame.to_hdf`.

        Raises
        ------
        ValueError
            if both `complib` and `compression` are specified.
        """
        if compression_level is None:
            compression_level = 9
        else:
            compression_level = compression_level

        # a little hack to overwrite the existing file?
        with h5py.File(filename, "w") as hf:
            for datasetname in hf.keys():
                del hf[datasetname]

        if complib is None and compression is None:
            comp = None
        elif complib is not None and compression is None:
            comp = complib
        elif complib is None and compression is not None:
            comp = compression
        if complib is not None and compression is not None:
            raise ValueError(
                "Please specify only complib or compression. They do the same thing."
            )

        # now to actually saving
        data = self.data.copy()
        data = sanitize_data(data)
        data = sanitize_data_for_saving(data)
        data.to_hdf(
            filename,
            "data",
            complib=comp,
            complevel=compression_level,
            **kwargs,
        )

        if self.metadata is not None:
            metadata = self.metadata.copy()
            for col in metadata.columns:
                weird = (
                    metadata[[col]].applymap(type)
                    != metadata[[col]].iloc[0].apply(type)
                ).any(axis=1)
                if len(metadata[weird]) > 0:
                    metadata[col] = metadata[col].where(
                        pd.notnull(metadata[col]), ""
                    )
            metadata.to_hdf(
                filename,
                "metadata",
                complib=comp,
                complevel=compression_level,
                format="table",
                nan_rep=np.nan,
                **kwargs,
            )

        graph_counter = 0
        try:
            for g in self.graph:
                G = nx.to_pandas_adjacency(g, nonedge=np.nan)
                G.to_hdf(
                    filename,
                    "graph/graph_" + str(graph_counter),
                    complib=comp,
                    complevel=compression_level,
                    **kwargs,
                )
                graph_counter += 1
        except:
            pass

        with h5py.File(filename, "a") as hf:
            try:
                layout_counter = 0
                for l in self.layout:
                    try:
                        hf.create_group("layout/layout_" + str(layout_counter))
                    except:
                        pass
                    for k in l.keys():
                        hf["layout/layout_" + str(layout_counter)].attrs[k] = l[
                            k
                        ]
                    layout_counter += 1
            except:
                pass

            if len(self.germline) > 0:
                try:
                    hf.create_group("germline")
                except:
                    pass
                for k in self.germline.keys():
                    hf["germline"].attrs[k] = self.germline[k]
            if self.threshold is not None:
                tr = self.threshold
                hf.create_dataset("threshold", data=tr)

    write = write_h5ddl


class Query:
    """Query class"""

    def __init__(self, data, verbose=False):
        self.data = data.copy()
        self.Cell = Tree()
        for contig, row in tqdm(
            data.iterrows(),
            desc="Setting up data",
            disable=not verbose,
        ):
            self.Cell[row["cell_id"]][contig].update(row)

    @property
    def querydtype(self):
        """Check dtype."""
        return str(self.data[self.query].dtype)

    def retrieve(self, query, retrieve_mode):
        """Retrieve query."""
        self.query = query
        ret = {}
        for cell in self.Cell:
            cols, vdj, vj = {}, [], []
            for _, contig in self.Cell[cell].items():
                if isinstance(contig, dict):
                    if contig["locus"] in ["IGH", "TRB", "TRD"]:
                        vdj.append(contig[query])
                    elif contig["locus"] in ["IGK", "IGL", "TRA", "TRG"]:
                        vj.append(contig[query])
            if retrieve_mode == "split and unique only":
                if len(vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_VDJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(vdj))
                                if present(x)
                            )
                        }
                    )
                if len(vj) > 0:
                    cols.update(
                        {
                            query
                            + "_VJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(vj))
                                if present(x)
                            )
                        }
                    )
            elif retrieve_mode == "split and merge":
                if len(vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_VDJ": "|".join(
                                str(x) for x in vdj if present(x)
                            )
                        }
                    )
                if len(vj) > 0:
                    cols.update(
                        {
                            query
                            + "_VJ": "|".join(str(x) for x in vj if present(x))
                        }
                    )
            elif retrieve_mode == "merge and unique only":
                cols.update(
                    {
                        query: "|".join(
                            str(x) for x in set(vdj + vj) if present(x)
                        )
                    }
                )
            elif retrieve_mode == "split and sum":
                if len(vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_VDJ": np.sum(
                                [float(x) for x in vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_VDJ": np.nan})
                if len(vj) > 0:
                    cols.update(
                        {
                            query
                            + "_VJ": np.sum(
                                [float(x) for x in vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_VJ": np.nan})
            elif retrieve_mode == "split and average":
                if len(vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_VDJ": np.mean(
                                [float(x) for x in vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_VDJ": np.nan})
                if len(vj) > 0:
                    cols.update(
                        {
                            query
                            + "_VJ": np.mean(
                                [float(x) for x in vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_VJ": np.nan})
            elif retrieve_mode == "merge":
                cols.update(
                    {query: "|".join(x for x in (vdj + vj) if present(x))}
                )
            elif retrieve_mode == "split":
                if len(vdj) > 0:
                    for i in range(1, len(vdj) + 1):
                        cols.update({query + "_VDJ_" + str(i): vdj[i - 1]})
                if len(vj) > 0:
                    for i in range(1, len(vj) + 1):
                        cols.update({query + "_VJ_" + str(i): vj[i - 1]})
            elif retrieve_mode == "sum":
                cols.update(
                    {query: np.sum([float(x) for x in vdj + vj if present(x)])}
                )
                if not present(cols[query]):
                    cols.update({query: np.nan})
            elif retrieve_mode == "average":
                cols.update(
                    {query: np.mean([float(x) for x in vdj + vj if present(x)])}
                )
                if not present(cols[query]):
                    cols.update({query: np.nan})
            ret.update({cell: cols})
        out = pd.DataFrame.from_dict(ret, orient="index")
        if retrieve_mode not in [
            "split and sum",
            "split and average",
            "sum",
            "average",
        ]:
            if retrieve_mode == "split":
                for x in out:
                    try:
                        out[x] = pd.to_numeric(out[x])
                    except:
                        out[x].fillna("None", inplace=True)
            else:
                out.fillna("None", inplace=True)
        return out

    def retrieve_celltype(self, query, retrieve_mode):
        """Retrieve query."""
        self.query = query
        ret = {}
        for cell in self.Cell:
            cols, abt_vdj, gdt_vdj, b_vdj, abt_vj, gdt_vj, b_vj = (
                {},
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for _, contig in self.Cell[cell].items():
                if isinstance(contig, dict):
                    if contig["locus"] in ["IGH"]:
                        b_vdj.append(contig[query])
                    elif contig["locus"] in ["IGK", "IGL"]:
                        b_vj.append(contig[query])
                    elif contig["locus"] in ["TRB"]:
                        abt_vdj.append(contig[query])
                    elif contig["locus"] in ["TRD"]:
                        gdt_vdj.append(contig[query])
                    elif contig["locus"] in ["TRA"]:
                        abt_vj.append(contig[query])
                    elif contig["locus"] in ["TRG"]:
                        gdt_vj.append(contig[query])
            if retrieve_mode == "split and unique only":
                if len(b_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VDJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(b_vdj))
                                if present(x)
                            )
                        }
                    )
                if len(b_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(b_vj))
                                if present(x)
                            )
                        }
                    )

                if len(abt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VDJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(abt_vdj))
                                if present(x)
                            )
                        }
                    )
                if len(abt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(abt_vj))
                                if present(x)
                            )
                        }
                    )

                if len(gdt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VDJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(gdt_vdj))
                                if present(x)
                            )
                        }
                    )
                if len(gdt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VJ": "|".join(
                                str(x)
                                for x in list(dict.fromkeys(gdt_vj))
                                if present(x)
                            )
                        }
                    )
            elif retrieve_mode == "split and merge":
                if len(b_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VDJ": "|".join(
                                str(x) for x in b_vdj if present(x)
                            )
                        }
                    )
                if len(b_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VJ": "|".join(
                                str(x) for x in b_vj if present(x)
                            )
                        }
                    )

                if len(abt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VDJ": "|".join(
                                str(x) for x in abt_vdj if present(x)
                            )
                        }
                    )
                if len(abt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VJ": "|".join(
                                str(x) for x in abt_vj if present(x)
                            )
                        }
                    )

                if len(gdt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VDJ": "|".join(
                                str(x) for x in gdt_vdj if present(x)
                            )
                        }
                    )
                if len(gdt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VJ": "|".join(
                                str(x) for x in gdt_vj if present(x)
                            )
                        }
                    )

            elif retrieve_mode == "merge and unique only":
                cols.update(
                    {
                        query: "|".join(
                            str(x)
                            for x in set(
                                b_vdj
                                + abt_vdj
                                + gdt_vdj
                                + b_vj
                                + abt_vj
                                + gdt_vj
                            )
                            if present(x)
                        )
                    }
                )
            elif retrieve_mode == "split and sum":
                if len(b_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VDJ": np.sum(
                                [float(x) for x in b_vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_B_VDJ": np.nan})
                if len(b_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VJ": np.sum(
                                [float(x) for x in b_vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_B_VJ": np.nan})

                if len(abt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VDJ": np.sum(
                                [float(x) for x in abt_vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_abT_VDJ": np.nan})
                if len(abt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VJ": np.sum(
                                [float(x) for x in abt_vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_abT_VJ": np.nan})

                if len(gdt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VDJ": np.sum(
                                [float(x) for x in gdt_vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_gdT_VDJ": np.nan})
                if len(gdt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VJ": np.sum(
                                [float(x) for x in gdt_vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_gdT_VJ": np.nan})
            elif retrieve_mode == "split and average":
                if len(b_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VDJ": np.mean(
                                [float(x) for x in b_vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_B_VDJ": np.nan})
                if len(b_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_B_VJ": np.mean(
                                [float(x) for x in b_vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_B_VJ": np.nan})

                if len(abt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VDJ": np.mean(
                                [float(x) for x in abt_vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_abT_VDJ": np.nan})
                if len(abt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_abT_VJ": np.mean(
                                [float(x) for x in abt_vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_abT_VJ": np.nan})

                if len(gdt_vdj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VDJ": np.mean(
                                [float(x) for x in gdt_vdj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_gdT_VDJ": np.nan})
                if len(gdt_vj) > 0:
                    cols.update(
                        {
                            query
                            + "_gdT_VJ": np.mean(
                                [float(x) for x in gdt_vj if present(x)]
                            )
                        }
                    )
                else:
                    cols.update({query + "_gdT_VJ": np.nan})
            elif retrieve_mode == "merge":
                cols.update(
                    {
                        query: "|".join(
                            x
                            for x in (
                                b_vdj
                                + abt_vdj
                                + gdt_vdj
                                + b_vj
                                + abt_vj
                                + gdt_vj
                            )
                            if present(x)
                        )
                    }
                )
            elif retrieve_mode == "split":
                if len(b_vdj) > 0:
                    for i in range(1, len(b_vdj) + 1):
                        cols.update({query + "_B_VDJ_" + str(i): b_vdj[i - 1]})
                if len(b_vj) > 0:
                    for i in range(1, len(b_vj) + 1):
                        cols.update({query + "_B_VJ_" + str(i): b_vj[i - 1]})
                if len(abt_vdj) > 0:
                    for i in range(1, len(abt_vdj) + 1):
                        cols.update(
                            {query + "_abT_VDJ_" + str(i): abt_vdj[i - 1]}
                        )
                if len(abt_vj) > 0:
                    for i in range(1, len(abt_vj) + 1):
                        cols.update(
                            {query + "_abT_VJ_" + str(i): abt_vj[i - 1]}
                        )
                if len(gdt_vdj) > 0:
                    for i in range(1, len(gdt_vdj) + 1):
                        cols.update(
                            {query + "_gdT_VDJ_" + str(i): gdt_vdj[i - 1]}
                        )
                if len(gdt_vj) > 0:
                    for i in range(1, len(gdt_vj) + 1):
                        cols.update(
                            {query + "_gdT_VJ_" + str(i): gdt_vj[i - 1]}
                        )
            elif retrieve_mode == "sum":
                cols.update(
                    {
                        query: np.sum(
                            [
                                float(x)
                                for x in b_vdj
                                + abt_vdj
                                + gdt_vdj
                                + b_vj
                                + abt_vj
                                + gdt_vj
                                if present(x)
                            ]
                        )
                    }
                )
                if not present(cols[query]):
                    cols.update({query: np.nan})
            elif retrieve_mode == "average":
                cols.update(
                    {
                        query: np.mean(
                            [
                                float(x)
                                for x in b_vdj
                                + abt_vdj
                                + gdt_vdj
                                + b_vj
                                + abt_vj
                                + gdt_vj
                                if present(x)
                            ]
                        )
                    }
                )
                if not present(cols[query]):
                    cols.update({query: np.nan})
            ret.update({cell: cols})
        out = pd.DataFrame.from_dict(ret, orient="index")
        if retrieve_mode not in [
            "split and sum",
            "split and average",
            "sum",
            "average",
        ]:
            if retrieve_mode == "split":
                for x in out:
                    try:
                        out[x] = pd.to_numeric(out[x])
                    except:
                        out[x].fillna("None", inplace=True)
            else:
                out.fillna("None", inplace=True)
        return out


def initialize_metadata(
    vdj_data, cols: List[str], clonekey: str, collapse_alleles: bool
):
    """Initialize Dandelion metadata."""
    init_dict = {}
    for col in cols:
        init_dict.update(
            {
                col: {
                    "query": col,
                    "retrieve_mode": "split and merge",
                }
            }
        )
    if clonekey in init_dict:
        init_dict.update(
            {
                clonekey: {
                    "query": clonekey,
                    "retrieve_mode": "merge and unique only",
                }
            }
        )
    if "sample_id" in init_dict:
        init_dict.update(
            {
                "sample_id": {
                    "query": "sample_id",
                    "retrieve_mode": "merge and unique only",
                }
            }
        )
    update_rearrangement_status(vdj_data)

    if "ambiguous" in vdj_data.data:
        dataq = vdj_data.data[vdj_data.data["ambiguous"] == "F"]
    else:
        dataq = vdj_data.data
    if vdj_data.querier is None:
        querier = Query(dataq)
        vdj_data.querier = querier
    else:
        if vdj_data.metadata is not None:
            if any(~vdj_data.metadata_names.isin(vdj_data.data.cell_id)):
                querier = Query(dataq)
                vdj_data.querier = querier
            else:
                querier = vdj_data.querier
        else:
            querier = vdj_data.querier

    meta_ = defaultdict(dict)
    for k, v in init_dict.copy().items():
        if all_missing(vdj_data.data[k]):
            init_dict.pop(k)
            continue
        meta_[k] = querier.retrieve(**v)
        if k in [
            "v_call",
            "v_call_genotyped",
            "d_call",
            "j_call",
            "c_call",
            "productive",
        ]:
            meta_[k + "_split"] = querier.retrieve_celltype(**v)
        if k in ["duplicate_count"]:
            v.update({"retrieve_mode": "split and sum"})
            meta_[k] = querier.retrieve_celltype(**v)
        if k in ["mu_count", "mu_freq"]:
            v.update({"retrieve_mode": "split and average"})
            meta_[k] = querier.retrieve_celltype(**v)

    tmp_metadata = pd.concat(meta_.values(), axis=1, join="inner")

    reqcols1 = [
        "locus_VDJ",
    ]
    vcall = (
        "v_call_genotyped" if "v_call_genotyped" in vdj_data.data else "v_call"
    )
    reqcols2 = [
        "locus_VJ",
        "productive_VDJ",
        "productive_VJ",
        vcall + "_VDJ",
        "d_call_VDJ",
        "j_call_VDJ",
        vcall + "_VJ",
        "j_call_VJ",
        "c_call_VDJ",
        "c_call_VJ",
        "junction_VDJ",
        "junction_VJ",
        "junction_aa_VDJ",
        "junction_aa_VJ",
        vcall + "_B_VDJ",
        "d_call_B_VDJ",
        "j_call_B_VDJ",
        vcall + "_B_VJ",
        "j_call_B_VJ",
        "c_call_B_VDJ",
        "c_call_B_VJ",
        "productive_B_VDJ",
        "productive_B_VJ",
        vcall + "_abT_VDJ",
        "d_call_abT_VDJ",
        "j_call_abT_VDJ",
        vcall + "_abT_VJ",
        "j_call_abT_VJ",
        "c_call_abT_VDJ",
        "c_call_abT_VJ",
        "productive_abT_VDJ",
        "productive_abT_VJ",
        vcall + "_gdT_VDJ",
        "d_call_gdT_VDJ",
        "j_call_gdT_VDJ",
        vcall + "_gdT_VJ",
        "j_call_gdT_VJ",
        "c_call_gdT_VDJ",
        "c_call_gdT_VJ",
        "productive_gdT_VDJ",
        "productive_gdT_VJ",
    ]
    reqcols = reqcols1 + reqcols2
    for rc in reqcols:
        if rc not in tmp_metadata:
            tmp_metadata[rc] = ""
    for dc in ["d_call_VJ", "d_call_B_VJ", "d_call_abT_VJ", "d_call_gdT_VJ"]:
        if dc in tmp_metadata:
            tmp_metadata.drop(dc, axis=1, inplace=True)

    vcalldict = {
        vcall: "v_call",
        "d_call": "d_call",
        "j_call": "j_call",
        "c_call": "c_call",
    }
    for _call in [vcall, "d_call", "j_call", "c_call"]:
        tmp_metadata[vcalldict[_call] + "_VDJ_main"] = [
            return_none_call(x) for x in tmp_metadata[_call + "_VDJ"]
        ]
        if _call != "d_call":
            tmp_metadata[vcalldict[_call] + "_VJ_main"] = [
                return_none_call(x) for x in tmp_metadata[_call + "_VJ"]
            ]

    for mode in ["B", "abT", "gdT"]:
        tmp_metadata[vcalldict[vcall] + "_" + mode + "_VDJ_main"] = [
            return_none_call(x)
            for x in tmp_metadata[vcall + "_" + mode + "_VDJ"]
        ]
        tmp_metadata["d_call_" + mode + "_VDJ_main"] = [
            return_none_call(x) for x in tmp_metadata["d_call_" + mode + "_VDJ"]
        ]
        tmp_metadata["j_call_" + mode + "_VDJ_main"] = [
            return_none_call(x) for x in tmp_metadata["j_call_" + mode + "_VDJ"]
        ]
        tmp_metadata[vcalldict[vcall] + "_" + mode + "_VJ_main"] = [
            return_none_call(x)
            for x in tmp_metadata[vcall + "_" + mode + "_VJ"]
        ]
        tmp_metadata["j_call_" + mode + "_VJ_main"] = [
            return_none_call(x) for x in tmp_metadata["j_call_" + mode + "_VJ"]
        ]

    if "locus_VDJ" in tmp_metadata:
        suffix_vdj = "_VDJ"
        suffix_vj = "_VJ"
    else:
        suffix_vdj = ""
        suffix_vj = ""

    if clonekey in init_dict:
        tmp_metadata[str(clonekey)].replace("", "None", inplace=True)

        clones = tmp_metadata[str(clonekey)].str.split("|", expand=False)
        tmpclones = []
        for i in clones:
            while "None" in i:
                i.remove("None")
                if len(i) == 1:
                    break
            tmpclones.append(i)
        tmpclones = [
            "|".join(sorted(list(set(x)), key=cmp_to_key(cmp_str_emptylast)))
            for x in tmpclones
        ]
        tmpclonesdict = dict(zip(tmp_metadata.index, tmpclones))
        tmp_metadata[str(clonekey)] = pd.Series(tmpclonesdict)
        tmp = tmp_metadata[str(clonekey)].str.split("|", expand=True).stack()
        tmp = tmp.reset_index(drop=False)
        tmp.columns = ["cell_id", "tmp", str(clonekey)]
        clone_size = tmp[str(clonekey)].value_counts()
        if "" in clone_size.index:
            clone_size = clone_size.drop("", axis=0)
        clonesize_dict = dict(clone_size)
        size_of_clone = pd.DataFrame.from_dict(clonesize_dict, orient="index")
        size_of_clone.reset_index(drop=False, inplace=True)
        size_of_clone.columns = [str(clonekey), "clone_size"]
        size_of_clone[str(clonekey) + "_by_size"] = size_of_clone.index + 1
        size_dict = dict(
            zip(
                size_of_clone[clonekey],
                size_of_clone[str(clonekey) + "_by_size"],
            )
        )
        size_dict.update({"": "None"})
        tmp_metadata[str(clonekey) + "_by_size"] = [
            "|".join(
                sorted(list(set([str(size_dict[c_]) for c_ in c.split("|")])))
            )
            if len(c.split("|")) > 1
            else str(size_dict[c])
            for c in tmp_metadata[str(clonekey)]
        ]
        tmp_metadata[str(clonekey) + "_by_size"] = tmp_metadata[
            str(clonekey) + "_by_size"
        ].astype("category")
        tmp_metadata = tmp_metadata[
            [str(clonekey), str(clonekey) + "_by_size"]
            + [
                cl
                for cl in tmp_metadata
                if cl not in [str(clonekey), str(clonekey) + "_by_size"]
            ]
        ]

    conversion_dict = {
        "igha": "IgA",
        "igha1": "IgA",
        "igha2": "IgA",
        "ighd": "IgD",
        "ighe": "IgE",
        "ighg": "IgG",
        "ighg1": "IgG",
        "ighg2": "IgG",
        "ighg3": "IgG",
        "ighg4": "IgG",
        "ighg2a": "IgG",
        "ighg2b": "IgG",
        "ighg2c": "IgG",
        "ighga": "IgG",
        "ighgb": "IgG",
        "ighgc": "IgG",
        "ighm": "IgM",
        "igkc": "IgK",
        "iglc": "IgL",
        "iglc1": "IgL",
        "iglc2": "IgL",
        "iglc3": "IgL",
        "iglc4": "IgL",
        "iglc5": "IgL",
        "iglc6": "IgL",
        "iglc7": "IgL",
        "na": "None",
        "nan": "None",
        "": "None",
        "none": "None",
        "trac": "None",
        "trbc": "None",
        "trbc1": "None",
        "trbc2": "None",
        "trdc": "None",
        "trgc": "None",
        "trgc1": "None",
        "trgc2": "None",
        "trgc3": "None",
        "trgc4": "None",
        "unassigned": "None",
        None: "None",
        np.nan: "None",
    }

    isotype = []
    multi, multic = {}, {}

    if "c_call" + suffix_vdj in tmp_metadata:
        for k in tmp_metadata["c_call" + suffix_vdj]:
            if isinstance(k, str):
                isotype.append(
                    "|".join(
                        [
                            str(z)
                            for z in [
                                conversion_dict[y.split(",")[0].lower()]
                                for y in [
                                    re.sub("[0-9]", "", x) for x in k.split("|")
                                ]
                            ]
                        ]
                    )
                )
            else:
                isotype.append("None")
        tmp_metadata["isotype"] = isotype
        tmp_metadata["isotype_status"] = [
            "IgM/IgD"
            if (i == "IgM|IgD") or (i == "IgD|IgM")
            else "Multi"
            if "|" in i
            else i
            for i in tmp_metadata["isotype"]
        ]

    vdj_gene_calls = ["v_call", "d_call", "j_call"]
    if collapse_alleles:
        for x in vdj_gene_calls:
            if x in vdj_data.data:
                for c in tmp_metadata:
                    if x in c:
                        tmp_metadata[c] = [
                            "|".join(
                                [
                                    ",".join(list(set(yy.split(","))))
                                    for yy in list(
                                        [
                                            re.sub("[*][0-9][0-9]", "", tx)
                                            for tx in t.split("|")
                                        ]
                                    )
                                ]
                            )
                            for t in tmp_metadata[c]
                        ]

    tmp_metadata["locus_status"] = format_locus(tmp_metadata)
    tmp_metadata["chain_status"] = format_chain_status(
        tmp_metadata["locus_status"]
    )

    if "isotype" in tmp_metadata:
        if all(tmp_metadata["isotype"] == "None"):
            tmp_metadata.drop(
                ["isotype", "isotype_status"], axis=1, inplace=True
            )
    for rc in reqcols:
        tmp_metadata[rc].replace("", "None", inplace=True)
    if clonekey in init_dict:
        tmp_metadata[clonekey].replace("", "None", inplace=True)

    tmp_metadata = movecol(
        tmp_metadata,
        cols_to_move=[rc2 for rc2 in reqcols2 if rc2 in tmp_metadata],
        ref_col="locus_VDJ",
    )

    for tmpm in tmp_metadata:
        if all_missing2(tmp_metadata[tmpm]):
            tmp_metadata.drop(tmpm, axis=1, inplace=True)

    tmpxregstat = querier.retrieve(
        query="rearrangement_status", retrieve_mode="split and unique only"
    )

    for x in tmpxregstat:
        tmpxregstat[x] = [
            "chimeric"
            if re.search("chimeric", y)
            else "Multi"
            if "|" in y
            else y
            for y in tmpxregstat[x]
        ]
        tmp_metadata[x] = pd.Series(tmpxregstat[x])

    tmp_metadata = movecol(
        tmp_metadata,
        cols_to_move=[
            rs
            for rs in [
                "rearrangement_status_VDJ",
                "rearrangement_status_VJ",
            ]
            if rs in tmp_metadata
        ],
        ref_col="chain_status",
    )
    # if metadata already exist, just overwrite the default columns?
    if vdj_data.metadata is not None:
        if any(~vdj_data.metadata_names.isin(vdj_data.data.cell_id)):
            vdj_data.metadata = tmp_metadata.copy()  # reindex and replace.
        else:
            for col in tmp_metadata:
                vdj_data.metadata[col] = pd.Series(tmp_metadata[col])
    else:
        vdj_data.metadata = tmp_metadata.copy()


def update_metadata(
    vdj_data: Dandelion,
    retrieve: Optional[Union[List[str], str]] = None,
    clone_key: Optional[str] = None,
    retrieve_mode: Literal[
        "split and unique only",
        "merge and unique only",
        "split and merge",
        "split and sum",
        "split and average",
        "split",
        "merge",
        "sum",
        "average",
    ] = "split and merge",
    collapse_alleles: bool = True,
    reinitialize: bool = True,
    verbose: bool = False,
    by_celltype: bool = False,
):
    """
    A `Dandelion` initialisation function to update and populate the `.metadata` slot.

    Parameters
    ----------
    vdj_data : Dandelion
        input `Dandelion` object.
    retrieve : Optional[Union[List[str], str]], optional
        column name in `.data` slot to retrieve and update the metadata.
    clone_key : Optional[str], optional
        column name of clone id. None defaults to 'clone_id'.
    retrieve_mode : Literal["split and unique only", "merge and unique only", "split and merge", "split and sum", "split and average", "split", "merge", "sum", "average", ], optional
        one of:
            `split and unique only`
                returns the retrieval splitted into two columns,
                i.e. one for VDJ and one for VJ chains, separated by `|` for unique elements.
            `merge and unique only`
                returns the retrieval merged into one column,
                separated by `|` for unique elements.
            `split and merge`
                returns the retrieval splitted into two columns,
                i.e. one for VDJ and one for VJ chains, separated by `|` for every elements.
            `split`
                returns the retrieval splitted into separate columns for each contig.
            `merge`
                returns the retrieval merged into one columns for each contig,
                separated by `|` for unique elements.
            `split and sum`
                returns the retrieval sumed in the VDJ and VJ columns (separately).
            `split and average`
                returns the retrieval averaged in the VDJ and VJ columns (separately).
            `sum`
                returns the retrieval sumed into one column for all contigs.
            `average`
                returns the retrieval averaged into one column for all contigs.
    collapse_alleles : bool, optional
        returns the V(D)J genes with allelic calls if False.
    reinitialize : bool, optional
        whether or not to reinitialize the current metadata.
        useful when updating older versions of `dandelion` to newer version.
    verbose : bool, optional
        whether to print progress.
    by_celltype : bool, optional
        whether to return the query/update by celltype.

    Raises
    ------
    KeyError
        if columns provided not found in Dandelion.data.
    ValueError
        if missing columns in Dandelion.data.
    """

    if clone_key is None:
        clonekey = "clone_id"
    else:
        clonekey = clone_key

    cols = [
        "sequence_id",
        "cell_id",
        "locus",
        "productive",
        "v_call",
        "d_call",
        "j_call",
        "c_call",
        "duplicate_count",
        "junction",
        "junction_aa",
    ]

    if "duplicate_count" not in vdj_data.data:
        raise ValueError(
            "Unable to initialize metadata due to missing keys. "
            "Please ensure either 'umi_count' or 'duplicate_count' is in the input data."
        )

    if not all([c in vdj_data.data for c in cols]):
        raise ValueError(
            "Unable to initialize metadata due to missing keys. "
            "Please ensure the input data contains all the following columns: {}".format(
                cols
            )
        )

    if "sample_id" in vdj_data.data:
        cols = ["sample_id"] + cols

    if "v_call_genotyped" in vdj_data.data:
        cols = list(
            map(lambda x: "v_call_genotyped" if x == "v_call" else x, cols)
        )

    for c in ["sequence_id", "cell_id"]:
        cols.remove(c)

    if clonekey in vdj_data.data:
        if not all(pd.isnull(vdj_data.data[clonekey])):
            cols = [clonekey] + cols

    metadata_status = vdj_data.metadata
    if (metadata_status is None) or reinitialize:
        initialize_metadata(vdj_data, cols, clonekey, collapse_alleles)

    tmp_metadata = vdj_data.metadata.copy()

    if retrieve is not None:
        ret_dict = {}
        if type(retrieve) is str:
            retrieve = [retrieve]
        if vdj_data.querier is None:
            querier = Query(vdj_data.data)
            vdj_data.querier = querier
        else:
            if any([r not in vdj_data.querier.data for r in retrieve]):
                querier = Query(vdj_data.data)
                vdj_data.querier = querier
            else:
                querier = vdj_data.querier

        if type(retrieve_mode) is str:
            retrieve_mode = [retrieve_mode]
            if len(retrieve) > len(retrieve_mode):
                retrieve_mode = [x for x in retrieve_mode for i in retrieve]
        for ret, mode in zip(retrieve, retrieve_mode):
            ret_dict.update(
                {
                    ret: {
                        "query": ret,
                        "retrieve_mode": mode,
                    }
                }
            )

        vdj_gene_ret = ["v_call", "d_call", "j_call"]

        retrieve_ = defaultdict(dict)
        for k, v in ret_dict.items():
            if k in vdj_data.data.columns:
                if by_celltype:
                    retrieve_[k] = querier.retrieve_celltype(**v)
                else:
                    retrieve_[k] = querier.retrieve(**v)
            else:
                raise KeyError(
                    "Cannot retrieve '%s' : Unknown column name." % k
                )
        ret_metadata = pd.concat(retrieve_.values(), axis=1, join="inner")
        ret_metadata.dropna(axis=1, how="all", inplace=True)
        for col in ret_metadata:
            if all_missing(ret_metadata[col]):
                ret_metadata.drop(col, axis=1, inplace=True)

        if collapse_alleles:
            for k in ret_dict.keys():
                if k in vdj_gene_ret:
                    for c in ret_metadata:
                        if k in c:
                            ret_metadata[c] = [
                                "|".join(
                                    [
                                        "|".join(list(set(yy.split(","))))
                                        for yy in list(
                                            set(
                                                [
                                                    re.sub(
                                                        "[*][0-9][0-9]", "", tx
                                                    )
                                                    for tx in t.split("|")
                                                ]
                                            )
                                        )
                                    ]
                                )
                                for t in ret_metadata[c]
                            ]

        for r in ret_metadata:
            tmp_metadata[r] = pd.Series(ret_metadata[r])
        vdj_data.metadata = tmp_metadata.copy()


def _normalize_indices(
    index: Optional[Index], names0: pd.Index, names1: pd.Index
) -> Tuple[slice, str]:
    """return indices"""
    # deal with tuples of length 1
    if isinstance(index, tuple) and len(index) == 1:
        index = index[0]
    # deal with pd.Series
    if isinstance(index, pd.Series):
        index = index.values
    if isinstance(index, tuple):
        if len(index) > 2:
            raise ValueError(
                "Dandelion can only be sliced in data or metadata rows."
            )
        # deal with pd.Series
        # TODO: The series should probably be aligned first
        if isinstance(index[1], pd.Series):
            index = index[0], index[1].values
        if isinstance(index[0], pd.Series):
            index = index[0].values, index[1]
    ax0_, _ = unpack_index(index)
    if all(ax_ in names0 for ax_ in ax0_):
        ax0 = _normalize_index(ax0_, names0)
        axtype = "metadata"
    elif all(ax_ in names1 for ax_ in ax0_):
        ax0 = _normalize_index(ax0_, names1)
        axtype = "data"
    return ax0, axtype


def return_none_call(call: str) -> str:
    """Return None if not present."""
    return call.split("|")[0] if not call in ["None", ""] else "None"
