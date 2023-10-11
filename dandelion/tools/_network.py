#!/usr/bin/env python
import networkx as nx
import numpy as np
import pandas as pd

from polyleven import levenshtein
from scanpy import logging as logg
from scipy.spatial.distance import pdist, squareform
from time import sleep
from tqdm import tqdm
from typing import Union, Tuple, Optional

try:
    from networkx.utils import np_random_state as random_state
except:
    from networkx.utils import random_state

from dandelion.utilities._core import *
from dandelion.utilities._io import *
from dandelion.utilities._utilities import *


def generate_network(
    vdj_data: Union[Dandelion, pd.DataFrame, str],
    key: Optional[str] = None,
    clone_key: Optional[str] = None,
    min_size: int = 2,
    downsample: Optional[int] = None,
    verbose: bool = True,
    compute_layout: bool = True,
    layout_method: Literal["sfdp", "mod_fr"] = "sfdp",
    **kwargs,
) -> Dandelion:
    """
    Generate a Levenshtein distance network based on full length VDJ sequence alignments for heavy and light chain(s).

    The distance matrices are then combined into a singular matrix.

    Parameters
    ----------
    vdj_data : Union[Dandelion, pd.DataFrame, str]
        `Dandelion` object, pandas `DataFrame` in changeo/airr format, or file path to changeo/airr file after clones
        have been determined.
    key : Optional[str], optional
        column name for distance calculations. None defaults to 'sequence_alignment_aa'.
    clone_key : Optional[str], optional
        column name to build network on.
    min_size : int, optional
        For visualization purposes, two graphs are created where one contains all cells and a trimmed second graph.
        This value specifies the minimum number of edges required otherwise node will be trimmed in the secondary graph.
    downsample : Optional[int], optional
        whether or not to downsample the number of cells prior to construction of network. If provided, cells will be
        randomly sampled to the integer provided. A new Dandelion class will be returned.
    verbose : bool, optional
        whether or not to print the progress bars.
    compute_layout : bool, optional
        whether or not to generate the layout. May be time consuming if too many cells.
    layout_method : Literal["sfdp", "mod_fr"], optional
        accepts one of 'sfdp' or 'mod_fr'. 'sfdp' refers to `sfdp_layout` from `graph_tool` (C++ implementation; fast)
        whereas 'mod_fr' refers to modified Fruchterman-Reingold layout originally implemented in dandelion (python
        implementation; slow).
    **kwargs
        additional kwargs passed to options specified in `networkx.drawing.layout.spring_layout` or
        `graph_tool.draw.sfdp_layout`.

    Returns
    -------
    Dandelion
        `Dandelion` object with `.edges`, `.layout`, `.graph` initialized.

    Raises
    ------
    ValueError
        if any errors with dandelion input.

    """
    start = logg.info("Generating network")

    if isinstance(vdj_data, Dandelion):
        dat = load_data(vdj_data.data)
        if "ambiguous" in vdj_data.data:
            dat = dat[dat["ambiguous"] == "F"].copy()
    else:
        dat = load_data(data)

    if key is None:
        key_ = "sequence_alignment_aa"  # default
    else:
        key_ = key

    if key_ not in dat:
        raise ValueError("key {} not found in input table.".format(key_))

    if clone_key is None:
        clonekey = "clone_id"
    else:
        clonekey = clone_key
    if clonekey not in dat:
        raise ValueError(
            "Data does not contain clone information. Please run find_clones."
        )

    # calculate distance

    if downsample is not None:
        # if downsample >= dat_h.shape[0]:
        if downsample >= vdj_data.metadata.shape[0]:
            logg.info(
                "Cannot downsample to {} cells. Using all {} cells.".format(
                    str(downsample), vdj_data.metadata.shape[0]
                )
            )
        else:
            logg.info("Downsampling to {} cells.".format(str(downsample)))
            dat_h = dat[np.in1d(dat["locus"], ["IGH", "TRB", "TRD"])]
            dat_l = dat[np.in1d(dat["locus"], ["IGK", "IGL", "TRA", "TRG"])]

            dat_h = dat_h.sample(downsample)
            dat_l = dat_l[dat_l["cell_id"].isin(list(dat_h["cell_id"]))]
            dat_ = pd.concat([dat_h, dat_l], ignore_index=True)
            dat_ = sanitize_data(dat_, ignore=clonekey)
    else:
        dat_ = sanitize_data(dat, ignore=clonekey)

    querier = Query(dat_, verbose=verbose)
    dat_seq = querier.retrieve(query=key_, retrieve_mode="split")
    dat_seq.columns = [re.sub(key_ + "_", "", i) for i in dat_seq.columns]
    dat_clone = querier.retrieve(
        query=clonekey, retrieve_mode="merge and unique only"
    )

    dat_clone = dat_clone[clonekey].str.split("|", expand=True)
    membership = Tree()
    for i, j in dat_clone.iterrows():
        jjj = [jj for jj in j if present(jj)]
        for ij in jjj:
            membership[ij][i].value = 1
    membership = {i: list(j) for i, j in dict(membership).items()}
    tmp_ = np.zeros((dat_seq.shape[0], dat_seq.shape[0]))
    dict_ = {i: {j: tmp_[i, j] for j in range(dat_seq.shape[0])} for i in range(dat_seq.shape[0])}

    # df = pd.DataFrame(tmp_)
    # df.index = dat_seq.index
    # df.columns = dat_seq.index
    # Equivalent of df.index
    index = list(dict_.keys())

    # Equivalent of df.columns
    columns = list(dict_[index[0]].keys())
    dmat = Tree()
    for t in tqdm(
        membership,
        desc="Calculating distances... ",
        disable=not verbose,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        tmp = dat_seq.loc[membership[t]]
        if tmp.shape[0] > 1:
            tmp = tmp.replace(
                "[.]", "", regex=True
            )  # replace gaps before calculating distances
            for x in tmp.columns:
                tdarray = np.array(np.array(tmp[x])).reshape(-1, 1)
                d_mat_tmp = squareform(
                    pdist(
                        tdarray,
                        lambda x, y: levenshtein(x[0], y[0])
                        if (x[0] == x[0]) and (y[0] == y[0])
                        else 0,
                    )
                )
                dmat[x][t] = pd.DataFrame(
                    d_mat_tmp, index=tmp.index, columns=tmp.index
                )
    if len(dmat) > 0:
        for x in dmat:
            dmat[x] = pd.concat(dmat[x])
            dmat[x] = dmat[x].droplevel(level=0)
            if any(dmat[x].index.duplicated()):
                tmp_dmat = dmat[x].copy()
                dup_indices = tmp_dmat.index[tmp_dmat.index.duplicated()]
                tmp_dmatx = tmp_dmat.drop(dup_indices)
                for di in list(set(dup_indices)):
                    _tmpdmat = tmp_dmat.loc[di]
                    _tmpdmat = _tmpdmat.apply(lambda r: sum_col(r), axis=0)
                    tmp_dmatx = pd.concat(
                        [tmp_dmatx, pd.DataFrame(_tmpdmat, columns=[di]).T]
                    )
                dmat[x] = tmp_dmatx.copy()
            dmat[x] = dmat[x].reindex(index=dat_seq.index, columns=dat_seq.index)
            dmat[x] = dmat[x].values

        dist_mat_list = [dmat[x] for x in dmat if type(dmat[x]) is np.ndarray]

        total_dist = np.sum(dist_mat_list, axis=0)
        np.fill_diagonal(total_dist, np.nan)

        # free up memory
        del dmat
        del dist_mat_list

        # generate edge list
        if isinstance(vdj_data, Dandelion):
            out = vdj_data.copy()
            if downsample is not None:
                out = Dandelion(dat_)
        else:  # re-initiate a Dandelion class object
            out = Dandelion(dat_)

        tmp_totaldist = pd.DataFrame(
            total_dist, index=dat_seq.index, columns=dat_seq.index
        )
        tmp_clusterdist = Tree()
        overlap = []
        for i in out.metadata.index:
            if len(out.metadata.loc[i, str(clonekey)].split("|")) > 1:
                overlap.append(
                    [
                        c
                        for c in out.metadata.loc[i, str(clonekey)].split("|")
                        if c != "None"
                    ]
                )
                for c in out.metadata.loc[i, str(clonekey)].split("|"):
                    if c != "None":
                        tmp_clusterdist[c][i].value = 1
            else:
                cx = out.metadata.loc[i, str(clonekey)]
                if cx != "None":
                    tmp_clusterdist[cx][i].value = 1
        tmp_clusterdist2 = {}
        for x in tmp_clusterdist:
            tmp_clusterdist2[x] = list(tmp_clusterdist[x])
        cluster_dist = {}
        for c_ in tmp_clusterdist2:
            if c_ in list(flatten(overlap)):
                for ol in overlap:
                    if c_ in ol:
                        idx = list(
                            set(flatten([tmp_clusterdist2[c_x] for c_x in ol]))
                        )
                        if len(list(set(idx))) > 1:
                            dist_mat_ = tmp_totaldist.loc[idx, idx]
                            s1, s2 = dist_mat_.shape
                            if s1 > 1 and s2 > 1:
                                cluster_dist["|".join(ol)] = dist_mat_
            else:
                dist_mat_ = tmp_totaldist.loc[
                    tmp_clusterdist2[c_], tmp_clusterdist2[c_]
                ]
                s1, s2 = dist_mat_.shape
                if s1 > 1 and s2 > 1:
                    cluster_dist[c_] = dist_mat_

        # to improve the visulisation and plotting efficiency, i will build a minimum spanning tree for
        # each group/clone to connect the shortest path
        mst_tree = mst(cluster_dist)

        edge_list = Tree()
        for c in tqdm(
            mst_tree,
            desc="Generating edge list ",
            disable=not verbose,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ):
            edge_list[c] = nx.to_pandas_edgelist(mst_tree[c])
            if edge_list[c].shape[0] > 0:
                edge_list[c]["weight"] = edge_list[c]["weight"] - 1
                edge_list[c]["weight"][
                    edge_list[c]["weight"] < 0
                ] = 0  # just in case

        clone_ref = dict(out.metadata[clonekey])
        clone_ref = {k: r for k, r in clone_ref.items() if r != "None"}
        tmp_clone_tree = Tree()
        for x in out.metadata.index:
            if x in clone_ref:
                if "|" in clone_ref[x]:
                    for x_ in clone_ref[x].split("|"):
                        if x_ != "None":
                            tmp_clone_tree[x_][x].value = 1
                else:
                    tmp_clone_tree[clone_ref[x]][x].value = 1
        tmp_clone_tree2 = Tree()
        for x in tmp_clone_tree:
            tmp_clone_tree2[x] = list(tmp_clone_tree[x])

        tmp_clone_tree3 = Tree()
        tmp_clone_tree3_overlap = Tree()
        for x in tqdm(
            tmp_clone_tree2,
            desc="Computing overlap ",
            disable=not verbose,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ):
            # this is to catch all possible cells that may potentially match up with this clone that's joined together
            if x in list(flatten(overlap)):
                for ol in overlap:
                    if x in ol:
                        if len(tmp_clone_tree2[x]) > 1:
                            for x_ in tmp_clone_tree2[x]:
                                tmp_clone_tree3_overlap["|".join(ol)][
                                    "".join(x_)
                                ].value = 1
                        else:
                            tmp_clone_tree3_overlap["|".join(ol)][
                                "".join(tmp_clone_tree2[x])
                            ].value = 1
            else:
                tmp_ = pd.DataFrame(
                    index=tmp_clone_tree2[x], columns=tmp_clone_tree2[x]
                )
                tmp_ = pd.DataFrame(
                    np.tril(tmp_) + 1,
                    index=tmp_clone_tree2[x],
                    columns=tmp_clone_tree2[x],
                )
                tmp_.fillna(0, inplace=True)
                tmp_clone_tree3[x] = tmp_

        for x in tmp_clone_tree3_overlap:  # repeat for the overlap clones
            tmp_ = pd.DataFrame(
                index=tmp_clone_tree3_overlap[x],
                columns=tmp_clone_tree3_overlap[x],
            )
            tmp_ = pd.DataFrame(
                np.tril(tmp_) + 1,
                index=tmp_clone_tree3_overlap[x],
                columns=tmp_clone_tree3_overlap[x],
            )
            tmp_.fillna(0, inplace=True)
            tmp_clone_tree3[x] = tmp_

        # free up memory
        del tmp_clone_tree2
        # here I'm using a temporary edge list to catch all cells that were identified as clones to forcefully
        # link them up if they were identical but clipped off during the mst step

        # create a data frame to recall the actual distance quickly
        tmp_totaldiststack = tmp_totaldist.stack().reset_index()

        # free up memory
        del tmp_totaldist
        tmp_totaldiststack.columns = ["source", "target", "weight"]
        tmp_totaldiststack.index = [
            str(s) + "|" + str(t)
            for s, t in zip(
                tmp_totaldiststack["source"], tmp_totaldiststack["target"]
            )
        ]
        tmp_totaldiststack["keep"] = [
            False if len(list(set(i.split("|")))) == 1 else True
            for i in tmp_totaldiststack.index
        ]

        tmp_totaldiststack = tmp_totaldiststack[tmp_totaldiststack.keep].drop(
            "keep", axis=1
        )


        tmp_edge_list = Tree()
        for c in tqdm(
            tmp_clone_tree3,
            desc="Linking edges ",
            disable=not verbose,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ):
            if len(tmp_clone_tree3[c]) > 1:
                G = nx.from_pandas_adjacency(tmp_clone_tree3[c])
                tmp_edge_list[c] = nx.to_pandas_edgelist(G)
                tmp_edge_list[c].index = [
                    str(s) + "|" + str(t)
                    for s, t in zip(
                        tmp_edge_list[c]["source"], tmp_edge_list[c]["target"]
                    )
                ]
                tmp_edge_list[c]["weight"].update(tmp_totaldiststack["weight"])
                # keep only edges when there is 100% identity, to minimise crowding
                tmp_edge_list[c] = tmp_edge_list[c][
                    tmp_edge_list[c]["weight"] == 0
                ]
                tmp_edge_list[c].reset_index(inplace=True)

        # try to catch situations where there's no edge (only singletons)
        try:
            edge_listx = pd.concat([edge_list[x] for x in edge_list])
            edge_listx.index = [
                str(s) + "|" + str(t)
                for s, t in zip(edge_listx["source"], edge_listx["target"])
            ]

            tmp_edge_listx = pd.concat(
                [tmp_edge_list[x] for x in tmp_edge_list]
            )
            tmp_edge_listx.drop("index", axis=1, inplace=True)
            tmp_edge_listx.index = [
                str(s) + "|" + str(t)
                for s, t in zip(
                    tmp_edge_listx["source"], tmp_edge_listx["target"]
                )
            ]

            edge_list_final = edge_listx.combine_first(tmp_edge_listx)
            edge_list_final["weight"].update(tmp_totaldiststack["weight"])
            # return the edge list
            edge_list_final.reset_index(drop=True, inplace=True)
        except:
            edge_list_final = None

        # free up memory
        del tmp_totaldiststack
        del tmp_edge_list

        vertice_list = list(out.metadata.index)
    else:
        edge_list_final = None
        vertice_list = list(index)
    # and finally the vertex list which is super easy

    # and now to actually generate the network
    g, g_, lyt, lyt_ = _generate_layout(
        vertice_list,
        edge_list_final,
        min_size=min_size,
        weight=None,
        verbose=verbose,
        compute_layout=compute_layout,
        layout_method=layout_method,
        **kwargs,
    )

    logg.info(
        " finished",
        time=start,
        deep=(
            "Updated Dandelion object: \n"
            "   'data', contig-indexed clone table\n"
            "   'metadata', cell-indexed clone table\n"
            "   'layout', graph layout\n"
            "   'graph', network constructed from distance matrices of VDJ- and VJ- chains"
        ),
    )
    if isinstance(vdj_data, Dandelion):
        if vdj_data.germline is not None:
            germline_ = vdj_data.germline
        else:
            germline_ = None
        if vdj_data.threshold is not None:
            threshold_ = vdj_data.threshold
        else:
            threshold_ = None
        if downsample is not None:
            if (lyt and lyt_) is not None:
                out = Dandelion(
                    data=dat_,
                    layout=(lyt, lyt_),
                    graph=(g, g_),
                    germline=germline_,
                )
            else:
                out = Dandelion(
                    data=dat_,
                    graph=(g, g_),
                    germline=germline_,
                )
            out.threshold = threshold_
            return out
        else:
            if (lyt and lyt_) is not None:
                vdj_data.__init__(
                    data=vdj_data.data,
                    metadata=vdj_data.metadata,
                    layout=(lyt, lyt_),
                    graph=(g, g_),
                    germline=germline_,
                    initialize=False,
                )
            else:
                vdj_data.__init__(
                    data=vdj_data.data,
                    metadata=vdj_data.metadata,
                    layout=None,
                    graph=(g, g_),
                    germline=germline_,
                    initialize=False,
                )
            vdj_data.threshold = threshold_
    else:
        if (lyt and lyt_) is not None:
            out = Dandelion(
                data=dat_,
                layout=(lyt, lyt_),
                graph=(g, g_),
                clone_key=clone_key,
            )
        else:
            out = Dandelion(
                data=dat_,
                layout=None,
                graph=(g, g_),
                clone_key=clone_key,
            )
        return out


def mst(mat: dict) -> Tree:
    """
    Construct minimum spanning tree based on supplied matrix in dictionary.

    Parameters
    ----------
    mat : dict
        Dictionary containing numpy ndarrays.

    Returns
    -------
    Tree
        Dandelion `Tree` object holding DataFrames of constructed minimum spanning trees.
    """
    mst_tree = Tree()
    for c in mat:
        tmp = mat[c] + 1
        tmp[np.isnan(tmp)] = 0
        G = nx.from_pandas_adjacency(tmp)
        mst_tree[c] = nx.minimum_spanning_tree(G)
    return mst_tree


def clone_degree(
    vdj_data: Dandelion, weight: Optional[str] = None, verbose: bool = True
) -> Dandelion:
    """
    Calculate node degree in BCR/TCR network.

    Parameters
    ----------
    vdj_data : Dandelion
        `Dandelion` object after `tl.generate_network` has been run.
    weight : Optional[str], optional
        Attribute name for retrieving edge weight in graph. None defaults to ignoring this. See `networkx.Graph.degree`.
    verbose : bool, optional
        Whether or not to show logging information.

    Raises
    ------
    AttributeError
        if graph not found.
    TypeError
        if input is not Dandelion class.
    """
    start = logg.info("Calculating node degree")
    if isinstance(vdj_data, Dandelion):
        if vdj_data.graph is None:
            raise AttributeError(
                "Graph not found. Please run tl.generate_network."
            )
        else:
            G = vdj_data.graph[0]
            cd = pd.DataFrame.from_dict(G.degree(weight=weight))
            cd.set_index(0, inplace=True)
            vdj_data.metadata["clone_degree"] = pd.Series(cd[1])
            logg.info(
                " finished",
                time=start,
                deep=("Updated Dandelion metadata\n"),
            )
    else:
        raise TypeError("Input object must be of {}".format(Dandelion))


def clone_centrality(vdj_data: Dandelion, verbose: bool = True):
    """
    Calculate node closeness centrality in BCR/TCR network.

    Parameters
    ----------
    vdj_data : Dandelion
        `Dandelion` object after `tl.generate_network` has been run.
    verbose : bool, optional
        Whether or not to show logging information.

    Raises
    ------
    AttributeError
        if graph not found.
    TypeError
        if input is not Dandelion class.
    """
    start = logg.info("Calculating node closeness centrality")
    if isinstance(vdj_data, Dandelion):
        if vdj_data.graph is None:
            raise AttributeError(
                "Graph not found. Please run tl.generate_network."
            )
        else:
            G = vdj_data.graph[0]
            cc = nx.closeness_centrality(G)
            cc = pd.DataFrame.from_dict(
                cc, orient="index", columns=["clone_centrality"]
            )
            vdj_data.metadata["clone_centrality"] = pd.Series(
                cc["clone_centrality"]
            )
            logg.info(
                " finished",
                time=start,
                deep=("Updated Dandelion metadata\n"),
            )
    else:
        raise TypeError("Input object must be of {}".format(Dandelion))


def _generate_layout(
    vertices: list,
    edges: pd.DataFrame = None,
    min_size: int = 2,
    weight: Optional[str] = None,
    verbose: bool = True,
    compute_layout: bool = True,
    layout_method: Literal["sfdp", "mod_fr"] = "sfdp",
    **kwargs,
) -> Tuple[nx.Graph, nx.Graph, dict, dict]:
    """Generate layout.

    Parameters
    ----------
    vertices : list
        list of vertices
    edges : pd.DataFrame, optional
        edge list in a pandas data frame.
    min_size : int, optional
        minimum clone size.
    weight : Optional[str], optional
        name of weight column.
    verbose : bool, optional
        whether to print progress
    compute_layout : bool, optional
        whether or not to compute layout.
    layout_method : Literal["sfdp", "mod_fr"], optional
        layout method.
    **kwargs
        passed to fruchterman_reingold_layout.

    Returns
    -------
    Tuple[nx.Graph, nx.Graph, dict, dict]
        graphs and layout positions.
    """
    G = nx.Graph()
    G.add_nodes_from(vertices)
    if edges is not None:
        G.add_weighted_edges_from(
            [
                (x, y, z)
                for x, y, z in zip(
                    edges["source"], edges["target"], edges["weight"]
                )
            ]
        )
    degree = G.degree()
    G_ = G.copy()
    if min_size == 2:
        if edges is not None:
            G_.remove_nodes_from(nx.isolates(G))
        else:
            pass
    elif min_size > 2:
        if edges is not None:
            for component in list(nx.connected_components(G_)):
                if len(component) < min_size:
                    for node in component:
                        G_.remove_node(node)
        else:
            pass
    if compute_layout:
        logg.info("generating network layout")
        if layout_method == "mod_fr":
            pos = _fruchterman_reingold_layout(G, weight=weight, **kwargs)
            pos_ = _fruchterman_reingold_layout(G_, weight=weight, **kwargs)
        elif layout_method == "sfdp":
            try:
                from graph_tool.all import sfdp_layout
            except ImportError:
                print(
                    "To benefit from faster layout computation, please install graph-tool: "
                    "conda install -c conda-forge graph-tool"
                )
                nographtool = True
            if "nographtool" in locals():
                pos = _fruchterman_reingold_layout(G, weight=weight, **kwargs)
                pos_ = _fruchterman_reingold_layout(G_, weight=weight, **kwargs)
            else:
                gtg = nx2gt(G)
                gtg_ = nx2gt(G_)
                posx = sfdp_layout(gtg, **kwargs)
                posx_ = sfdp_layout(gtg_, **kwargs)
                pos = dict(zip(list(gtg.vertex_properties["id"]), list(posx)))
                pos_ = dict(
                    zip(list(gtg_.vertex_properties["id"]), list(posx_))
                )
        return (G, G_, pos, pos_)
    else:
        return (G, G_, None, None)


# when dealing with a lot of unconnected vertices, the pieces fly out to infinity and the original fr layout can't be
# used
# work around from https://stackoverflow.com/questions/14283341/how-to-increase-node-spacing-for-networkx-spring-layout
# code chunk from networkx's layout.py https://github.com/networkx/networkx/blob/master/networkx/drawing/layout.py


def _process_params(G, center, dim):
    """Some boilerplate code."""
    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center


def _fruchterman_reingold_layout(
    G,
    k=None,
    pos=None,
    fixed=None,
    iterations=50,
    threshold=1e-4,
    weight="weight",
    scale=1,
    center=None,
    dim=2,
    seed=None,
    **kwargs,
):
    """
    Position nodes using Fruchterman-Reingold force-directed algorithm.

    The algorithm simulates a force-directed representation of the network
    treating edges as springs holding nodes close, while treating nodes
    as repelling objects, sometimes called an anti-gravity force.
    Simulation continues until the positions are close to an equilibrium.
    There are some hard-coded values: minimal distance between
    nodes (0.01) and "temperature" of 0.1 to ensure nodes don't fly away.
    During the simulation, `k` helps determine the distance between nodes,
    though `scale` and `center` determine the size and place after
    rescaling occurs at the end of the simulation.
    Fixing some nodes doesn't allow them to move in the simulation.
    It also turns off the rescaling feature at the simulation's end.
    In addition, setting `scale` to `None` turns off rescaling.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.
    k : float (default=None)
        Optimal distance between nodes.  If None the distance is set to
        1/sqrt(n) where n is the number of nodes.  Increase this value
        to move nodes farther apart.
    pos : dict or None  Optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        random initial positions.
    fixed : list or None  Optional (default=None)
        Nodes to keep fixed at initial position.
        ValueError raised if `fixed` specified and `pos` not.
    iterations : int  Optional (default=50)
        Maximum number of iterations taken
    threshold: float Optional (default = 1e-4)
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold.
    weight : string or None   Optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.
    scale : number or None (default: 1)
        Scale factor for positions. Not used unless `fixed is None`.
        If scale is None, no rescaling is performed.
    center : array-like or None
        Coordinate pair around which to center the layout.
        Not used unless `fixed is None`.
    dim : int
        Dimension of layout.
    seed : int, RandomState instance or None  Optional (default=None)
        Set the random state for deterministic node layouts.
        If int, `seed` is the seed used by the random number generator,
        if numpy.random.RandomState instance, `seed` is the random
        number generator,
        if None, the random number generator is the RandomState instance used
        by numpy.random.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.spring_layout(G)
    # The same using longer but equivalent function name
    >>> pos = nx.fruchterman_reingold_layout(G)
    """
    G, center = _process_params(G, center, dim)

    if fixed is not None:
        if pos is None:
            raise ValueError("nodes are fixed without positions given")
        for node in fixed:
            if node not in pos:
                raise ValueError("nodes are fixed without positions given")
        nfixed = {node: i for i, node in enumerate(G)}
        fixed = np.asarray([nfixed[node] for node in fixed])

    if pos is not None:
        # Determine size of existing domain to adjust initial positions
        dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
        if dom_size == 0:
            dom_size = 1
        pos_arr = seed.rand(len(G), dim) * dom_size + center

        for i, n in enumerate(G):
            if n in pos:
                pos_arr[i] = np.asarray(pos[n])
    else:
        pos_arr = None
        dom_size = 1

    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G.nodes()): center}

    try:
        # Sparse matrix
        if len(G) < 500:  # sparse solver for large graphs
            raise ValueError
        if int(nx.__version__[0]) > 2:
            A = nx.to_scipy_sparse_array(G, weight=weight, dtype="f")
        else:
            A = nx.to_scipy_sparse_matrix(G, weight=weight, dtype="f")
        if k is None and fixed is not None:
            # We must adjust k by domain size for layouts not near 1x1
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)
        pos = _sparse_fruchterman_reingold(
            A, k, pos_arr, fixed, iterations, threshold, dim, seed
        )
    except ValueError:
        A = nx.to_numpy_array(G, weight=weight)
        if k is None and fixed is not None:
            # We must adjust k by domain size for layouts not near 1x1
            nnodes, _ = A.shape
            k = dom_size / np.sqrt(nnodes)
        pos = _fruchterman_reingold(
            A, k, pos_arr, fixed, iterations, threshold, dim, seed
        )
    if fixed is None and scale is not None:
        pos = _rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))
    return pos


@random_state(7)
def _fruchterman_reingold(
    A,
    k=None,
    pos=None,
    fixed=None,
    iterations=50,
    threshold=1e-4,
    dim=2,
    seed=None,
    **kwargs,
):
    """Fruchterman Reingold algorithm."""
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    import numpy as np

    try:
        nnodes, _ = A.shape
    except AttributeError as e:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from e

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    # the inscrutable (but fast) version
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        # matrix of difference between points
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # distance between points
        distance = np.linalg.norm(delta, axis=-1)
        # enforce minimum distance of 0.01
        np.clip(distance, 0.001, None, out=distance)
        # displacement "force"
        displacement = np.einsum(
            "ijk,ij->ik", delta, (k * k / distance**2 - A * distance / k)
        )
        displacement = displacement - pos / (k * np.sqrt(nnodes))
        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        pos += delta_pos
        # cool temperature
        t -= dt
        err = np.linalg.norm(delta_pos) / nnodes
        if err < threshold:
            break
    return pos


@random_state(7)
def _sparse_fruchterman_reingold(
    A,
    k=None,
    pos=None,
    fixed=None,
    iterations=50,
    threshold=1e-4,
    dim=2,
    seed=None,
    **kwargs,
):
    """Sparse Fruchterman Reingold algorithm."""
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    # Sparse version
    import numpy as np

    try:
        nnodes, _ = A.shape
    except AttributeError as e:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from e
    try:
        from scipy.sparse import coo_matrix
    except ImportError as e:
        msg = "_sparse_fruchterman_reingold() scipy numpy: http://scipy.org/ "
        raise ImportError(msg) from e
    # make sure we have a LIst of Lists representation
    try:
        A = A.tolil()
    except AttributeError:
        A = (coo_matrix(A)).tolil()

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # no fixed nodes
    if fixed is None:
        fixed = []

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)

    displacement = np.zeros((dim, nnodes))
    for iteration in range(iterations):
        displacement *= 0
        # loop over rows
        for i in range(A.shape[0]):
            if i in fixed:
                continue
            # difference between this row's node position and all others
            delta = (pos[i] - pos).T
            # distance between points
            distance = np.sqrt((delta**2).sum(axis=0))
            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)
            # the adjacency matrix row
            Ai = np.asarray(A.getrowview(i).toarray())
            # displacement "force"
            displacement[:, i] += (
                delta * (k * k / distance**2 - Ai * distance / k)
            ).sum(axis=1)
        displacement = displacement - pos / (k * np.sqrt(nnodes))
        # update positions
        length = np.sqrt((displacement**2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (displacement * t / length).T
        pos += delta_pos
        # cool temperature
        t -= dt
        err = np.linalg.norm(delta_pos) / nnodes
        if err < threshold:
            break
    return pos


def _rescale_layout(pos, scale=1):
    """
    Return scaled position array to (-scale, scale) in all axes.

    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.
    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).

    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.
    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.
    """
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos


def extract_edge_weights(
    vdj_data: Dandelion, expanded_only: bool = False
) -> list:
    """
    Retrieve edge weights (BCR levenshtein distance) from graph.

    Parameters
    ----------
    vdj_data : Dandelion
        `Dandelion` object after `tl.generate_network` has been run.
    expanded_only : bool, optional
        whether to retrieve the edge weights from the expanded only graph or entire graph.

    Returns
    -------
    list
        list of edge weights.
    """
    if expanded_only:
        try:
            edges, weights = zip(
                *nx.get_edge_attributes(vdj_data.graph[1], "weight").items()
            )
        except ValueError as e:
            print(
                "{} i.e. the graph does not contain edges. Therefore, edge weights not returned.".format(
                    e
                )
            )
    else:
        try:
            edges, weights = zip(
                *nx.get_edge_attributes(vdj_data.graph[0], "weight").items()
            )
        except ValueError as e:
            print(
                "{} i.e. the graph does not contain edges. Therefore, edge weights not returned.".format(
                    e
                )
            )
    if "weights" in locals():
        return weights


# from https://bbengfort.github.io/2016/06/graph-tool-from-networkx/
def nx2gt(nxG):
    """Convert a networkx graph to a graph-tool graph."""
    try:
        import graph_tool as gt
    except ImportError:
        raise ImportError(
            "Please install graph-tool: conda install -c conda-forge graph-tool"
        )
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname)  # Create the PropertyMap
        gtG.graph_properties[key] = prop  # Set the PropertyMap
        gtG.graph_properties[key] = value  # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set()  # cache keys to only add properties once
    for node, data in list(nxG.nodes(data=True)):
        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops:
                continue  # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname)  # Create the PropertyMap
            gtG.vertex_properties[key] = prop  # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties["id"] = gtG.new_vertex_property("string")

    # Add the edge properties second
    eprops = set()  # cache keys to only add properties once
    for src, dst, data in list(nxG.edges(data=True)):
        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops:
                continue  # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname)  # Create the PropertyMap
            gtG.edge_properties[key] = prop  # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {}  # vertex mapping for tracking edges later
    for node, data in list(nxG.nodes(data=True)):
        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data["id"] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value  # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in list(nxG.edges(data=True)):
        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value  # ep is short for edge_properties

    # Done, finally!
    return gtG


def get_prop_type(value, key=None):
    """
    Perform typing and value conversion for the graph_tool PropertyMap class.

    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    # Deal with the value
    if isinstance(value, bool):
        tname = "bool"

    elif isinstance(value, int):
        tname = "float"
        value = float(value)

    elif isinstance(value, float):
        tname = "float"

    elif isinstance(value, dict):
        tname = "object"

    else:
        tname = "string"
        value = str(value)

    return tname, value, key
