#!/usr/bin/env python
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from anndata import AnnData
from itertools import combinations, cycle
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import networkx as nx
from plotnine import (
    aes,
    geom_line,
    ggplot,
    ggtitle,
    labs,
    options,
    scale_color_manual,
    theme_classic,
    xlab,
    ylab,
)
from scanpy.plotting import palettes
from scanpy.plotting._tools.scatterplots import embedding
from time import sleep
from tqdm import tqdm
from typing import Union, Tuple, Dict, Optional

import dandelion.external.nxviz as nxv
from dandelion.external.nxviz import annotate
from dandelion.tools._diversity import rarefun
from dandelion.utilities._core import *
from dandelion.utilities._io import *
from dandelion.utilities._utilities import *


def clone_rarefaction(
    vdj_data: Union[AnnData, Dandelion],
    color: str,
    clone_key: Optional[str] = None,
    palette: Optional[List[str]] = None,
    figsize: Tuple[Union[int, float], Union[int, float]] = (5, 3),
    chain_status_include: List[
        Literal[
            "Single pair",
            "Orphan VDJ",
            "Orphan VDJ-exception",
            "Orphan VJ",
            "Orphan VJ-exception",
            "Extra pair",
            "Extra pair-exception",
        ]
    ] = [
        "Single pair",
        "Orphan VDJ",
        "Orphan VDJ-exception",
        "Extra pair",
        "Extra pair-exception",
    ],
    save: Optional[str] = None,
) -> ggplot:
    """
    Plot rarefaction curve for cell numbers vs clone size.

    Parameters
    ----------
    vdj_data : Union[AnnData, Dandelion]
        `AnnData` or `Dandelion` object.
    color : str
        Column name to split the calculation of clone numbers for a given number of cells for e.g. sample, patient etc.
    clone_key : Optional[str], optional
        Column name specifying the clone_id column in metadata/obs.
    palette : Optional[List[str]], optional
        Color mapping for unique elements in color. Will try to retrieve from AnnData `.uns` slot if present.
    figsize : Tuple[Union[int, float], Union[int, float]], optional
        Size of plot.
    chain_status_include : List[Literal["Single pair", "Orphan VDJ", "Orphan VDJ-exception", "Orphan VJ", "Orphan VJ-exception", "Extra pair", "Extra pair-exception", ]], optional
        chain statuses to include.
    save : Optional[str], optional
        Save path.

    Returns
    -------
    ggplot
        rarefaction plot.
    """
    if isinstance(vdj_data, AnnData):
        metadata = vdj_data.obs.copy()
    elif isinstance(vdj_data, Dandelion):
        metadata = vdj_data.metadata.copy()
    if clone_key is None:
        clonekey = "clone_id"
    else:
        clonekey = clone_key

    groups = list(set(metadata[color]))
    if "contig_QC_pass" in metadata:
        metadata = metadata[metadata["contig_QC_pass"].isin(TRUES)]
    elif "chain_status" in metadata:
        metadata = metadata[metadata["chain_status"].isin(chain_status_include)]

    if type(metadata[clonekey]) == "category":
        metadata[clonekey] = metadata[clonekey].cat.remove_unused_categories()
    res = {}
    for g in groups:
        _metadata = metadata[metadata[color] == g]
        res[g] = _metadata[clonekey].value_counts()
    res_ = pd.DataFrame.from_dict(res, orient="index")

    # remove those with no counts
    print(
        "removing due to zero counts:",
        ", ".join(
            [res_.index[i] for i, x in enumerate(res_.sum(axis=1) == 0) if x]
        ),
    )
    sleep(0.5)
    res_ = res_[~(res_.sum(axis=1) == 0)]

    # set up for calculating rarefaction
    tot = res_.apply(sum, axis=1)
    # S = res_.apply(lambda x: x[x > 0].shape[0], axis=1)
    nr = res_.shape[0]

    # append the results to a dictionary
    rarecurve = {}
    for i in tqdm(
        range(0, nr),
        desc="Calculating rarefaction curve ",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        n = np.arange(1, tot[i], step=10)
        if n[-1:] != tot[i]:
            n = np.append(n, tot[i])
        rarecurve[res_.index[i]] = [
            rarefun(
                np.array(res_.iloc[i,]),
                z,
            )
            for z in n
        ]
    y = pd.DataFrame([rarecurve[c] for c in rarecurve]).T
    pred = pd.DataFrame(
        [np.append(np.arange(1, s, 10), s) for s in res_.sum(axis=1)],
        index=res_.index,
    ).T

    y = y.melt()
    pred = pred.melt()
    pred["yhat"] = y["value"]

    options.figure_size = figsize
    if palette is None:
        if isinstance(vdj_data, AnnData):
            try:
                pal = vdj_data.uns[str(color) + "_colors"]
            except:
                if len(list(set((pred.variable)))) <= 20:
                    pal = palettes.default_20
                elif len(list(set((pred.variable)))) <= 28:
                    pal = palettes.default_28
                elif len(list(set((pred.variable)))) <= 102:
                    pal = palettes.default_102
                else:
                    pal = cycle(palettes.default_102)

            if pal is not None:
                p = (
                    ggplot(pred, aes(x="value", y="yhat", color="variable"))
                    + theme_classic()
                    + xlab("number of cells")
                    + ylab("number of clones")
                    + ggtitle("rarefaction curve")
                    + labs(color=color)
                    + scale_color_manual(values=(pal))
                    + geom_line()
                )
            else:
                p = (
                    ggplot(pred, aes(x="value", y="yhat", color="variable"))
                    + theme_classic()
                    + xlab("number of cells")
                    + ylab("number of clones")
                    + ggtitle("rarefaction curve")
                    + labs(color=color)
                    + geom_line()
                )
        else:
            if len(list(set((pred.variable)))) <= 20:
                pal = palettes.default_20
            elif len(list(set((pred.variable)))) <= 28:
                pal = palettes.default_28
            elif len(list(set((pred.variable)))) <= 102:
                pal = palettes.default_102
            else:
                pal = None

            if pal is not None:
                p = (
                    ggplot(pred, aes(x="value", y="yhat", color="variable"))
                    + theme_classic()
                    + xlab("number of cells")
                    + ylab("number of clones")
                    + ggtitle("rarefaction curve")
                    + labs(color=color)
                    + scale_color_manual(values=(pal))
                    + geom_line()
                )
            else:
                p = (
                    ggplot(pred, aes(x="value", y="yhat", color="variable"))
                    + theme_classic()
                    + xlab("number of cells")
                    + ylab("number of clones")
                    + ggtitle("rarefaction curve")
                    + labs(color=color)
                    + geom_line()
                )
    else:
        p = (
            ggplot(pred, aes(x="value", y="yhat", color="variable"))
            + theme_classic()
            + xlab("number of cells")
            + ylab("number of clones")
            + ggtitle("rarefaction curve")
            + labs(color=color)
            + geom_line()
        )
    if save:
        p.save(
            filename="figures/rarefaction" + str(save),
            height=plt.rcParams["figure.figsize"][0],
            width=plt.rcParams["figure.figsize"][1],
            units="in",
            dpi=plt.rcParams["savefig.dpi"],
        )

    return p


def clone_network(
    adata: AnnData, basis: str = "vdj", edges: bool = True, **kwargs
) -> Union[Figure, Axes, None]:
    """
    Using scanpy's plotting module to plot the network.

    Only thing that is changed is the default options:
    `basis = 'bcr'` and `edges = True`.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    basis : str, optional
        key for embedding.
    edges : bool, optional
        whether or not to plot edges.
    **kwargs
        passed `sc.pl.embedding`.
    """
    embedding(adata, basis=basis, edges=edges, **kwargs)


def barplot(
    vdj_data: Union[AnnData, Dandelion],
    color: str,
    palette: str = "Set1",
    figsize: Tuple[Union[int, float], Union[int, float]] = (8, 3),
    normalize: bool = True,
    sort_descending: bool = True,
    title: Optional[str] = None,
    xtick_fontsize: Optional[int] = None,
    xtick_rotation: Optional[Union[int, float]] = None,
    min_clone_size: int = 1,
    clone_key: Optional[str] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    A barplot function to plot usage of V/J genes in the data.

    Parameters
    ----------
    vdj_data : Union[AnnData, Dandelion]
        `Dandelion` or `AnnData` object.
    color : str
        column name in metadata for plotting in bar plot.
    palette : str, optional
        Colors to use for the different levels of the color variable.
        Should be something that can be interpreted by
        [color_palette](https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette),
        or a dictionary mapping hue levels to matplotlib colors.
        See [seaborn.barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html).
    figsize : Tuple[Union[int, float], Union[int, float]], optional
        figure size.
    normalize : bool, optional
        if True, will return as proportion out of 1.
        Otherwise False will return counts.
    sort_descending : bool, optional
        whether or not to sort the order of the plot.
    title : Optional[str], optional
        title of plot.
    xtick_fontsize : Optional[int], optional
        size of x tick labels
    xtick_rotation : Optional[Union[int, float]], optional
        rotation of x tick labels.
    min_clone_size : int, optional
        minimum clone size to keep.
    clone_key : Optional[str], optional
        column name for clones. None defaults to 'clone_id'.
    **kwargs
        passed to `sns.barplot`.

    Returns
    -------
    Tuple[Figure, Axes]
        bar plot.

    """
    if isinstance(vdj_data, Dandelion):
        data = vdj_data.metadata.copy()
    elif isinstance(vdj_data, AnnData):
        data = vdj_data.obs.copy()

    min_size = min_clone_size

    if clone_key is None:
        clone_ = "clone_id"
    else:
        clone_ = clone_key

    size = data[clone_].value_counts()
    keep = list(size[size >= min_size].index)
    data_ = data[data[clone_].isin(keep)]

    sns.set_style("whitegrid", {"axes.grid": False})
    res = pd.DataFrame(data_[color].value_counts(normalize=normalize))
    if not sort_descending:
        res = res.sort_index()
    res.reset_index(drop=False, inplace=True)

    # Initialize the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)

    # plot
    try:
        sns.barplot(x="index", y=color, data=res, palette=palette, **kwargs)
    except ValueError:
        yname = "proportion" if normalize else "count"
        sns.barplot(x=color, y=yname, data=res, palette=palette, **kwargs)
    # change some parts
    if title is None:
        ax.set_title(color.replace("_", " ") + " usage")
    else:
        ax.set_title(title)
    if normalize:
        ax.set_ylabel("proportion")
    else:
        ax.set_ylabel("count")
    ax.set_xlabel("")
    # modify the x ticks accordingly
    xtick_params = {}
    if xtick_rotation is None:
        xtick_params["rotation"] = 90
    else:
        xtick_params["rotation"] = xtick_rotation
    if xtick_fontsize is not None:
        xtick_params["fontsize"] = xtick_fontsize
    plt.xticks(**xtick_params)
    return fig, ax


def stackedbarplot(
    vdj_data: Union[AnnData, Dandelion],
    color: str,
    groupby: Optional[str],
    figsize: Tuple[Union[int, float], Union[int, float]] = (8, 3),
    normalize: bool = False,
    title: Optional[str] = None,
    sort_descending: bool = True,
    xtick_fontsize: Optional[int] = None,
    xtick_rotation: Optional[Union[int, float]] = None,
    hide_legend: bool = False,
    legend_options: Tuple[str, Tuple[float, float], int] = (
        "upper left",
        (1, 1),
        1,
    ),
    labels: Optional[List[str]] = None,
    min_clone_size: int = 1,
    clone_key: Optional[str] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    A stacked bar plot function to plot usage of V/J genes in the data split by groups.

    Parameters
    ----------
    vdj_data : Union[AnnData, Dandelion]
        `Dandelion` or `AnnData` object.
    color : str
        column name in metadata for plotting in bar plot.
    groupby : Optional[str]
        column name in metadata to split by during plotting.
    figsize : Tuple[Union[int, float], Union[int, float]], optional
        figure size.
    normalize : bool, optional
        if True, will return as proportion out of 1, otherwise False will return counts.
    title : Optional[str], optional
        title of plot.
    sort_descending : bool, optional
        whether or not to sort the order of the plot.
    xtick_fontsize : Optional[int], optional
        size of x tick labels
    xtick_rotation : Optional[Union[int, float]], optional
        rotation of x tick labels.
    hide_legend : bool, optional
        whether or not to hide the legend.
    legend_options : Tuple[str, Tuple[float, float], int], optional
        a tuple holding 3 options for specify legend options: 1) loc (string), 2) bbox_to_anchor (tuple), 3) ncol (int).
    labels : Optional[List[str]], optional
        Names of objects will be used for the legend if list of multiple data frames supplied.
    min_clone_size : int, optional
        minimum clone size to keep.
    clone_key : Optional[str], optional
        column name for clones. None defaults to 'clone_id'.
    **kwargs
        other kwargs passed to `matplotlib.plt`.

    Returns
    -------
    Tuple[Figure, Axes]
        stacked barplot.
    """
    if isinstance(vdj_data, Dandelion):
        data = vdj_data.metadata.copy()
    elif isinstance(vdj_data, AnnData):
        data = vdj_data.obs.copy()
    # quick fix to prevent dropping of nan
    data[groupby] = [str(l) for l in data[groupby]]

    min_size = min_clone_size

    if clone_key is None:
        clone_ = "clone_id"
    else:
        clone_ = clone_key

    size = data[clone_].value_counts()
    keep = list(size[size >= min_size].index)
    data_ = data[data[clone_].isin(keep)]

    dat_ = pd.DataFrame(
        data_.groupby(color)[groupby]
        .value_counts(normalize=normalize)
        .unstack(fill_value=0)
        .stack(),
        columns=["value"],
    )
    dat_.reset_index(drop=False, inplace=True)
    dat_order = pd.DataFrame(data[color].value_counts(normalize=normalize))
    dat_ = dat_.pivot(index=color, columns=groupby, values="value")
    if sort_descending is True:
        dat_ = dat_.reindex(dat_order.index)
    elif sort_descending is False:
        dat_ = dat_.reindex(dat_order.index[::-1])
    elif sort_descending is None:
        dat_ = dat_.sort_index()

    def _plot_bar_stacked(
        dfall: pd.DataFrame,
        labels: Optional[List[str]] = None,
        figsize: Tuple[Union[int, float], Union[int, float]] = (8, 3),
        title: str = "multiple stacked bar plot",
        xtick_fontsize: Optional[int] = None,
        xtick_rotation: Optional[Union[int, float]] = None,
        legend_options: Tuple[str, Tuple[float, float], int] = None,
        hide_legend: bool = False,
        H: Literal["/"] = "/",
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Given a list of data frames, with identical columns and index, create a clustered stacked bar plot.

        Parameters
        ----------
        dfall : pd.DataFrame
            data frame for plotting.
        labels : Optional[List[str]], optional
            a list of the data frame objects. Names of objects will be used for the legend.
        figsize : Tuple[Union[int, float], Union[int, float]], optional
            size of figure.
        title : str, optional
            string for the title of the plot
        xtick_fontsize : Optional[int], optional
            xtick fontsize.
        xtick_rotation : Optional[Union[int, float]], optional
            rotation of xtick labels
        legend_options : Tuple[str, Tuple[float, float], int], optional
            legend options.
        hide_legend : bool, optional
            whether to show legend.
        H : Literal["/"], optional
            is the hatch used for identification of the different data frames
        **kwargs
            other kwargs passed to matplotlib.plt

        Returns
        -------
        Tuple[Figure, Axes]
            stacked barplot.
        """
        if type(dfall) is not list:
            dfall = [dfall]
        n_df = len(dfall)
        n_col = len(dfall[0].columns)
        n_ind = len(dfall[0].index)
        # Initialize the matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        for df in dfall:  # for each data frame
            ax = df.plot(
                kind="bar",
                linewidth=0,
                stacked=True,
                ax=ax,
                legend=False,
                grid=False,
                **kwargs,
            )  # make bar plots
        (
            h,
            l,
        ) = ax.get_legend_handles_labels()  # get the handles we want to modify
        for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
            for j, pa in enumerate(h[i : i + n_col]):
                for rect in pa.patches:  # for each index
                    rect.set_x(
                        rect.get_x() + 1 / float(n_df + 1) * i / float(n_col)
                    )
                    rect.set_hatch(H * int(i / n_col))  # edited part
                    rect.set_width(1 / float(n_df + 1))
        ax.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.0)
        ax.set_xticklabels(df.index, rotation=0)
        ax.set_title(title)
        if normalize:
            ax.set_ylabel("proportion")
        else:
            ax.set_ylabel("count")
        # Add invisible data to add another legend
        n = []
        for i in range(n_df):
            n.append(ax.bar(0, 0, color="grey", hatch=H * i))
        if legend_options is None:
            Legend = ("center right", (1.15, 0.5), 1)
        else:
            Legend = legend_options
        if hide_legend is False:
            l1 = ax.legend(
                h[:n_col],
                l[:n_col],
                loc=Legend[0],
                bbox_to_anchor=Legend[1],
                ncol=Legend[2],
                frameon=False,
            )
            if labels is not None:
                l2 = plt.legend(
                    n,
                    labels,
                    loc=Legend[0],
                    bbox_to_anchor=Legend[1],
                    ncol=Legend[2],
                    frameon=False,
                )
                ax.add_artist(l2)
            ax.add_artist(l1)
        # modify the x ticks accordingly
        xtick_params = {}
        if xtick_rotation is None:
            xtick_params["rotation"] = 90
        else:
            xtick_params["rotation"] = xtick_rotation
        if xtick_fontsize is not None:
            xtick_params["fontsize"] = xtick_fontsize
        plt.xticks(**xtick_params)

        return fig, ax

    if title is None:
        title = (
            "multiple stacked bar plot : " + color.replace("_", " ") + " usage"
        )
    else:
        title = title

    return _plot_bar_stacked(
        dat_,
        labels=labels,
        figsize=figsize,
        title=title,
        xtick_fontsize=xtick_fontsize,
        xtick_rotation=xtick_rotation,
        legend_options=legend_options,
        hide_legend=hide_legend,
        **kwargs,
    )


def spectratype(
    vdj_data: Dandelion,
    color: str,
    groupby: str,
    locus: str,
    figsize: Tuple[Union[int, float], Union[int, float]] = (5, 3),
    width: Optional[Union[int, float]] = None,
    title: Optional[str] = None,
    xtick_fontsize: Optional[int] = None,
    xtick_rotation: Optional[Union[int, float]] = None,
    hide_legend: bool = False,
    legend_options: Tuple[str, Tuple[float, float], int] = (
        "upper left",
        (1, 1),
        1,
    ),
    labels: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    A spectratype function to plot usage of CDR3 length.

    Parameters
    ----------
    vdj_data : Dandelion
        `Dandelion` object.
    color : str
        column name in metadata for plotting in bar plot.
    groupby : str
        column name in metadata to split by during plotting.
    locus : str
        either IGH or IGL.
    figsize : Tuple[Union[int, float], Union[int, float]], optional
        figure size.
    width : Optional[Union[int, float]], optional
        width of bars.
    title : Optional[str], optional
        title of plot.
    xtick_fontsize : Optional[int], optional
        size of x tick labels
    xtick_rotation : Optional[Union[int, float]], optional
        rotation of x tick labels.
    hide_legend : bool, optional
        whether or not to hide the legend.
    legend_options : Tuple[str, Tuple[float, float], int], optional
        a tuple holding 3 options for specify legend options:
        1) loc (string), 2) bbox_to_anchor (tuple), 3) ncol (int).
    labels : Optional[List[str]], optional
        Names of objects will be used for the legend if list of
        multiple data frames supplied.
    **kwargs
        other kwargs passed to matplotlib.pyplot.plot

    Returns
    -------
    Tuple[Figure, Axes]
        spectratype plot.
    """
    data = vdj_data.data.copy()
    if "ambiguous" in vdj_data.data:
        data = data[data["ambiguous"] == "F"].copy()

    if type(locus) is not list:
        locus = [locus]
    data = data[data["locus"].isin(locus)].copy()
    data[groupby] = [str(l) for l in data[groupby]]
    dat_ = pd.DataFrame(
        data.groupby(color)[groupby]
        .value_counts(normalize=False)
        .unstack(fill_value=0)
        .stack(),
        columns=["value"],
    )
    dat_.reset_index(drop=False, inplace=True)
    dat_[color] = pd.to_numeric(dat_[color], errors="coerce")
    dat_.sort_values(by=color)
    dat_2 = dat_.pivot(index=color, columns=groupby, values="value")
    new_index = range(0, int(dat_[color].max()) + 1)
    dat_2 = dat_2.reindex(new_index, fill_value=0)

    def _plot_spectra_stacked(
        dfall: pd.DataFrame,
        labels: Optional[List[str]] = None,
        figsize: Tuple[Union[int, float], Union[int, float]] = (5, 3),
        title: str = "multiple stacked bar plot",
        width: Optional[Union[int, float]] = None,
        xtick_fontsize: Optional[int] = None,
        xtick_rotation: Optional[Union[int, float]] = None,
        legend_options: Tuple[str, Tuple[float, float], int] = None,
        hide_legend: bool = False,
        H: Literal["/"] = "/",
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Stacked spectratype plots.

        Parameters
        ----------
        dfall : pd.DataFrame
            data frame for plotting.
        labels : Optional[List[str]], optional
            a list of the data frame objects. Names of objects will be used for the legend.
        figsize : Tuple[Union[int, float], Union[int, float]], optional
            size of figure.
        title : str, optional
            string for the title of the plot.
        width : Optional[Union[int, float]], optional
            width of bars.
        xtick_fontsize : Optional[int], optional
            size of x tick labels
        xtick_rotation : Optional[Union[int, float]], optional
            rotation of x tick labels.
        legend_options : Tuple[str, Tuple[float, float], int], optional
            a tuple holding 3 options for specify legend options:
            1) loc (string), 2) bbox_to_anchor (tuple), 3) ncol (int).
        hide_legend : bool, optional
            whether or not to hide the legend.
        H : Literal["/"], optional
            not sure.
        **kwargs
            other kwargs passed to matplotlib.plt

        Returns
        -------
        Tuple[Figure, Axes]
            spectratype plot.
        """
        if type(dfall) is not list:
            dfall = [dfall]
        n_df = len(dfall)
        n_col = len(dfall[0].columns)
        n_ind = len(dfall[0].index)
        if width is None:
            wdth = 0.1 * n_ind / 60 + 0.8
        else:
            wdth = width
        # Initialize the matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        for df in dfall:  # for each data frame
            ax = df.plot(
                kind="bar",
                linewidth=0,
                stacked=True,
                ax=ax,
                legend=False,
                grid=False,
                **kwargs,
            )  # make bar plots
        (
            h,
            l,
        ) = ax.get_legend_handles_labels()  # get the handles we want to modify
        for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
            for j, pa in enumerate(h[i : i + n_col]):
                for rect in pa.patches:  # for each index
                    rect.set_x(
                        rect.get_x() + 1 / float(n_df + 1) * i / float(n_col)
                    )
                    rect.set_hatch(H * int(i / n_col))  # edited part
                    # need to see if there's a better way to toggle this.
                    rect.set_width(wdth)

        n = 5  # Keeps every 5th label visible and hides the rest
        [
            l.set_visible(False)
            for (i, l) in enumerate(ax.xaxis.get_ticklabels())
            if i % n != 0
        ]
        ax.set_title(title)
        ax.set_ylabel("count")
        # Add invisible data to add another legend
        n = []
        for i in range(n_df):
            n.append(ax.bar(0, 0, color="gray", hatch=H * i))
        if legend_options is None:
            Legend = ("center right", (1.25, 0.5), 1)
        else:
            Legend = legend_options
        if hide_legend is False:
            l1 = ax.legend(
                h[:n_col],
                l[:n_col],
                loc=Legend[0],
                bbox_to_anchor=Legend[1],
                ncol=Legend[2],
                frameon=False,
            )
            if labels is not None:
                l2 = plt.legend(
                    n,
                    labels,
                    loc=Legend[0],
                    bbox_to_anchor=Legend[1],
                    ncol=Legend[2],
                    frameon=False,
                )
            ax.add_artist(l1)
        # modify the x ticks accordingly
        xtick_params = {}
        if xtick_rotation is None:
            xtick_params["rotation"] = 90
        else:
            xtick_params["rotation"] = xtick_rotation
        if xtick_fontsize is not None:
            xtick_params["fontsize"] = xtick_fontsize
        plt.xticks(**xtick_params)

        return fig, ax

    return _plot_spectra_stacked(
        dat_2,
        labels=labels,
        figsize=figsize,
        title=title,
        width=width,
        xtick_fontsize=xtick_fontsize,
        xtick_rotation=xtick_rotation,
        legend_options=legend_options,
        hide_legend=hide_legend,
        **kwargs,
    )


def clone_overlap(
    adata: AnnData,
    groupby: str,
    colorby: Optional[str] = None,
    weighted_overlap: bool = False,
    clone_key: Optional[str] = None,
    color_mapping: Optional[Union[list, dict]] = None,
    node_labels: bool = True,
    return_graph: bool = False,
    save: Optional[str] = None,
    legend_kwargs: dict = {
        "ncol": 2,
        "bbox_to_anchor": (1, 0.5),
        "frameon": False,
        "loc": "center left",
    },
    node_label_size: int = 10,
    as_heatmap: bool = False,
    **kwargs,
) -> nxv.CircosPlot:
    """
    A plot function to visualise clonal overlap as a circos-style plot.

    Originally written with nxviz < 0.7.3. Ported from https://github.com/zktuong/nxviz/tree/custom_color_mapping_circos_nodes_and_edges

    Parameters
    ----------
    adata : AnnData
        `AnnData` object.
    groupby : str
        column name in obs for collapsing to nodes in circos plot.
    colorby : Optional[str], optional
        column name in obs for grouping and color of nodes in plot. Must be a same or subcategory of the `groupby` categories e.g. `groupby="group_tissue", colorby="tissue"`.
    weighted_overlap : bool, optional
        if True, instead of collapsing to overlap to binary, edge thickness will reflect the number of
        cells found in the overlap. In the future, there will be the option to use something like a jaccard
        index instead.
    clone_key : Optional[str], optional
        column name for clones. None defaults to 'clone_id'.
    color_mapping : Optional[Union[list, dict]], optional
        custom color mapping provided as a sequence (correpsonding to order of categories or
        alpha-numeric order ifdtype is not category), or dictionary containing custom {category:color} mapping.
    node_labels : bool, optional
        whether to use node objects as labels or not
    return_graph : bool, optional
        whether or not to return the graph for fine tuning.
    save : Optional[str], optional
        file path for saving plot
    legend_kwargs : dict, optional
        options for adjusting legend placement
    node_label_size : int, optional
        size of labels if node_labels = True
    as_heatmap : bool, optional
        whether to return plot as heatmap.
    **kwargs
        passed to `matplotlib.pyplot.savefig`.

    Returns
    -------
    nxv.CircosPlot
        a `nxviz.CircosPlot` object.

    Raises
    ------
    KeyError
        if `clone_overlap` not found in `adata.uns`.
    ValueError
        if input is not `AnnData`.
    """

    if clone_key is None:
        clone_ = "clone_id"
    else:
        clone_ = clone_key

    if isinstance(adata, AnnData):
        data = adata.obs.copy()
        # get rid of problematic rows that appear because of category conversion?
        if "clone_overlap" in adata.uns:
            overlap = adata.uns["clone_overlap"].copy()
        else:
            raise KeyError(
                "`clone_overlap` not found in `adata.uns`. Did you run `tl.clone_overlap`?"
            )
    else:
        raise ValueError("Please provide a AnnData object.")

    edges = {}
    if not weighted_overlap:
        for x in overlap.index:
            if overlap.loc[x].sum() > 1:
                edges[x] = [
                    y + ({str(clone_): x},)
                    for y in list(
                        combinations(
                            [
                                i
                                for i in overlap.loc[x][
                                    overlap.loc[x] > 0
                                ].index
                            ],
                            2,
                        )
                    )
                ]
    else:
        tmp_overlap = overlap.astype(bool).sum(axis=1)
        combis = {
            x: list(
                combinations(
                    [i for i in overlap.loc[x][overlap.loc[x] > 0].index], 2
                )
            )
            for x in tmp_overlap.index
            if tmp_overlap.loc[x] > 1
        }

        tmp_edge_weight_dict = defaultdict(list)
        for k_clone, val_pair in combis.items():
            for pair in val_pair:
                tmp_edge_weight_dict[pair].append(
                    overlap.loc[k_clone, list(pair)].sum()
                )
        for combix in tmp_edge_weight_dict:
            tmp_edge_weight_dict[combix] = sum(tmp_edge_weight_dict[combix])
        for x in overlap.index:
            if overlap.loc[x].sum() > 1:
                edges[x] = [
                    y
                    + (
                        {
                            str(clone_): x,
                            "weight": tmp_edge_weight_dict[y],
                        },
                    )
                    for y in list(
                        combinations(
                            [
                                i
                                for i in overlap.loc[x][
                                    overlap.loc[x] > 0
                                ].index
                            ],
                            2,
                        )
                    )
                ]

    colorby = groupby if colorby is None else colorby
    # create graph
    G = nx.Graph()
    # add in the nodes
    G.add_nodes_from(
        [(p, {str(colorby): d}) for p, d in zip(data[groupby], data[colorby])]
    )

    # unpack the edgelist and add to the graph
    for edge in edges:
        G.add_edges_from(edges[edge])

    if not weighted_overlap:
        weighted_attr = None
    else:
        weighted_attr = "weight"

    if color_mapping is None:
        if str(colorby) + "_colors" in adata.uns:
            if pd.api.types.is_categorical_dtype(adata.obs[groupby]):
                colorby_dict = dict(
                    zip(
                        list(adata.obs[str(colorby)].cat.categories),
                        adata.uns[str(colorby) + "_colors"],
                    )
                )
            else:
                colorby_dict = dict(
                    zip(
                        list(adata.obs[str(colorby)].unique()),
                        adata.uns[str(colorby) + "_colors"],
                    )
                )
        else:
            if len(adata.obs[str(colorby)].unique()) <= 20:
                pal = cycle(palettes.default_20)
            elif len(adata.obs[str(colorby)].unique()) <= 28:
                pal = cycle(palettes.default_28)
            else:
                pal = cycle(palettes.default_102)
            colorby_dict = dict(
                zip(list(adata.obs[str(colorby)].unique()), pal)
            )
    else:
        if type(color_mapping) is dict:
            colorby_dict = color_mapping
        else:
            if pd.api.types.is_categorical_dtype(data[groupby]):
                colorby_dict = dict(
                    zip(list(data[str(colorby)].cat.categories), color_mapping)
                )
            else:
                colorby_dict = dict(
                    zip(sorted(list(set(data[str(colorby)]))), color_mapping)
                )
    df = data[[groupby, colorby]]
    if groupby == colorby:
        df = data[[groupby]]
        df = (
            df.sort_values(groupby)
            .drop_duplicates(subset=groupby, keep="first")
            .reset_index(drop=True)
        )
    else:
        df = (
            df.sort_values(colorby)
            .drop_duplicates(subset=groupby, keep="first")
            .reset_index(drop=True)
        )

    if as_heatmap:
        hm = nx.to_pandas_adjacency(G)
        sns.clustermap(hm, **kwargs)
    else:
        ax = nxv.circos(
            G,
            group_by=colorby,
            node_color_by=colorby,
            edge_lw_by=weighted_attr,
            node_palette=colorby_dict,
        )  # group_by
        if node_labels:
            annotate.circos_group(
                G,
                group_by=colorby,
                midpoint=False,
                fontdict={"size": node_label_size},
            )
        annotate.node_colormapping(
            G,
            color_by=colorby,
            palette=colorby_dict,
            legend_kwargs=legend_kwargs,
        )
    if save is not None:
        plt.savefig(save, bbox_inches="tight", **kwargs)
    if return_graph:
        return G


def productive_ratio(
    adata: AnnData,
    figsize: Tuple[Union[int, float], Union[int, float]] = (8, 4),
    palette: List = ["lightblue", "darkblue"],
    fontsize: Union[int, float] = 8,
    rotation: Union[int, float] = 90,
    legend_kwargs: Dict = {
        "bbox_to_anchor": (1, 0.5),
        "loc": "center left",
        "frameon": False,
    },
):
    """Plot productive/non-productive contig ratio from AnnData (cell level).

    Parameters
    ----------
    adata : AnnData
        AnnData object with `.uns['productive_ratio']` computed from
        `tl.productive_ratio`.
    figsize : Tuple[Union[int, float], Union[int, float]], optional
        Size of figure.
    palette : List, optional
        List of colours to plot non-productive and productive respectively.
    fontsize : Union[int, float], optional
        Font size of x and y tick labels.
    rotation : Union[int, float], optional
        Rotation of x tick labels.
    legend_kwargs : Dict, optional
        Any additional kwargs to `plt.legend`
    """
    res = adata.uns["productive_ratio"]["results"]
    locus = adata.uns["productive_ratio"]["locus"]
    groupby = adata.uns["productive_ratio"]["groupby"]

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        x=groupby, y="productive+non-productive", data=res, color=palette[0]
    )
    ax = sns.barplot(
        x=groupby, y="productive", data=res, color=palette[1], ax=ax
    )
    legend = [
        mpatches.Patch(
            color=palette[0], label="% with non-productive " + locus
        ),
        mpatches.Patch(color=palette[1], label="% with productive " + locus),
    ]
    plt.xticks(fontsize=fontsize, rotation=rotation)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("")
    plt.ylabel("")
    ax.set(ylim=(0, 100))
    plt.title(locus)
    # add legend
    plt.legend(handles=legend, **legend_kwargs)
