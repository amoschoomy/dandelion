#!/usr/bin/env python
import dandelion as ddl
import pandas as pd
import pytest


@pytest.mark.usefixtures(
    "create_testfolder", "annotation_10x", "dummy_adata_mouse"
)
def test_clone_overlap(
    create_testfolder, annotation_10x_mouse, dummy_adata_mouse
):
    """test_clone_overlap"""
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x_mouse.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(create_testfolder)
    ddl.pp.filter_contigs(vdj)
    ddl.tl.find_clones(vdj)
    assert vdj.data.shape[0] == 1987
    assert vdj.metadata.shape[0] == 547
    ddl.tl.transfer(dummy_adata_mouse, vdj)
    assert dummy_adata_mouse.n_obs == 547
    # create a sample column
    label = []
    for x in range(0, dummy_adata_mouse.n_obs):
        if x < 100:
            label.append("A")
        elif x < 200:
            label.append("B")
        elif x < 300:
            label.append("C")
        elif x < 400:
            label.append("D")
        elif x < 500:
            label.append("E")
        else:
            label.append("F")
    dummy_adata_mouse.obs["sample_idx"] = label
    with pytest.raises(KeyError):
        ddl.pl.clone_overlap(
            dummy_adata_mouse,
            groupby="sample_idx",
        )
    ddl.tl.clone_overlap(dummy_adata_mouse, groupby="sample_idx")
    assert "clone_overlap" in dummy_adata_mouse.uns
    ddl.pl.clone_overlap(
        dummy_adata_mouse,
        groupby="sample_idx",
    )
    with pytest.raises(ValueError):
        ddl.pl.clone_overlap(
            vdj,
            groupby="sample_idx",
        )
    G = ddl.pl.clone_overlap(
        dummy_adata_mouse,
        groupby="sample_idx",
        weighted_overlap=False,
        save=create_testfolder / "test.png",
        return_graph=True,
    )
    assert G is not None

    G = ddl.pl.clone_overlap(
        dummy_adata_mouse,
        groupby="sample_idx",
        weighted_overlap=True,
        scale_edge_lambda=lambda x: x * 10,
        return_graph=True,
    )
    assert G is not None

    ddl.pl.clone_overlap(
        dummy_adata_mouse,
        groupby="sample_idx",
        as_heatmap=True,
    )

    out = ddl.pl.clone_overlap(
        dummy_adata_mouse,
        groupby="sample_idx",
        as_heatmap=True,
        return_heatmap_data=True,
    )
    assert isinstance(out, pd.DataFrame)
