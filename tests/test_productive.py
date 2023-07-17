#!/usr/bin/env python
import dandelion as ddl
from itertools import cycle
import pytest


@pytest.mark.usefixtures(
    "create_testfolder", "annotation_10x_mouse", "dummy_adata_mouse"
)
def test_productive_ratio(
    create_testfolder, annotation_10x_mouse, dummy_adata_mouse
):
    """test_productive_ratio"""
    annot_file = create_testfolder / "test_filtered_contig_annotations.csv"
    annotation_10x_mouse.to_csv(annot_file, index=False)
    vdj = ddl.read_10x_vdj(create_testfolder)
    vdj.data["ambiguous"] = "F"
    group = cycle(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    groups = [next(group) for i in dummy_adata_mouse.obs_names]
    dummy_adata_mouse.obs["group"] = groups
    ddl.tl.productive_ratio(
        dummy_adata_mouse, vdj, groupby="group", locus="IGH"
    )
    assert "productive_ratio" in dummy_adata_mouse.uns
    ddl.tl.productive_ratio(
        dummy_adata_mouse,
        vdj,
        groupby="group",
        locus="IGH",
        groups=["A", "B", "C"],
    )
    assert "productive_ratio" in dummy_adata_mouse.uns
    ddl.pl.productive_ratio(dummy_adata_mouse)


@pytest.mark.usefixtures("create_testfolder", "dummy_adata_mouse")
def test_vj_usage_pca(create_testfolder, dummy_adata_mouse):
    """Test vj usage pca."""
    vdj = ddl.read_10x_vdj(create_testfolder)
    _, adata = ddl.pp.check_contigs(vdj, dummy_adata_mouse)
    group = cycle(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    groups = [next(group) for i in adata.obs_names]
    groups2 = [next(group) for i in adata.obs_names]
    adata.obs["group"] = groups
    adata.obs["group2"] = groups2
    new_adata = ddl.tl.vj_usage_pca(
        adata,
        groupby="group",
        mode="B",
        n_comps=5,
        transfer_mapping=["group2"],
    )
    assert "X_pca" in new_adata.obsm
    adata2 = adata.copy()
    adata2.obs = adata2.obs.rename(
        columns={
            "v_call_VDJ": "v_call_genotyped_VDJ",
            "v_call_VJ": "v_call_genotyped_VJ",
            "v_call_B_VDJ": "v_call_genotyped_B_VDJ",
            "v_call_B_VJ": "v_call_genotyped_B_VJ",
        }
    )
    new_adata2 = ddl.tl.vj_usage_pca(
        adata2,
        groupby="group",
        mode="B",
        n_comps=5,
        transfer_mapping=["group2"],
    )
    assert "X_pca" in new_adata2.obsm
