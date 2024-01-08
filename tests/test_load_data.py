#!/usr/bin/env python
import dandelion as ddl
import pytest


@pytest.mark.usefixtures("airr_reannotated")
def test_load_data(airr_reannotated):
    """test load_data"""
    vdj = ddl.Dandelion(airr_reannotated)
    assert all(
        [x != y for x, y in zip(vdj.data["cell_id"], vdj.data["sequence_id"])]
    )
    cell_ids = list(vdj.data["cell_id"])
    tmp = vdj.data.drop("cell_id", axis=1)
    vdj = ddl.Dandelion(tmp)
    assert all([x == y for x, y in zip(vdj.data["cell_id"], cell_ids)])


@pytest.mark.usefixtures("airr_generic")
def test_slice_data(airr_generic):
    """test load_data"""
    vdj = ddl.Dandelion(airr_generic)
    assert vdj.data.shape[0] == 114
    assert vdj.metadata.shape[0] == 45
    vdj2 = vdj[vdj.data["productive"] == "T"]
    assert vdj2.data.shape[0] == 103
    assert vdj2.metadata.shape[0] == 43
    vdj2 = vdj[vdj.metadata["productive_VDJ"] == "T"]
    assert vdj2.data.shape[0] == 41
    assert vdj2.metadata.shape[0] == 20
    vdj2 = vdj[
        vdj.metadata_names.isin(
            [
                "IGHA+IGHM+IGHD+IGLv2",
                "IGHA+IGHM+IGHD+IGLv3",
                "IGHM+IGHD+IGL+IGHA",
                "IGHM+IGHD+IGL+IGHAnp",
                "IGHM+IGHD+IGL+IGHM",
            ]
        )
    ]
    assert vdj2.data.shape[0] == 20
    assert vdj2.metadata.shape[0] == 5
    vdj2 = vdj[
        vdj.data_names.isin(
            [
                "IGHM+IGHD+IGL+IGHAnp_contig_1",
                "IGHM+IGHD+IGL+IGHAnp_contig_2",
                "IGHM+IGHD+IGL+IGHAnp_contig_4",
                "IGHM+IGHD+IGL+IGHAnp_contig_3",
                "IGHM+IGHD+IGL+IGHM_contig_1",
                "IGHM+IGHD+IGL+IGHM_contig_2",
                "IGHM+IGHD+IGL+IGHM_contig_4",
                "IGHM+IGHD+IGL+IGHM_contig_3",
                "IGHM+IGHD+IGL+IGK_contig_1",
                "IGHM+IGHD+IGL+IGK_contig_2",
                "IGHM+IGHD+IGL+IGK_contig_3",
                "IGHM+IGHD+IGL+IGK_contig_4",
                "IGHM+IGHM+IGL_contig_3",
                "IGHM+IGHM+IGL_contig_2",
                "IGHM+IGHM+IGL_contig_1",
                "IGHM+TRA_contig_1",
                "IGHM+TRA_contig_2",
                "IGHM+TRG_contig_2",
                "IGHM+TRG_contig_1",
                "IGK+IGL_contig_1",
                "IGK+IGL_contig_2",
                "TRA+TRG_contig_2",
                "TRA+TRG_contig_1",
                "TRB+IGL_contig_1",
                "TRB+IGL_contig_2",
                "TRB+TRG_contig_1",
                "TRB+TRG_contig_2",
                "TRBV+TRAJ+TRAC__TRAV+TRAJ_contig_1",
                "TRBV+TRAJ+TRAC__TRAV+TRAJ_contig_2",
                "TRBV+TRAJ+TRBC__TRAV+TRAJ_contig_1",
            ]
        )
    ]
    assert vdj2.data.shape[0] == 30
    assert vdj2.metadata.shape[0] == 12


@pytest.mark.usefixtures("airr_generic")
def test_names(airr_generic):
    """test load_data"""
    vdj = ddl.Dandelion(airr_generic)
    assert all(i == j for i, j in zip(vdj.data_names, vdj.data.index))
    assert all(i == j for i, j in zip(vdj.metadata_names, vdj.metadata.index))


@pytest.mark.usefixtures("airr_generic")
def test_slice_data_with_graph(airr_generic):
    """Test slicing data with graph"""
    vdj = ddl.Dandelion(airr_generic)
    vdj = ddl.pp.check_contigs(vdj, productive_only=False)
    ddl.tl.find_clones(vdj)
    ddl.tl.generate_network(vdj, key="junction", layout_method="mod_fr")
    vdj2 = vdj[vdj.data["productive"] == "T"]
    assert vdj2.data.shape[0] == 103
    assert vdj2.metadata.shape[0] == 43
    vdj2 = vdj[vdj.metadata["productive_VDJ"] == "T"]
    assert vdj2.data.shape[0] == 30
    assert vdj2.metadata.shape[0] == 15
    vdj2 = vdj[
        vdj.metadata_names.isin(
            [
                "IGHA+IGHM+IGHD+IGLv2",
                "IGHA+IGHM+IGHD+IGLv3",
                "IGHM+IGHD+IGL+IGHA",
                "IGHM+IGHD+IGL+IGHAnp",
                "IGHM+IGHD+IGL+IGHM",
            ]
        )
    ]
    assert vdj2.data.shape[0] == 20
    assert vdj2.metadata.shape[0] == 5
    assert len(vdj2.layout[0]) == 5
    assert len(vdj2.layout[1]) == 5
    assert len(vdj2.graph[0]) == 5
    assert len(vdj2.graph[1]) == 5
    vdj2 = vdj[
        vdj.data_names.isin(
            [
                "IGHM+IGHD+IGL+IGHAnp_contig_1",
                "IGHM+IGHD+IGL+IGHAnp_contig_2",
                "IGHM+IGHD+IGL+IGHAnp_contig_4",
                "IGHM+IGHD+IGL+IGHAnp_contig_3",
                "IGHM+IGHD+IGL+IGHM_contig_1",
                "IGHM+IGHD+IGL+IGHM_contig_2",
                "IGHM+IGHD+IGL+IGHM_contig_4",
                "IGHM+IGHD+IGL+IGHM_contig_3",
                "IGHM+IGHD+IGL+IGK_contig_1",
                "IGHM+IGHD+IGL+IGK_contig_2",
                "IGHM+IGHD+IGL+IGK_contig_3",
                "IGHM+IGHD+IGL+IGK_contig_4",
                "IGHM+IGHM+IGL_contig_3",
                "IGHM+IGHM+IGL_contig_2",
                "IGHM+IGHM+IGL_contig_1",
                "IGHM+TRA_contig_1",
                "IGHM+TRA_contig_2",
                "IGHM+TRG_contig_2",
                "IGHM+TRG_contig_1",
                "IGK+IGL_contig_1",
                "IGK+IGL_contig_2",
                "TRA+TRG_contig_2",
                "TRA+TRG_contig_1",
                "TRB+IGL_contig_1",
                "TRB+IGL_contig_2",
                "TRB+TRG_contig_1",
                "TRB+TRG_contig_2",
                "TRBV+TRAJ+TRAC__TRAV+TRAJ_contig_1",
                "TRBV+TRAJ+TRAC__TRAV+TRAJ_contig_2",
                "TRBV+TRAJ+TRBC__TRAV+TRAJ_contig_1",
            ]
        )
    ]
    assert vdj2.data.shape[0] == 30
    assert vdj2.metadata.shape[0] == 12
    assert len(vdj2.layout[0]) == 12
    # assert len(vdj2.layout[1]) == 4
    assert len(vdj2.layout[1]) == 10
    assert len(vdj2.graph[0]) == 12
    # assert len(vdj2.graph[1]) == 4
    assert len(vdj2.graph[1]) == 10
