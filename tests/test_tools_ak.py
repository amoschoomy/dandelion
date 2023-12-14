#!/usr/bin/env python
import pytest
import json
import pandas as pd
import dandelion as ddl
import scanpy as sc


# convert from airr_Reannotate to airr, replicate this, run scirpy chainqc, test mudata as well
@pytest.mark.usefixtures(
    "create_testfolder", "airr_reannotated", "airr_reannotated2", "dummy_adata"
)
def test_setup(
    create_testfolder, airr_reannotated, airr_reannotated2, dummy_adata
):
    """test setup"""
    vdj, adata = ddl.pp.filter_contigs(airr_reannotated, dummy_adata)
    vdj2 = ddl.pp.filter_contigs(airr_reannotated2)
    assert airr_reannotated.shape[0] == 8
    assert airr_reannotated2.shape[0] == 15
    assert vdj.data.shape[0] == 7
    assert vdj2.data.shape[0] == 13
    assert vdj.metadata.shape[0] == 4
    assert vdj2.metadata.shape[0] == 7
    assert adata.n_obs == 5
    f = create_testfolder / "test.h5ddl"
    f2 = create_testfolder / "test2.h5ddl"
    vdj.write_h5ddl(f)
    vdj2.write_h5ddl(f2)
    assert len(list(create_testfolder.iterdir())) == 2
    vdj3 = ddl.read_h5ddl(f)
    assert vdj3.metadata is not None


@pytest.mark.usefixtures("create_testfolder")
def test_find_clones(create_testfolder):
    """test find clones"""
    f = create_testfolder / "test.h5ddl"
    f2 = create_testfolder / "test2.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj2 = ddl.read_h5ddl(f2)
    ddl.tl.find_clones(vdj)
    ddl.tl.find_clones(vdj2)
    assert not vdj.data.clone_id.empty
    assert not vdj.metadata.clone_id.empty
    assert not vdj2.data.clone_id.empty
    assert not vdj2.metadata.clone_id.empty
    assert len(set(x for x in vdj.metadata["clone_id"] if pd.notnull(x))) == 4
    assert len(set(x for x in vdj2.metadata["clone_id"] if pd.notnull(x))) == 4
    vdj.write_h5ddl(f)
    vdj2.write_h5ddl(f2)


@pytest.mark.usefixtures("create_testfolder")
def test_clone_size(create_testfolder):
    """test clone_size"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    ddl.tl.clone_size(vdj)
    assert not vdj.metadata.clone_id_size.empty
    ddl.tl.clone_size(vdj, max_size=3)
    assert not vdj.metadata.clone_id_size.empty


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "resample,expected", [pytest.param(None, 7), pytest.param(3, 3)]
)
def test_generate_network(create_testfolder, resample, expected):
    """test generate network"""
    f = create_testfolder / "test.h5ddl"
    f2 = create_testfolder / "test2.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj2 = ddl.read_h5ddl(f2)
    if resample is not None:
        vdj = ddl.tl.generate_network(
            vdj, downsample=resample, layout_method="mod_fr"
        )
        assert vdj.n_obs == expected
        assert vdj.layout is not None
        assert vdj.graph is not None
    else:
        ddl.tl.generate_network(vdj2, layout_method="mod_fr")
        assert vdj2.n_obs == expected
        assert vdj2.layout is not None
        assert vdj2.graph is not None
    vdj.data["clone_id"] = "1"
    vdj = ddl.Dandelion(vdj.data)
    assert vdj.data.clone_id.dtype == "object"
    ddl.tl.generate_network(vdj, layout_method="mod_fr")
    assert vdj.layout is not None


@pytest.mark.usefixtures("create_testfolder")
def test_find_clones_key(create_testfolder):
    """test different clone key"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    ddl.tl.find_clones(vdj, key_added="test_clone")
    assert not vdj.metadata.test_clone.empty
    assert vdj.data.test_clone.dtype == "object"
    ddl.tl.generate_network(vdj, clone_key="test_clone", layout_method="mod_fr")
    assert vdj.layout is not None
    assert vdj.graph is not None


@pytest.mark.usefixtures("create_testfolder", "dummy_adata2")
def test_transfer(create_testfolder, dummy_adata2):
    """test transfer"""
    f = create_testfolder / "test2.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj, adata = ddl.pp.filter_contigs(vdj, dummy_adata2)
    ddl.tl.transfer(dummy_adata2, vdj)
    assert "clone_id" in dummy_adata2.obs
    ddl.tl.generate_network(vdj, layout_method="mod_fr")
    ddl.tl.transfer(dummy_adata2, vdj)
    assert "X_vdj" in dummy_adata2.obsm
    f2 = create_testfolder / "test2.h5ad"
    dummy_adata2.write_h5ad(f2)


@pytest.mark.usefixtures("create_testfolder")
def test_diversity_gini(create_testfolder):
    """test gini"""
    f = create_testfolder / "test2.h5ddl"
    vdj = ddl.read_h5ddl(f)
    ddl.tl.clone_diversity(vdj, groupby="sample_id")
    assert not vdj.metadata.clone_network_vertex_size_gini.empty
    assert not vdj.metadata.clone_network_cluster_size_gini.empty
    ddl.tl.generate_network(vdj, layout_method="mod_fr")
    ddl.tl.clone_diversity(vdj, groupby="sample_id", metric="clone_centrality")
    assert not vdj.metadata.clone_centrality_gini.empty
    assert not vdj.metadata.clone_size_gini.empty
    tmp = ddl.tl.clone_diversity(
        vdj,
        groupby="sample_id",
        metric="clone_centrality",
        update_obs_meta=False,
    )
    assert isinstance(tmp, pd.DataFrame)


@pytest.mark.usefixtures("create_testfolder")
def test_diversity_gini2(create_testfolder):
    """test gini 2"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    ddl.tl.clone_diversity(vdj, groupby="sample_id")
    tmp = ddl.tl.clone_diversity(
        vdj, groupby="sample_id", update_obs_meta=False
    )
    assert isinstance(tmp, pd.DataFrame)


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize("resample", [True, False])
def test_diversity_chao(create_testfolder, resample):
    """test chao"""
    f = create_testfolder / "test2.h5ddl"
    vdj = ddl.read_h5ddl(f)
    if resample:
        ddl.tl.clone_diversity(
            vdj,
            groupby="sample_id",
            method="chao1",
            resample=resample,
            downsample=6,
        )
    else:
        ddl.tl.clone_diversity(
            vdj, groupby="sample_id", method="chao1", resample=resample
        )
    assert not vdj.metadata.clone_size_chao1.empty
    tmp = ddl.tl.clone_diversity(
        vdj, groupby="sample_id", method="chao1", update_obs_meta=False
    )
    assert isinstance(tmp, pd.DataFrame)


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "method,diversitykey",
    [
        pytest.param("chao1", None),
        pytest.param("chao1", "test_diversity_key"),
        pytest.param("shannon", None),
        pytest.param("shannon", "test_diversity_key"),
    ],
)
def test_diversity_anndata(create_testfolder, method, diversitykey):
    """test div anndata"""
    f = create_testfolder / "test2.h5ad"
    adata = sc.read_h5ad(f)
    ddl.tl.clone_diversity(
        adata, groupby="sample_id", method=method, diversity_key=diversitykey
    )
    if diversitykey is None:
        assert "diversity" in adata.uns
    else:
        assert "test_diversity_key" in adata.uns


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "resample,normalize",
    [
        pytest.param(True, True),
        pytest.param(False, True),
        pytest.param(True, False),
        pytest.param(False, False),
    ],
)
def test_diversity_shannon(create_testfolder, resample, normalize):
    """test shannon"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    if resample:
        ddl.tl.clone_diversity(
            vdj,
            groupby="sample_id",
            method="shannon",
            resample=resample,
            normalize=normalize,
            downsample=6,
        )
    else:
        ddl.tl.clone_diversity(
            vdj,
            groupby="sample_id",
            method="shannon",
            resample=resample,
            normalize=normalize,
        )
    if normalize:
        assert not vdj.metadata.clone_size_normalized_shannon.empty
    else:
        assert not vdj.metadata.clone_size_shannon.empty
    tmp = ddl.tl.clone_diversity(
        vdj, groupby="sample_id", method="shannon", update_obs_meta=False
    )
    assert isinstance(tmp, pd.DataFrame)


@pytest.mark.usefixtures("create_testfolder", "json_10x_cr6", "dummy_adata_cr6")
def test_setup2(create_testfolder, json_10x_cr6, dummy_adata_cr6):
    """test setup 2"""
    json_file = create_testfolder / "test_all_contig_annotations.json"
    with open(json_file, "w") as outfile:
        json.dump(json_10x_cr6, outfile)
    vdj = ddl.read_10x_vdj(create_testfolder)
    vdj, adata = ddl.pp.filter_contigs(vdj, dummy_adata_cr6)
    assert vdj.data.shape[0] == 17
    assert vdj.metadata.shape[0] == 8
    ddl.tl.find_clones(vdj)
    ddl.tl.generate_network(vdj, key="sequence", layout_method="mod_fr")
    ddl.tl.transfer(adata, vdj)
    f = create_testfolder / "test.h5ddl"
    vdj.write_h5ddl(f)
    f2 = create_testfolder / "test.h5ad"
    adata.write_h5ad(f2)


@pytest.mark.usefixtures("create_testfolder")
def test_diversity_rarefaction(create_testfolder):
    """test rarefaction"""
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    ddl.tl.clone_rarefaction(adata, groupby="sample_id")
    assert "diversity" in adata.uns
    ddl.tl.clone_rarefaction(
        adata, groupby="sample_id", diversity_key="test_diversity_key"
    )
    assert "test_diversity_key" in adata.uns
    p = ddl.pl.clone_rarefaction(adata, color="sample_id")
    assert p is not None


@pytest.mark.usefixtures("create_testfolder")
def test_diversity_rarefaction2(create_testfolder):
    """test rarefaction2"""
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    ddl.tl.clone_rarefaction(adata, groupby="sample_id", clone_key="clone_id")
    assert "diversity" in adata.uns
    p = ddl.pl.clone_rarefaction(adata, color="sample_id")
    assert p is not None
    adata = sc.read_h5ad(f)
    p = ddl.pl.clone_rarefaction(adata, color="sample_id")
    assert p is not None


@pytest.mark.usefixtures("create_testfolder")
def test_diversity_rarefaction3(create_testfolder):
    """test rarefaction3"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj.data["sample_id"] = "sample_test"
    vdj.data["contig_QC_pass"] = "True"
    ddl.update_metadata(
        vdj,
        retrieve=["sample_id", "contig_QC_pass"],
        retrieve_mode=["merge and unique only", "merge and unique only"],
    )
    df = ddl.tl.clone_rarefaction(vdj, groupby="sample_id")
    assert isinstance(df, dict)
    p = ddl.pl.clone_rarefaction(vdj, color="sample_id")
    assert p is not None


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "metric", ["clone_network", None, "clone_degree", "clone_centrality"]
)
def test_diversity_gini3(create_testfolder, metric):
    """test gini more"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj.data["sample_id"] = "sample_test"
    vdj.data["contig_QC_pass"] = "True"
    ddl.update_metadata(
        vdj,
        retrieve=["sample_id", "contig_QC_pass"],
        retrieve_mode=["merge and unique only", "merge and unique only"],
    )
    ddl.tl.clone_diversity(
        vdj,
        groupby="sample_id",
        resample=True,
        downsample=6,
        key="sequence",
        n_resample=5,
        metric=metric,
    )
    if metric == "clone_network" or metric is None:
        assert not vdj.metadata.clone_network_cluster_size_gini.empty
        assert not vdj.metadata.clone_network_vertex_size_gini.empty
    if metric == "clone_degree":
        assert not vdj.metadata.clone_degree.empty
        assert not vdj.metadata.clone_size_gini.empty
        assert not vdj.metadata.clone_degree_gini.empty
    if metric == "clone_centrality":
        assert not vdj.metadata.clone_centrality.empty
        assert not vdj.metadata.clone_centrality_gini.empty


@pytest.mark.usefixtures("create_testfolder")
def test_diversity2a(create_testfolder):
    """test div"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj.data["sample_id"] = "sample_test"
    vdj.data["contig_QC_pass"] = "True"
    ddl.update_metadata(
        vdj,
        retrieve=["sample_id", "contig_QC_pass"],
        retrieve_mode=["merge and unique only", "merge and unique only"],
    )
    ddl.tl.clone_diversity(
        vdj, groupby="sample_id", reconstruct_network=False, key="sequence"
    )
    assert not vdj.metadata.clone_network_cluster_size_gini.empty
    assert not vdj.metadata.clone_network_vertex_size_gini.empty


@pytest.mark.usefixtures("create_testfolder")
def test_diversity2b(create_testfolder):
    """test div2"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj.data["sample_id"] = "sample_test"
    vdj.data["contig_QC_pass"] = "True"
    ddl.update_metadata(
        vdj,
        retrieve=["sample_id", "contig_QC_pass"],
        retrieve_mode=["merge and unique only", "merge and unique only"],
    )
    ddl.tl.clone_diversity(
        vdj, groupby="sample_id", use_contracted=True, key="sequence"
    )
    assert not vdj.metadata.clone_network_cluster_size_gini.empty
    assert not vdj.metadata.clone_network_vertex_size_gini.empty


@pytest.mark.usefixtures("create_testfolder")
def test_diversity2c(create_testfolder):
    """test div3"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    vdj.data["sample_id"] = "sample_test"
    vdj.data["contig_QC_pass"] = "True"
    ddl.update_metadata(
        vdj,
        retrieve=["sample_id", "contig_QC_pass"],
        retrieve_mode=["merge and unique only", "merge and unique only"],
    )
    x = ddl.tl.clone_diversity(
        vdj, groupby="sample_id", key="sequence", update_obs_meta=False
    )
    assert isinstance(x, pd.DataFrame)


@pytest.mark.usefixtures("create_testfolder")
def test_extract_edge_weights(create_testfolder):
    """test edge weights"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    x = ddl.tl.extract_edge_weights(vdj)
    assert x is None
    x = ddl.tl.extract_edge_weights(vdj, expanded_only=True)
    assert x is None


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "method",
    [
        pytest.param("chao1"),
        pytest.param("shannon"),
    ],
)
def test_diversity_anndata2(create_testfolder, method):
    """test div4"""
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    tmp = ddl.tl.clone_diversity(
        adata, groupby="sample_id", method=method, update_obs_meta=False
    )
    assert isinstance(tmp, pd.DataFrame)
