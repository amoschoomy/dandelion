#!/usr/bin/env python
import pytest
import dandelion as ddl
import scanpy as sc


@pytest.mark.usefixtures("create_testfolder", "airr_reannotated", "dummy_adata")
def test_setup(create_testfolder, airr_reannotated, dummy_adata):
    """test_setup"""
    vdj, adata = ddl.pp.filter_contigs(airr_reannotated, dummy_adata)
    assert airr_reannotated.shape[0] == 8
    assert vdj.data.shape[0] == 7
    assert vdj.metadata.shape[0] == 4
    assert adata.n_obs == 5
    vdj.data["clone_id"] = ["A", "A", "A", "A", "A", "A", "A"]
    vdj = ddl.Dandelion(vdj.data)
    ddl.tl.generate_network(vdj, layout_method="mod_fr")
    ddl.tl.transfer(adata, vdj)
    assert "clone_id" in adata.obs
    assert "X_vdj" in adata.obsm
    f1 = create_testfolder / "test.h5ddl"
    f2 = create_testfolder / "test.h5ad"
    vdj.write_h5ddl(f1)
    adata.write_h5ad(f2)


@pytest.mark.usefixtures("create_testfolder")
def test_plot_network(create_testfolder):
    """test_plot_network"""
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    ddl.pl.clone_network(adata, color=["isotype"], show=False, return_fig=False)


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize(
    "sort,norm",
    [
        pytest.param(True, True),
        pytest.param(True, False),
        pytest.param(False, True),
        pytest.param(False, False),
    ],
)
def test_plot_bar(create_testfolder, sort, norm):
    """test_plot_bar"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    ax = ddl.pl.barplot(vdj, color="v_call_genotyped_VDJ")
    assert ax is not None
    ax = ddl.pl.barplot(vdj, color="v_call_genotyped_VDJ", sort_descending=sort)
    assert ax is not None
    ax = ddl.pl.barplot(vdj, color="v_call_genotyped_VDJ", normalize=norm)
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
def test_plot_bar2(create_testfolder):
    """test_plot_bar2"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    ax = ddl.pl.barplot(
        vdj,
        color="v_call_genotyped_VDJ",
        min_clone_size=2,
        clone_key="clone_id",
        title="test",
        xtick_rotation=90,
    )
    assert ax is not None
    ax = ddl.pl.barplot(adata, color="v_call_genotyped_VDJ")
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize("norm", [True, False])
def test_plot_stackedbar(create_testfolder, norm):
    """test_plot_stackedbar"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    ax = ddl.pl.stackedbarplot(
        vdj, color="v_call_genotyped_VDJ", groupby="isotype", normalize=norm
    )
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
@pytest.mark.parametrize("norm", [True, False])
def test_plot_stackedbar2(create_testfolder, norm):
    """test_plot_stackedbar2"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    f = create_testfolder / "test.h5ad"
    adata = sc.read_h5ad(f)
    ax = ddl.pl.stackedbarplot(
        vdj,
        color="v_call_genotyped_VDJ",
        groupby="isotype",
        min_clone_size=2,
        clone_key="clone_id",
        title="test",
        xtick_rotation=90,
        normalize=norm,
    )
    assert ax is not None
    ax = ddl.pl.stackedbarplot(
        adata,
        color="v_call_genotyped_VDJ",
        groupby="isotype",
        normalize=norm,
    )
    assert ax is not None


@pytest.mark.usefixtures("create_testfolder")
def test_plot_spectratype(create_testfolder):
    """test_plot_spectratype"""
    f = create_testfolder / "test.h5ddl"
    vdj = ddl.read_h5ddl(f)
    ax = ddl.pl.spectratype(
        vdj, color="junction_length", groupby="c_call", locus="IGH"
    )
    assert ax is not None
    ax = ddl.pl.spectratype(
        vdj,
        color="junction_length",
        groupby="c_call",
        locus="IGH",
        hide_legend=False,
        width=1,
        xtick_rotation=90,
        title="test",
        labels="test",
    )
    assert ax is not None
