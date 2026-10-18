import matplotlib
matplotlib.use("Agg")

import json
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pytest
from matplotlib.figure import Figure

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


@pytest.fixture
def reduction_session_and_tools():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load build_dtm_tool to prepare feature_matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Load tools under test from tools_reduction.py
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    # Load fixture dataset
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    return session, build_dtm_tool, reduce_dimensions_tool, binarize_labels_tool


def test_reduce_dimensions_pca(reduction_session_and_tools):
    session, build_dtm_tool, reduce_dimensions_tool, _ = reduction_session_and_tools

    # Build DTM first
    build_dtm_tool.invoke({})
    orig_matrix = session.feature_matrix.toarray().copy()

    # Known Answer #7 (PCA)
    pca_res = reduce_dimensions_tool.invoke({"method": "pca"})
    json.dumps(pca_res)

    assert pca_res["status"] == "success"
    assert pca_res["n_samples"] == 8
    assert pca_res["n_components"] == 2

    # Check 8 coordinates
    expected_pca_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    actual_coords = np.array(pca_res["first_10_coords"])
    np.testing.assert_allclose(actual_coords, expected_pca_coords, atol=1e-3)

    # Check explained_variance_ratio
    expected_evr = [0.7532, 0.1965]
    np.testing.assert_allclose(pca_res["explained_variance_ratio"], expected_evr, atol=1e-3)

    # Check pending_figure is matplotlib Figure
    assert isinstance(session.pending_figure, Figure)

    # Check feature matrix is unchanged
    np.testing.assert_array_equal(session.feature_matrix.toarray(), orig_matrix)


def test_reduce_dimensions_umap(reduction_session_and_tools):
    session, build_dtm_tool, reduce_dimensions_tool, _ = reduction_session_and_tools

    build_dtm_tool.invoke({})
    orig_matrix = session.feature_matrix.toarray().copy()

    # Known Answer #9 (UMAP)
    umap_res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })
    json.dumps(umap_res)

    assert umap_res["status"] == "success"
    assert umap_res["n_samples"] == 8
    assert umap_res["n_components"] == 2

    expected_umap_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])
    actual_coords = np.array(umap_res["first_10_coords"])
    np.testing.assert_allclose(actual_coords, expected_umap_coords, atol=1e-2)

    assert isinstance(session.pending_figure, Figure)
    np.testing.assert_array_equal(session.feature_matrix.toarray(), orig_matrix)


def test_reduce_dimensions_tsne(reduction_session_and_tools):
    session, build_dtm_tool, reduce_dimensions_tool, _ = reduction_session_and_tools

    build_dtm_tool.invoke({})
    orig_matrix = session.feature_matrix.toarray().copy()

    # Known Answer #8 (t-SNE check)
    tsne_res1 = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })
    json.dumps(tsne_res1)

    assert tsne_res1["status"] == "success"
    coords1 = np.array(tsne_res1["first_10_coords"])
    assert coords1.shape == (8, 2)

    # Calling again with same random_state yields identical result
    tsne_res2 = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })
    coords2 = np.array(tsne_res2["first_10_coords"])
    np.testing.assert_array_equal(coords1, coords2)

    # Documents 5 and 6 (identical text) get identical coordinates
    np.testing.assert_allclose(coords1[5], coords1[6], atol=1e-3)

    assert isinstance(session.pending_figure, Figure)
    np.testing.assert_array_equal(session.feature_matrix.toarray(), orig_matrix)


def test_reduce_dimensions_capping_15_docs():
    session = SessionState()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Set up 15 synthesized documents with 5 features
    np.random.seed(42)
    session.feature_matrix = sp.csr_matrix(np.random.rand(15, 5))
    session.dataframe = pd.DataFrame({"category_name": ["catA"] * 8 + ["catB"] * 7})

    res = reduce_dimensions_tool.invoke({"method": "pca", "random_state": 42})
    json.dumps(res)

    assert res["status"] == "success"
    # Summary capping check: summary holds exactly 10 coordinates
    assert len(res["first_10_coords"]) == 10

    # full_report check: holds all 15 rows
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]
    assert full_report is not None
    assert len(full_report) == 15

    # session.artifacts check: holds all 15 rows
    artifact_coords = session.artifacts["coords_pca"]
    assert len(artifact_coords) == 15


def test_reduce_dimensions_errors():
    # Fresh session with no DTM
    empty_session = SessionState()
    reduction_tools = make_reduction_tools(empty_session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Error: No DTM
    res_no_dtm = reduce_dimensions_tool.invoke({"method": "pca"})
    assert res_no_dtm.get("status") == "error"

    # Setup fixture dataset with DTM
    premade_tools, _ = load_workspace_tools("premade_tools", empty_session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    dtm_tools = make_dtm_tools(empty_session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    # Error: t-SNE with default perplexity (=30 >= 8 samples)
    res_tsne_err = reduce_dimensions_tool.invoke({"method": "tsne"})
    assert res_tsne_err.get("status") == "error"

    # Error: UMAP with default n_neighbors (=15 >= 8 samples)
    res_umap_err = reduce_dimensions_tool.invoke({"method": "umap"})
    assert res_umap_err.get("status") == "error"

    # Error: PCA on matrix with 1 feature
    empty_session.feature_matrix = sp.csr_matrix(np.ones((8, 1)))
    res_pca_1f = reduce_dimensions_tool.invoke({"method": "pca"})
    assert res_pca_1f.get("status") == "error"


def test_binarize_labels_tool(reduction_session_and_tools):
    session, _, _, binarize_labels_tool = reduction_session_and_tools

    orig_df = session.dataframe.copy()

    # Binarize labels on fixture
    res = binarize_labels_tool.invoke({})
    json.dumps(res)

    assert res["status"] == "success"
    assert res["categories"] == ["catA", "catB"]
    assert res["shape"] == [8, 2]
    assert res["counts_per_category"] == {"catA": 4, "catB": 4}

    # Verify 0/1 values per document
    expected_binary = [
        [1, 0],  # row 0: catA
        [1, 0],  # row 1: catA
        [1, 0],  # row 2: catA
        [0, 1],  # row 3: catB
        [0, 1],  # row 4: catB
        [0, 1],  # row 5: catB
        [0, 1],  # row 6: catB
        [1, 0],  # row 7: catA
    ]
    actual_binary = [[row["catA"], row["catB"]] for row in session.artifacts["binarized_labels"].to_dict(orient="records")]
    assert actual_binary == expected_binary

    # Verify session.dataframe is unchanged
    pd_testing = pytest.importorskip("pandas.testing")
    pd_testing.assert_frame_equal(session.dataframe, orig_df)

    # Error: status "error" when no dataframe loaded
    empty_session = SessionState()
    empty_binarize_tool = next(t for t in make_reduction_tools(empty_session) if t.name == "binarize_labels_tool")
    res_err = empty_binarize_tool.invoke({})
    assert res_err.get("status") == "error"
