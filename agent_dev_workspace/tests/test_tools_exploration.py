import json
import pytest
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.figure

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools

def setup_fixture_session():
    """Helper to set up a shared session loaded with sample_fixture.csv and DTM built."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Import build_dtm_tool from tools_dtm to populate feature_matrix and feature_names
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})
    
    return session

def test_cosine_similarity_known_answers():
    session = setup_fixture_session()
    tools = make_tools(session)
    cosine_tool = next(t for t in tools if t.name == "cosine_similarity_tool")
    
    # Known answer 16 pairs
    # 0 vs 1
    res_0_1 = cosine_tool.invoke({"doc_index_a": 0, "doc_index_b": 1})
    assert res_0_1["status"] == "success"
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-4)
    
    # 0 vs 3
    res_0_3 = cosine_tool.invoke({"doc_index_a": 0, "doc_index_b": 3})
    assert res_0_3["status"] == "success"
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-4)
    
    # 3 vs 4
    res_3_4 = cosine_tool.invoke({"doc_index_a": 3, "doc_index_b": 4})
    assert res_3_4["status"] == "success"
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-4)
    
    # 0 vs 7 (empty text)
    res_0_7 = cosine_tool.invoke({"doc_index_a": 0, "doc_index_b": 7})
    assert res_0_7["status"] == "success"
    assert res_0_7["similarity"] == 0.0
    assert not np.isnan(res_0_7["similarity"])
    
    # Document compared with itself
    res_0_0 = cosine_tool.invoke({"doc_index_a": 0, "doc_index_b": 0})
    assert res_0_0["similarity"] == pytest.approx(1.0, abs=1e-4)

def test_cosine_similarity_errors_and_json():
    session = SessionState() # No DTM built
    tools = make_tools(session)
    cosine_tool = next(t for t in tools if t.name == "cosine_similarity_tool")
    
    # No DTM error
    res_no_dtm = cosine_tool.invoke({"doc_index_a": 0, "doc_index_b": 1})
    assert res_no_dtm["status"] == "error"
    
    # Setup fixture for index range checks
    session = setup_fixture_session()
    tools = make_tools(session)
    cosine_tool = next(t for t in tools if t.name == "cosine_similarity_tool")
    
    # Negative first index
    res_neg_a = cosine_tool.invoke({"doc_index_a": -1, "doc_index_b": 0})
    assert res_neg_a["status"] == "error"

    # Negative second index
    res_neg_b = cosine_tool.invoke({"doc_index_a": 0, "doc_index_b": -1})
    assert res_neg_b["status"] == "error"
    
    # Out of range first index
    res_out_a = cosine_tool.invoke({"doc_index_a": 100, "doc_index_b": 0})
    assert res_out_a["status"] == "error"

    # Out of range second index
    res_out_b = cosine_tool.invoke({"doc_index_a": 0, "doc_index_b": 100})
    assert res_out_b["status"] == "error"
    
    # Check json serializability
    res_valid = cosine_tool.invoke({"doc_index_a": 0, "doc_index_b": 1})
    assert json.dumps(res_valid) is not None

def test_feature_correlation_matrix_known_answers():
    session = setup_fixture_session()
    tools = make_tools(session)
    feat_corr_tool = next(t for t in tools if t.name == "feature_correlation_matrix_tool")
    
    # Copy feature_matrix and feature_names to verify they remain unchanged after tool call
    matrix_before = session.feature_matrix.toarray().copy()
    names_before = list(session.feature_names)

    res = feat_corr_tool.invoke({})
    assert res["status"] == "success"
    
    # Check session matrix and names are unchanged
    np.testing.assert_array_equal(session.feature_matrix.toarray(), matrix_before)
    assert session.feature_names == names_before

    # Known answer 19
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["top_terms"] == expected_terms
    
    expected_matrix = [
        [ 1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718,  1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882,  0.8307,  1.0000, -0.7746,  0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [ 0.2601,  0.3140,  0.3780,  0.2928,  1.0000]
    ]
    
    corr_matrix = np.array(res["correlation_matrix"])
    assert np.allclose(corr_matrix, expected_matrix, atol=1e-4)
    
    # Diagonal is 1.0
    assert np.allclose(np.diag(corr_matrix), 1.0, atol=1e-4)
    
    # Symmetric
    assert np.allclose(corr_matrix, corr_matrix.T, atol=1e-4)
    
    # Figure state
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

def test_feature_correlation_matrix_errors():
    session = SessionState()
    tools = make_tools(session)
    feat_corr_tool = next(t for t in tools if t.name == "feature_correlation_matrix_tool")
    
    # No DTM
    res_no_dtm = feat_corr_tool.invoke({})
    assert res_no_dtm["status"] == "error"
    
    # Too few documents or terms (< 2)
    session.feature_matrix = csr_matrix(np.array([[1], [2]])) # 2 docs, 1 term
    session.feature_names = ["term1"]
    res_few_terms = feat_corr_tool.invoke({})
    assert res_few_terms["status"] == "error"
    
    session.feature_matrix = csr_matrix(np.array([[1, 2]])) # 1 doc, 2 terms
    session.feature_names = ["term1", "term2"]
    res_few_docs = feat_corr_tool.invoke({})
    assert res_few_docs["status"] == "error"

def test_feature_correlation_matrix_zero_variance():
    session = SessionState()
    tools = make_tools(session)
    feat_corr_tool = next(t for t in tools if t.name == "feature_correlation_matrix_tool")
    
    # DTM with a constant column (zero variance)
    # term0: constant 5 (zero variance), term1: [1, 2, 3]
    dense_matrix = np.array([
        [5, 1],
        [5, 2],
        [5, 3]
    ])
    session.feature_matrix = csr_matrix(dense_matrix)
    session.feature_names = ["const_term", "var_term"]
    
    res = feat_corr_tool.invoke({})
    assert res["status"] == "success"
    corr = np.array(res["correlation_matrix"])
    
    # Check no NaN values exist
    assert not np.isnan(corr).any()
    
    # Check row for constant term: 0 correlation with var_term, 1.0 on diagonal
    const_idx = res["top_terms"].index("const_term")
    var_idx = res["top_terms"].index("var_term")
    
    assert corr[const_idx, const_idx] == pytest.approx(1.0)
    assert corr[const_idx, var_idx] == pytest.approx(0.0)

def test_feature_correlation_matrix_top_20_cutoff():
    session = SessionState()
    tools = make_tools(session)
    feat_corr_tool = next(t for t in tools if t.name == "feature_correlation_matrix_tool")
    
    # Matrix with 25 terms and 10 documents
    np.random.seed(42)
    data = np.random.randint(0, 10, size=(10, 25))
    session.feature_matrix = csr_matrix(data)
    session.feature_names = [f"term_{i}" for i in range(25)]
    
    res = feat_corr_tool.invoke({})
    assert res["status"] == "success"
    
    # Exactly 20 terms returned
    assert len(res["top_terms"]) == 20
    assert np.array(res["correlation_matrix"]).shape == (20, 20)
    
    # Verify they are indeed the 20 highest variance terms
    variances = np.var(data, axis=0, ddof=0)
    expected_top_20_indices = np.argsort(variances)[::-1][:20]
    expected_top_20_names = [session.feature_names[i] for i in expected_top_20_indices]
    
    assert res["top_terms"] == expected_top_20_names
