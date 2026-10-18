import matplotlib
matplotlib.use("Agg")

import json
import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


@pytest.fixture
def dtm_session_and_tools():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load student-built tools from tools_dtm.py
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Load fixture dataset
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    return session, build_dtm_tool, term_frequency_tool, dtm_heatmap_tool


def test_build_dtm_tool(dtm_session_and_tools):
    session, build_dtm_tool, _, _ = dtm_session_and_tools

    # Known Answer #4
    dtm_res = build_dtm_tool.invoke({})
    
    # Assert JSON serializable
    json.dumps(dtm_res)

    assert dtm_res["n_documents"] == 8
    assert dtm_res["n_terms"] == 5
    assert dtm_res["non_zero"] == 21
    assert dtm_res["total_elements"] == 40
    assert dtm_res["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Check vocabulary terms
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]

    # Check row 7 (empty text) is all zeros in feature_matrix
    row_7_counts = session.feature_matrix[7, :].toarray().flatten()
    assert np.all(row_7_counts == 0)

    # Check count_vectorizer saved in session.artifacts
    assert "count_vectorizer" in session.artifacts
    assert session.artifacts["count_vectorizer"] is not None

    # Check session.categories and session.labels match fixture labels in row order
    expected_labels = np.array(["catA", "catA", "catA", "catB", "catB", "catB", "catB", "catA"])
    assert session.categories == ["catA", "catB"]
    np.testing.assert_array_equal(session.labels, expected_labels)


def test_term_frequency_tool(dtm_session_and_tools):
    session, build_dtm_tool, term_frequency_tool, _ = dtm_session_and_tools

    # 1. Calling term_frequency_tool before build_dtm_tool returns error
    pre_dtm_res = term_frequency_tool.invoke({})
    assert pre_dtm_res.get("status") == "error"

    # First build DTM
    build_dtm_tool.invoke({})

    # 2. Test invalid top_n parameter (0 and -1)
    err_0 = term_frequency_tool.invoke({"top_n": 0})
    assert err_0.get("status") == "error"

    err_neg1 = term_frequency_tool.invoke({"top_n": -1})
    assert err_neg1.get("status") == "error"

    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }

    # 3. Call term_frequency_tool with NO args (default top_n) -> compares whole dict to expected_frequencies
    tf_noargs_res = term_frequency_tool.invoke({})
    json.dumps(tf_noargs_res)
    assert tf_noargs_res["term_frequencies"] == expected_frequencies

    # 4. Known Answer #14 with top_n=10
    tf_res = term_frequency_tool.invoke({"top_n": 10})
    json.dumps(tf_res)
    assert tf_res["term_frequencies"] == expected_frequencies

    # Check full_report by position and column order: first column term, second frequency, sorted descending
    result_id = tf_res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]
    assert full_report is not None
    assert list(full_report.columns[:2]) == ["term", "frequency"]
    
    frequencies = full_report["frequency"].tolist()
    assert frequencies == sorted(frequencies, reverse=True)

    # Restored assertions: check length and dict zip of full_report
    assert len(full_report) == 5
    report_dict = dict(zip(full_report["term"], full_report["frequency"]))
    assert report_dict == expected_frequencies

    # 5. Test top_n=2 parameter
    tf_top2_res = term_frequency_tool.invoke({"top_n": 2})
    json.dumps(tf_top2_res)
    expected_top2 = {
        "always": 7,
        "gamma": 7
    }
    assert tf_top2_res["term_frequencies"] == expected_top2

    # Check full_report still holds all 5 terms
    top2_result_id = tf_top2_res["result_id"]
    top2_stored_entry = session.results_store[top2_result_id]
    top2_full_report = top2_stored_entry["full_report"]
    assert top2_full_report is not None
    assert len(top2_full_report) == 5


def test_dtm_heatmap_tool(dtm_session_and_tools):
    session, build_dtm_tool, _, dtm_heatmap_tool = dtm_session_and_tools

    # 1. Calling dtm_heatmap_tool before build_dtm_tool returns error
    pre_dtm_res = dtm_heatmap_tool.invoke({})
    assert pre_dtm_res.get("status") == "error"

    # First build DTM
    build_dtm_tool.invoke({})

    # 2. Test invalid n_terms and n_documents parameters (0 and -1)
    assert dtm_heatmap_tool.invoke({"n_terms": 0}).get("status") == "error"
    assert dtm_heatmap_tool.invoke({"n_terms": -1}).get("status") == "error"
    assert dtm_heatmap_tool.invoke({"n_documents": 0}).get("status") == "error"
    assert dtm_heatmap_tool.invoke({"n_documents": -1}).get("status") == "error"

    # Known Answer #21 (default 20x20 parameters return full 8x5 matrix)
    heatmap_res = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})
    json.dumps(heatmap_res)

    assert heatmap_res["n_documents"] == 8
    assert heatmap_res["n_terms"] == 5

    expected_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert heatmap_res["matrix_slice"] == expected_matrix

    # Check pending_figure is set and verify plotted image data matches
    assert session.pending_figure is not None
    ax = session.pending_figure.axes[0]
    mesh = ax.collections[0]
    plotted_array = mesh.get_array().data
    np.testing.assert_array_equal(plotted_array, np.array(expected_matrix))

    # Test explicit matrix slicing with n_terms=2, n_documents=3 (Known Answer #21 slice)
    sliced_res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    json.dumps(sliced_res)

    assert sliced_res["n_documents"] == 3
    assert sliced_res["n_terms"] == 2
    assert sliced_res["terms"] == ["alpha", "always"]
    assert sliced_res["documents"] == ["Doc_0", "Doc_1", "Doc_2"]
    assert sliced_res["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
