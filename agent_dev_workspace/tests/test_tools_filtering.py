"""Tests for variance_filter_tool, pearson_filter_tool, and spearman_filter_tool in tools_filtering.py."""

import json
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import spearmanr

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def _setup_fixture_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    return session


def test_variance_filter_known_answer_5():
    session = _setup_fixture_session()
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Run filter with threshold=0.15
    res = variance_filter_tool.invoke({"threshold": 0.15})
    assert res["status"] == "success"
    assert res["total_terms"] == 5
    assert res["retained_terms"] == 4
    assert res["removed_terms"] == 1
    assert [t["term"] for t in res["removed_terms_preview"]] == ["always"]

    # Verify full_report and exact known answer #5 variances
    stored_result = session.results_store[res["result_id"]]
    full_df = stored_result["full_report"]

    assert list(full_df.columns) == ["term", "variance", "keep"]
    
    # Check sorting: descending by variance
    assert full_df["variance"].is_monotonic_decreasing

    # Known answer #5 expected variances
    expected_variances = {
        "alpha": 1.1875,
        "gamma": 1.109375,
        "delta": 0.25,
        "beta": 0.234375,
        "always": 0.109375,
    }

    term_to_var = dict(zip(full_df["term"], full_df["variance"]))
    for term, exp_var in expected_variances.items():
        assert term_to_var[term] == pytest.approx(exp_var, abs=1e-4)

    # Check keep column at threshold 0.15
    term_to_keep = dict(zip(full_df["term"], full_df["keep"]))
    assert term_to_keep["alpha"] is True or term_to_keep["alpha"] == True
    assert term_to_keep["gamma"] is True or term_to_keep["gamma"] == True
    assert term_to_keep["delta"] is True or term_to_keep["delta"] == True
    assert term_to_keep["beta"] is True or term_to_keep["beta"] == True
    assert term_to_keep["always"] is False or term_to_keep["always"] == False

    # Summary must be JSON serializable
    json_str = json.dumps(res)
    assert isinstance(json_str, str)


def test_variance_filter_threshold_equal_value_removed():
    session = _setup_fixture_session()
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # threshold=0.25: alpha (1.1875) and gamma (1.109375) > 0.25 are kept (retained_terms=2);
    # delta (0.25), beta (0.234375), and always (0.109375) <= 0.25 are removed (removed_terms=3)
    res = variance_filter_tool.invoke({"threshold": 0.25})
    assert res["status"] == "success"
    assert res["retained_terms"] == 2
    assert res["removed_terms"] == 3

    stored_result = session.results_store[res["result_id"]]
    full_df = stored_result["full_report"]
    term_to_keep = dict(zip(full_df["term"], full_df["keep"]))

    assert term_to_keep["alpha"] is True or term_to_keep["alpha"] == True
    assert term_to_keep["gamma"] is True or term_to_keep["gamma"] == True
    assert term_to_keep["delta"] is False or term_to_keep["delta"] == False
    assert term_to_keep["beta"] is False or term_to_keep["beta"] == False
    assert term_to_keep["always"] is False or term_to_keep["always"] == False


def test_variance_filter_side_effects_free_and_max_preview():
    session = _setup_fixture_session()
    matrix_before = session.feature_matrix.copy()
    names_before = list(session.feature_names)

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    res = variance_filter_tool.invoke({"threshold": 0.15, "max_preview": 0})
    assert res["status"] == "success"
    assert res["removed_terms_preview"] == []

    # Check session feature matrix/names are unmodified
    assert np.array_equal(session.feature_matrix.toarray(), matrix_before.toarray())
    assert session.feature_names == names_before


def test_variance_filter_error_cases():
    session = SessionState()
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 1. No DTM built
    res_no_dtm = variance_filter_tool.invoke({"threshold": 0.0})
    assert res_no_dtm["status"] == "error"
    assert "No document-term matrix available" in res_no_dtm["message"]

    # Build DTM to test parameter errors
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    # 2. Negative threshold
    res_neg_thresh = variance_filter_tool.invoke({"threshold": -0.1})
    assert res_neg_thresh["status"] == "error"
    assert "non-negative" in res_neg_thresh["message"]

    # 3. Negative max_preview
    res_neg_prev = variance_filter_tool.invoke({"max_preview": -1})
    assert res_neg_prev["status"] == "error"
    assert "non-negative" in res_neg_prev["message"]


def test_variance_filter_constant_column():
    session = SessionState()
    # 300 docs x 2 terms: term 'constant' has value 3 in all 300 rows, term 'varying' varies
    data = np.zeros((300, 2), dtype=int)
    data[:, 0] = 3
    data[:, 1] = np.tile([0, 1, 2], 100)

    session.feature_matrix = sparse.csr_matrix(data)
    session.feature_names = ["constant", "varying"]

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    res = variance_filter_tool.invoke({"threshold": 0.0})
    assert res["status"] == "success"
    assert res["retained_terms"] == 1
    assert res["removed_terms"] == 1

    stored_result = session.results_store[res["result_id"]]
    full_df = stored_result["full_report"]
    term_to_var = dict(zip(full_df["term"], full_df["variance"]))
    term_to_keep = dict(zip(full_df["term"], full_df["keep"]))

    assert term_to_var["constant"] == 0.0
    assert term_to_keep["constant"] is False or term_to_keep["constant"] == False
    assert term_to_keep["varying"] is True or term_to_keep["varying"] == True


# --- Tests for pearson_filter_tool ---

def test_pearson_filter_known_answer_6():
    session = _setup_fixture_session()
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    res = pearson_filter_tool.invoke({"target_class": "catB"})
    assert res["status"] == "success"
    assert res["target_class"] == "catB"

    # Previews checks
    assert res["top_positive_preview"][0]["term"] == "delta"
    assert res["top_negative_preview"][0]["term"] == "beta"

    # Verify full_report and exact known answer #6 Pearson values
    stored_result = session.results_store[res["result_id"]]
    full_df = stored_result["full_report"]

    assert list(full_df.columns) == ["term", "pearson_r"]

    # Verify sorted descending by absolute pearson_r magnitude
    abs_r_values = full_df["pearson_r"].abs().values
    assert np.all(abs_r_values[:-1] >= abs_r_values[1:])

    # Known answer #6 expected Pearson correlations for target_class="catB"
    expected_pearson = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    term_to_r = dict(zip(full_df["term"], full_df["pearson_r"]))
    for term, exp_r in expected_pearson.items():
        assert term_to_r[term] == pytest.approx(exp_r, abs=1e-4)

    # Summary must be JSON serializable
    json_str = json.dumps(res)
    assert isinstance(json_str, str)


def test_pearson_filter_previews_and_side_effects():
    session = _setup_fixture_session()
    matrix_before = session.feature_matrix.copy()
    names_before = list(session.feature_names)

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    res = pearson_filter_tool.invoke({"target_class": "catB", "max_preview": 0})
    assert res["status"] == "success"
    assert res["top_positive_preview"] == []
    assert res["top_negative_preview"] == []

    # Check session feature matrix/names are unmodified
    assert np.array_equal(session.feature_matrix.toarray(), matrix_before.toarray())
    assert session.feature_names == names_before


def test_pearson_filter_error_cases():
    session = SessionState()
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # 1. No DTM built
    res_no_dtm = pearson_filter_tool.invoke({"target_class": "catB"})
    assert res_no_dtm["status"] == "error"
    assert "No document-term matrix available" in res_no_dtm["message"]

    # Setup fixture DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    # 2. Negative max_preview
    res_neg_prev = pearson_filter_tool.invoke({"target_class": "catB", "max_preview": -1})
    assert res_neg_prev["status"] == "error"
    assert "non-negative" in res_neg_prev["message"]

    # 3. Unknown target_class (must list valid categories catA, catB)
    res_unknown_cat = pearson_filter_tool.invoke({"target_class": "catC"})
    assert res_unknown_cat["status"] == "error"
    assert "Unknown target_class 'catC'" in res_unknown_cat["message"]
    assert "catA, catB" in res_unknown_cat["message"]

    # 4. Dataset with only one category
    single_cat_session = SessionState()
    single_cat_session.feature_matrix = sparse.csr_matrix(np.array([[1], [2]]))
    single_cat_session.feature_names = ["term1"]
    single_cat_session.labels = np.array(["catA", "catA"])

    filtering_tools_single = make_filtering_tools(single_cat_session)
    pearson_single = next(t for t in filtering_tools_single if t.name == "pearson_filter_tool")

    res_single_cat = pearson_single.invoke({"target_class": "catA"})
    assert res_single_cat["status"] == "error"
    assert "At least two distinct categories" in res_single_cat["message"]


def test_pearson_filter_constant_column():
    session = SessionState()
    # 100 docs x 2 terms: term 'constant' has value 5 across all docs, term 'varying' varies
    data = np.zeros((100, 2), dtype=int)
    data[:, 0] = 5
    data[:, 1] = np.tile([0, 1], 50)

    session.feature_matrix = sparse.csr_matrix(data)
    session.feature_names = ["constant", "varying"]
    session.labels = np.array(["catA", "catB"] * 50)

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    res = pearson_filter_tool.invoke({"target_class": "catB"})
    assert res["status"] == "success"

    stored_result = session.results_store[res["result_id"]]
    full_df = stored_result["full_report"]
    term_to_r = dict(zip(full_df["term"], full_df["pearson_r"]))

    # Constant term must have r = 0.0 (within 1e-6) and no NaN
    assert not np.isnan(term_to_r["constant"])
    assert term_to_r["constant"] == pytest.approx(0.0, abs=1e-6)


def test_pearson_filter_all_zeros_column():
    session = SessionState()
    # 100 docs x 2 terms: term 'unused' has 0 counts across all docs (exact 0 std)
    data = np.zeros((100, 2), dtype=int)
    data[:, 1] = np.tile([0, 1], 50)

    session.feature_matrix = sparse.csr_matrix(data)
    session.feature_names = ["unused", "varying"]
    session.labels = np.array(["catA", "catB"] * 50)

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    res = pearson_filter_tool.invoke({"target_class": "catB"})
    assert res["status"] == "success"

    stored_result = session.results_store[res["result_id"]]
    full_df = stored_result["full_report"]
    term_to_r = dict(zip(full_df["term"], full_df["pearson_r"]))

    # All-zeros term std is exactly 0.0: r must be exactly 0.0 with no NaN
    assert not np.isnan(term_to_r["unused"])
    assert term_to_r["unused"] == 0.0

    # Must not appear in positive preview (r > 0) or negative preview (r < 0)
    pos_terms = [t["term"] for t in res["top_positive_preview"]]
    neg_terms = [t["term"] for t in res["top_negative_preview"]]
    assert "unused" not in pos_terms
    assert "unused" not in neg_terms


# --- Tests for spearman_filter_tool ---

def test_spearman_filter_known_answer_6():
    session = _setup_fixture_session()
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    res = spearman_filter_tool.invoke({"target_class": "catB"})
    assert res["status"] == "success"
    assert res["target_class"] == "catB"

    # Previews checks
    assert res["top_positive_preview"][0]["term"] == "delta"
    assert res["top_negative_preview"][0]["term"] == "beta"

    # Verify full_report and exact known answer #6 Spearman values
    stored_result = session.results_store[res["result_id"]]
    full_df = stored_result["full_report"]

    assert list(full_df.columns) == ["term", "spearman_r"]

    # Verify sorted descending by absolute spearman_r magnitude
    abs_r_values = full_df["spearman_r"].abs().values
    assert np.all(abs_r_values[:-1] >= abs_r_values[1:])

    # Known answer #6 expected Spearman correlations for target_class="catB"
    expected_spearman = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    term_to_r = dict(zip(full_df["term"], full_df["spearman_r"]))
    for term, exp_r in expected_spearman.items():
        assert term_to_r[term] == pytest.approx(exp_r, abs=1e-4)

    # Summary must be JSON serializable
    json_str = json.dumps(res)
    assert isinstance(json_str, str)


def test_spearman_filter_previews_and_side_effects():
    session = _setup_fixture_session()
    matrix_before = session.feature_matrix.copy()
    names_before = list(session.feature_names)

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    res = spearman_filter_tool.invoke({"target_class": "catB", "max_preview": 0})
    assert res["status"] == "success"
    assert res["top_positive_preview"] == []
    assert res["top_negative_preview"] == []

    # Check session feature matrix/names are unmodified
    assert np.array_equal(session.feature_matrix.toarray(), matrix_before.toarray())
    assert session.feature_names == names_before


def test_spearman_filter_error_cases():
    session = SessionState()
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # 1. No DTM built
    res_no_dtm = spearman_filter_tool.invoke({"target_class": "catB"})
    assert res_no_dtm["status"] == "error"
    assert "No document-term matrix available" in res_no_dtm["message"]

    # Setup fixture DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    # 2. Negative max_preview
    res_neg_prev = spearman_filter_tool.invoke({"target_class": "catB", "max_preview": -1})
    assert res_neg_prev["status"] == "error"
    assert "non-negative" in res_neg_prev["message"]

    # 3. Unknown target_class (must list valid categories catA, catB)
    res_unknown_cat = spearman_filter_tool.invoke({"target_class": "catC"})
    assert res_unknown_cat["status"] == "error"
    assert "Unknown target_class 'catC'" in res_unknown_cat["message"]
    assert "catA, catB" in res_unknown_cat["message"]

    # 4. Dataset with only one category
    single_cat_session = SessionState()
    single_cat_session.feature_matrix = sparse.csr_matrix(np.array([[1], [2]]))
    single_cat_session.feature_names = ["term1"]
    single_cat_session.labels = np.array(["catA", "catA"])

    filtering_tools_single = make_filtering_tools(single_cat_session)
    spearman_single = next(t for t in filtering_tools_single if t.name == "spearman_filter_tool")

    res_single_cat = spearman_single.invoke({"target_class": "catA"})
    assert res_single_cat["status"] == "error"
    assert "At least two distinct categories" in res_single_cat["message"]


def test_spearman_filter_constant_and_all_zeros():
    session = SessionState()
    # 100 docs x 3 terms:
    # term0 ('constant'): 5 across all docs
    # term1 ('unused'): 0 across all docs
    # term2 ('varying'): varies
    data = np.zeros((100, 3), dtype=int)
    data[:, 0] = 5
    data[:, 2] = np.tile([0, 1], 50)

    session.feature_matrix = sparse.csr_matrix(data)
    session.feature_names = ["constant", "unused", "varying"]
    session.labels = np.array(["catA", "catB"] * 50)

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    res = spearman_filter_tool.invoke({"target_class": "catB"})
    assert res["status"] == "success"

    stored_result = session.results_store[res["result_id"]]
    full_df = stored_result["full_report"]
    term_to_r = dict(zip(full_df["term"], full_df["spearman_r"]))

    # Constant and unused terms must have r = 0.0 with no NaN
    assert not np.isnan(term_to_r["constant"])
    assert term_to_r["constant"] == 0.0

    assert not np.isnan(term_to_r["unused"])
    assert term_to_r["unused"] == 0.0

    pos_terms = [t["term"] for t in res["top_positive_preview"]]
    neg_terms = [t["term"] for t in res["top_negative_preview"]]
    assert "constant" not in pos_terms and "constant" not in neg_terms
    assert "unused" not in pos_terms and "unused" not in neg_terms


def test_spearman_filter_multi_block():
    session = SessionState()
    np.random.seed(42)
    n_docs = 20
    n_terms = 4500  # Exceeds BLOCK_SIZE = 2000 so block loop runs > 2 times

    # Random sparse data
    dense_data = np.random.randint(0, 5, size=(n_docs, n_terms))
    session.feature_matrix = sparse.csr_matrix(dense_data)
    session.feature_names = [f"term_{i}" for i in range(n_terms)]
    binary_y = np.array([0, 1] * (n_docs // 2))
    session.labels = np.array(["catA" if y == 0 else "catB" for y in binary_y])

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    res = spearman_filter_tool.invoke({"target_class": "catB"})
    assert res["status"] == "success"

    stored_result = session.results_store[res["result_id"]]
    full_df = stored_result["full_report"]
    term_to_r = dict(zip(full_df["term"], full_df["spearman_r"]))

    # Test indices near block boundaries and middle: 0, 1999, 2000, 3999, 4000, 4499
    check_indices = [0, 1999, 2000, 3999, 4000, 4499]
    for idx in check_indices:
        term_name = f"term_{idx}"
        col = dense_data[:, idx]
        expected_r, _ = spearmanr(col, binary_y)
        if np.isnan(expected_r):
            expected_r = 0.0
        assert term_to_r[term_name] == pytest.approx(expected_r, abs=1e-4)
