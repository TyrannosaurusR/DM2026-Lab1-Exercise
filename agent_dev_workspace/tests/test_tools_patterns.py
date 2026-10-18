import os
import json
import numpy as np
import pandas as pd
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def test_known_answers_known_fixture():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # 1. Variance and term_frequency filtering both yield 0 surviving terms
    for filter_m in ["variance", "term_frequency"]:
        res = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": filter_m,
            "algorithm": "fpgrowth",
            "min_sup": 1
        })
        assert res["status"] == "success"
        assert res["n_surviving_terms"] == 0
        assert res["patterns"] == []
        art_key = f"patterns_catA_{filter_m}_fpgrowth"
        assert art_key in session.artifacts
        assert session.artifacts[art_key] == []

    # 2. tfidf + fpgrowth for catA (alpha, support 3) and catB (gamma, support 4) at min_sup=1
    res_catA_fp = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catA_fp["status"] == "success"
    assert res_catA_fp["n_surviving_terms"] == 1
    assert res_catA_fp["patterns"] == [{"terms": ["alpha"], "support": 3}]
    assert "patterns_catA_tfidf_fpgrowth" in session.artifacts
    assert session.artifacts["patterns_catA_tfidf_fpgrowth"] == [{"terms": ["alpha"], "support": 3}]

    res_catB_fp = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catB_fp["status"] == "success"
    assert res_catB_fp["n_surviving_terms"] == 1
    assert res_catB_fp["patterns"] == [{"terms": ["gamma"], "support": 4}]
    assert "patterns_catB_tfidf_fpgrowth" in session.artifacts
    assert session.artifacts["patterns_catB_tfidf_fpgrowth"] == [{"terms": ["gamma"], "support": 4}]

    # 2b. Check min_sup=4 sensitivity (Known Answer #17): catA finds 0 patterns, catB finds {gamma} support 4
    res_catA_sup4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_catA_sup4["status"] == "success"
    assert res_catA_sup4["patterns"] == []

    res_catB_sup4 = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_catB_sup4["status"] == "success"
    assert res_catB_sup4["patterns"] == [{"terms": ["gamma"], "support": 4}]

    # 3. tfidf + topk for catA and catB
    res_catA_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_catA_topk["status"] == "success"
    assert res_catA_topk["patterns"] == [{"terms": ["alpha"], "support": 3}]

    res_catB_topk = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_catB_topk["status"] == "success"
    assert res_catB_topk["patterns"] == [{"terms": ["gamma"], "support": 4}]

    # 4. tfidf + maxfpgrowth yields 0 patterns for both categories
    for cat in ["catA", "catB"]:
        res_max = mine_patterns_tool.invoke({
            "category_name": cat,
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": 1
        })
        assert res_max["status"] == "success"
        assert res_max["n_surviving_terms"] == 1
        assert res_max["patterns"] == []


def test_error_handling_and_validations():
    session = SessionState()
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    res_no_df = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_no_df["status"] == "error"
    assert "No dataframe loaded" in res_no_df["message"]

    session.dataframe = pd.DataFrame({
        "text": ["apple banana", "apple cherry"],
        "category_name": ["fruit", "fruit"]
    })

    # Unknown category
    res_bad_cat = mine_patterns_tool.invoke({
        "category_name": "veggie",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_bad_cat["status"] == "error"
    assert "Category 'veggie' not found" in res_bad_cat["message"]

    # Missing/invalid min_sup for fpgrowth
    res_no_sup = mine_patterns_tool.invoke({
        "category_name": "fruit",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth"
    })
    assert res_no_sup["status"] == "error"
    assert "requires min_sup" in res_no_sup["message"]

    res_bad_sup = mine_patterns_tool.invoke({
        "category_name": "fruit",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 0
    })
    assert res_bad_sup["status"] == "error"
    assert "requires min_sup" in res_bad_sup["message"]

    # Missing/invalid min_sup for maxfpgrowth
    res_no_sup_max = mine_patterns_tool.invoke({
        "category_name": "fruit",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth"
    })
    assert res_no_sup_max["status"] == "error"
    assert "requires min_sup" in res_no_sup_max["message"]

    # Missing/invalid k for topk
    res_no_k = mine_patterns_tool.invoke({
        "category_name": "fruit",
        "filtering_method": "tfidf",
        "algorithm": "topk"
    })
    assert res_no_k["status"] == "error"
    assert "requires k" in res_no_k["message"]

    res_bad_k = mine_patterns_tool.invoke({
        "category_name": "fruit",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 0
    })
    assert res_bad_k["status"] == "error"
    assert "requires k" in res_bad_k["message"]


def test_side_effects_and_json_serializable_and_clean_dir(tmp_path, monkeypatch):
    # Change current working directory to isolated temp path to strictly check no files created
    monkeypatch.chdir(tmp_path)

    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["apple banana", "apple cherry", "apple banana"],
        "category_name": ["catA", "catA", "catA"]
    })

    df_before = session.dataframe.copy()

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    # Assert summary is JSON serializable
    assert json.dumps(res) is not None

    # Assert session.dataframe is unchanged
    pd.testing.assert_frame_equal(session.dataframe, df_before)

    # Assert current working directory remains completely empty after execution
    assert os.listdir(tmp_path) == []


def test_missing_text_nan_none():
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["apple banana", None, np.nan, "apple cherry", "apple banana"],
        "category_name": ["catX", "catX", "catX", "catX", "catX"]
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    res = mine_patterns_tool.invoke({
        "category_name": "catX",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res["status"] == "success"
    assert res["n_surviving_terms"] > 0
    assert len(res["patterns"]) > 0
    assert res["patterns"][0]["terms"] == ["apple"]


def test_inclusive_percentile_filtering_ties():
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": [
            "t1 t5 t5 t5 t5 t5 t5 t5 t5 t5 t5",
            "t2",
            "t3",
            "t4"
        ],
        "category_name": ["catT", "catT", "catT", "catT"]
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Test term_frequency filtering tie behavior
    res_tf = mine_patterns_tool.invoke({
        "category_name": "catT",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tf["status"] == "success"
    assert res_tf["n_surviving_terms"] == 4

    # Test variance filtering tie behavior
    res_var = mine_patterns_tool.invoke({
        "category_name": "catT",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var["status"] == "success"
    assert res_var["n_surviving_terms"] == 4
