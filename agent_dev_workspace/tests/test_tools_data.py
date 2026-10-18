import json
import pandas as pd
import pytest
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def load_fixture_data():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    tools = {t.name: t for t in make_tools(session)}
    return session, tools

def test_check_missing_tool_known_answer_2():
    session, tools = load_fixture_data()
    initial_shape = session.dataframe.shape
    
    res = tools["check_missing_tool"].invoke({})
    
    assert res["status"] == "success"
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]
    assert session.dataframe.shape == initial_shape
    json.dumps(res)

def test_check_duplicates_tool_known_answer_3():
    # Test check without drop
    session, tools = load_fixture_data()
    res = tools["check_duplicates_tool"].invoke({"drop": False})
    assert res["status"] == "success"
    assert res["duplicate_count"] == 1
    assert res["initial_rows"] == 8
    assert res["final_rows"] == 8
    assert res["cleared_dtm"] is False
    assert len(session.dataframe) == 8
    json.dumps(res)

    # Test drop without DTM
    session, tools = load_fixture_data()
    res = tools["check_duplicates_tool"].invoke({"drop": True})
    assert res["status"] == "success"
    assert res["duplicate_count"] == 1
    assert res["initial_rows"] == 8
    assert res["final_rows"] == 6
    assert res["cleared_dtm"] is False
    assert len(session.dataframe) == 6
    json.dumps(res)

    # Test drop with DTM clearing
    session, tools = load_fixture_data()
    session.feature_matrix = "fake_matrix"
    session.feature_names = ["fake_term"]
    res = tools["check_duplicates_tool"].invoke({"drop": True})
    assert res["status"] == "success"
    assert res["cleared_dtm"] is True
    assert session.feature_matrix is None
    assert session.feature_names is None
    json.dumps(res)

def test_sample_data_tool_known_answer_13():
    session, tools = load_fixture_data()
    initial_len = len(session.dataframe)
    
    res = tools["sample_data_tool"].invoke({"n": 4, "random_state": 42})
    assert res["status"] == "success"
    assert res["sample_size"] == 4
    assert res["sample_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == initial_len
    json.dumps(res)

    # Test invalid n
    err_res1 = tools["sample_data_tool"].invoke({"n": 0})
    assert err_res1["status"] == "error"

    err_res2 = tools["sample_data_tool"].invoke({"n": 9})
    assert err_res2["status"] == "error"

def test_describe_data_tool_known_answer_18():
    session, tools = load_fixture_data()
    
    res = tools["describe_data_tool"].invoke({})
    assert res["status"] == "success"
    json.dumps(res)

    # Verify text_length persistent column was not added
    assert "text_length" not in session.dataframe.columns

    # Verify pending figure
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # Known Answer 18 checks: Overall stats
    overall = res["overall_stats"]
    assert overall["count"] == pytest.approx(8, abs=1e-3)
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == pytest.approx(0, abs=1e-3)
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == pytest.approx(30, abs=1e-3)

    # Per-category stats: catA
    catA = res["per_category_stats"]["catA"]
    assert catA["count"] == pytest.approx(4, abs=1e-3)
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == pytest.approx(0, abs=1e-3)
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == pytest.approx(29, abs=1e-3)

    # Per-category stats: catB
    catB = res["per_category_stats"]["catB"]
    assert catB["count"] == pytest.approx(4, abs=1e-3)
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == pytest.approx(18, abs=1e-3)
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == pytest.approx(30, abs=1e-3)

def test_inspect_data_tool():
    session, tools = load_fixture_data()
    
    res = tools["inspect_data_tool"].invoke({"n": 5})
    assert res["status"] == "success"
    assert res["shape"] == [8, 3]  # text, category, category_name
    assert "text" in res["columns"]
    assert "category_name" in res["columns"]
    assert len(res["preview"]) == 5
    json.dumps(res)

    # Check long text truncation logic
    session.dataframe.at[0, "text"] = "a" * 150
    res_trunc = tools["inspect_data_tool"].invoke({"n": 1})
    assert res_trunc["preview"][0]["text"].endswith("...")
    assert len(res_trunc["preview"][0]["text"]) == 103

    # Error handling for n < 1
    err_res = tools["inspect_data_tool"].invoke({"n": 0})
    assert err_res["status"] == "error"

def test_tokenize_tool_known_answer_10():
    session, tools = load_fixture_data()
    initial_shape = session.dataframe.shape
    initial_cols = list(session.dataframe.columns)

    res = tools["tokenize_tool"].invoke({})
    assert res["status"] == "success"
    assert res["n_documents"] == 8
    assert res["total_tokens"] == 27
    json.dumps(res)

    # session.dataframe must remain unchanged
    assert session.dataframe.shape == initial_shape
    assert list(session.dataframe.columns) == initial_cols

    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    assert session.artifacts["tokens"] == expected_tokens
    assert res["first_3_tokens"] == expected_tokens[:3]

    # Custom edge-case dataframe test (None, whitespace, mixed case)
    custom_session = SessionState()
    custom_session.dataframe = pd.DataFrame({
        "text": [None, "   ", "Hello WORLD"]
    })
    custom_tools = {t.name: t for t in make_tools(custom_session)}
    custom_res = custom_tools["tokenize_tool"].invoke({})

    assert custom_res["status"] == "success"
    assert custom_res["n_documents"] == 3
    assert custom_session.artifacts["tokens"] == [[], [], ["hello", "world"]]
    json.dumps(custom_res)

def test_list_files_tool_known_answer_11():
    session = SessionState()
    tools = {t.name: t for t in make_tools(session)}

    # Known answer 11
    res = tools["list_files_tool"].invoke({"subdirectory": "newdataset"})
    assert res["status"] == "success"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]
    json.dumps(res)

    # Valid internal traversal staying inside repo root
    res_internal = tools["list_files_tool"].invoke({"subdirectory": "agent_dev/../newdataset"})
    assert res_internal["status"] == "success"
    assert res_internal["directories"] == []
    assert res_internal["files"] == ["Reddit-stock-sentiment.csv"]
    json.dumps(res_internal)

    # Path traversal attempts escaping repo
    invalid_paths = ["..", "../..", "../outside", "/etc", "newdataset/../..", "C:\\", "newdataset\x00path"]
    for path in invalid_paths:
        err_res = tools["list_files_tool"].invoke({"subdirectory": path})
        assert err_res["status"] == "error"
        assert "message" in err_res
        json.dumps(err_res)

    # Non-existent directory
    non_existent_res = tools["list_files_tool"].invoke({"subdirectory": "nonexistent_dir_xyz"})
    assert non_existent_res["status"] == "error"

    # File path passed instead of directory
    file_res = tools["list_files_tool"].invoke({"subdirectory": "agent_dev/sample_fixture.csv"})
    assert file_res["status"] == "error"

def test_unloaded_dataframe_handling():
    empty_session = SessionState()
    tools = {t.name: t for t in make_tools(empty_session)}

    for name, tool_obj in tools.items():
        if name == "list_files_tool":
            continue
        if name in ["sample_data_tool", "inspect_data_tool"]:
            res = tool_obj.invoke({"n": 5})
        else:
            res = tool_obj.invoke({})
        assert res["status"] == "error"
        assert "message" in res
        json.dumps(res)
