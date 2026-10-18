import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
import nltk

def make_tools(session):
    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks for missing, empty, or whitespace-only text values in session.dataframe.

        Does not modify session.dataframe.

        Returns:
            Dict containing status, result_id, missing_count, and missing_indices.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded."}
        
        df = session.dataframe
        text_col = df["text"].fillna("")
        missing_mask = (df["text"].isna()) | (text_col.astype(str).str.strip() == "")
        missing_indices = df[missing_mask].index.tolist()
        
        res_id = session.next_result_id("missing")
        summary = {
            "status": "success",
            "result_id": res_id,
            "missing_count": len(missing_indices),
            "missing_indices": missing_indices
        }
        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Checks for duplicate documents in session.dataframe and optionally drops them.

        When dropping, removes every copy of a duplicated row (keep=False).
        If rows are dropped, clears session.feature_matrix and session.feature_names.

        Args:
            drop: If True, removes all copies of duplicated rows from session.dataframe.

        Returns:
            Dict containing status, result_id, duplicate_count, initial_rows, final_rows, and cleared_dtm flag.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded."}
        
        df = session.dataframe
        duplicate_count = int(df.duplicated().sum())
        initial_rows = len(df)
        cleared_dtm = False

        if drop and duplicate_count > 0:
            df_cleaned = df.drop_duplicates(keep=False).reset_index(drop=True)
            session.dataframe = df_cleaned
            
            if session.feature_matrix is not None or session.feature_names is not None:
                session.feature_matrix = None
                session.feature_names = None
                cleared_dtm = True
            
            final_rows = len(df_cleaned)
        else:
            final_rows = initial_rows

        res_id = session.next_result_id("duplicates")
        summary = {
            "status": "success",
            "result_id": res_id,
            "duplicate_count": duplicate_count,
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "cleared_dtm": cleared_dtm
        }
        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int, random_state: Optional[int] = None) -> Dict[str, Any]:
        """Subsamples n rows from session.dataframe without modifying the dataset.

        Args:
            n: Number of rows to sample (must be >= 1 and <= total rows).
            random_state: Seed for random number generator reproducibility.

        Returns:
            Dict containing status, result_id, sample_indices, and preview records.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded."}
        
        df = session.dataframe
        if n < 1:
            return {"status": "error", "message": "n must be at least 1."}
        if n > len(df):
            return {"status": "error", "message": f"n ({n}) cannot be greater than dataframe length ({len(df)})."}

        sampled_df = df.sample(n=n, random_state=random_state)
        sample_indices = sampled_df.index.tolist()
        
        # Safe preview with text truncated to 100 chars
        preview = []
        for idx, row in sampled_df.iterrows():
            rec = row.to_dict()
            if "text" in rec and isinstance(rec["text"], str):
                rec["text"] = rec["text"][:100] + ("..." if len(rec["text"]) > 100 else "")
            preview.append(rec)

        res_id = session.next_result_id("sample")
        summary = {
            "status": "success",
            "result_id": res_id,
            "sample_size": n,
            "sample_indices": sample_indices,
            "preview": preview
        }
        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Calculates text length statistics overall and per category, and builds a boxplot figure.

        Does not add text_length as a persistent column to session.dataframe.

        Returns:
            Dict containing status, result_id, overall_stats, and per_category_stats.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded."}
        
        df = session.dataframe
        text_series = df["text"].fillna("").astype(str)
        text_length = text_series.apply(len)

        overall_stats = text_length.describe().to_dict()

        per_category_stats = {}
        if "category_name" in df.columns:
            for cat, group in df.groupby("category_name"):
                grp_lengths = group["text"].fillna("").astype(str).apply(len)
                per_category_stats[str(cat)] = grp_lengths.describe().to_dict()

        # Plotting boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = pd.DataFrame({
            "text_length": text_length,
            "category_name": df["category_name"] if "category_name" in df.columns else "All"
        })
        sns.boxplot(data=plot_df, x="category_name", y="text_length", ax=ax)
        ax.set_title("Text Length Distribution by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        session.pending_figure = fig

        res_id = session.next_result_id("describe")
        summary = {
            "status": "success",
            "result_id": res_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats
        }
        session.store_result("describe_data_tool", {}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> Dict[str, Any]:
        """Inspects first n rows of session.dataframe, returning dataset shape, columns, and head preview.

        Args:
            n: Number of rows to inspect (must be >= 1).

        Returns:
            Dict containing status, result_id, shape, columns, and head preview.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded."}
        
        if n < 1:
            return {"status": "error", "message": "n must be at least 1."}

        df = session.dataframe
        head_df = df.head(n)
        
        preview = []
        for idx, row in head_df.iterrows():
            rec = row.to_dict()
            if "text" in rec and isinstance(rec["text"], str):
                rec["text"] = rec["text"][:100] + ("..." if len(rec["text"]) > 100 else "")
            preview.append(rec)

        res_id = session.next_result_id("inspect")
        summary = {
            "status": "success",
            "result_id": res_id,
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "preview": preview
        }
        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes each document's text in session.dataframe into lowercased unigram tokens using nltk.word_tokenize.

        Does not modify session.dataframe. Stores full token lists in session.artifacts["tokens"].

        Returns:
            Dict containing status, result_id, n_documents, total_tokens, and first_3_tokens preview.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded."}

        df = session.dataframe
        tokens_list = []
        total_tokens = 0

        for text_val in df["text"]:
            if pd.isna(text_val) or not str(text_val).strip():
                tokens = []
            else:
                lowercased = str(text_val).lower()
                tokens = nltk.word_tokenize(lowercased)
            tokens_list.append(tokens)
            total_tokens += len(tokens)

        session.artifacts["tokens"] = tokens_list

        res_id = session.next_result_id("tokenize")
        summary = {
            "status": "success",
            "result_id": res_id,
            "n_documents": len(df),
            "total_tokens": total_tokens,
            "first_3_tokens": tokens_list[:3]
        }
        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists files and directories inside a given subdirectory relative to the repository root.

        Prevents directory traversal outside the repository root and rejects absolute paths.

        Args:
            subdirectory: Path relative to repo root (defaults to ".").

        Returns:
            Dict containing status, result_id, subdirectory, directories, and files.
        """
        try:
            repo_root = Path(".").resolve()
            sub_path = Path(subdirectory)

            if sub_path.is_absolute():
                return {"status": "error", "message": "Absolute paths are not allowed."}

            target_path = (repo_root / sub_path).resolve()

            # Check for path traversal outside repo root
            try:
                rel_target = target_path.relative_to(repo_root)
            except ValueError:
                return {"status": "error", "message": "Path traversal outside repository root is not allowed."}

            if not target_path.exists():
                return {"status": "error", "message": f"Path '{subdirectory}' does not exist."}

            if not target_path.is_dir():
                return {"status": "error", "message": f"Path '{subdirectory}' is not a directory."}

            directories = []
            files = []

            for entry in target_path.iterdir():
                if entry.is_dir():
                    directories.append(entry.name)
                else:
                    files.append(entry.name)

            directories.sort()
            files.sort()

            res_id = session.next_result_id("list_files")
            summary = {
                "status": "success",
                "result_id": res_id,
                "subdirectory": str(rel_target),
                "directories": directories,
                "files": files
            }
            session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
            return summary

        except Exception as e:
            return {"status": "error", "message": str(e)}

    return [
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
        inspect_data_tool,
        tokenize_tool,
        list_files_tool
    ]
