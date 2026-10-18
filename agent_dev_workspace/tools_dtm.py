import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool() -> Dict[str, Any]:
        """Builds a Document-Term Matrix (DTM) from document text in session.dataframe.

        Uses CountVectorizer defaults to vectorize text, updates session state
        (feature_matrix, feature_names, labels, artifacts["count_vectorizer"]),
        and computes matrix sparsity metrics.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset loaded or 'text' column missing in session.dataframe."}

        # 1. Instantiate CountVectorizer with defaults and fit_transform on text
        vectorizer = CountVectorizer()
        text_data = session.dataframe["text"].fillna("").astype(str)
        feature_matrix = vectorizer.fit_transform(text_data)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # 2. Update session attributes
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Align session.labels with category_name per document
        if "category_name" in session.dataframe.columns:
            labels_array = session.dataframe["category_name"].to_numpy()
            session.set_labels(labels_array)
        elif "label" in session.dataframe.columns:
            labels_array = session.dataframe["label"].to_numpy()
            session.set_labels(labels_array)

        # 3. Compute matrix sparsity
        n_documents, n_terms = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_documents * n_terms)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        summary = {
            "result_id": session.next_result_id("dtm"),
            "n_documents": n_documents,
            "n_terms": n_terms,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result("build_dtm_tool", {}, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: int = 10) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in the global DTM.

        Args:
            top_n: Number of top-frequent terms to include in the returned summary. Default is 10.
        """
        if top_n < 1:
            return {"status": "error", "message": "top_n must be >= 1."}

        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No DTM found. Call build_dtm_tool first."}

        # Calculate total counts for every column term across all documents
        counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten()

        # Create DataFrame with all terms and counts, sorted descending
        df_full = pd.DataFrame({
            "term": session.feature_names,
            "frequency": [int(c) for c in counts]
        }).sort_values(by="frequency", ascending=False).reset_index(drop=True)

        # Top N terms as dict for the summary
        df_top = df_full.head(top_n)
        top_term_counts = dict(zip(df_top["term"], df_top["frequency"]))

        summary = {
            "result_id": session.next_result_id("term_freq"),
            "vocabulary_size": len(session.feature_names),
            "top_n": top_n,
            "term_frequencies": top_term_counts
        }

        session.store_result(
            "term_frequency_tool",
            {"top_n": top_n},
            summary,
            full_report=df_full
        )
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generates a heatmap visualization of a positional slice of the global DTM.

        Args:
            n_terms: Number of terms (columns) to include from the start of feature_names. Default is 20.
            n_documents: Number of documents (rows) to include from the start of dataframe. Default is 20.
        """
        if n_terms < 1:
            return {"status": "error", "message": "n_terms must be >= 1."}
        if n_documents < 1:
            return {"status": "error", "message": "n_documents must be >= 1."}

        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No DTM found. Call build_dtm_tool first."}

        # Slice matrix according to n_documents and n_terms bounds
        total_docs, total_terms = session.feature_matrix.shape
        actual_docs = min(n_documents, total_docs)
        actual_terms = min(n_terms, total_terms)

        matrix_slice = session.feature_matrix[0:actual_docs, 0:actual_terms].toarray()
        sliced_terms = session.feature_names[0:actual_terms]
        doc_labels = [f"Doc_{i}" for i in range(actual_docs)]

        # Generate HeatMap plot and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            matrix_slice,
            xticklabels=sliced_terms,
            yticklabels=doc_labels,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            ax=ax
        )
        ax.set_title(f"DTM Heatmap (First {actual_docs} Docs x First {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        summary = {
            "result_id": session.next_result_id("dtm_heatmap"),
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "terms": sliced_terms,
            "documents": doc_labels,
            "matrix_slice": matrix_slice.tolist()
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
