import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_index_a: int, doc_index_b: int) -> dict:
        """Compute the cosine similarity between two documents by their row indices in the DTM.

        Args:
            doc_index_a: Integer row index of the first document in session.feature_matrix.
            doc_index_b: Integer row index of the second document in session.feature_matrix.

        Returns:
            dict: Summary containing status, similarity score, and result_id.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature matrix found in session. Build DTM first."}

        n_docs = session.feature_matrix.shape[0]
        if doc_index_a < 0 or doc_index_a >= n_docs or doc_index_b < 0 or doc_index_b >= n_docs:
            return {"status": "error", "message": f"Document index out of range [0, {n_docs - 1}]."}

        vec_a = session.feature_matrix[doc_index_a].toarray().ravel()
        vec_b = session.feature_matrix[doc_index_b].toarray().ravel()

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

        res_id = session.next_result_id("cosine")
        summary = {
            "result_id": res_id,
            "status": "success",
            "doc_index_a": doc_index_a,
            "doc_index_b": doc_index_b,
            "similarity": similarity
        }
        session.store_result("cosine_similarity_tool", {"doc_index_a": doc_index_a, "doc_index_b": doc_index_b}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """Compute Pearson correlation matrix for top-20 terms by variance and render a heatmap.

        Returns:
            dict: Summary containing feature names, correlation matrix as list of lists, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No DTM found in session. Build DTM first."}

        n_docs, n_terms = session.feature_matrix.shape
        if n_docs < 2 or n_terms < 2:
            return {"status": "error", "message": "At least 2 documents and 2 terms are required for feature correlation matrix."}

        dense_dtm = session.feature_matrix.toarray()
        variances = np.var(dense_dtm, axis=0, ddof=0)

        k = min(20, n_terms)
        top_k_indices = np.argsort(variances)[::-1][:k]

        selected_names = [session.feature_names[i] for i in top_k_indices]
        selected_matrix = dense_dtm[:, top_k_indices]

        # Handle zero-variance terms explicitly to avoid NaN from np.corrcoef division by zero
        corr_matrix = np.corrcoef(selected_matrix, rowvar=False)

        # Replace any NaNs caused by constant columns
        if np.isnan(corr_matrix).any():
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.fill_diagonal(corr_matrix, 1.0)

        # Build Seaborn heatmap and store on session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=selected_names,
            yticklabels=selected_names,
            cmap="vlag",
            ax=ax
        )
        ax.set_title("Feature Correlation Matrix (Top Variance Terms)")
        plt.tight_layout()
        session.pending_figure = fig

        corr_list = corr_matrix.tolist()
        res_id = session.next_result_id("featcorr")
        summary = {
            "result_id": res_id,
            "status": "success",
            "top_terms": selected_names,
            "correlation_matrix": corr_list
        }
        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
