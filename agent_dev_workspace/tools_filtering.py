"""Filtering tools for feature selection in data mining pipelines."""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from langchain_core.tools import tool


def _validate_target_class(session: Any, target_class: str) -> Dict[str, Any] | np.ndarray:
    """Helper to validate DTM existence, labels, and target_class for class correlation tools.

    Returns error dict if validation fails, or binary target array (1 for target_class, 0 otherwise).
    """
    if session.feature_matrix is None or session.feature_names is None:
        return {
            "status": "error",
            "message": "No document-term matrix available. Build DTM first.",
        }

    # Derive document category labels from session
    if session.labels is not None:
        labels = np.asarray(session.labels)
    elif session.dataframe is not None and "category_name" in session.dataframe.columns:
        labels = session.dataframe["category_name"].values
    elif session.dataframe is not None and "category" in session.dataframe.columns:
        labels = session.dataframe["category"].values
    else:
        return {
            "status": "error",
            "message": "No document category labels available in session.",
        }

    if len(labels) == 0:
        return {
            "status": "error",
            "message": "Document-term matrix is empty.",
        }

    unique_classes = sorted(list(set(labels)))

    if len(unique_classes) < 2:
        return {
            "status": "error",
            "message": "At least two distinct categories are required to compute class correlation.",
        }

    if target_class not in unique_classes:
        valid_str = ", ".join(map(str, unique_classes))
        return {
            "status": "error",
            "message": f"Unknown target_class '{target_class}'. Valid categories: {valid_str}",
        }

    # Binary vector: 1 where category matches target_class, 0 elsewhere
    binary_y = (labels == target_class).astype(float)
    return binary_y


def make_tools(session: Any) -> List[Any]:
    """Factory function returning tools for feature filtering bound to session."""

    @tool
    def variance_filter_tool(
        threshold: float = 0.0,
        max_preview: int = 10,
    ) -> Dict[str, Any]:
        """Flags near-constant terms by variance for potential removal.

        Computes the population variance (ddof=0) for every term across all documents
        in the document-term matrix (session.feature_matrix). Terms with variance
        strictly greater than threshold are flagged to keep (keep=True), while terms
        with variance <= threshold are flagged for removal (keep=False).

        Does NOT mutate session.feature_matrix or session.feature_names in place.

        Args:
            threshold: Minimum variance required to keep a term. Default 0.0.
            max_preview: Maximum number of removed terms to include in summary preview.

        Returns:
            Dict containing execution status, result_id, variance threshold, summary counts,
            and a list preview of removed terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix available. Build DTM first.",
            }

        if threshold < 0:
            return {
                "status": "error",
                "message": "Threshold must be non-negative.",
            }

        if max_preview < 0:
            return {
                "status": "error",
                "message": "max_preview must be non-negative.",
            }

        matrix = session.feature_matrix
        feature_names = list(session.feature_names)
        n_docs = matrix.shape[0]

        if n_docs == 0 or len(feature_names) == 0:
            return {
                "status": "error",
                "message": "Document-term matrix is empty.",
            }

        # Sparse population variance computation without dense conversion:
        # Var(X) = E[X^2] - (E[X])^2
        # E[X] (mean per term/column)
        means = np.asarray(matrix.mean(axis=0)).ravel()

        # E[X^2] (mean of squared counts per term/column)
        means_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()

        variances = means_sq - (means ** 2)

        # Clip tiny floating point noise around zero so constant terms equal 0.0 exactly
        variances = np.maximum(variances, 0.0)
        variances[np.isclose(variances, 0.0, atol=1e-12)] = 0.0

        # Flag terms: variance > threshold
        keep_mask = variances > threshold

        full_df = pd.DataFrame(
            {
                "term": feature_names,
                "variance": variances,
                "keep": keep_mask,
            }
        )

        # Sort descending by variance for full_report
        full_df.sort_values(by="variance", ascending=False, inplace=True)
        full_df.reset_index(drop=True, inplace=True)

        removed_df = full_df[~full_df["keep"]].sort_values(by="variance", ascending=True)

        removed_terms_preview = removed_df.head(max_preview)[["term", "variance"]].to_dict(
            orient="records"
        )

        total_terms = len(feature_names)
        retained_terms = int(keep_mask.sum())
        removed_terms = total_terms - retained_terms

        result_id = session.next_result_id("variance_filter")

        summary = {
            "status": "success",
            "result_id": result_id,
            "threshold": float(threshold),
            "total_terms": total_terms,
            "retained_terms": retained_terms,
            "removed_terms": removed_terms,
            "removed_terms_preview": removed_terms_preview,
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold, "max_preview": max_preview},
            summary=summary,
            full_report=full_df,
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        max_preview: int = 10,
    ) -> Dict[str, Any]:
        """Computes Pearson correlation between each term's count and a target class.

        Performs one-vs-rest correlation analysis to identify terms strongly
        associated (positively or negatively) with documents of target_class.

        Does NOT mutate session.feature_matrix or session.feature_names in place.

        Args:
            target_class: Category name to correlate against (1 for target_class, 0 rest).
            max_preview: Number of top positive and negative terms to preview.

        Returns:
            Dict containing execution status, result_id, target_class, top positive
            preview terms, and top negative preview terms.
        """
        if max_preview < 0:
            return {
                "status": "error",
                "message": "max_preview must be non-negative.",
            }

        validation_res = _validate_target_class(session, target_class)
        if isinstance(validation_res, dict):
            return validation_res

        binary_y = validation_res  # shape (n_docs,)
        matrix = session.feature_matrix
        feature_names = list(session.feature_names)

        # Sparse-friendly computation of Pearson correlation r for all columns at once:
        # r = Cov(X, y) / (std(X) * std(y))
        # Cov(X, y) = E[X * y] - E[X] * E[y]
        
        # y centered
        y_mean = np.mean(binary_y)
        y_std = np.std(binary_y, ddof=0)

        # Matrix-vector product gives sum_i (x_ij * y_i) for each term j
        # matrix is (n_docs, n_terms), binary_y is (n_docs,)
        xy_sum = matrix.T.dot(binary_y)  # shape (n_terms,)
        xy_mean = xy_sum / matrix.shape[0]

        x_means = np.asarray(matrix.mean(axis=0)).ravel()
        
        # Covariance per term
        cov_xy = xy_mean - (x_means * y_mean)

        # Standard deviation per term: std(X) = sqrt(E[X^2] - (E[X])^2)
        x_means_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        x_vars = np.maximum(x_means_sq - (x_means ** 2), 0.0)
        x_stds = np.sqrt(x_vars)

        # Avoid division by zero when x_stds == 0 or y_std == 0
        denom = x_stds * y_std
        pearson_r = np.zeros_like(cov_xy)
        valid_mask = denom > 0
        pearson_r[valid_mask] = cov_xy[valid_mask] / denom[valid_mask]

        # Clip potential floating point inaccuracies outside [-1.0, 1.0]
        pearson_r = np.clip(pearson_r, -1.0, 1.0)

        full_df = pd.DataFrame(
            {
                "term": feature_names,
                "pearson_r": pearson_r,
            }
        )

        # Sort by absolute pearson_r descending, biggest magnitude first
        full_df["abs_r"] = full_df["pearson_r"].abs()
        full_df.sort_values(by="abs_r", ascending=False, inplace=True)
        full_df.drop(columns=["abs_r"], inplace=True)
        full_df.reset_index(drop=True, inplace=True)

        # Previews: highest positive correlation and lowest negative correlation
        pos_df = full_df[full_df["pearson_r"] > 0].sort_values(by="pearson_r", ascending=False)
        neg_df = full_df[full_df["pearson_r"] < 0].sort_values(by="pearson_r", ascending=True)

        top_positive_preview = pos_df.head(max_preview)[["term", "pearson_r"]].to_dict(
            orient="records"
        )
        top_negative_preview = neg_df.head(max_preview)[["term", "pearson_r"]].to_dict(
            orient="records"
        )

        result_id = session.next_result_id("pearson_filter")

        summary = {
            "status": "success",
            "result_id": result_id,
            "target_class": str(target_class),
            "top_positive_preview": top_positive_preview,
            "top_negative_preview": top_negative_preview,
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "max_preview": max_preview},
            summary=summary,
            full_report=full_df,
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        max_preview: int = 10,
    ) -> Dict[str, Any]:
        """Computes Spearman rank correlation between each term's count and a target class.

        Performs one-vs-rest rank correlation analysis to evaluate monotonic relationships
        between term frequencies and target_class membership across documents.

        Does NOT mutate session.feature_matrix or session.feature_names in place.

        Args:
            target_class: Category name to correlate against (1 for target_class, 0 rest).
            max_preview: Number of top positive and negative terms to preview.

        Returns:
            Dict containing execution status, result_id, target_class, top positive
            preview terms, and top negative preview terms.
        """
        if max_preview < 0:
            return {
                "status": "error",
                "message": "max_preview must be non-negative.",
            }

        validation_res = _validate_target_class(session, target_class)
        if isinstance(validation_res, dict):
            return validation_res

        binary_y = validation_res  # shape (n_docs,)
        matrix = session.feature_matrix
        feature_names = list(session.feature_names)
        n_docs, n_terms = matrix.shape

        # Rank the target class vector y (ties get average rank)
        ranked_y = rankdata(binary_y, method="average")
        y_mean = np.mean(ranked_y)
        y_std = np.std(ranked_y, ddof=0)

        BLOCK_SIZE = 2000
        spearman_r = np.zeros(n_terms, dtype=float)

        for start_col in range(0, n_terms, BLOCK_SIZE):
            end_col = min(start_col + BLOCK_SIZE, n_terms)
            block_sparse = matrix[:, start_col:end_col]
            block_dense = block_sparse.toarray()

            # Rank each column across documents (axis=0)
            ranked_block = rankdata(block_dense, method="average", axis=0)

            # Pearson correlation of ranks
            x_means = np.mean(ranked_block, axis=0)
            x_stds = np.std(ranked_block, axis=0, ddof=0)

            # Covariance between ranked_block and ranked_y
            # shape of ranked_block is (n_docs, block_len), ranked_y is (n_docs,)
            xy_mean = np.dot(ranked_y, ranked_block) / n_docs
            cov_xy = xy_mean - (x_means * y_mean)

            denom = x_stds * y_std
            valid_mask = denom > 0
            
            block_r = np.zeros(end_col - start_col, dtype=float)
            block_r[valid_mask] = cov_xy[valid_mask] / denom[valid_mask]
            spearman_r[start_col:end_col] = np.clip(block_r, -1.0, 1.0)

        full_df = pd.DataFrame(
            {
                "term": feature_names,
                "spearman_r": spearman_r,
            }
        )

        # Sort by absolute spearman_r descending, biggest magnitude first
        full_df["abs_r"] = full_df["spearman_r"].abs()
        full_df.sort_values(by="abs_r", ascending=False, inplace=True)
        full_df.drop(columns=["abs_r"], inplace=True)
        full_df.reset_index(drop=True, inplace=True)

        # Previews: highest positive correlation and lowest negative correlation
        pos_df = full_df[full_df["spearman_r"] > 0].sort_values(by="spearman_r", ascending=False)
        neg_df = full_df[full_df["spearman_r"] < 0].sort_values(by="spearman_r", ascending=True)

        top_positive_preview = pos_df.head(max_preview)[["term", "spearman_r"]].to_dict(
            orient="records"
        )
        top_negative_preview = neg_df.head(max_preview)[["term", "spearman_r"]].to_dict(
            orient="records"
        )

        result_id = session.next_result_id("spearman_filter")

        summary = {
            "status": "success",
            "result_id": result_id,
            "target_class": str(target_class),
            "top_positive_preview": top_positive_preview,
            "top_negative_preview": top_negative_preview,
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "max_preview": max_preview},
            summary=summary,
            full_report=full_df,
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
