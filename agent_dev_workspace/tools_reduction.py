import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Literal
from langchain_core.tools import tool

from agent_pipeline.session_state import SessionState


def make_tools(session: SessionState):
    """
    Factory function returning reduction tools bound to the provided session.
    """

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"],
        random_state: Optional[int] = None,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> dict:
        """
        Reduces feature matrix to 2 dimensions using PCA, t-SNE, or UMAP.
        Generates a 2D scatter plot colored by category on session.pending_figure.

        Args:
            method: Dimensionality reduction technique ('pca', 'tsne', or 'umap').
            random_state: Seed for random number generator for reproducibility.
            perplexity: Perplexity parameter for t-SNE (default 30.0).
            n_neighbors: Number of neighbors for UMAP (default 15).

        Returns:
            Dict containing status, result_id, method, coordinates summary,
            and explained variance ratio (for PCA).
        """
        if session.feature_matrix is None or session.feature_matrix.shape[0] == 0:
            return {"status": "error", "message": "No feature matrix available. Run build_dtm_tool first."}

        n_samples, n_features = session.feature_matrix.shape

        if n_samples < 2:
            return {"status": "error", "message": "At least 2 documents are required for dimensionality reduction."}

        if method not in ["pca", "tsne", "umap"]:
            return {"status": "error", "message": f"Unknown reduction method '{method}'. Must be 'pca', 'tsne', or 'umap'."}

        if method == "tsne" and perplexity >= n_samples:
            return {
                "status": "error",
                "message": f"perplexity ({perplexity}) must be less than number of samples ({n_samples}).",
            }

        if method == "umap" and n_neighbors >= n_samples:
            return {
                "status": "error",
                "message": f"n_neighbors ({n_neighbors}) must be less than number of samples ({n_samples}).",
            }

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else session.feature_matrix
        explained_variance_ratio = None

        try:
            if method == "pca":
                if n_features < 2:
                    return {"status": "error", "message": f"PCA requires at least 2 features, but matrix has {n_features}."}
                from sklearn.decomposition import PCA

                model = PCA(n_components=2, random_state=random_state)
                coords = model.fit_transform(X)
                explained_variance_ratio = model.explained_variance_ratio_.tolist()

            elif method == "tsne":
                from sklearn.manifold import TSNE

                model = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
                coords = model.fit_transform(X)

            elif method == "umap":
                import umap

                model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
                coords = model.fit_transform(X)

        except Exception as e:
            return {"status": "error", "message": f"Error running {method}: {str(e)}"}

        # Caching in artifacts and creating full report
        result_id = session.next_result_id(f"reduce_{method}")
        
        # Extract categories for plotting and reporting
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            categories = session.dataframe["category_name"].tolist()
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            categories = session.dataframe["category"].tolist()
        else:
            categories = ["unknown"] * n_samples

        full_report = pd.DataFrame(
            {
                "dim_1": coords[:, 0],
                "dim_2": coords[:, 1],
                "category": categories,
            }
        )

        session.artifacts[f"coords_{method}"] = full_report.to_dict(orient="records")

        # Create plot and save on session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_cats = sorted(list(set(categories)))
        for cat in unique_cats:
            mask = [c == cat for c in categories]
            ax.scatter(coords[mask, 0], coords[mask, 1], label=str(cat), alpha=0.7)

        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title="Category")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        session.pending_figure = fig

        summary = {
            "status": "success",
            "result_id": result_id,
            "method": method,
            "n_samples": n_samples,
            "n_components": 2,
            "first_10_coords": coords[:10].tolist(),
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result("reduce_dimensions_tool", {"method": method}, summary, full_report=full_report)

        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """
        One-hot encodes the category labels into a binary 0/1 matrix with one
        column per category (in sorted category order). Stores the result in
        session.artifacts['binarized_labels'].

        Returns:
            Dict containing status, categories, counts per category, matrix shape,
            and first 5 rows of the one-hot matrix.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded."}

        if "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"]
        elif "category" in session.dataframe.columns:
            labels = session.dataframe["category"]
        else:
            return {"status": "error", "message": "No category labels found in session.dataframe."}

        unique_categories = sorted(labels.dropna().unique().tolist())
        if not unique_categories:
            return {"status": "error", "message": "No category labels available to binarize."}

        # One-hot encoding explicitly for all categories (one column per unique category)
        one_hot_df = pd.DataFrame(0, index=session.dataframe.index, columns=unique_categories)
        for cat in unique_categories:
            one_hot_df[cat] = (labels == cat).astype(int)

        session.artifacts["binarized_labels"] = one_hot_df

        counts_per_category = labels.value_counts().to_dict()

        result_id = session.next_result_id("binarize_labels")

        summary = {
            "status": "success",
            "result_id": result_id,
            "categories": unique_categories,
            "counts_per_category": counts_per_category,
            "shape": list(one_hot_df.shape),
            "first_5_rows": one_hot_df.head(5).to_dict(orient="records"),
        }

        session.store_result("binarize_labels_tool", {}, summary, full_report=one_hot_df)

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
