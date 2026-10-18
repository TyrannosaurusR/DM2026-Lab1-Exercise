import tempfile
import os
import numpy as np
import pandas as pd
from typing import Optional, Literal, List, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# Import PAMI algorithms and converter
from PAMI.extras.convert import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mines frequent itemsets/patterns from document text for a specific category.

        Performs per-category vectorization with stopword removal, filters terms using
        the selected percentile band method (variance, term_frequency, or tfidf), converts
        surviving terms into a transactional database, and runs the requested PAMI pattern
        mining algorithm (FPGrowth, Top-K FAE, or MaxFPGrowth).

        Args:
            category_name: The category label string to filter session.dataframe by.
            filtering_method: Method to select vocabulary ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth', >= 1).
            k: Top K patterns limit (required for 'topk', >= 1).

        Returns:
            Dict containing status, result_id, category_name, filtering_method, algorithm,
            n_category_documents, n_surviving_terms, and mined patterns (top 10).
        """
        # Validate dataframe existence
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataframe loaded in session. Load a dataset first.",
            }

        df = session.dataframe

        # Validate category_name
        if "category_name" not in df.columns or category_name not in df["category_name"].values:
            return {
                "status": "error",
                "message": f"Category '{category_name}' not found in dataset.",
            }

        # Validate filtering_method
        if filtering_method not in ["variance", "tfidf", "term_frequency"]:
            return {
                "status": "error",
                "message": f"Unknown filtering method '{filtering_method}'. Expected 'variance', 'tfidf', or 'term_frequency'.",
            }

        # Validate algorithm
        if algorithm not in ["fpgrowth", "topk", "maxfpgrowth"]:
            return {
                "status": "error",
                "message": f"Unknown algorithm '{algorithm}'. Expected 'fpgrowth', 'topk', or 'maxfpgrowth'.",
            }

        # Validate parameters for specific algorithms
        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or min_sup < 1:
                return {
                    "status": "error",
                    "message": f"Algorithm '{algorithm}' requires min_sup as an integer >= 1.",
                }
        elif algorithm == "topk":
            if k is None or k < 1:
                return {
                    "status": "error",
                    "message": "Algorithm 'topk' requires k as an integer >= 1.",
                }

        # Filter dataset to specific category
        cat_df = df[df["category_name"] == category_name]
        n_category_documents = len(cat_df)

        # Sanitize text column (handle NaN / None values) matching build_dtm_tool
        text_series = cat_df["text"].fillna("").astype(str)

        # 1. Build fresh, per-category CountVectorizer (with English stopwords dropped)
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X_cat = vectorizer.fit_transform(text_series)
        except ValueError:
            # E.g. empty vocabulary after stopword removal
            X_cat = None
            feature_names = np.array([])
        else:
            feature_names = np.array(vectorizer.get_feature_names_out())

        surviving_feature_indices = []

        if len(feature_names) > 0 and X_cat is not None:
            # Convert to dense matrix for filtering metric calculations
            X_dense = X_cat.toarray()

            if filtering_method == "variance":
                # Sample variance (ddof=1) matching pandas var and text_filtering_helpers.py
                variances = np.var(X_dense, axis=0, ddof=1)
                p5, p95 = np.percentile(variances, [5, 95])
                # Match text_filtering_helpers.py (inclusive boundaries >= and <=)
                surviving_feature_indices = np.where((variances >= p5) & (variances <= p95))[0]

            elif filtering_method == "term_frequency":
                col_sums = np.sum(X_dense, axis=0)
                p5, p95 = np.percentile(col_sums, [5, 95])
                # Match text_filtering_helpers.py (inclusive boundaries >= and <=)
                surviving_feature_indices = np.where((col_sums >= p5) & (col_sums <= p95))[0]

            elif filtering_method == "tfidf":
                transformer = TfidfTransformer()
                tfidf_matrix = transformer.fit_transform(X_cat).toarray()
                mean_tfidf = np.mean(tfidf_matrix, axis=0)
                p20 = np.percentile(mean_tfidf, 20)
                surviving_feature_indices = np.where(mean_tfidf >= p20)[0]

        surviving_terms = feature_names[surviving_feature_indices]
        n_surviving_terms = len(surviving_terms)

        result_id = session.next_result_id("patterns")
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        # 3. If zero terms survived filtering, return success with empty patterns
        if n_surviving_terms == 0:
            session.artifacts[artifact_key] = []
            summary = {
                "status": "success",
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_category_documents": n_category_documents,
                "n_surviving_terms": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result(
                "mine_patterns_tool",
                {
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                },
                summary,
            )
            return summary

        # 4. Convert surviving terms into a binary transaction DataFrame
        surviving_X = X_dense[:, surviving_feature_indices]
        binary_df = pd.DataFrame(surviving_X > 0, columns=surviving_terms)

        all_patterns = []

        try:
            # Run PAMI inside a temporary directory to auto-clean files
            with tempfile.TemporaryDirectory() as tmpdir:
                db_file = os.path.join(tmpdir, "transactions.db")
                output_file = os.path.join(tmpdir, "output.txt")

                # Pass DataFrame directly to DF2DB, matching Master: DF2DB(df).convert2TransactionalDatabase(path, ">=", 1)
                converter = DF2DB.DF2DB(binary_df)
                converter.convert2TransactionalDatabase(db_file, ">=", 1)

                # Mine patterns based on chosen algorithm
                if algorithm == "fpgrowth":
                    miner = FPGrowth.FPGrowth(db_file, minSup=min_sup, sep="\t")
                    miner.mine()
                    miner.save(output_file)
                elif algorithm == "topk":
                    miner = FAE.FAE(db_file, k=k, sep="\t")
                    miner.mine()
                    miner.save(output_file)
                elif algorithm == "maxfpgrowth":
                    miner = MaxFPGrowth.MaxFPGrowth(db_file, minSup=min_sup, sep="\t")
                    miner.mine()
                    miner.save(output_file)

                # Parse saved patterns output file
                if os.path.exists(output_file):
                    with open(output_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            # PAMI format typically: item1\titem2\t:support
                            if ":" in line:
                                parts = line.rsplit(":", 1)
                                items_str = parts[0].strip()
                                support_str = parts[1].strip()
                                try:
                                    support_val = int(float(support_str))
                                except ValueError:
                                    continue
                                terms_list = (
                                    items_str.split("\t")
                                    if "\t" in items_str
                                    else items_str.split()
                                )
                                all_patterns.append(
                                    {"terms": terms_list, "support": support_val}
                                )
        except Exception as e:
            return {
                "status": "error",
                "message": f"PAMI mining failed: {str(e)}",
            }

        # Sort patterns by support descending
        all_patterns.sort(key=lambda x: x["support"], reverse=True)

        # Save complete list to artifacts
        session.artifacts[artifact_key] = all_patterns

        # Top 10 for summary
        top_10 = all_patterns[:10]

        note = (
            "mining complete"
            if len(all_patterns) > 0
            else "no patterns found for threshold/algorithm"
        )

        summary = {
            "status": "success",
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_category_documents": n_category_documents,
            "n_surviving_terms": n_surviving_terms,
            "patterns": top_10,
            "total_patterns_found": len(all_patterns),
            "note": note,
        }

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
        )

        return summary

    return [mine_patterns_tool]
