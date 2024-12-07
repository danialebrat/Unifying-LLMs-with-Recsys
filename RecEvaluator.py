import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


class RecEvaluator:
    def __init__(
            self,
            test_df: pd.DataFrame,
            recommendations_df: pd.DataFrame,
            user_id_col: str = "user_id",
            item_id_col: str = "movie_id",
            rating_col: str = "rating",
            rank_col: str = "recommendation_rank",
            relevance_threshold: float = 4.0
    ):
        """
        Initialize the RecEvaluator with test and recommendation data.

        Parameters
        ----------
        test_df : pd.DataFrame
            DataFrame containing actual user-item interactions and their ratings.
        recommendations_df : pd.DataFrame
            DataFrame containing recommended items for each user with a ranking.
        user_id_col : str, optional
            Name of the user ID column.
        item_id_col : str, optional
            Name of the item ID column.
        rating_col : str, optional
            Name of the rating column in test_df.
        rank_col : str, optional
            Name of the ranking column in recommendations_df.
        relevance_threshold : float, optional
            Threshold to consider an item as relevant.
        """
        self.test_df = test_df
        self.recommendations_df = recommendations_df
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.rating_col = rating_col
        self.rank_col = rank_col
        self.relevance_threshold = relevance_threshold

        self._prepare_data()

    def _prepare_data(self):
        """Prepare internal data structures for faster computation."""
        # Get relevant items (those that meet or exceed the threshold) from the test set
        self.test_relevant = self.test_df[self.test_df[self.rating_col] >= self.relevance_threshold]

        # For each user, get the set of relevant items
        self.user_relevant_items = self.test_relevant.groupby(self.user_id_col)[self.item_id_col].apply(set).to_dict()

        # For each user, get the full set of test items (to calculate coverage-based metrics if needed)
        self.user_all_items = self.test_df.groupby(self.user_id_col)[self.item_id_col].apply(set).to_dict()

        # For faster evaluation: create a dictionary {user_id: [(item, rank), ...]} from recommendations
        self.user_recommendations = (
            self.recommendations_df
            .sort_values([self.user_id_col, self.rank_col])
            .groupby(self.user_id_col)[[self.item_id_col, self.rank_col]]
            .apply(lambda x: list(zip(x[self.item_id_col], x[self.rank_col])))
            .to_dict()
        )

        # In case a user from recommendations isn't in test, handle that:
        self.users_in_test = set(self.user_all_items.keys())
        self.users_in_rec = set(self.user_recommendations.keys())
        self.evaluation_users = list(self.users_in_test.intersection(self.users_in_rec))

    def precision_at_k(self, k: int) -> float:
        """Compute Precision@K across all users."""
        precisions = []
        for user in self.evaluation_users:
            # Get top-k recommended items for this user
            recommended_items = [i for i, r in self.user_recommendations[user] if r <= k]
            if not recommended_items:
                # No recommendations at K for this user (or fewer than K)
                continue

            rel_items = self.user_relevant_items.get(user, set())
            hit_count = len(set(recommended_items) & rel_items)
            precisions.append(hit_count / len(recommended_items))
        return np.mean(precisions) if precisions else 0.0

    def recall_at_k(self, k: int) -> float:
        """Compute Recall@K across all users."""
        recalls = []
        for user in self.evaluation_users:
            recommended_items = [i for i, r in self.user_recommendations[user] if r <= k]
            rel_items = self.user_relevant_items.get(user, set())
            if len(rel_items) == 0:
                # If the user has no relevant items in test, skip
                continue
            hit_count = len(set(recommended_items) & rel_items)
            recalls.append(hit_count / len(rel_items))
        return np.mean(recalls) if recalls else 0.0

    def _dcg_at_k(self, recommended_items: List[int], relevant_items: set, k: int) -> float:
        """Compute Discounted Cumulative Gain at K."""
        dcg = 0.0
        for idx, item in enumerate(recommended_items[:k], start=1):
            rel = 1.0 if item in relevant_items else 0.0
            dcg += rel / np.log2(idx + 1)
        return dcg

    def ndcg_at_k(self, k: int) -> float:
        """Compute NDCG@K across all users."""
        ndcgs = []
        for user in self.evaluation_users:
            recommended_items = [i for i, r in self.user_recommendations[user] if r <= k]
            if not recommended_items:
                continue
            rel_items = self.user_relevant_items.get(user, set())

            # Calculate DCG
            dcg = self._dcg_at_k(recommended_items, rel_items, k)

            # Calculate IDCG (Ideal DCG) - best ordering would put all relevant items first
            ideal_list = list(rel_items)
            # If user has fewer relevant items than k, no problem, just take as many as available
            ideal_list = ideal_list[:min(len(ideal_list), k)]
            idcg = self._dcg_at_k(ideal_list, rel_items, k)

            if idcg > 0:
                ndcgs.append(dcg / idcg)
            else:
                # If user has no relevant items, skip or treat as 0.
                # Usually these users don't contribute to NDCG calculation.
                continue
        return np.mean(ndcgs) if ndcgs else 0.0

    def mrr_at_k(self, k: int) -> float:
        """Compute MRR@K (Mean Reciprocal Rank) across all users."""
        mrrs = []
        for user in self.evaluation_users:
            recommended_items = [i for i, r in self.user_recommendations[user] if r <= k]
            rel_items = self.user_relevant_items.get(user, set())
            # Find the first relevant item in the ranked list
            rr = 0.0
            for idx, item in enumerate(recommended_items, start=1):
                if item in rel_items:
                    rr = 1.0 / idx
                    break
            if rr > 0:
                mrrs.append(rr)
        return np.mean(mrrs) if mrrs else 0.0

    def average_precision_at_k(self, k: int) -> float:
        """Compute MAP@K (Mean Average Precision) across all users."""
        ap_values = []
        for user in self.evaluation_users:
            recommended_items = [i for i, r in self.user_recommendations[user] if r <= k]
            rel_items = self.user_relevant_items.get(user, set())
            if not rel_items:
                # no relevant items for this user
                continue

            hits = 0
            precision_sum = 0.0
            for idx, item in enumerate(recommended_items, start=1):
                if item in rel_items:
                    hits += 1
                    precision_sum += hits / idx
            if hits > 0:
                ap_values.append(precision_sum / len(rel_items))

        return np.mean(ap_values) if ap_values else 0.0

    def evaluate_all_metrics(self, k_values: List[int]) -> pd.DataFrame:
        """
        Evaluate all metrics at given k values and return a DataFrame.

        Parameters
        ----------
        k_values : list of int
            Values of K to evaluate metrics at.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: K, Precision, Recall, NDCG, MRR, MAP
        """
        results = []
        for k in k_values:
            precision = self.precision_at_k(k)
            recall = self.recall_at_k(k)
            ndcg = self.ndcg_at_k(k)
            mrr = self.mrr_at_k(k)
            ap = self.average_precision_at_k(k)

            results.append({
                "K": k,
                "Precision": precision,
                "Recall": recall,
                "NDCG": ndcg,
                "MRR": mrr,
                "MAP": ap
            })

        self.results_df = pd.DataFrame(results)
        return self.results_df

    def plot_metrics(self, metrics: Optional[List[str]] = None, save_path: Optional[str] = None):
        """
        Plot selected metrics over K.

        Parameters
        ----------
        metrics : list of str, optional
            Metrics to plot. Default: ["Precision", "Recall", "NDCG", "MRR", "MAP"]
        save_path : str, optional
            If provided, save the plot to this path.
        """
        if metrics is None:
            metrics = ["Precision", "Recall", "NDCG", "MRR", "MAP"]

        if not hasattr(self, 'results_df'):
            raise ValueError("No results found. Run evaluate_all_metrics first.")

        plt.figure(figsize=(10, 6))
        for metric in metrics:
            if metric in self.results_df.columns:
                plt.plot(self.results_df["K"], self.results_df[metric], marker='o', label=metric)

        plt.xlabel("K")
        plt.ylabel("Metric Value")
        plt.title("Recommendation Metrics Over K")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_results_to_csv(self, csv_path: str):
        """
        Save the evaluation results to a CSV file.

        Parameters
        ----------
        csv_path : str
            Path to the output CSV file.
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("No results found. Run evaluate_all_metrics first.")
        self.results_df.to_csv(csv_path, index=False)
