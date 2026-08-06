import math
from typing import Any, Dict, List

from shared.logging import get_logger

logger = get_logger(__name__)


class RecommendationEvaluator:
    """Evaluates top-K recommendation quality using standard information retrieval & ranking metrics."""

    @staticmethod
    def precision_at_k(actual: List[Any], recommended: List[Any], k: int) -> float:
        """Computes Precision at K."""
        if not recommended:
            return 0.0
        recommended_k = recommended[:k]
        relevant_count = sum(1 for item in recommended_k if item in actual)
        return relevant_count / len(recommended_k)

    @staticmethod
    def recall_at_k(actual: List[Any], recommended: List[Any], k: int) -> float:
        """Computes Recall at K."""
        if not actual:
            return 0.0
        recommended_k = recommended[:k]
        relevant_count = sum(1 for item in recommended_k if item in actual)
        return relevant_count / len(actual)

    @staticmethod
    def _average_precision_at_k(actual: List[Any], recommended: List[Any], k: int) -> float:
        """Helper to compute average precision at K for a single list."""
        if not actual or not recommended:
            return 0.0
        recommended_k = recommended[:k]
        score = 0.0
        num_hits = 0
        for i, p in enumerate(recommended_k):
            if p in actual:
                num_hits += 1
                score += num_hits / (i + 1.0)
        return score / min(len(actual), k)

    @classmethod
    def map_at_k(cls, actual_list: List[List[Any]], recommended_list: List[List[Any]], k: int) -> float:
        """Computes Mean Average Precision at K across a list of recommendations."""
        if not actual_list:
            return 0.0
        ap_sum = sum(
            cls._average_precision_at_k(actual, recommended, k)
            for actual, recommended in zip(actual_list, recommended_list)
        )
        return ap_sum / len(actual_list)

    @staticmethod
    def _dcg_at_k(actual: List[Any], recommended: List[Any], k: int) -> float:
        """Helper to compute Discounted Cumulative Gain at K."""
        recommended_k = recommended[:k]
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in actual:
                dcg += 1.0 / math.log2(i + 2)
        return dcg

    @classmethod
    def ndcg_at_k(cls, actual: List[Any], recommended: List[Any], k: int) -> float:
        """Computes Normalized Discounted Cumulative Gain at K."""
        if not actual:
            return 0.0
        dcg = cls._dcg_at_k(actual, recommended, k)
        idcg = cls._dcg_at_k(actual, actual, k)
        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    def evaluate_all(self, test_user_actuals: Dict[Any, List[Any]], recommender: Any, k: int = 10) -> Dict[str, float]:
        """
        Computes Precision@K, Recall@K, MAP@K, NDCG@K across all test users.
        """
        if not test_user_actuals:
            return {"precision_at_k": 0.0, "recall_at_k": 0.0, "map_at_k": 0.0, "ndcg_at_k": 0.0}

        actual_list = []
        recommended_list = []

        for user_id, actual in test_user_actuals.items():
            try:
                # Expects recommender to return a list of dicts with an 'item_id' key
                recs = recommender.recommend(user_id=user_id, top_k=k)
                recommended_items = [rec.get("item_id") for rec in recs if isinstance(rec, dict)]
            except Exception as e:
                logger.error(f"Error generating recommendations for user {user_id}: {e}")
                recommended_items = []

            actual_list.append(actual)
            recommended_list.append(recommended_items)

        map_k = self.map_at_k(actual_list, recommended_list, k)

        precisions = [self.precision_at_k(a, r, k) for a, r in zip(actual_list, recommended_list)]
        recalls = [self.recall_at_k(a, r, k) for a, r in zip(actual_list, recommended_list)]
        ndcgs = [self.ndcg_at_k(a, r, k) for a, r in zip(actual_list, recommended_list)]

        return {
            "precision_at_k": sum(precisions) / len(precisions) if precisions else 0.0,
            "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
            "map_at_k": map_k,
            "ndcg_at_k": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        }
