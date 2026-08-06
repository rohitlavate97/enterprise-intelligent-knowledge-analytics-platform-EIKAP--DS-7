import pandas as pd
from typing import Any, Dict, List, Optional
from shared.logging import get_logger

# Import CF and CB recommenders (assumes they are available in the project)
try:
    from recommendations.collaborative import CollaborativeRecommender
    from recommendations.content_based import ContentBasedRecommender
except ImportError:
    # Dummy classes for standalone execution in case they don't exist yet.
    class CollaborativeRecommender:
        def fit(self, interactions_df, user_col, item_col, rating_col): return self
        def recommend_for_user(self, user_id, top_k): return []
    class ContentBasedRecommender:
        def fit(self, items_df, item_col, item_text_col): return self
        def recommend_for_user_profile(self, user_item_ids, top_k): return []

logger = get_logger(__name__)


class HybridRecommender:
    """
    Standard maturity recommendation engine combining Collaborative Filtering
    and Content-Based Filtering.
    """

    def __init__(self, cf_weight: float = 0.5, fallback_popularity: bool = True):
        self.cf_weight = cf_weight
        self.fallback_popularity = fallback_popularity
        self.cf_recommender = CollaborativeRecommender()
        self.cb_recommender = ContentBasedRecommender()
        self.item_popularity: Dict[Any, float] = {}
        self.known_users = set()

    def fit(
        self,
        interactions_df: pd.DataFrame,
        items_df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
        item_text_col: str = "description"
    ) -> 'HybridRecommender':
        """
        Fits CollaborativeRecommender and ContentBasedRecommender and computes
        global item popularity scores for cold-start mitigation.
        """
        logger.info("Fitting HybridRecommender.")

        # Fit Collaborative Filtering
        self.cf_recommender.fit(
            df=interactions_df,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col
        )

        # Fit Content-Based Filtering
        self.cb_recommender.fit(
            item_df=items_df,
            item_col=item_col,
            text_col=item_text_col
        )

        # Track known users to determine cold-start conditions
        self.known_users = set(interactions_df[user_col].unique())

        # Compute popularity fallback
        if self.fallback_popularity:
            popularity_counts = interactions_df.groupby(item_col)[rating_col].sum()
            if not popularity_counts.empty:
                max_pop = popularity_counts.max()
                if max_pop > 0:
                    self.item_popularity = (popularity_counts / max_pop).to_dict()
                else:
                    self.item_popularity = {k: 0.0 for k in popularity_counts.index}
        
        logger.info("HybridRecommender fitting complete.")
        return self

    def recommend(
        self,
        user_id: Any,
        user_history_item_ids: Optional[List[Any]] = None,
        top_k: int = 10,
        cf_weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Provides recommendations for a user.
        Enforces weighted hybrid combination, resolving cold-start via content-based or popularity.
        Returns a list of dictionaries with item_id, score, rank, and source.
        """
        weight = cf_weight if cf_weight is not None else self.cf_weight
        
        # Cold-Start Mitigation
        if user_id not in self.known_users:
            if user_history_item_ids:
                cb_recs = self.cb_recommender.recommend_for_user_profile(user_item_ids=user_history_item_ids, top_k=top_k)
                results = []
                for idx, rec in enumerate(cb_recs):
                    results.append({
                        "item_id": rec.get("item_id"),
                        "score": rec.get("score", 0.0),
                        "rank": idx + 1,
                        "source": "content_based"
                    })
                return results
            else:
                if self.fallback_popularity and self.item_popularity:
                    sorted_pop = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)[:top_k]
                    return [{
                        "item_id": item,
                        "score": score,
                        "rank": idx + 1,
                        "source": "popularity"
                    } for idx, (item, score) in enumerate(sorted_pop)]
                return []

        # Standard Hybrid Recommendation
        cf_recs = self.cf_recommender.recommend_for_user(user_id=user_id, top_k=top_k * 2)
        cf_scores = {rec["item_id"]: rec.get("score", 0.0) for rec in cf_recs if "item_id" in rec}
        
        cb_scores = {}
        if user_history_item_ids:
            cb_recs = self.cb_recommender.recommend_for_user_profile(user_item_ids=user_history_item_ids, top_k=top_k * 2)
            cb_scores = {rec["item_id"]: rec.get("score", 0.0) for rec in cb_recs if "item_id" in rec}
        
        all_candidates = set(cf_scores.keys()).union(set(cb_scores.keys()))
        hybrid_scores = []
        for item in all_candidates:
            s_cf = cf_scores.get(item, 0.0)
            s_cb = cb_scores.get(item, 0.0)
            final_score = weight * s_cf + (1.0 - weight) * s_cb
            
            source = "hybrid"
            if item in cf_scores and item not in cb_scores:
                source = "cf"
            elif item in cb_scores and item not in cf_scores:
                source = "content_based"
                
            hybrid_scores.append({
                "item_id": item,
                "score": final_score,
                "source": source
            })
            
        hybrid_scores.sort(key=lambda x: x["score"], reverse=True)
        hybrid_scores = hybrid_scores[:top_k]
        
        for idx, rec in enumerate(hybrid_scores):
            rec["rank"] = idx + 1
            
        return hybrid_scores
