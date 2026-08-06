import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Dict, List, Optional
from shared.logging import get_logger
from shared.exceptions import PipelineError

logger = get_logger(__name__)

class ContentBasedRecommender:
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words="english")
        self.item_ids = []
        self.item_id_to_idx = {}
        self.item_embeddings = None
        self.similarity_matrix = None

    def fit(self, item_df: pd.DataFrame, item_col: str = "item_id", text_col: str = "description", metadata_cols: Optional[List[str]] = None) -> 'ContentBasedRecommender':
        logger.info("Fitting ContentBasedRecommender")
        try:
            self.item_ids = item_df[item_col].tolist()
            self.item_id_to_idx = {i_id: i for i, i_id in enumerate(self.item_ids)}
            
            texts = item_df[text_col].fillna("").astype(str)
            
            if metadata_cols:
                for col in metadata_cols:
                    texts += " " + item_df[col].fillna("").astype(str)
            
            self.item_embeddings = self.vectorizer.fit_transform(texts)
            self.similarity_matrix = cosine_similarity(self.item_embeddings)
            
            logger.info("ContentBasedRecommender fitting complete.")
            return self
        except Exception as e:
            logger.error(f"Error in fitting ContentBasedRecommender: {e}")
            raise PipelineError(f"Error in fitting ContentBasedRecommender: {e}")

    def recommend_similar_items(self, item_id: Any, top_k: int = 10) -> List[Dict[str, Any]]:
        if item_id not in self.item_id_to_idx:
            logger.warning(f"Unknown item_id: {item_id}")
            return []
            
        item_idx = self.item_id_to_idx[item_id]
        sim_scores = self.similarity_matrix[item_idx].copy()
        
        # Exclude the item itself
        sim_scores[item_idx] = -np.inf
        
        top_items_idx = np.argsort(sim_scores)[::-1][:top_k]
        
        recommendations = []
        for rank, idx in enumerate(top_items_idx):
            if sim_scores[idx] != -np.inf:
                recommendations.append({
                    "item_id": self.item_ids[idx],
                    "similarity_score": float(sim_scores[idx]),
                    "rank": rank + 1
                })
                
        return recommendations

    def recommend_for_user_profile(self, user_item_ids: List[Any], top_k: int = 10) -> List[Dict[str, Any]]:
        valid_idxs = [self.item_id_to_idx[i_id] for i_id in user_item_ids if i_id in self.item_id_to_idx]
        if not valid_idxs:
            logger.warning("No known items in user_item_ids to build profile")
            return []
            
        # Aggregate user's historical item vectors
        user_profile_vec = np.asarray(self.item_embeddings[valid_idxs].mean(axis=0))
        
        sim_scores = cosine_similarity(user_profile_vec, self.item_embeddings).flatten()
        
        # Filter out items already in the profile
        for idx in valid_idxs:
            sim_scores[idx] = -np.inf
            
        top_items_idx = np.argsort(sim_scores)[::-1][:top_k]
        
        recommendations = []
        for rank, idx in enumerate(top_items_idx):
            if sim_scores[idx] != -np.inf:
                recommendations.append({
                    "item_id": self.item_ids[idx],
                    "similarity_score": float(sim_scores[idx]),
                    "rank": rank + 1
                })
                
        return recommendations
