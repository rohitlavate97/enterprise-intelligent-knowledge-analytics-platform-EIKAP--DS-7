import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Dict, List, Optional
from shared.logging import get_logger
from shared.exceptions import PipelineError

logger = get_logger(__name__)

class CollaborativeRecommender:
    def __init__(self, method: str = "svd", n_components: int = 20):
        if method not in ["svd", "user_based", "item_based"]:
            raise PipelineError(f"Invalid method: {method}")
        self.method = method
        self.n_components = n_components
        
        self.user_item_matrix = None
        self.user_ids = []
        self.item_ids = []
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        self.similarity_matrix = None

    def fit(self, df: pd.DataFrame, user_col: str = "user_id", item_col: str = "item_id", rating_col: str = "rating") -> 'CollaborativeRecommender':
        logger.info(f"Fitting CollaborativeRecommender with method {self.method}")
        
        try:
            # Create user-item matrix
            self.user_ids = df[user_col].unique().tolist()
            self.item_ids = df[item_col].unique().tolist()
            
            self.user_id_to_idx = {u: i for i, u in enumerate(self.user_ids)}
            self.item_id_to_idx = {i_id: i for i, i_id in enumerate(self.item_ids)}
            
            row = df[user_col].map(self.user_id_to_idx).values
            col = df[item_col].map(self.item_id_to_idx).values
            data = df[rating_col].values
            
            self.user_item_matrix = sp.csr_matrix((data, (row, col)), shape=(len(self.user_ids), len(self.item_ids)))
            
            if self.method == "svd":
                self.svd = TruncatedSVD(n_components=min(self.n_components, len(self.item_ids) - 1, len(self.user_ids) - 1), random_state=42)
                self.user_factors = self.svd.fit_transform(self.user_item_matrix)
                self.item_factors = self.svd.components_.T
            elif self.method == "user_based":
                self.similarity_matrix = cosine_similarity(self.user_item_matrix)
            elif self.method == "item_based":
                self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
                
            logger.info("CollaborativeRecommender fitting complete.")
            return self
        except Exception as e:
            logger.error(f"Error in fitting CollaborativeRecommender: {e}")
            raise PipelineError(f"Error in fitting CollaborativeRecommender: {e}")

    def recommend_for_user(self, user_id: Any, top_k: int = 10, filter_already_interacted: bool = True) -> List[Dict[str, Any]]:
        if user_id not in self.user_id_to_idx:
            logger.warning(f"Unknown user_id: {user_id}")
            return []
            
        user_idx = self.user_id_to_idx[user_id]
        
        if self.method == "svd":
            user_vec = self.user_factors[user_idx, :]
            scores = np.dot(user_vec, self.item_factors.T)
        elif self.method == "user_based":
            sim_users = self.similarity_matrix[user_idx, :]
            scores = sim_users.dot(self.user_item_matrix.toarray()) / (np.sum(np.abs(sim_users)) + 1e-9)
        elif self.method == "item_based":
            user_ratings = self.user_item_matrix[user_idx, :].toarray().flatten()
            scores = self.similarity_matrix.dot(user_ratings) / (np.sum(np.abs(self.similarity_matrix), axis=1) + 1e-9)
        else:
            raise PipelineError(f"Unsupported method: {self.method}")
            
        if filter_already_interacted:
            interacted_items_idx = self.user_item_matrix[user_idx, :].nonzero()[1]
            scores[interacted_items_idx] = -np.inf
            
        top_items_idx = np.argsort(scores)[::-1][:top_k]
        
        recommendations = []
        for rank, idx in enumerate(top_items_idx):
            if scores[idx] != -np.inf:
                recommendations.append({
                    "item_id": self.item_ids[idx],
                    "score": float(scores[idx]),
                    "rank": rank + 1
                })
        
        return recommendations

    def predict_rating(self, user_id: Any, item_id: Any) -> float:
        if user_id not in self.user_id_to_idx or item_id not in self.item_id_to_idx:
            return 0.0
            
        user_idx = self.user_id_to_idx[user_id]
        item_idx = self.item_id_to_idx[item_id]
        
        if self.method == "svd":
            return float(np.dot(self.user_factors[user_idx, :], self.item_factors[item_idx, :]))
        elif self.method == "user_based":
            sim_users = self.similarity_matrix[user_idx, :]
            item_ratings = self.user_item_matrix[:, item_idx].toarray().flatten()
            mask = item_ratings > 0
            if not np.any(mask):
                return 0.0
            return float(np.dot(sim_users[mask], item_ratings[mask]) / (np.sum(np.abs(sim_users[mask])) + 1e-9))
        elif self.method == "item_based":
            sim_items = self.similarity_matrix[item_idx, :]
            user_ratings = self.user_item_matrix[user_idx, :].toarray().flatten()
            mask = user_ratings > 0
            if not np.any(mask):
                return 0.0
            return float(np.dot(sim_items[mask], user_ratings[mask]) / (np.sum(np.abs(sim_items[mask])) + 1e-9))
        return 0.0
