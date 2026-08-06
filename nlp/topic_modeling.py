from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from shared.logging import get_logger
from shared.exceptions import PipelineError

logger = get_logger(__name__)

class TopicModeler:
    """Topic modeling for text data."""

    def __init__(self, num_topics: int = 5, method: str = "lda", n_components: Optional[int] = None):
        self.num_topics = n_components if n_components is not None else num_topics
        self.method = method.lower()
        
        if self.method == "lda":
            self.model = LatentDirichletAllocation(n_components=self.num_topics, random_state=42)
            self.vectorizer = CountVectorizer(stop_words='english')
        elif self.method == "nmf":
            self.model = NMF(n_components=self.num_topics, random_state=42, init='nndsvda')
            self.vectorizer = TfidfVectorizer(stop_words='english')
        else:
            raise ValueError(f"Unsupported topic modeling method: {self.method}")
            
        self._is_fitted = False

    def fit(self, texts: List[str]) -> 'TopicModeler':
        """Fits the topic model on the given texts."""
        if not texts:
            logger.warning("Empty texts provided to fit.")
            return self
            
        try:
            safe_texts = [str(t) if t is not None else "" for t in texts]
            if all(not t.strip() for t in safe_texts):
                logger.warning("All texts are empty.")
                return self
                
            X = self.vectorizer.fit_transform(safe_texts)
            self.model.fit(X)
            self._is_fitted = True
            return self
        except Exception as e:
            logger.error(f"Error fitting topic model: {e}")
            raise PipelineError(f"Error fitting topic model: {e}")

    def get_topics(self, top_n_words: int = 10) -> List[Dict[str, Any]]:
        """Returns the topics and their top words."""
        if not self._is_fitted:
            raise PipelineError("TopicModeler is not fitted yet. Call fit first.")
            
        try:
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(self.model.components_):
                top_features_ind = topic.argsort()[:-top_n_words - 1:-1]
                top_features = [feature_names[i] for i in top_features_ind]
                weights = [float(topic[i]) for i in top_features_ind]
                
                topic_dict = {
                    "topic_id": topic_idx,
                    "words": top_features,
                    "weights": weights
                }
                topics.append(topic_dict)
                
            return topics
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            raise PipelineError(f"Error extracting topics: {e}")

    def extract_topics(self, texts: List[str]) -> List[Dict[str, float]]:
        """Fits and transforms texts into topic probability dicts per document."""
        self.fit(texts)
        probas = self.transform(texts)
        res = []
        for row in probas:
            doc_dict = {}
            for i, p in enumerate(row):
                doc_dict[f"topic_{i}"] = float(p)
            res.append(doc_dict)
        return res

    def transform(self, texts: List[str]) -> np.ndarray:
        """Returns topic probability distributions per document."""
        if not self._is_fitted:
            raise PipelineError("TopicModeler is not fitted yet. Call fit first.")
            
        if not texts:
            return np.array([])
            
        try:
            safe_texts = [str(t) if t is not None else "" for t in texts]
            X = self.vectorizer.transform(safe_texts)
            return self.model.transform(X)
        except Exception as e:
            logger.error(f"Error transforming documents to topics: {e}")
            raise PipelineError(f"Error transforming documents to topics: {e}")
