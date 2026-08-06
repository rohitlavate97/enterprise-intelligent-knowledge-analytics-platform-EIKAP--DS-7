import pytest
import pandas as pd
from recommendations.collaborative import CollaborativeRecommender
from recommendations.content_based import ContentBasedRecommender
from recommendations.hybrid import HybridRecommender

@pytest.fixture
def sample_interactions():
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3, 3, 4],
        "item_id": [101, 102, 103, 101, 104, 102, 105, 106],
        "rating": [5.0, 4.0, 3.0, 5.0, 4.0, 4.0, 5.0, 3.0]
    })

@pytest.fixture
def sample_items():
    return pd.DataFrame({
        "item_id": [101, 102, 103, 104, 105, 106],
        "description": [
            "Action movie with car chases",
            "Romantic comedy in Paris",
            "Sci-fi adventure in space",
            "Action movie with explosions",
            "Romantic drama",
            "Documentary about space"
        ],
        "category": ["Action", "Romance", "Sci-Fi", "Action", "Romance", "Documentary"]
    })

@pytest.fixture
def collaborative_recommender(sample_interactions):
    recommender = CollaborativeRecommender(method="svd", n_components=2)
    recommender.fit(sample_interactions)
    return recommender

@pytest.fixture
def content_recommender(sample_items):
    recommender = ContentBasedRecommender()
    recommender.fit(sample_items, metadata_cols=["category"])
    return recommender

@pytest.fixture
def hybrid_recommender(sample_interactions, sample_items):
    recommender = HybridRecommender(cf_weight=0.5)
    recommender.fit(sample_interactions, sample_items)
    return recommender
