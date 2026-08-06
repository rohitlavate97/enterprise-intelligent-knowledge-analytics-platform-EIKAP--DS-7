from recommendations.evaluation import RecommendationEvaluator

def test_precision_recall_at_k():
    """Test Precision@K and Recall@K calculations."""
    actual = [1, 2, 3]
    recommended = [2, 3, 4, 5, 6]
    
    # Precision@3 = 2/3 (2 and 3 are relevant)
    p3 = RecommendationEvaluator.precision_at_k(actual, recommended, 3)
    assert abs(p3 - (2.0 / 3.0)) < 1e-6
    
    # Recall@3 = 2/3 (2 out of 3 actual items found)
    r3 = RecommendationEvaluator.recall_at_k(actual, recommended, 3)
    assert abs(r3 - (2.0 / 3.0)) < 1e-6

def test_ndcg_at_k():
    """Test NDCG@K calculation."""
    actual = [1, 2, 3]
    recommended = [2, 4, 1]
    
    ndcg = RecommendationEvaluator.ndcg_at_k(actual, recommended, 3)
    assert 0.0 <= ndcg <= 1.0
    
    # Perfect ranking
    perfect = RecommendationEvaluator.ndcg_at_k(actual, actual, 3)
    assert abs(perfect - 1.0) < 1e-6
