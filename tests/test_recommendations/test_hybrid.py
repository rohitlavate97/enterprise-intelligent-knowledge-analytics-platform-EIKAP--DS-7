def test_hybrid_recommendation(hybrid_recommender):
    """Test hybrid recommendation combining CF and CB."""
    recs = hybrid_recommender.recommend(user_id=1, top_k=2)
    
    assert len(recs) <= 2
    if len(recs) > 0:
        assert "item_id" in recs[0]
        assert "score" in recs[0]
        assert "source" in recs[0]

def test_hybrid_cold_start_fallback(hybrid_recommender):
    """Test cold start fallback for unknown user with history."""
    recs = hybrid_recommender.recommend(user_id=999, user_history_item_ids=[101, 104], top_k=2)
    
    assert len(recs) <= 2
    if len(recs) > 0:
        assert "item_id" in recs[0]
        assert "score" in recs[0]
        assert recs[0]["source"] == "content_based"
        
    # Test completely cold user without history
    recs_pop = hybrid_recommender.recommend(user_id=999, user_history_item_ids=[], top_k=2)
    
    assert len(recs_pop) <= 2
    if len(recs_pop) > 0:
        assert recs_pop[0]["source"] == "popularity"
