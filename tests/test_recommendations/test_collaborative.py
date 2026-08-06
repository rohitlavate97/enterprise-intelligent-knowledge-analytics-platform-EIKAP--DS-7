def test_cf_fit_recommend(collaborative_recommender):
    """Test fitting and recommending with CollaborativeRecommender."""
    recs = collaborative_recommender.recommend_for_user(user_id=1, top_k=2)
    
    assert len(recs) <= 2
    if len(recs) > 0:
        assert "item_id" in recs[0]
        assert "score" in recs[0]
        
    # User 1 has interacted with 101, 102, 103. The recommendations should NOT include these
    rec_item_ids = [r["item_id"] for r in recs]
    assert 101 not in rec_item_ids
    assert 102 not in rec_item_ids
    assert 103 not in rec_item_ids

    # Cold user should return empty
    cold_recs = collaborative_recommender.recommend_for_user(user_id=999, top_k=2)
    assert len(cold_recs) == 0
