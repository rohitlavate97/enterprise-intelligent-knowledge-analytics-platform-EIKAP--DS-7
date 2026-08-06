def test_content_similar_items(content_recommender):
    """Test finding similar items using content-based filtering."""
    recs = content_recommender.recommend_similar_items(item_id=101, top_k=2)
    
    assert len(recs) <= 2
    if len(recs) > 0:
        assert "item_id" in recs[0]
        assert "similarity_score" in recs[0]
        
    # Item 101 is "Action movie with car chases"
    # Expected similar item is 104 "Action movie with explosions"
    rec_item_ids = [r["item_id"] for r in recs]
    assert 104 in rec_item_ids

def test_content_user_profile(content_recommender):
    """Test recommending items based on user history profile."""
    # User likes Action (101, 104)
    recs = content_recommender.recommend_for_user_profile(user_item_ids=[101, 104], top_k=2)
    
    # Verify we got recommendations
    assert len(recs) <= 2
    
    rec_item_ids = [r["item_id"] for r in recs]
    # Shouldn't recommend items already in profile
    assert 101 not in rec_item_ids
    assert 104 not in rec_item_ids
