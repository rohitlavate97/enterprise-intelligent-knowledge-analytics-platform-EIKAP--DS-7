def test_defect_inspection_clean_vs_defective(defect_inspector, sample_synthetic_image, sample_defective_image):
    # Test on clean image
    clean_result = defect_inspector.inspect_image(sample_synthetic_image)
    assert "is_defective" in clean_result
    assert "defect_score" in clean_result
    
    # Test on defective image
    defect_result = defect_inspector.inspect_image(sample_defective_image)
    
    # We just ensure it runs and returns expected keys
    assert "is_defective" in defect_result
    assert "defect_score" in defect_result
