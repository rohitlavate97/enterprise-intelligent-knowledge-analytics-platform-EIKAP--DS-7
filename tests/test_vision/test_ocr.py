def test_ocr_parse_receipt(ocr_parser, sample_receipt_image):
    result = ocr_parser.parse_receipt(sample_receipt_image)
    
    assert "raw_text" in result
    assert "detected_bounding_boxes" in result
    assert "merchant_name" in result
    assert "total_amount" in result
    
    # In our mock implementation, it might not detect exact numbers, but fields should exist
    assert result["detected_bounding_boxes"] is not None
