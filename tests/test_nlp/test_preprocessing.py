def test_clean_text(preprocessor):
    text = "Hello, World! 123"
    cleaned = preprocessor.clean_text(text)
    assert cleaned == "hello world 123"

def test_preprocess_pipeline(preprocessor):
    texts = ["Test 1!", "Another-Test."]
    cleaned_texts = preprocessor.preprocess_pipeline(texts)
    assert cleaned_texts == ["test 1", "anothertest"]
