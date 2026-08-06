def test_topic_modeling_lda():
    from nlp import TopicModeler
    modeler = TopicModeler(n_components=2)
    texts = [
        "Python is a programming language.",
        "Machine learning is cool.",
        "Python and machine learning are related.",
        "Data science uses Python."
    ]
    topics = modeler.extract_topics(texts)
    assert len(topics) == 4
    for doc_topics in topics:
        assert "topic_0" in doc_topics
        assert "topic_1" in doc_topics
        assert abs(doc_topics["topic_0"] + doc_topics["topic_1"] - 1.0) < 1e-5
