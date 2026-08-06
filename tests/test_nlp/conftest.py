import pytest
from nlp import TextPreprocessor, TextEmbedder, SentimentAnalyzer, ResumeScreener, SupportTicketClassifier

@pytest.fixture
def sample_texts():
    return ["This is a good text.", "This is a bad text.", "Urgent support ticket."]

@pytest.fixture
def sample_resume():
    return "Python engineer with 5 years experience. Graduated in 1995. He lives in Zip 12345."

@pytest.fixture
def sample_job_description():
    return "Looking for a Python engineer with experience."

@pytest.fixture
def sample_tickets():
    return [
        {"text": "Urgent: password reset needed", "label": "access_issue"},
        {"text": "General billing question", "label": "billing"}
    ]

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

@pytest.fixture
def embedder():
    return TextEmbedder()

@pytest.fixture
def sentiment_analyzer():
    return SentimentAnalyzer()

@pytest.fixture
def resume_screener():
    return ResumeScreener()

@pytest.fixture
def ticket_classifier():
    return SupportTicketClassifier()
