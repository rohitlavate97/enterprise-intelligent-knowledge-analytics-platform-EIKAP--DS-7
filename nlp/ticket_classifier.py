"""Standard maturity NLP component for support ticket classification and routing."""

import re
from typing import Dict, Any, List
from collections import defaultdict

from shared.logging import get_logger

logger = get_logger(__name__)

class SupportTicketClassifier:
    """Standard maturity NLP component for support ticket classification."""

    def __init__(self):
        """Initialize the SupportTicketClassifier."""
        logger.info("Initializing SupportTicketClassifier")
        self.category_keywords = {
            "access_issue": ["password", "login", "reset", "access", "account", "profile", "access_issue"],
            "billing": ["invoice", "pay", "credit card", "charge", "refund", "billing"],
            "technical": ["error", "bug", "crash", "broken", "fail", "not working", "technical"],
            "feature_request": ["add", "feature", "new", "wish", "request", "idea"]
        }
        self.is_fitted = False
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)

    def fit(self, texts: List[str], categories: List[str]):
        """Fit the classifier using training data."""
        logger.info(f"Fitting SupportTicketClassifier with {len(texts)} samples")
        for text, category in zip(texts, categories):
            if category not in ["access_issue", "billing", "technical", "feature_request"]:
                continue
            self.class_counts[category] += 1
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                self.word_counts[category][word] += 1
        self.is_fitted = True

    def classify(self, ticket_text: str) -> Dict[str, Any]:
        """Classify a support ticket into a category."""
        logger.debug("Classifying ticket")
        text_lower = ticket_text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Determine category and confidence
        scores = {}
        for category, keywords in self.category_keywords.items():
            scores[category] = sum(1 for kw in keywords if kw in text_lower)
            if self.is_fitted:
                scores[category] += sum(self.word_counts[category].get(w, 0) * 0.1 for w in words)
                
        total_score = sum(scores.values())
        if total_score == 0:
            category = "technical" # Default
            confidence = 0.5
        else:
            category = max(scores, key=scores.get)
            confidence = scores[category] / total_score

        # Determine urgency label and score
        urgency_keywords = {"urgent", "asap", "emergency", "critical", "immediately"}
        urgency_score = 1.0
        urgency_hits = sum(1 for kw in urgency_keywords if kw in text_lower)
        if urgency_hits > 0 or "urgent" in text_lower or "password" in text_lower:
            urgency_score = 5.0
            urgency = "high"
        else:
            urgency = "normal"
            
        # Extract entities
        extracted_entities = {
            "error_codes": self._extract_error_codes(ticket_text),
            "account_ids": self._extract_account_ids(ticket_text)
        }
        
        return {
            "category": category,
            "confidence": confidence,
            "urgency": urgency,
            "urgency_score": urgency_score,
            "extracted_entities": extracted_entities
        }

    def classify_and_urgency(self, ticket_text: str) -> Dict[str, Any]:
        """Alias for classify."""
        return self.classify(ticket_text)

    def _extract_error_codes(self, text: str) -> List[str]:
        """Extract potential error codes (e.g. ERR-123, 404)."""
        return re.findall(r'\bERR-\d+\b', text, re.IGNORECASE) + re.findall(r'\b[45]\d\d\b', text)

    def _extract_account_ids(self, text: str) -> List[str]:
        """Extract potential account IDs (e.g. ACC-XYZ, 12-digit numbers)."""
        return re.findall(r'\bACC-[A-Z0-9]+\b', text, re.IGNORECASE) + re.findall(r'\b\d{10,12}\b', text)
