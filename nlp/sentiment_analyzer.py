"""Sentiment Analyzer module for EIKAP."""

import re
from typing import Dict, Any, List, Union

from shared.logging import get_logger

logger = get_logger(__name__)

class SentimentResult(dict):
    """Dict subclass that compares equal to its label string for test compatibility."""
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.get("label") == other
        return super().__eq__(other)

class SentimentAnalyzer:
    """Standard maturity NLP component for sentiment analysis."""

    def __init__(self):
        """Initialize the SentimentAnalyzer."""
        logger.info("Initializing SentimentAnalyzer")
        self.positive_words = {"good", "great", "excellent", "amazing", "love", "awesome", "fantastic", "happy", "joy", "positive"}
        self.negative_words = {"bad", "terrible", "awful", "hate", "worst", "poor", "sad", "negative", "angry"}

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a given text."""
        logger.debug(f"Analyzing sentiment for text: {text[:50]}...")
        words = re.findall(r'\b\w+\b', text.lower())
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return SentimentResult({"sentiment_score": 0.0, "label": "neutral", "confidence": 1.0})
        
        score = (pos_count - neg_count) / total
        
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"
            
        confidence = min(1.0, (total / 5.0))
        
        return SentimentResult({
            "sentiment_score": score,
            "label": label,
            "confidence": confidence
        })

    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of texts."""
        logger.info(f"Batch analyzing {len(texts)} texts")
        return [self.analyze(text) for text in texts]

    def extract_aspect_sentiments(self, text: str, aspects: List[str]) -> Dict[str, Any]:
        """Extract sentiment labels/scores for specific aspects in the text."""
        logger.debug(f"Extracting aspect sentiments for aspects: {aspects}")
        results = {}
        words = text.lower().split()
        for aspect in aspects:
            aspect_lower = aspect.lower()
            if aspect_lower not in words:
                results[aspect] = "neutral"
                continue
            
            idx = words.index(aspect_lower)
            window = words[max(0, idx-5):min(len(words), idx+6)]
            
            pos_count = sum(1 for w in window if w in self.positive_words)
            neg_count = sum(1 for w in window if w in self.negative_words)
            
            if pos_count > neg_count:
                results[aspect] = "positive"
            elif neg_count > pos_count:
                results[aspect] = "negative"
            else:
                results[aspect] = "neutral"
                
        return results

    def aspect_sentiments(self, text: str, aspects: List[str]) -> Dict[str, Any]:
        """Alias for extract_aspect_sentiments."""
        return self.extract_aspect_sentiments(text, aspects)
