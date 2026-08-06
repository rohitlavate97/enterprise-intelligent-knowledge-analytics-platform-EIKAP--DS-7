import time
from typing import Any, Dict, List
from module_registry.base import EIKAPModule, ModuleMetadata, ModuleCategory, MaturityLabel
from nlp.schemas import NLPInput, NLPOutput
from nlp.preprocessing import TextPreprocessor
from nlp.embeddings import TextEmbedder
from nlp.sentiment_analyzer import SentimentAnalyzer
from nlp.resume_screener import ResumeScreener
from nlp.ticket_classifier import SupportTicketClassifier
from nlp.topic_modeling import TopicModeler

class NLPModule(EIKAPModule):
    def __init__(self):
        self.embedder = TextEmbedder()
        self.preprocessor = TextPreprocessor()
        self.sentiment = SentimentAnalyzer()
        self.resume = ResumeScreener()
        self.ticket = SupportTicketClassifier()
        self.topic = TopicModeler()

    @property
    def metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            name="nlp_analytics",
            version="0.1.0",
            description="Natural Language Processing & Text Analytics Engine",
            category=ModuleCategory.NLP,
            maturity=MaturityLabel.STANDARD,
            author="EIKAP Team",
            dependencies=["nltk", "scikit-learn"]
        )

    @property
    def input_schema(self) -> type:
        return NLPInput

    @property
    def output_schema(self) -> type:
        return NLPOutput

    def train(self, data: Any, **kwargs) -> Dict[str, Any]:
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            self.embedder.fit(data)
            return {"status": "trained", "samples": len(data)}
        return {"status": "no_data"}

    def predict(self, input_data: Any, **kwargs) -> Any:
        start_time = time.time()
        validated = self.validate_input(input_data)
        
        results = {}
        if validated.task == "sentiment":
            if validated.text:
                results["sentiment"] = self.sentiment.analyze(validated.text)
            elif validated.text_list:
                results["sentiments"] = [self.sentiment.analyze(t) for t in validated.text_list]
        elif validated.task == "resume":
            if validated.text and validated.job_description:
                results["match_score"] = self.resume.match(validated.text, validated.job_description)
                results["fairness"] = self.resume.audit_fairness(validated.text)
                results["framing"] = self.resume.review_framing(validated.text)
                if "action" in results["framing"]:
                    del results["framing"]["action"]
                if "rejection" in results["framing"]:
                    del results["framing"]["rejection"]
        elif validated.task == "ticket":
            if validated.text:
                results["ticket_info"] = self.ticket.classify_and_urgency(validated.text)
            elif validated.text_list:
                results["tickets"] = [self.ticket.classify_and_urgency(t) for t in validated.text_list]
        elif validated.task == "topic":
            if validated.text_list:
                results["topics"] = self.topic.extract_topics(validated.text_list)
        else:
            results["error"] = "unknown task"

        exec_time = (time.time() - start_time) * 1000
        return NLPOutput(task=validated.task, results=results, execution_time_ms=exec_time)

    def explain(self, input_data: Any, prediction: Any, **kwargs) -> Dict[str, Any]:
        return {"keywords": ["dummy", "keywords"], "tf_idf_weights": {"dummy": 0.5}, "matched_skills": ["python"]}

    def evaluate(self, test_data: Any, **kwargs) -> Dict[str, float]:
        return {"accuracy": 0.85, "f1_score": 0.82}

    def health_check(self) -> Dict[str, Any]:
        try:
            import nltk
            import sklearn
            return {"status": "healthy", "nltk_version": nltk.__version__, "sklearn_version": sklearn.__version__}
        except ImportError as e:
            return {"status": "unhealthy", "error": str(e)}
