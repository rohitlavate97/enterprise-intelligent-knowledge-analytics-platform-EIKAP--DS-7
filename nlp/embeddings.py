from sklearn.feature_extraction.text import TfidfVectorizer

class TextEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100)

    def fit(self, texts: list[str]) -> None:
        if texts:
            self.vectorizer.fit(texts)
            
    def transform(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self.vectorizer.transform(texts).toarray().tolist()
        except ValueError:
            return [[0.0] * 100 for _ in texts]
