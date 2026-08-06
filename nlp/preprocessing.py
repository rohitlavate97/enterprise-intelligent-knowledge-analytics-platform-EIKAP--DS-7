import re

class TextPreprocessor:
    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    def preprocess_pipeline(self, texts: list[str]) -> list[str]:
        return [self.clean_text(t) for t in texts]
