"""NER inference for the app and the benchmarks: load a trained spaCy model and extract entity spans."""

from __future__ import annotations

from pathlib import Path


class SpacyNer:
    """A trained spaCy NER model (the one downloaded from S3 into models/ner/<name>/model-best)."""

    def __init__(self, model_dir: Path | str):
        import spacy

        self.nlp = spacy.load(str(model_dir))

    @staticmethod
    def _entities(doc) -> list[dict]:
        return [{"text": e.text, "label": e.label_, "span": [e.start_char, e.end_char]} for e in doc.ents]

    def predict(self, text: str) -> list[dict]:
        return self._entities(self.nlp(text))

    def predict_many(self, texts: list[str], batch_size: int = 64) -> list[list[dict]]:
        return [self._entities(d) for d in self.nlp.pipe(texts, batch_size=batch_size)]
