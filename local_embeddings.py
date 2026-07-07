from __future__ import annotations

import hashlib
import math
from collections import Counter

from langchain_core.embeddings import Embeddings


class LocalHashEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _tokenize(self, text: str) -> list[str]:
        text = " ".join(text.split()).lower()
        if not text:
            return []
        chars = list(text)
        bigrams = [text[i : i + 2] for i in range(max(len(text) - 1, 0))]
        return chars + bigrams

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        counts = Counter(self._tokenize(text))
        for token, weight in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * float(weight)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
