from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable


def _hash_embedding(text: str, dims: int = 128) -> list[float]:
    vec = [0.0] * dims
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = digest[0] % dims
        sign = 1.0 if (digest[1] % 2 == 0) else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class Embedder:
    def __init__(self, backend: str = "fallback"):
        self.backend = backend
        self.model = None
        if backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                self.backend = "fallback"

    def embed(self, text: str) -> list[float]:
        if self.backend == "sentence-transformers" and self.model is not None:
            arr = self.model.encode([text], normalize_embeddings=True)[0]
            return [float(x) for x in arr.tolist()]
        return _hash_embedding(text)

    def dumps(self, vector: Iterable[float]) -> str:
        return json.dumps([float(v) for v in vector])
