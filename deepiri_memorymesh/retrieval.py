from __future__ import annotations

import json
import math
from typing import Iterable


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    aa = list(a)
    bb = list(b)
    dot = sum(x * y for x, y in zip(aa, bb))
    na = math.sqrt(sum(x * x for x in aa)) or 1.0
    nb = math.sqrt(sum(y * y for y in bb)) or 1.0
    return dot / (na * nb)


def rank_rows(query_vec: list[float], rows: list[dict], top_k: int = 8) -> list[dict]:
    scored: list[tuple[float, dict]] = []
    for row in rows:
        emb = json.loads(row["embedding_json"])
        score = cosine(query_vec, emb)
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    for score, row in scored[:top_k]:
        item = dict(row)
        item["score"] = score
        out.append(item)
    return out
