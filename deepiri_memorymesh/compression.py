from __future__ import annotations

import re
from collections import Counter


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def compress_conversation(text: str, target_chars: int = 1200) -> str:
    sents = _sentences(text)
    if not sents:
        return ""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    freqs = Counter(words)
    scored: list[tuple[float, str]] = []
    for s in sents:
        score = sum(freqs.get(w.lower(), 0) for w in re.findall(r"[a-zA-Z]{3,}", s))
        scored.append((float(score), s))
    ranked = sorted(scored, key=lambda x: x[0], reverse=True)
    picked: list[str] = []
    size = 0
    for _, s in ranked:
        if size + len(s) > target_chars and picked:
            continue
        picked.append(s)
        size += len(s) + 1
        if size >= target_chars:
            break
    original_order = [s for s in sents if s in set(picked)]
    return " ".join(original_order)[:target_chars]
