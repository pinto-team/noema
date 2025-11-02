# -*- coding: utf-8 -*-
"""NOEMA • skills/reply_from_memory.py — Reply using DEMO memory.


Reads pairs from data/demo_memory.jsonl and returns the best-matching output.

Optionally uses TF-IDF artifacts (data/demo_index.npz, data/demo_vocab.json);

falls back to a tiny n-gram overlap when unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any

import json


def _simple_similarity(q: str, t: str, n: int = 3) -> float:
    """n-gram Jaccard overlap as a very lightweight similarity."""
    if not q or not t:
        return 0.0
    q = q.strip()
    t = t.strip()
    if not q or not t:
        return 0.0
    n = max(1, n)
    qng = {q[i : i + n] for i in range(max(1, len(q) - n + 1))}
    tng = {t[i : i + n] for i in range(max(1, len(t) - n + 1))}
    if not qng or not tng:
        return 0.0
    inter = qng & tng
    union = qng | tng
    return len(inter) / float(len(union) or 1)


def _load_demos(path: Path) -> List[dict]:
    if not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    demos = []
    for ln in lines:
        if not ln:
            continue
        try:
            demos.append(json.loads(ln))
        except Exception:
            continue
    return demos


def run(user_text: str = "", **kwargs) -> Dict[str, Any]:
    mem_path = Path("data/demo_memory.jsonl")
    demos = _load_demos(mem_path)
    if not demos:
        return {
            "intent": "memory.reply",
            "outcome": {},
            "text_out": "No demo memory yet — record a DEMO and try again.",
            "meta": {"confidence": 0.35, "u": 0.5, "risk": 0.0, "r_total": 0.0},
            "extras": {},
            "label_ok": True,
        }

    # Try TF-IDF if artifacts/deps are available
    try:
        import numpy as np  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from scipy.sparse import csr_matrix  # type: ignore

        idx_path = Path("data/demo_index.npz")
        vocab_path = Path("data/demo_vocab.json")
        if idx_path.exists() and vocab_path.exists():
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            vectorizer = TfidfVectorizer(vocabulary=vocab)
            query_vec = vectorizer.fit_transform([user_text or ""])
            data = np.load(idx_path)
            matrix = csr_matrix((data["data"], data["indices"], data["indptr"]), shape=data["shape"])
            scores = (matrix @ query_vec.T).toarray().ravel()
            best_idx = int(scores.argmax())
            score = float(scores[best_idx])
            output = str(demos[best_idx].get("output", "").strip())
            confidence = 0.5 + 0.45 * (1.0 if score > 0 else 0.0)
            return {
                "intent": "memory.reply",
                "outcome": {"match_index": best_idx, "score": score},
                "text_out": output,
                "meta": {"confidence": min(0.95, confidence), "u": 0.2, "risk": 0.0, "r_total": 0.0},
                "extras": {},
                "label_ok": True,
            }
    except Exception:
        pass

    # Fallback: n-gram similarity
    sims: List[Tuple[float, int]] = []
    for idx, demo in enumerate(demos):
        score = _simple_similarity(user_text or "", str(demo.get("input", "")))
        sims.append((score, idx))
    sims.sort(key=lambda item: item[0], reverse=True)
    best_score, best_idx = sims[0]
    output = str(demos[best_idx].get("output", "").strip())
    confidence = 0.5 + 0.4 * min(1.0, float(best_score))

    return {
        "intent": "memory.reply",
        "outcome": {"match_index": int(best_idx), "score": float(best_score)},
        "text_out": output or "No suitable memory found yet.",
        "meta": {"confidence": min(0.9, confidence), "u": 0.3, "risk": 0.0, "r_total": 0.0},
        "extras": {},
        "label_ok": True,
    }
