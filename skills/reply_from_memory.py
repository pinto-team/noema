# -*- coding: utf-8 -*-
"""NOEMA • skills.reply_from_memory — Retrieve replies from episodic memory (fixed & simplified)."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json, math, time, re, unicodedata

# -------------------- Optional episodic index (kept, but not required) --------------------
try:
    from perception.encoder import encode as _encode_text  # type: ignore
except Exception:
    _encode_text = None  # type: ignore

try:
    from memory import EpisodeStore, build_from_episode_store, make_key_vector  # type: ignore
except Exception:
    EpisodeStore = None
    build_from_episode_store = None
    make_key_vector = None

_INDEX_CACHE: Optional[Any] = None
_INDEX_DIM: int = 64
_INDEX_STAMP: float = 0.0
_INDEX_TTL: float = 90.0

def _soft_hash(text: str, dim: int = 32) -> List[float]:
    vec = [0.0] * dim
    for i, ch in enumerate(text or ""):
        vec[i % dim] += (ord(ch) % 23) / 23.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]

def _pad(vec: List[float], dim: int) -> List[float]:
    return (vec + [0.0] * max(0, dim - len(vec)))[:dim]

def _encode_vector(text: str, dim: int) -> List[float]:
    if callable(_encode_text):
        try:
            got = list(_encode_text(text))
            if got:
                return _pad(got, dim)
        except Exception:
            pass
    return _pad(_soft_hash(text, max(dim, 1)), dim)

def _guess_dim(store: Any) -> int:
    if store is None or make_key_vector is None:
        return 64
    try:
        tail = store.tail(1)
        if not tail:
            return 64
        key = make_key_vector(tail[0], mode="mean")
        return len(key) if key else 64
    except Exception:
        return 64

def _ensure_index(refresh: bool = False) -> Optional[Tuple[Any, int, str]]:
    global _INDEX_CACHE, _INDEX_DIM, _INDEX_STAMP
    if EpisodeStore is None or build_from_episode_store is None:
        return None
    now = time.time()
    if _INDEX_CACHE is not None and not refresh and (now - _INDEX_STAMP) < _INDEX_TTL:
        metric = getattr(getattr(_INDEX_CACHE, "cfg", None), "metric", "ip")
        return _INDEX_CACHE, _INDEX_DIM, str(metric)
    try:
        store = EpisodeStore()
        dim = _guess_dim(store)
        idx = build_from_episode_store(store, key_mode="mean", dim=max(dim, 16), normalize=True)
        if idx.ntotal() == 0:
            _INDEX_CACHE = None
            return None
    except Exception:
        _INDEX_CACHE = None
        _INDEX_DIM = 64
        _INDEX_STAMP = now
        return None
    _INDEX_CACHE = idx
    _INDEX_DIM = int(getattr(idx.cfg, "dim", max(dim, 16)))
    _INDEX_STAMP = now
    metric = getattr(idx.cfg, "metric", "ip")
    return idx, _INDEX_DIM, str(metric)

def _score_from_distance(dist: float, metric: str) -> float:
    return 1.0 / (1.0 + float(dist)) if metric == "l2" else max(-1.0, min(1.0, float(dist)))

def _search_memory(text: str, top_k: int, refresh: bool = False) -> List[Dict[str, Any]]:
    info = _ensure_index(refresh=refresh)
    if not info:
        return []
    idx, dim, metric = info
    qv = _encode_vector(text, dim)
    try:
        dists, inds, metas = idx.search(qv, k=max(1, top_k))
    except Exception:
        return []
    hits: List[Dict[str, Any]] = []
    for dist, ind, meta in zip(list(dists), list(inds), list(metas)):
        if int(ind) < 0:
            continue
        payload = dict(meta or {})
        payload["score"] = _score_from_distance(float(dist), metric)
        payload["score_raw"] = float(dist)
        payload["metric"] = metric
        hits.append(payload)
    hits.sort(key=lambda it: it.get("score", 0.0), reverse=True)
    return hits

# ------------------------------ DEMO memory ------------------------------

def _normalize_fa(text: str) -> str:
    if not text: return ""
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"[\u200c\u200d\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", t)  # ZW* & bidi
    t = (t.replace("ي", "ی").replace("ك", "ک")
           .replace("ة", "ه").replace("ۀ", "ه")
           .replace("ؤ", "و")
           .replace("إ", "ا").replace("أ", "ا"))
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def _simple_similarity(q: str, t: str, n: int = 3) -> float:
    q = _normalize_fa(q); t = _normalize_fa(t)
    if not q or not t: return 0.0
    n = max(1, n)
    qg = {q[i:i+n] for i in range(max(1, len(q)-n+1))}
    tg = {t[i:i+n] for i in range(max(1, len(t)-n+1))}
    return len(qg & tg) / float(len(qg | tg) or 1)

def _load_demos(path: Path) -> List[Dict[str, Any]]:
    if not path.exists(): return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out

def _reply_from_demos(user_text: str) -> Dict[str, Any]:
    mem_path = Path(__file__).resolve().parents[1] / "data" / "demo_memory.jsonl"
    demos = _load_demos(mem_path)
    if not demos:
        return {
            "intent": "memory.reply",
            "text_out": "No episodic memory yet.",
            "meta": {"confidence": 0.35, "u": 0.55, "risk": 0.0, "r_total": 0.0},
            "extras": {"matches": []},
            "label_ok": True,
        }

    # Try TF-IDF if artifacts exist
    try:
        import numpy as np  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from scipy.sparse import csr_matrix  # type: ignore

        idx_path = Path("data/demo_index.npz")
        vocab_path = Path("data/demo_vocab.json")
        if idx_path.exists() and vocab_path.exists():
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            vectorizer = TfidfVectorizer(vocabulary=vocab)
            q = vectorizer.fit_transform([_normalize_fa(user_text) or ""])
            data = np.load(idx_path)
            mat = csr_matrix((data["data"], data["indices"], data["indptr"]), shape=data["shape"])
            scores = (mat @ q.T).toarray().ravel()
            bi = int(scores.argmax())
            out = str(demos[bi].get("output", "")).strip()
            conf = 0.7 if float(scores[bi]) > 0 else 0.55
            return {
                "intent": "memory.reply",
                "text_out": out or "No episodic memory yet.",
                "meta": {"confidence": min(0.9, conf), "u": 0.25, "risk": 0.0, "r_total": 0.0},
                "extras": {"matches": [{"index": bi, "score": float(scores[bi])}]},
                "label_ok": True,
            }
    except Exception:
        pass

    # Fallback: character n-gram overlap
    sims = [(_simple_similarity(user_text, str(d.get("input", ""))), i) for i, d in enumerate(demos)]
    sims.sort(key=lambda it: it[0], reverse=True)
    best_score, bi = sims[0]
    out = str(demos[bi].get("output", "")).strip()
    if best_score < 0.05 or not out:
        out = "No episodic memory yet."
    conf = 0.55 + 0.35 * min(1.0, float(best_score))
    return {
        "intent": "memory.reply",
        "text_out": out,
        "meta": {"confidence": min(0.9, conf), "u": 0.25, "risk": 0.0, "r_total": 0.0},
        "extras": {"matches": [{"index": int(bi), "score": float(best_score)}]},
        "label_ok": True,
    }

# ------------------------------ Public run ------------------------------

_BAD_TEXTS = {
    "Please share a bit more detail so I can help better.",
    "Could you clarify what you mean?",
}

def run(user_text: str = "", *, k: int = 3, refresh_index: bool = False, **kwargs) -> Dict[str, Any]:
    query = str(user_text or "")

    # 1) DEMO-FIRST
    demo = _reply_from_demos(query)
    if demo.get("text_out") and demo["text_out"].strip() and demo["text_out"] != "No episodic memory yet.":
        return demo

    # 2) episodic (optional), filter out clarify loops
    hits = _search_memory(query, top_k=max(1, int(k)), refresh=bool(refresh_index))
    if hits:
        clean = []
        for h in hits:
            txt = str(h.get("text_out") or h.get("text_in") or "").strip()
            act = str(h.get("action") or "")
            if txt and txt not in _BAD_TEXTS and act != "ask_clarify":
                clean.append(h)
        if clean:
            best = clean[0]
            score = float(best.get("score", 0.0))
            text_out = str(best.get("text_out") or best.get("text_in") or "").strip()
            conf = 0.6 + 0.3 * max(0.0, min(1.0, score))
            return {
                "intent": "memory.reply",
                "text_out": text_out,
                "meta": {"confidence": min(0.92, conf), "u": max(0.15, 0.5 - conf * 0.3), "risk": 0.0, "r_total": 0.0},
                "extras": {"matches": clean[:max(1, int(k))]},
                "label_ok": True,
            }

    # 3) No episodic → demo result (even if fallback)
    return demo
