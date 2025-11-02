# -*- coding: utf-8 -*-
"""
NOEMA • perception/encoder.py — minimal text encoder (V0)

Goal
-----
Turn raw text into a stable, compact vector (latent z ∈ R^d) using
character n-gram hashing + lightweight, language-agnostic normalization.

Public API
----------
- encode(text: str) -> List[float]
- encode_batch(texts: List[str]) -> List[List[float]]
- normalize_text(text: str) -> str
- set_config(...), TextEncoderV0, EncoderConfig

Notes
-----
- Uses deterministic hashing (blake2b), independent of PYTHONHASHSEED.
- Normalization is language-agnostic: Unicode NFKC, remove combining marks,
  casefold, collapse whitespace, and map ANY decimal digit to ASCII 0-9.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple, Optional
from pathlib import Path
import hashlib
import json
import math
import re
import unicodedata

# ------------------------- Normalization -------------------------

def normalize_text(text: str) -> str:
    """
    Language-agnostic normalization:
      - Unicode NFKC (compatibility decomposition + composition)
      - strip combining marks (accents/diacritics)
      - map any Unicode decimal digit (Nd) to ASCII 0-9
      - casefold
      - collapse whitespace
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)

    # remove combining marks
    t = "".join(ch for ch in t if not unicodedata.combining(ch))

    # map any decimal digit to ASCII '0'..'9'
    out_chars: List[str] = []
    for ch in t:
        if ch.isdecimal():
            try:
                out_chars.append(str(unicodedata.decimal(ch)))
            except Exception:
                out_chars.append(ch)
        else:
            out_chars.append(ch)
    t = "".join(out_chars)

    # casefold + collapse spaces
    t = t.casefold()
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ------------------------- Hashing & n-grams -------------------------

def _stable_hash_to_bucket(s: str, d: int) -> int:
    """Deterministic blake2b hash to bucket index in [0, d)."""
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(h, byteorder="big", signed=False)
    return v % d

def _char_ngrams(t: str, n_values: Tuple[int, ...] = (2, 3)) -> Iterable[str]:
    """Character n-grams (works well for short texts)."""
    for n in n_values:
        if len(t) < n:
            continue
        for i in range(len(t) - n + 1):
            yield t[i : i + n]

# ------------------------- Encoder -------------------------

@dataclass
class EncoderConfig:
    dim: int = 64
    ngrams: Tuple[int, ...] = (2, 3)
    l2_normalize: bool = True

class TextEncoderV0:
    """
    Lightweight n-gram hashing encoder:
      - Each char n-gram maps into a bucket in a dim-sized vector.
      - TF counting + optional L2 normalization.
    """

    def __init__(self, cfg: Optional[EncoderConfig] = None):
        self.cfg = cfg or EncoderConfig()

    def encode(self, text: str) -> List[float]:
        t = normalize_text(text)
        if not t:
            return [0.0] * self.cfg.dim

        vec = [0.0] * self.cfg.dim
        for ng in _char_ngrams(t, self.cfg.ngrams):
            j = _stable_hash_to_bucket(ng, self.cfg.dim)
            vec[j] += 1.0

        if self.cfg.l2_normalize:
            self._l2_inplace(vec)
        return vec

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(x) for x in texts]

    @staticmethod
    def _l2_inplace(v: List[float]) -> None:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        inv = 1.0 / n
        for i, x in enumerate(v):
            v[i] = x * inv

    # ---- config I/O (no learned weights here) ----
    def save_config(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self.cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_config(cls, path: str | Path) -> "TextEncoderV0":
        p = Path(path)
        cfg = EncoderConfig(**json.loads(p.read_text(encoding="utf-8")))
        return cls(cfg)

# ------------------------- Module-level convenience -------------------------

_DEFAULT_ENCODER = TextEncoderV0()

def set_config(dim: int = 64, ngrams: Tuple[int, ...] = (2, 3), l2_normalize: bool = True) -> None:
    """Override the default encoder configuration (call before use)."""
    global _DEFAULT_ENCODER
    _DEFAULT_ENCODER = TextEncoderV0(EncoderConfig(dim=dim, ngrams=ngrams, l2_normalize=l2_normalize))

def encode(text: str) -> List[float]:
    """Encode a single text with the default encoder (compatible with app/main.py)."""
    return _DEFAULT_ENCODER.encode(text)

def encode_batch(texts: List[str]) -> List[List[float]]:
    return _DEFAULT_ENCODER.encode_batch(texts)

# ------------------------- Manual test -------------------------

if __name__ == "__main__":
    samples = [
        "سلام!", "سلام!!", "خداحافظ", "۲ + ۲ ؟",  # non-Latin digits handled
        "Hello", "HELLO", "héllö  café", "١٢٣۴۵ six ７ eight ９",
    ]
    print("dim =", _DEFAULT_ENCODER.cfg.dim)
    for s in samples:
        v = encode(s)
        norm = math.sqrt(sum(x*x for x in v))
        print(f"{s!r} -> ||z||≈{norm:.2f}, first5={v[:5]}")
