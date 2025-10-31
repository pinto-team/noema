# -*- coding: utf-8 -*-
"""
NOEMA • perception/encoder.py — رمزگذار متنی مینیمال (V0)
- هدف: تبدیل متن خام به یک بردار پایدار و فشرده (Latent z ∈ R^d).
- بدون وابستگی سنگین: از n-gram + هشِ پایدار + نرمال‌سازی فارسی/عربی.
- اگر بعداً مدل آموختنی اضافه شد، همین API ثابت می‌ماند.

API عمومی:
    encode(text: str) -> List[float]
    encode_batch(texts: List[str]) -> List[List[float]]

نکته:
- هَش داخلی «پایدار» است (hashlib) و به مقدار PYTHONHASHSEED وابسته نیست.
- نرمال‌سازی شامل یکسان‌سازی «ی/ك→ی/ک»، حذف کشیده/اعراب، تبدیل ارقام فارسی→لاتین.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Iterable, Tuple, Optional, Dict
from pathlib import Path
import math, re, json, unicodedata, hashlib

# ----------------- نرمال‌سازی متن فارسی/عربی -----------------

_ARABIC_TO_PERSIAN = {
    "\u064a": "\u06cc",  # ي -> ی
    "\u0643": "\u06a9",  # ك -> ک
}

_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_LATIN_DIGITS   = "0123456789"
_DIGIT_MAP = {ord(p): ord(l) for p, l in zip(_PERSIAN_DIGITS, _LATIN_DIGITS)}

_ZWNJ = "\u200c"
_TATWEEL = "\u0640"

def normalize_text(text: str) -> str:
    """یکسان‌سازی سبک برای فارسی: ی/ک عربی، حذف کشیده/اعراب، تبدیل ارقام، فاصله‌های اضافی."""
    if not text:
        return ""
    # NFC → حذف ترکیبات عجیب
    t = unicodedata.normalize("NFC", text)
    # جایگزینی حروف عربی به فارسی
    for a, p in _ARABIC_TO_PERSIAN.items():
        t = t.replace(a, p)
    # حذف کشیده و ZWNJ اضافی
    t = t.replace(_TATWEEL, " ").replace(_ZWNJ, " ")
    # حذف اعراب/نقاط ترکیبی
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    # تبدیل ارقام فارسی→لاتین
    t = t.translate(_DIGIT_MAP)
    # حروف کوچک + تراش فاصله‌ها
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ----------------- هش پایدار و n-gram -----------------

def _stable_hash_to_bucket(s: str, d: int) -> int:
    """
    هشِ پایدار (blake2b) → اندیسِ سطل [0..d-1]
    """
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    # به int تبدیل و مد بگیریم
    v = int.from_bytes(h, byteorder="big", signed=False)
    return v % d

def _char_ngrams(t: str, n_values: Tuple[int, ...] = (2, 3)) -> Iterable[str]:
    """استخراج n-gramهای کاراکتری (برای متون کوتاه هم خوب جواب می‌دهد)."""
    for n in n_values:
        if len(t) < n:
            continue
        for i in range(len(t) - n + 1):
            yield t[i : i + n]

# ----------------- رمزگذار مینیمال -----------------

@dataclass
class EncoderConfig:
    dim: int = 64
    ngrams: Tuple[int, ...] = (2, 3)
    l2_normalize: bool = True

class TextEncoderV0:
    """
    رمزگذار سبک مبتنی بر n-gram hashing:
      - هر n-gram به یک سطل در بردار dim-بعدی نگاشت می‌شود.
      - وزن‌دهی TF ساده (تعداد وقوع)، سپس L2 نرمال‌سازی.
    """

    def __init__(self, cfg: Optional[EncoderConfig] = None):
        self.cfg = cfg or EncoderConfig()

    # --- API اصلی ---
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

    # --- ابزارها ---
    @staticmethod
    def _l2_inplace(v: List[float]) -> None:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        for i, x in enumerate(v):
            v[i] = x / n

    # --- ذخیره/بارگذاری پیکربندی (بدون وزن) ---
    def save_config(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self.cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_config(cls, path: str | Path) -> "TextEncoderV0":
        p = Path(path)
        cfg = EncoderConfig(**json.loads(p.read_text(encoding="utf-8")))
        return cls(cfg)

# ----------------- Singleton ساد‌ه و توابع سطح-ماژول -----------------

# یک نمونه‌ی پیش‌فرض برای راحتیِ استفاده از سایر ماژول‌ها
_DEFAULT_ENCODER = TextEncoderV0()

def set_config(dim: int = 64, ngrams: Tuple[int, ...] = (2, 3), l2_normalize: bool = True) -> None:
    """تنظیم پیکربندی رمزگذار پیش‌فرض (قبل از استفاده صدا بزنید)."""
    global _DEFAULT_ENCODER
    _DEFAULT_ENCODER = TextEncoderV0(EncoderConfig(dim=dim, ngrams=ngrams, l2_normalize=l2_normalize))

def encode(text: str) -> List[float]:
    """رمزگذاری یک متن با رمزگذار پیش‌فرض (سازگار با app/main.py)."""
    return _DEFAULT_ENCODER.encode(text)

def encode_batch(texts: List[str]) -> List[List[float]]:
    return _DEFAULT_ENCODER.encode_batch(texts)

# ----------------- اجرای مستقیم برای تست سریع -----------------

if __name__ == "__main__":
    samples = [
        "سلام!",
        "سلام!!",
        "خداحافظ",
        "۲ + ۲ ؟",
        "Hello",
        "HELLO",
    ]
    print("dim =", _DEFAULT_ENCODER.cfg.dim)
    for s in samples:
        v = encode(s)
        print(f"{s!r} -> ||z||≈{math.sqrt(sum(x*x for x in v)):.2f}, first5={v[:5]}")
