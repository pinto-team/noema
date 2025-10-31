# -*- coding: utf-8 -*-
"""
NOEMA • memory/index_faiss.py — ایندکس برداری (ANN) مینیمال با FAISS (V0)
- هدف: جست‌وجوی اپیزودهای مشابه بر اساس بردار کلید (z/s/میانگین).
- سازگار با فایل‌های JSONL اپیزود (memory/episodic.py).

خلاصه‌ی طراحی:
- دو فایل کنار هم ذخیره می‌شوند:
    data/index/faiss.index         ← خودِ ایندکس FAISS
    data/index/faiss.meta.jsonl    ← متادیتای هر بردار (به همان ترتیب)
- اگر faiss نصب نباشد، یک fallback ساده با NumPy (brute-force) فعال می‌شود.

API اصلی:
    idx = FaissIndex(dim=64, kind="HNSW32", metric="ip")
    idx.add([vec1, vec2, ...], metas=[{...}, ...])       # افزودن دسته‌ای
    D, I, M = idx.search(query_vec, k=5)                  # نتایج (فاصله‌ها، شناسه‌ها، متا)
    idx.save(prefix="data/index/faiss")                   # ذخیره
    idx2 = FaissIndex.load(prefix="data/index/faiss")     # بارگذاری

ادمین:
    - برای cosine، بردارها را L2 normalize کنید و metric="ip" بگذارید.
    - برای L2، normalize نکنید و metric="l2" قرار دهید.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import json, os, math, numpy as np

# تلاش برای وارد کردن FAISS
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# ابزارها
def _as_float32(a: Iterable[float]) -> np.ndarray:
    return np.asarray(list(a), dtype=np.float32)

def _l2_normalize_inplace(X: np.ndarray) -> None:
    # X: [N, D]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms

# ---------------------- ایندکس fallback (بدون FAISS) ----------------------

class _BruteANN:
    """جایگزین ساده وقتی faiss موجود نیست؛ جست‌وجوی O(ND)."""
    def __init__(self, dim: int, metric: str = "ip"):
        assert metric in ("ip", "l2")
        self.dim = int(dim)
        self.metric = metric
        self._X: Optional[np.ndarray] = None  # [N, D]

    def add(self, X: np.ndarray) -> None:
        if self._X is None:
            self._X = X.copy()
        else:
            self._X = np.vstack([self._X, X])

    def ntotal(self) -> int:
        return 0 if self._X is None else int(self._X.shape[0])

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        assert q.shape == (self.dim,), f"query shape {q.shape} != ({self.dim},)"
        if self._X is None or self.ntotal() == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        X = self._X  # [N, D]
        if self.metric == "ip":
            # فرض: q و X نرمال شده‌اند (cosine≈dot)
            scores = X @ q  # [N]
            I = np.argsort(-scores)[:k]
            D = scores[I]
        else:  # l2
            dif = X - q[None, :]
            d2 = np.sum(dif * dif, axis=1)
            I = np.argsort(d2)[:k]
            D = d2[I]
        return D.astype(np.float32), I.astype(np.int64)

# ---------------------- ایندکس FAISS اصلی ----------------------

@dataclass
class IndexConfig:
    dim: int = 64
    kind: str = "HNSW32"     # "Flat", "HNSW32", "IVF100,PQ16" ...
    metric: str = "ip"       # "ip" (cos) یا "l2"
    l2_normalize: bool = True  # اگر metric="ip" و قصد cosine دارید، True باشد.

class FaissIndex:
    """
    لایه‌ی نازک روی FAISS + فایل متادیتا.
    """

    def __init__(self, cfg: IndexConfig, index_obj: Any = None, metas: Optional[List[Dict[str, Any]]] = None):
        self.cfg = cfg
        self.metas: List[Dict[str, Any]] = metas or []
        self._index = index_obj or self._create_index(cfg)
        self._fallback = None
        if not _HAS_FAISS:
            self._fallback = _BruteANN(cfg.dim, cfg.metric)

    # ---------- ساخت ایندکس ----------
    @staticmethod
    def _create_index(cfg: IndexConfig):
        if not _HAS_FAISS:
            return None
        metric = faiss.METRIC_INNER_PRODUCT if cfg.metric == "ip" else faiss.METRIC_L2
        if cfg.kind.upper() == "FLAT":
            index = faiss.IndexFlat(cfg.dim, metric)
        elif cfg.kind.upper().startswith("HNSW"):
            # HNSW32 → M=32
            try:
                m = int(cfg.kind.upper().replace("HNSW", ""))
            except Exception:
                m = 32
            index = faiss.IndexHNSWFlat(cfg.dim, m, metric)
            index.hnsw.efSearch = 64
            index.hnsw.efConstruction = 80
        else:
            # تلاش برای parse IVF,PQ؛ اگر نشد، Flat
            try:
                # e.g., "IVF100,PQ16"
                coarse = int(cfg.kind.split(",")[0].replace("IVF", ""))
                pqm = int(cfg.kind.split(",")[1].replace("PQ", ""))
                quantizer = faiss.IndexFlat(cfg.dim, metric)
                index = faiss.IndexIVFPQ(quantizer, cfg.dim, coarse, pqm, 8, metric)
                index.nprobe = min(8, coarse)
            except Exception:
                index = faiss.IndexFlat(cfg.dim, metric)
        return index

    # ---------- افزودن ----------
    def add(self, vecs: List[List[float]], metas: Optional[List[Dict[str, Any]]] = None) -> None:
        X = np.asarray(vecs, dtype=np.float32)
        assert X.ndim == 2 and X.shape[1] == self.cfg.dim, f"Expected [N,{self.cfg.dim}] got {X.shape}"
        if self.cfg.l2_normalize and self.cfg.metric == "ip":
            _l2_normalize_inplace(X)
        # FAISS
        if self._index is not None:
            # اگر IVF است و آموزش ندیده، یکبار آموزش بده
            if hasattr(self._index, "is_trained") and not self._index.is_trained:
                self._index.train(X)
            self._index.add(X)
        # Fallback
        if self._fallback is not None:
            self._fallback.add(X)
        # متادیتا
        metas = metas or [{} for _ in range(X.shape[0])]
        self.metas.extend(metas)

    # ---------- تعداد ----------
    def ntotal(self) -> int:
        if self._index is not None:
            return int(self._index.ntotal)
        if self._fallback is not None:
            return self._fallback.ntotal()
        return 0

    # ---------- جست‌وجو ----------
    def search(self, q: List[float], k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if self.ntotal() == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
        qv = _as_float32(q)
        assert qv.shape == (self.cfg.dim,), f"query dim mismatch: {qv.shape}"
        if self.cfg.l2_normalize and self.cfg.metric == "ip":
            qv = qv / (np.linalg.norm(qv) or 1.0)

        if self._index is not None:
            qv2 = qv[None, :]  # [1, D]
            D, I = self._index.search(qv2, min(k, self.ntotal()))
            D, I = D[0], I[0]
        else:
            D, I = self._fallback.search(qv, min(k, self.ntotal()))  # type: ignore

        metas = [self.metas[i] for i in I if 0 <= int(i) < len(self.metas)]
        return D, I, metas

    # ---------- ذخیره ----------
    def save(self, prefix: str | Path = "data/index/faiss") -> None:
        prefix = str(prefix)
        idx_path = Path(prefix + ".index")
        meta_path = Path(prefix + ".meta.jsonl")
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_FAISS and self._index is not None:
            faiss.write_index(self._index, str(idx_path))
        # متادیتا
        with meta_path.open("w", encoding="utf-8") as f:
            for m in self.metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ---------- بارگذاری ----------
    @classmethod
    def load(cls, prefix: str | Path = "data/index/faiss", cfg: Optional[IndexConfig] = None) -> "FaissIndex":
        prefix = str(prefix)
        idx_path = Path(prefix + ".index")
        meta_path = Path(prefix + ".meta.jsonl")
        metas: List[Dict[str, Any]] = []
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        metas.append(json.loads(ln))
                    except Exception:
                        continue
        # اگر faiss هست و فایل وجود دارد، بخوان
        index_obj = None
        if _HAS_FAISS and idx_path.exists():
            index_obj = faiss.read_index(str(idx_path))
            dim = index_obj.d
            metric = "ip" if index_obj.metric_type == faiss.METRIC_INNER_PRODUCT else "l2"
            if cfg is None:
                cfg = IndexConfig(dim=dim, metric=metric)
        if cfg is None:
            # حدس از متادیتا یا پیش‌فرض
            cfg = IndexConfig()
        return cls(cfg=cfg, index_obj=index_obj, metas=metas)

# ---------------------- یکپارچه با EpisodeStore ----------------------

def build_from_episode_store(
    store,
    *,
    key_mode: str = "mean",
    dim: int = 64,
    metric: str = "ip",
    kind: str = "HNSW32",
    normalize: bool = True,
    limit_days: Optional[int] = None,
) -> FaissIndex:
    """
    از اپیزودهای ذخیره‌شده یک ایندکس می‌سازد.
    - store: EpisodeStore (memory/episodic.EpisodeStore)
    - key_mode: "mean" | "z" | "s"
    - limit_days: اگر تعیین شد، فقط همین تعداد روز اخیر را می‌خواند.
    """
    from datetime import datetime, timedelta
    from memory.episodic import make_key_vector, Episode

    cfg = IndexConfig(dim=dim, kind=kind, metric=metric, l2_normalize=normalize and (metric == "ip"))
    idx = FaissIndex(cfg)

    # جمع‌آوری اپیزودها
    episodes: List[Episode] = []
    if limit_days is None:
        # ساده: tail زیاد بگیریم (برای V0)
        episodes = store.tail(n=5000)
    else:
        today = datetime.utcnow().date()
        start = today - timedelta(days=max(0, int(limit_days)-1))
        for ep in store.iter_days(start.isoformat(), today.isoformat()):
            episodes.append(ep)

    if not episodes:
        return idx

    vecs: List[List[float]] = []
    metas: List[Dict[str, Any]] = []
    for ep in episodes:
        key = make_key_vector(ep, mode=key_mode, dim_limit=dim)
        if not key:
            continue
        vecs.append(key)
        metas.append({
            "ts": ep.ts,
            "session_id": ep.session_id,
            "intent": ep.intent,
            "action": ep.action_name,
            "text_in": ep.text_in,
            "text_out": ep.text_out,
        })

    if vecs:
        idx.add(vecs, metas=metas)
    return idx

# ---------------------- اجرای مستقیم (تست دستی) ----------------------

if __name__ == "__main__":
    try:
        from memory.episodic import EpisodeStore
    except Exception as e:
        print("⚠️ برای تست، ابتدا memory/episodic.py را داشته باشید.")
        raise

    store = EpisodeStore()
    # اگر اپیزودی ندارید، یکی ثبت کنیم
    if not store.tail(1):
        store.log(
            session_id="S-TEST",
            text_in="سلام",
            text_out="سلام! خوش اومدی",
            intent="greeting",
            action_kind="skill",
            action_name="reply_greeting",
            r_total=1.0, r_int=0.2, r_ext=0.8,
            u=0.1, conf=0.9,
            s_vec=[0.1]*64, z_vec=[0.12]*64,
        )
        store.log(
            session_id="S-TEST",
            text_in="۲+۲؟",
            text_out="۴",
            intent="compute",
            action_kind="tool",
            action_name="invoke_calc",
            r_total=1.0, r_int=0.3, r_ext=0.7,
            u=0.1, conf=0.9,
            s_vec=[0.2]*64, z_vec=[0.21]*64,
        )

    idx = build_from_episode_store(store, key_mode="mean", dim=64, metric="ip", kind="HNSW32", normalize=True)
    print("ntotal:", idx.ntotal())

    # کوئری: یک بردار ساختگی نزدیک به «سلام»
    q = np.array([0.12]*64, dtype=np.float32)
    q /= (np.linalg.norm(q) or 1.0)
    D, I, M = idx.search(q.tolist(), k=3)
    print("search D:", D)
    print("search I:", I)
    print("search metas:", M[:2])

    # ذخیره
    idx.save("data/index/faiss")
    # بارگذاری
    idx2 = FaissIndex.load("data/index/faiss")
    print("loaded ntotal:", idx2.ntotal())
