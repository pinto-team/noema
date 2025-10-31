# -*- coding: utf-8 -*-
"""
NOEMA • memory/episodic.py — حافظه‌ی اپیزودیک (V0 سبک و فایل‌محور)
- هدف: ثبت اپیزودهای تعامل (ورودی/خروجی/کنش/پاداش/بردارها) به‌شکل JSONL روزانه،
  با قابلیت مرور، جست‌وجوی ساده، و فشرده‌سازی اختیاری به Parquet.
- ایندکس برداری (ANN) در فایل جداگانه‌ی memory/index_faiss.py انجام می‌شود.

طراحی:
  data/episodes/
    ├─ 2025-10-30.jsonl
    ├─ 2025-10-31.jsonl
    └─ parquet/ (اختیاری؛ خروجی فشرده‌سازی)

API اصلی:
  store = EpisodeStore(root="data/episodes")
  store.log( ... فیلدها ... )                      # افزودن یک رکورد
  for ep in store.iter_days("2025-10-30","2025-10-31"): ...
  tail = store.tail(n=20)                           # آخرین n رکورد
  store.compact_day_to_parquet("2025-10-31")        # (اختیاری) تبدیل یک روز به Parquet
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, date
import json, time, os

try:
    import pandas as pd  # اختیاری، فقط برای فشرده‌سازی Parquet
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

# ----------------------------- ساختار داده -----------------------------

@dataclass
class Episode:
    ts: float
    session_id: str
    text_in: Optional[str] = None
    text_out: Optional[str] = None
    intent: Optional[str] = None
    action_kind: Optional[str] = None
    action_name: Optional[str] = None
    action_args: Dict[str, Any] = field(default_factory=dict)

    r_total: float = 0.0
    r_int: float = 0.0
    r_ext: float = 0.0
    risk: float = 0.0
    energy: float = 0.0

    u: float = 0.0               # uncertainty
    conf: float = 0.0            # confidence

    s_vec: List[float] = field(default_factory=list)
    z_vec: List[float] = field(default_factory=list)

    tests: List[Dict[str, Any]] = field(default_factory=list)
    costs: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

# ----------------------------- ابزار تاریخ/مسیر -----------------------------

def _day_str(ts: float | None = None) -> str:
    dt = datetime.utcfromtimestamp(ts or time.time()).date()
    return dt.isoformat()

def _coerce_day_str(d: str | date | None = None) -> str:
    if d is None:
        return _day_str()
    if isinstance(d, date):
        return d.isoformat()
    return str(d)

# ----------------------------- EpisodeStore -----------------------------

class EpisodeStore:
    """
    ذخیره‌سازی JSONL روزانه با فشرده‌سازی اختیاری.
    Thread-safe کامل نیست، اما برای V0 شخصی کافی است.
    """

    def __init__(self, root: str | Path = "data/episodes"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "parquet").mkdir(parents=True, exist_ok=True)

    # ---------- مسیر فایل روز ----------
    def _file_for_day(self, day: str | None = None) -> Path:
        d = _coerce_day_str(day)
        return self.root / f"{d}.jsonl"

    # ---------- افزودن رکورد ----------
    def log(
        self,
        *,
        ts: Optional[float] = None,
        session_id: str = "S-LOCAL-001",
        text_in: Optional[str] = None,
        text_out: Optional[str] = None,
        intent: Optional[str] = None,
        action_kind: Optional[str] = None,
        action_name: Optional[str] = None,
        action_args: Optional[Dict[str, Any]] = None,
        r_total: float = 0.0,
        r_int: float = 0.0,
        r_ext: float = 0.0,
        risk: float = 0.0,
        energy: float = 0.0,
        u: float = 0.0,
        conf: float = 0.0,
        s_vec: Optional[List[float]] = None,
        z_vec: Optional[List[float]] = None,
        tests: Optional[List[Dict[str, Any]]] = None,
        costs: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        day: Optional[str] = None,
    ) -> Episode:
        rec = Episode(
            ts=float(ts or time.time()),
            session_id=session_id,
            text_in=text_in, text_out=text_out,
            intent=intent,
            action_kind=action_kind, action_name=action_name,
            action_args=dict(action_args or {}),
            r_total=float(r_total), r_int=float(r_int), r_ext=float(r_ext),
            risk=float(risk), energy=float(energy),
            u=float(u), conf=float(conf),
            s_vec=list(s_vec or []), z_vec=list(z_vec or []),
            tests=list(tests or []), costs=dict(costs or {}),
            tags=list(tags or []),
        )
        path = self._file_for_day(day)
        with path.open("a", encoding="utf-8") as f:
            f.write(rec.to_json() + "\n")
        return rec

    # ---------- نگاشت Transition → Episode ----------
    def log_transition(
        self,
        *,
        session_id: str,
        text_in: Optional[str],
        state: Dict[str, Any],
        latent: Dict[str, Any] | List[float],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        reward: Dict[str, Any],
        ts: Optional[float] = None,
        tags: Optional[List[str]] = None,
        day: Optional[str] = None,
    ) -> Episode:
        """
        برای سازگاری با خروجی app/main.py و سایر ماژول‌ها.
        state باید شامل u/conf و بردار s باشد؛ latent شامل z.
        """
        s_vec = state.get("s") if isinstance(state, dict) else state
        z_vec = latent.get("z") if isinstance(latent, dict) else latent

        return self.log(
            ts=ts,
            session_id=session_id,
            text_in=text_in,
            text_out=(outcome or {}).get("text_out"),
            intent=(outcome or {}).get("intent"),
            action_kind=action.get("kind"),
            action_name=action.get("name"),
            action_args=action.get("args", {}),
            r_total=float((reward or {}).get("r_total", 0.0)),
            r_int=float((reward or {}).get("r_int", 0.0)),
            r_ext=float((reward or {}).get("r_ext", 0.0)),
            risk=float((reward or {}).get("risk", 0.0)),
            energy=float((reward or {}).get("energy", 0.0)),
            u=float((state or {}).get("u", 0.0)),
            conf=float((state or {}).get("conf", 0.0)),
            s_vec=list(s_vec or []),
            z_vec=list(z_vec or []),
            tests=list((outcome or {}).get("tests", [])),
            costs=dict((outcome or {}).get("costs", {})),
            tags=list(tags or []),
            day=day,
        )

    # ---------- پیمایش ----------
    def iter_day(self, day: str | date | None = None) -> Iterator[Episode]:
        path = self._file_for_day(day)
        if not path.exists():
            return iter(())
        def _gen():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        yield Episode(**obj)
                    except Exception:
                        continue
        return _gen()

    def iter_days(self, start_day: str | date, end_day: str | date) -> Iterator[Episode]:
        """شامل هر دو انتها."""
        start = datetime.fromisoformat(_coerce_day_str(start_day)).date()
        end   = datetime.fromisoformat(_coerce_day_str(end_day)).date()
        cur = start
        while cur <= end:
            for ep in self.iter_day(cur.isoformat()):
                yield ep
            cur = date.fromordinal(cur.toordinal() + 1)

    # ---------- Tail ----------
    def tail(self, n: int = 20) -> List[Episode]:
        """
        آخرین n رکورد بین چند فایل روزانه.
        ساده: از امروز به عقب می‌خوانیم تا n پر شود.
        """
        out: List[Episode] = []
        today = datetime.utcnow().date()
        cur = today
        while len(out) < n:
            p = self._file_for_day(cur.isoformat())
            if p.exists():
                # ساده: کل فایل را بخوان، ولی اگر بزرگ شد، بهینه‌سازی کن
                lines = p.read_text(encoding="utf-8").splitlines()
                for line in reversed(lines):
                    if len(out) >= n:
                        break
                    try:
                        obj = json.loads(line)
                        out.append(Episode(**obj))
                    except Exception:
                        continue
            # روز قبل
            cur = date.fromordinal(cur.toordinal() - 1)
            # قطع امن پس از 365 روز
            if (today.toordinal() - cur.toordinal()) > 365:
                break
        return list(reversed(out))

    # ---------- فشرده‌سازی به Parquet ----------
    def compact_day_to_parquet(self, day: str | date) -> Optional[Path]:
        """
        فایل JSONL آن روز را به Parquet تبدیل می‌کند. نیازمند pandas است.
        خروجی: data/episodes/parquet/{day}.parquet
        """
        if not _HAS_PANDAS:
            print("⚠️ pandas در دسترس نیست؛ فشرده‌سازی انجام نشد.")
            return None
        d = _coerce_day_str(day)
        src = self._file_for_day(d)
        if not src.exists():
            print(f"⚠️ فایل روز {d} یافت نشد.")
            return None
        rows: List[Dict[str, Any]] = []
        with src.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    continue
        if not rows:
            print("⚠️ رکوردی برای فشرده‌سازی نیست.")
            return None
        df = pd.DataFrame(rows)
        out = self.root / "parquet" / f"{d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"✅ parquet written: {out}")
        return out

# ----------------------------- کلید برداری برای ANN -----------------------------

def make_key_vector(ep: Episode, mode: str = "mean", dim_limit: Optional[int] = None) -> List[float]:
    """
    کلید برداری برای ایندکس معنایی:
      - mode="mean": میانگین z و s (اگر باشد)
      - mode="z": فقط z
      - mode="s": فقط s
    dim_limit: اگر تعیین شود، بردار به آن طول بریده/پد می‌شود (برای سازگاری با ایندکس).
    """
    z = ep.z_vec or []
    s = ep.s_vec or []
    key: List[float]
    if mode == "z":
        key = list(z)
    elif mode == "s":
        key = list(s)
    else:
        if z and s and len(z) == len(s):
            key = [(zi + si) * 0.5 for zi, si in zip(z, s)]
        elif z:
            key = list(z)
        else:
            key = list(s)

    if dim_limit is not None:
        D = int(dim_limit)
        if len(key) > D:
            key = key[:D]
        elif len(key) < D:
            key = key + [0.0] * (D - len(key))
    return key

# ----------------------------- تست سریع -----------------------------

if __name__ == "__main__":
    store = EpisodeStore()
    # یک نمونه لاگ
    ep = store.log(
        session_id="S-TEST",
        text_in="۲+۲؟",
        text_out="۴",
        intent="compute",
        action_kind="tool",
        action_name="invoke_calc",
        action_args={"expr": "2+2"},
        r_total=1.0, r_int=0.2, r_ext=0.8,
        risk=0.0, energy=0.01,
        u=0.1, conf=0.9,
        s_vec=[0.1]*8, z_vec=[0.12]*8,
        tests=[{"name":"alt_eval","pass":True}],
        costs={"latency_ms": 5, "compute": 2},
        tags=["demo"],
    )
    print("wrote:", ep)

    print("tail(3):", [e.text_in for e in store.tail(3)])

    # فشرده‌سازی روز جاری (اگر pandas نصب باشد)
    store.compact_day_to_parquet(_day_str())
