# -*- coding: utf-8 -*-
"""
NOEMA • memory/wm.py — حافظه‌ی کاری (Working Memory) مینیمال V0
- بافر حلقه‌ای سبک برای نگه‌داری چند گام اخیر تعامل (z, s, action, outcome, reward).
- بدون وابستگی خارجی؛ فقط stdlib. برای استفاده مستقیم در app/main یا کنترل/پلنر.

API اصلی:
    wm = WorkingMemory(maxlen=16)
    wm.push(z, s, action, outcome, reward, text_in=None)
    wm.recent(k=8) -> List[WMItem]
    wm.z_hist(k=8) -> List[List[float]]
    wm.state_hist(k=8) -> List[List[float]]
    wm.context(k=4) -> List[Tuple[str,str]]     # (input, output)
    wm.clear()

توجه:
- این ماژول «اپیزودیک/دائمی» نیست؛ فقط حافظه‌ی کوتاه‌مدت است.
- برای ذخیره‌ی پایدار و جست‌وجوی معنایی از memory/episodic و index_faiss استفاده کنید.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
import time
import math

# ---------------------------- داده‌های کمینه ----------------------------

@dataclass
class WMItem:
    ts: float
    z: List[float]                          # بردار نهان (Latent)
    s: List[float]                          # بردار وضعیت (State)
    action: Dict[str, Any]                  # {"kind":.., "name":.., "args":{...}}
    outcome_text: Optional[str] = None
    reward_total: float = 0.0
    reward_int: float = 0.0
    reward_ext: float = 0.0
    text_in: Optional[str] = None           # ورودی خام (اختیاری برای context)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ---------------------------- WM ----------------------------

class WorkingMemory:
    """
    بافر حلقه‌ای سبک:
      - آخرین N گام را نگه می‌دارد (پیش‌فرض 16).
      - فقط برای استنتاج و زمینه‌ی کوتاه‌مدت استفاده می‌شود.
    """

    def __init__(self, maxlen: int = 16):
        self.maxlen: int = int(maxlen)
        self._buf: Deque[WMItem] = deque(maxlen=self.maxlen)

    # ---- نوشتن ----
    def push(
        self,
        z: List[float],
        s: List[float],
        action: Dict[str, Any],
        outcome: Dict[str, Any] | None = None,
        reward: Dict[str, Any] | None = None,
        text_in: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> WMItem:
        """
        یک رکورد به حافظه اضافه می‌کند.
        پارامترها:
          - z, s: بردارهای نهان/وضعیت (لیست float)
          - action: {"kind","name","args"}
          - outcome: {"text_out", ...}  (اختیاری)
          - reward: {"r_total","r_int","r_ext"} (اختیاری)
          - text_in: متن خام کاربر (اختیاری)
        """
        ts = ts if ts is not None else time.time()
        out_text = None
        if isinstance(outcome, dict):
            out_text = outcome.get("text_out") or outcome.get("text") or None
        r_total = float((reward or {}).get("r_total", 0.0))
        r_int   = float((reward or {}).get("r_int", 0.0))
        r_ext   = float((reward or {}).get("r_ext", 0.0))

        item = WMItem(
            ts=ts, z=list(z), s=list(s),
            action=dict(action or {}),
            outcome_text=out_text,
            reward_total=r_total, reward_int=r_int, reward_ext=r_ext,
            text_in=text_in
        )
        self._buf.append(item)
        return item

    # ---- خواندن ----
    def recent(self, k: int = 8) -> List[WMItem]:
        """آخرین k آیتم (قدیمی→جدید)."""
        k = max(0, int(k))
        if k == 0 or not self._buf:
            return []
        return list(self._buf)[-k:]

    def z_hist(self, k: int = 8) -> List[List[float]]:
        """آخرین k بردار z (قدیمی→جدید)."""
        return [it.z for it in self.recent(k)]

    def state_hist(self, k: int = 8) -> List[List[float]]:
        """آخرین k بردار s (قدیمی→جدید)."""
        return [it.s for it in self.recent(k)]

    def context(self, k: int = 4) -> List[Tuple[str, str]]:
        """
        جفت‌های (input, output) اخیر را برمی‌گرداند.
        فقط آیتم‌هایی که هر دو بخش را دارند انتخاب می‌شود.
        """
        pairs: List[Tuple[str, str]] = []
        for it in self.recent(k= max(k, 0)):
            if it.text_in and it.outcome_text:
                pairs.append((it.text_in, it.outcome_text))
        return pairs

    # ---- ابزارهای کمکی ----
    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def mean_z(self, k: int = 8) -> List[float]:
        """
        میانگین z در پنجره‌ی اخیر؛ اگر بافر خالی باشد، لیست خالی.
        """
        zs = self.z_hist(k)
        if not zs:
            return []
        D = len(zs[0])
        acc = [0.0] * D
        for v in zs:
            for i in range(D):
                acc[i] += v[i]
        n = float(len(zs))
        return [x / n for x in acc]

    def last_action(self) -> Optional[Dict[str, Any]]:
        return self._buf[-1].action if self._buf else None

    def last_output(self) -> Optional[str]:
        return self._buf[-1].outcome_text if self._buf else None

# ---------------------------- تست سریع ----------------------------

if __name__ == "__main__":
    wm = WorkingMemory(maxlen=4)
    for i in range(6):
        z = [0.1 + 0.01 * i] * 4
        s = [0.2 + 0.01 * i] * 4
        a = {"kind": "tool", "name": "invoke_calc", "args": {"expr": f"{i}+{i}"}}
        out = {"text_out": f"{i+i}"}
        r = {"r_total": 0.5, "r_int": 0.3, "r_ext": 0.2}
        wm.push(z, s, a, outcome=out, reward=r, text_in=f"ورودی {i}")
    print("len:", len(wm))
    print("z_hist last2:", wm.z_hist(2))
    print("state mean:", wm.mean_z(4))
    print("context:", wm.context(3))
