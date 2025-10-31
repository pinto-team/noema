# -*- coding: utf-8 -*-
"""
NOEMA • app/gws.py — فضای کار جهانی (Global Workspace) مینیمال برای V0
- هدف: یک لایه‌ی سبک برای هماهنگی توجه، بودجه، خواب و جذب رویدادهای مربی.
- وابستگی خارجی ندارد (فقط stdlib). می‌تواند بعداً به main.py وصل شود.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import time, json, math, os

# مسیر پیش‌فرض رویدادهای مربی
DEFAULT_EVENTS_PATH = Path("logs/teacher_events.jsonl")
DEFAULT_OFFSET_PATH = Path("logs/.teacher_events.offset")

@dataclass
class Focus:
    """خلاصه‌ی وضعیتِ «روی چه چیزی تمرکز کنیم» در این گام."""
    salience: float                  # 0..1
    mode: str                        # "normal" | "clarify-first" | "budget-tight" | "sleep-soon"
    intent: str                      # intent فعلی (برای اطلاع سایر بلوک‌ها)
    budget_ms: int                   # بودجه‌ی تقریبی این گام
    notes: Dict[str, Any] = field(default_factory=dict)

class GlobalWorkspace:
    """
    GWS مینیمال:
      - محاسبه‌ی salience از روی intent/متن
      - تصمیمِ حالت (clarify/budget-tight/sleep) با قواعد ساده
      - بودجه‌ی محاسبه‌ی یک گام
      - ingest رویدادهای مربی (از JSONL) با نگه‌داشت offset
    """

    def __init__(
        self,
        events_path: Path = DEFAULT_EVENTS_PATH,
        offset_path: Path = DEFAULT_OFFSET_PATH,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.events_path = Path(events_path)
        self.offset_path = Path(offset_path)
        self.cfg = {
            "base_budget_ms": 400,        # بودجه‌ی پیش‌فرض برای هر گام
            "extra_for_compute_ms": 300,  # اگر intent=compute
            "sleep_every_steps": 25,      # هر چند گام یک بار micro-nap
            "sleep_min_interval_s": 60,   # حداقل فاصله‌ی زمانی بین دو خواب
            "salience_unknown": 0.85,
            "salience_compute": 0.65,
            "salience_greeting": 0.25,
            "salience_question_bonus": 0.1,
            "u_hi": 0.5,                  # آستانه‌ی عدم‌قطعیت برای clarify-first
            "risk_hi": 0.05,              # اگر ریسک بیشتر از این بود: budget-tight
        }
        if config:
            self.cfg.update(config)

        self._turn = 0
        self._last_sleep_ts = 0.0
        self._intent_hist: List[str] = []
        self._last_offset = self._load_offset()
        self._events_buffer: List[Dict[str, Any]] = []   # آخرین رویدادهای خوانده‌شده

    # ---------- ابزارهای داخلی ----------
    def _load_offset(self) -> int:
        try:
            if self.offset_path.exists():
                return int(self.offset_path.read_text().strip() or "0")
        except Exception:
            pass
        return 0

    def _save_offset(self, n: int) -> None:
        try:
            self.offset_path.parent.mkdir(parents=True, exist_ok=True)
            self.offset_path.write_text(str(n))
        except Exception:
            pass

    # ---------- برآورد سالینس ----------
    def _salience_of(self, text: str, intent: str) -> float:
        s = 0.0
        if intent == "unknown":
            s = self.cfg["salience_unknown"]
        elif intent == "compute":
            s = self.cfg["salience_compute"]
        elif intent == "greeting":
            s = self.cfg["salience_greeting"]
        else:
            s = 0.5
        # اگر علامت سؤال دارد، اندکی مهم‌ترش کن
        if "؟" in text or "?" in text:
            s = min(1.0, s + self.cfg["salience_question_bonus"])
        # کمی novelty نسبت به ۵ intent آخر
        novelty = 0.15 if (not self._intent_hist or intent != self._intent_hist[-1]) else 0.0
        return max(0.0, min(1.0, s + novelty))

    # ---------- تصمیم حالت / بودجه ----------
    def _decide_mode_and_budget(self, intent: str, u_mean: float, risk: float) -> Tuple[str, int]:
        # clarify اگر عدم‌قطعیت بالاست
        if u_mean >= self.cfg["u_hi"]:
            return "clarify-first", int(self.cfg["base_budget_ms"] * 0.6)

        # اگر ریسک بالا رفت، بودجه را کوچک و محافظه‌کارانه کن
        if risk > self.cfg["risk_hi"]:
            return "budget-tight", int(self.cfg["base_budget_ms"] * 0.5)

        # intent محاسبه → کمی بودجه‌ی بیشتر
        budget = self.cfg["base_budget_ms"]
        if intent == "compute":
            budget += self.cfg["extra_for_compute_ms"]

        # اگر وقت خواب نزدیک است، اطلاع بده
        if self._sleep_due(soft_check=True):
            return "sleep-soon", budget

        return "normal", budget

    def _sleep_due(self, soft_check: bool = False) -> bool:
        # شرایط: تعداد گام‌ها + فاصله‌ی زمانی از آخرین خواب
        too_many_steps = (self._turn % self.cfg["sleep_every_steps"] == 0 and self._turn != 0)
        enough_time = (time.time() - self._last_sleep_ts) >= self.cfg["sleep_min_interval_s"]
        return (too_many_steps and enough_time) if soft_check else (enough_time and too_many_steps)

    # ---------- API عمومی ----------
    def tick(
        self,
        text: str,
        intent: str,
        self_state: Optional[Dict[str, Any]] = None,
        model_signals: Optional[Dict[str, float]] = None,
    ) -> Focus:
        """
        ورودی‌ها:
          - text: متن خام کاربر
          - intent: نیت تشخیص‌داده‌شده
          - self_state: مثلاً {"energy":..,"fatigue":..,"ece":..}
          - model_signals: مثلاً {"u_mean":..,"risk":..}
        خروجی: Focus (salience, mode, budget, notes)
        """
        self._turn += 1
        self._intent_hist.append(intent)
        self._intent_hist = self._intent_hist[-5:]

        u_mean = float((model_signals or {}).get("u_mean", 0.2))
        risk   = float((model_signals or {}).get("risk", 0.0))

        sal = self._salience_of(text, intent)
        mode, budget = self._decide_mode_and_budget(intent, u_mean, risk)

        notes: Dict[str, Any] = {
            "turn": self._turn,
            "u_mean": u_mean,
            "risk": risk,
            "sleep_due": self._sleep_due(soft_check=True),
        }
        return Focus(salience=sal, mode=mode, intent=intent, budget_ms=budget, notes=notes)

    def should_sleep_now(self) -> bool:
        """برای main: بررسی نهاییِ شروع خواب."""
        return self._sleep_due(soft_check=False)

    def mark_slept(self) -> None:
        """بعد از اجرای خواب صدا بزنید."""
        self._last_sleep_ts = time.time()

    # ---------- ingest رویدادهای مربی ----------
    def read_new_teacher_events(self) -> List[Dict[str, Any]]:
        """
        JSONL را از offset قبلی می‌خواند و فقط رکوردهای جدید را برمی‌گرداند.
        اگر فایل وجود نداشت، خروجی لیست خالی است.
        """
        if not self.events_path.exists():
            self._events_buffer = []
            return []

        size = self.events_path.stat().st_size
        offset = min(self._last_offset, size)
        new_events: List[Dict[str, Any]] = []

        with self.events_path.open("r", encoding="utf-8") as f:
            f.seek(offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    new_events.append(ev)
                except Exception:
                    continue
            self._last_offset = f.tell()

        self._save_offset(self._last_offset)
        self._events_buffer = new_events
        return new_events

    def summarize_events(self, events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        خلاصه‌ی سریع رویدادهای تازه: شمارش هر نوع، آخرین RULE/TEST.
        """
        events = events if events is not None else self._events_buffer
        counts: Dict[str, int] = {}
        last_rule, last_test = None, None
        for ev in events:
            t = ev.get("type", "UNKNOWN")
            counts[t] = counts.get(t, 0) + 1
            if t == "RULE":
                last_rule = ev
            elif t == "TEST":
                last_test = ev
        return {
            "counts": counts,
            "last_rule": last_rule,
            "last_test": last_test,
            "n_events": len(events),
        }

# ---------- استفاده‌ی نمونه (اجرای مستقیم) ----------
if __name__ == "__main__":
    gws = GlobalWorkspace()
    # شبیه‌سازی چند ورودی
    samples = [
        ("سلام!", "greeting", {"u_mean": 0.1, "risk": 0.0}),
        ("۲+۲؟", "compute", {"u_mean": 0.2, "risk": 0.0}),
        ("لیست را درست کن", "unknown", {"u_mean": 0.7, "risk": 0.0}),
    ]
    for text, intent, sig in samples:
        f = gws.tick(text, intent, model_signals=sig)
        print(f"[{intent}] salience={f.salience:.2f} mode={f.mode} budget={f.budget_ms}ms notes={f.notes}")

    # خواندن رویدادهای مربی (اگر باشد)
    evs = gws.read_new_teacher_events()
    summary = gws.summarize_events(evs)
    print("Teacher events summary:", json.dumps(summary, ensure_ascii=False, indent=2))
