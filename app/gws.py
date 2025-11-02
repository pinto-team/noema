# -*- coding: utf-8 -*-
"""
NOEMA • app/gws.py — Minimal Global Workspace (V0)

Purpose:
  A lightweight coordination layer for attention, per-step budget, micro-sleep
  scheduling, and ingestion of teacher events. Pure-stdlib, can be plugged into
  main.py later.

Key features:
  - Salience estimation from (intent, text)
  - Mode selection: "normal" | "clarify-first" | "budget-tight" | "sleep-soon"
  - Per-step compute budget heuristic
  - Teacher event ingestion from JSONL with offset tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time

# Default paths for teacher events and offset
DEFAULT_EVENTS_PATH = Path("logs/teacher_events.jsonl")
DEFAULT_OFFSET_PATH = Path("logs/.teacher_events.offset")


@dataclass
class Focus:
    """What to focus on for this step."""
    salience: float                     # 0..1
    mode: str                           # "normal" | "clarify-first" | "budget-tight" | "sleep-soon"
    intent: str                         # current plan intent (for other blocks)
    budget_ms: int                      # approximate per-step budget
    notes: Dict[str, Any] = field(default_factory=dict)


class GlobalWorkspace:
    """
    Minimal GWS:
      - salience estimation from intent/text
      - simple mode selection rules (clarify/budget-tight/sleep)
      - per-step budget estimation
      - ingest teacher events (JSONL) with offset retention
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
            "base_budget_ms": 400,        # default per-step budget
            "extra_for_compute_ms": 300,  # if intent == "compute"
            "sleep_every_steps": 25,      # micro-nap cadence
            "sleep_min_interval_s": 60,   # min seconds between naps
            "salience_unknown": 0.85,
            "salience_compute": 0.65,
            "salience_greeting": 0.25,
            "salience_question_bonus": 0.10,
            "u_hi": 0.5,                  # uncertainty threshold for clarify-first
            "risk_hi": 0.05,              # if risk above → budget-tight
            "novelty_bonus": 0.15,        # bonus when intent changes vs last turn
            "intent_hist_keep": 5,        # how many past intents to retain
        }
        if config:
            self.cfg.update(config)

        self._turn = 0
        self._last_sleep_ts = 0.0
        self._intent_hist: List[str] = []
        self._last_offset = self._load_offset()
        self._events_buffer: List[Dict[str, Any]] = []  # latest ingested events

    # ---------------- Internal helpers ----------------

    def _load_offset(self) -> int:
        try:
            if self.offset_path.exists():
                return int(self.offset_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            pass
        return 0

    def _save_offset(self, n: int) -> None:
        try:
            self.offset_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.offset_path.with_suffix(self.offset_path.suffix + ".tmp")
            tmp.write_text(str(n), encoding="utf-8")
            tmp.replace(self.offset_path)
        except Exception:
            # best-effort; ignore persistence issues
            pass

    # ---------------- Salience ----------------

    def _salience_of(self, text: str, intent: str) -> float:
        s = 0.5
        if intent == "unknown":
            s = float(self.cfg["salience_unknown"])
        elif intent == "compute":
            s = float(self.cfg["salience_compute"])
        elif intent == "greeting":
            s = float(self.cfg["salience_greeting"])

        # Slightly boost questions (both ASCII '?' and Arabic '؟')
        if ("?" in text) or ("\u061F" in text):
            s = min(1.0, s + float(self.cfg["salience_question_bonus"]))

        # Novelty vs the last intent
        novelty = float(self.cfg["novelty_bonus"]) if (not self._intent_hist or intent != self._intent_hist[-1]) else 0.0
        s = max(0.0, min(1.0, s + novelty))
        return s

    # ---------------- Mode & budget ----------------

    def _decide_mode_and_budget(self, intent: str, u_mean: float, risk: float) -> Tuple[str, int]:
        # High uncertainty → clarify-first with a smaller budget
        if u_mean >= float(self.cfg["u_hi"]):
            return "clarify-first", int(self.cfg["base_budget_ms"] * 0.6)

        # Elevated risk → tighten budget
        if risk > float(self.cfg["risk_hi"]):
            return "budget-tight", int(self.cfg["base_budget_ms"] * 0.5)

        # Compute intent → allocate extra budget
        budget = int(self.cfg["base_budget_ms"])
        if intent == "compute":
            budget += int(self.cfg["extra_for_compute_ms"])

        # If a micro-nap is due soon, surface that
        if self._sleep_due(soft_check=True):
            return "sleep-soon", budget

        return "normal", budget

    def _sleep_due(self, soft_check: bool = False) -> bool:
        # Conditions: step cadence + time since last sleep
        steps_due = (self._turn % int(self.cfg["sleep_every_steps"]) == 0 and self._turn != 0)
        time_ok = (time.time() - self._last_sleep_ts) >= float(self.cfg["sleep_min_interval_s"])
        return (steps_due and time_ok) if soft_check else (time_ok and steps_due)

    # ---------------- Public API ----------------

    def tick(
        self,
        text: str,
        intent: str,
        self_state: Optional[Dict[str, Any]] = None,
        model_signals: Optional[Dict[str, float]] = None,
    ) -> Focus:
        """
        Advance one step and return a Focus suggestion.

        Args:
            text: raw user text for salience heuristics
            intent: current plan intent (e.g., "compute", "greeting", "smalltalk", "unknown")
            self_state: optional agent state (e.g., {"energy":..., "fatigue":...})
            model_signals: optional model metrics, e.g., {"u_mean":..., "risk":...}

        Returns:
            Focus dataclass bundle.
        """
        self._turn += 1
        self._intent_hist.append(intent)
        self._intent_hist = self._intent_hist[-int(self.cfg["intent_hist_keep"]):]

        u_mean = float((model_signals or {}).get("u_mean", 0.2))
        risk = float((model_signals or {}).get("risk", 0.0))

        sal = self._salience_of(text or "", intent or "unknown")
        mode, budget = self._decide_mode_and_budget(intent or "unknown", u_mean, risk)

        notes: Dict[str, Any] = {
            "turn": self._turn,
            "u_mean": u_mean,
            "risk": risk,
            "sleep_due": self._sleep_due(soft_check=True),
        }
        return Focus(salience=sal, mode=mode, intent=intent, budget_ms=budget, notes=notes)

    def should_sleep_now(self) -> bool:
        """Final check for main: whether to start a nap now."""
        return self._sleep_due(soft_check=False)

    def mark_slept(self) -> None:
        """Call after a nap to reset the internal timer."""
        self._last_sleep_ts = time.time()

    # ---------------- Teacher events ingestion ----------------

    def read_new_teacher_events(self) -> List[Dict[str, Any]]:
        """
        Read JSONL from the last offset and return only new records.
        Handles file truncation/rotation by resetting the offset.
        """
        if not self.events_path.exists():
            self._events_buffer = []
            return []

        size = self.events_path.stat().st_size
        # If file shrank (rotation/truncation), reset offset
        if self._last_offset > size:
            self._last_offset = 0

        new_events: List[Dict[str, Any]] = []
        with self.events_path.open("r", encoding="utf-8") as f:
            f.seek(self._last_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    if isinstance(ev, dict):
                        new_events.append(ev)
                except Exception:
                    continue
            self._last_offset = f.tell()

        self._save_offset(self._last_offset)
        self._events_buffer = new_events
        return new_events

    def summarize_events(self, events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Quick summary: counts by type, last RULE and last TEST.
        """
        events = events if events is not None else self._events_buffer
        counts: Dict[str, int] = {}
        last_rule, last_test = None, None
        for ev in events:
            t = str(ev.get("type", "UNKNOWN")).upper()
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


# ---------------- Demo usage ----------------
if __name__ == "__main__":
    gws = GlobalWorkspace()
    samples = [
        ("hello!", "greeting", {"u_mean": 0.1, "risk": 0.0}),
        ("2+2?", "compute", {"u_mean": 0.2, "risk": 0.0}),
        ("fix the list", "unknown", {"u_mean": 0.7, "risk": 0.0}),
    ]
    for text, intent, sig in samples:
        f = gws.tick(text, intent, model_signals=sig)
        print(f"[{intent}] salience={f.salience:.2f} mode={f.mode} budget={f.budget_ms}ms notes={f.notes}")

    evs = gws.read_new_teacher_events()
    summary = gws.summarize_events(evs)
    print("Teacher events summary:", json.dumps(summary, ensure_ascii=False, indent=2))
