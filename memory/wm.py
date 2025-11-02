# -*- coding: utf-8 -*-
"""
NOEMA • memory/wm.py — Minimal Working Memory (V0)

A small ring buffer to keep the last few interaction steps:
  - latent/state vectors (z, s)
  - chosen action and short outcome text
  - reward signals
  - optional original user input for quick context

This is short-term only (NOT persistent). For persistent episodic storage and
semantic search, use memory/episodic.py and memory/index_faiss.py.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
import time


# ---------------------------- Data ----------------------------

@dataclass
class WMItem:
    ts: float
    z: List[float]                       # latent vector
    s: List[float]                       # state vector
    action: Dict[str, Any]               # {"kind":.., "name":.., "args":{...}}
    outcome_text: Optional[str] = None
    reward_total: float = 0.0
    reward_int: float = 0.0
    reward_ext: float = 0.0
    text_in: Optional[str] = None        # raw user input (optional)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------- Working Memory ----------------------------

class WorkingMemory:
    """A small ring buffer that keeps the last N items (default: 16)."""

    def __init__(self, maxlen: int = 16):
        self.maxlen: int = int(maxlen)
        self._buf: Deque[WMItem] = deque(maxlen=self.maxlen)

    # ---- write ----
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
        Append a new record into the buffer.

        Args:
          z, s: latent/state vectors (lists of floats)
          action: {"kind","name","args"}
          outcome: e.g. {"text_out": "..."} (optional)
          reward:  {"r_total","r_int","r_ext"} (optional)
          text_in: optional raw user input
        """
        ts = float(ts if ts is not None else time.time())
        out_text = None
        if isinstance(outcome, dict):
            out_text = outcome.get("text_out") or outcome.get("text") or None
        r_total = float((reward or {}).get("r_total", 0.0))
        r_int   = float((reward or {}).get("r_int", 0.0))
        r_ext   = float((reward or {}).get("r_ext", 0.0))

        item = WMItem(
            ts=ts,
            z=list(z),
            s=list(s),
            action=dict(action or {}),
            outcome_text=out_text,
            reward_total=r_total,
            reward_int=r_int,
            reward_ext=r_ext,
            text_in=text_in,
        )
        self._buf.append(item)
        return item

    # ---- read ----
    def recent(self, k: int = 8) -> List[WMItem]:
        """Return the last k items (oldest → newest)."""
        k = max(0, int(k))
        if k == 0 or not self._buf:
            return []
        return list(self._buf)[-k:]

    def z_hist(self, k: int = 8) -> List[List[float]]:
        """Return the last k latent vectors (oldest → newest)."""
        return [it.z for it in self.recent(k)]

    def state_hist(self, k: int = 8) -> List[List[float]]:
        """Return the last k state vectors (oldest → newest)."""
        return [it.s for it in self.recent(k)]

    def context(self, k: int = 4) -> List[Tuple[str, str]]:
        """
        Return recent (input, output) pairs where both are present.
        """
        pairs: List[Tuple[str, str]] = []
        for it in self.recent(k=max(k, 0)):
            if it.text_in and it.outcome_text:
                pairs.append((it.text_in, it.outcome_text))
        return pairs

    # ---- utils ----
    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def mean_z(self, k: int = 8) -> List[float]:
        """
        Average the last k latent vectors; returns [] if empty.
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


# ---------------------------- quick test ----------------------------
if __name__ == "__main__":
    wm = WorkingMemory(maxlen=4)
    for i in range(6):
        z = [0.1 + 0.01 * i] * 4
        s = [0.2 + 0.01 * i] * 4
        a = {"kind": "tool", "name": "invoke_calc", "args": {"expr": f"{i}+{i}"}}
        out = {"text_out": f"{i+i}"}
        r = {"r_total": 0.5, "r_int": 0.3, "r_ext": 0.2}
        wm.push(z, s, a, outcome=out, reward=r, text_in=f"input {i}")
    print("len:", len(wm))
    print("z_hist last2:", wm.z_hist(2))
    print("state mean_z:", wm.mean_z(4))
    print("context:", wm.context(3))
