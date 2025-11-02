# -*- coding: utf-8 -*-
"""
NOEMA • memory/episodic.py — File-based Episodic Memory (V0)

Goal:
  Append-only daily JSONL logs of interaction episodes (I/O, action, rewards, vectors),
  with simple traversal utilities and optional Parquet compaction.

Layout:
  data/episodes/
    ├─ YYYY-MM-DD.jsonl
    └─ parquet/  (optional)

Vector indexing is handled separately in memory/index_faiss.py.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, date
import json
import time

try:
    import pandas as pd  # optional, only for Parquet compaction
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


# ----------------------------- Data -----------------------------

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


# ----------------------------- Date/Path helpers -----------------------------

def _day_str(ts: float | None = None) -> str:
    dt = datetime.utcfromtimestamp(ts or time.time()).date()
    return dt.isoformat()

def _coerce_day_str(d: str | date | None = None) -> str:
    if d is None:
        return _day_str()
    if isinstance(d, date):
        return d.isoformat()
    return str(d)


# ----------------------------- Store -----------------------------

class EpisodeStore:
    """
    Daily JSONL store with optional Parquet compaction.
    This is not fully thread-safe but good enough for V0.
    """

    def __init__(self, root: str | Path = "data/episodes"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "parquet").mkdir(parents=True, exist_ok=True)

    # ---------- per-day file ----------
    def _file_for_day(self, day: str | None = None) -> Path:
        d = _coerce_day_str(day)
        return self.root / f"{d}.jsonl"

    # ---------- append ----------
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

    # ---------- Transition → Episode ----------
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
        Compatibility shim for app/main.py style transitions.
        `state` should include {u, conf, s}; `latent` should include {z}.
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

    # ---------- iterate ----------
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
        """Inclusive of both endpoints."""
        start = datetime.fromisoformat(_coerce_day_str(start_day)).date()
        end   = datetime.fromisoformat(_coerce_day_str(end_day)).date()
        cur = start
        while cur <= end:
            for ep in self.iter_day(cur.isoformat()):
                yield ep
            cur = date.fromordinal(cur.toordinal() + 1)

    # ---------- tail ----------
    def tail(self, n: int = 20) -> List[Episode]:
        """
        Return the last n records across recent day-files.
        Simple implementation: walk backwards day-by-day until n is collected (limit 365 days).
        """
        out: List[Episode] = []
        today = datetime.utcnow().date()
        cur = today
        while len(out) < n:
            p = self._file_for_day(cur.isoformat())
            if p.exists():
                lines = p.read_text(encoding="utf-8").splitlines()
                for line in reversed(lines):
                    if len(out) >= n:
                        break
                    try:
                        obj = json.loads(line)
                        out.append(Episode(**obj))
                    except Exception:
                        continue
            cur = date.fromordinal(cur.toordinal() - 1)
            if (today.toordinal() - cur.toordinal()) > 365:
                break
        return list(reversed(out))

    # ---------- Parquet compaction ----------
    def compact_day_to_parquet(self, day: str | date) -> Optional[Path]:
        """
        Convert a day's JSONL to Parquet (requires pandas).
        Output: data/episodes/parquet/{day}.parquet
        """
        if not _HAS_PANDAS:
            print("⚠️ pandas not available; skipping compaction.")
            return None
        d = _coerce_day_str(day)
        src = self._file_for_day(d)
        if not src.exists():
            print(f"⚠️ day file not found: {d}")
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
            print("⚠️ nothing to compact.")
            return None
        df = pd.DataFrame(rows)
        out = self.root / "parquet" / f"{d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"✅ parquet written: {out}")
        return out


# ----------------------------- Key vector for ANN -----------------------------

def make_key_vector(ep: Episode, mode: str = "mean", dim_limit: Optional[int] = None) -> List[float]:
    """
    Build a key vector for semantic indexing:
      - mode="mean": average of z and s if both present (same dim), else whichever is present
      - mode="z": only z
      - mode="s": only s

    dim_limit: if set, trim/pad the vector to this length.
    """
    z = ep.z_vec or []
    s = ep.s_vec or []
    if mode == "z":
        key: List[float] = list(z)
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


# ----------------------------- manual test -----------------------------
if __name__ == "__main__":
    store = EpisodeStore()
    ep = store.log(
        session_id="S-TEST",
        text_in="2+2?",
        text_out="4",
        intent="compute",
        action_kind="tool",
        action_name="invoke_calc",
        action_args={"expr": "2+2"},
        r_total=1.0, r_int=0.2, r_ext=0.8,
        risk=0.0, energy=0.01,
        u=0.1, conf=0.9,
        s_vec=[0.1]*8, z_vec=[0.12]*8,
        tests=[{"name": "alt_eval", "pass": True}],
        costs={"latency_ms": 5, "compute": 2},
        tags=["demo"],
    )
    print("wrote:", ep)

    print("tail(3):", [e.text_in for e in store.tail(3)])

    store.compact_day_to_parquet(_day_str())
