# -*- coding: utf-8 -*-
"""
NOEMA • sleep/offline.py — Offline consolidation for learned rules and DEMO memory.

This module consumes teacher events and (optionally) environment episodes to:
  - Write learned intent rules → config/learned_rules.yaml
  - Write demo memory           → data/demo_memory.jsonl
  - Optionally build TF-IDF index for demos (data/demo_index.npz, data/demo_vocab.json)
  - Optionally train an intent classifier from LABEL events (models/intent_clf.joblib)

It is language-agnostic; content is stored as-is (UTF-8).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import os

# ------------------------------- I/O utils -------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[Dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _write_yaml_rules(rules: List[Dict[str, Any]]) -> Path:
    config_dir = Path("config")
    config_dir.mkdir(parents=True, exist_ok=True)
    target = config_dir / "learned_rules.yaml"
    with target.open("w", encoding="utf-8") as fh:
        fh.write("rules:\n")
        for rule in rules:
            intent = (rule.get("intent") or "").strip()
            pats = list(rule.get("patterns") or [])
            pats = [str(p) for p in pats if str(p).strip()]
            if not intent or not pats:
                continue
            fh.write(f"  - intent: {intent}\n")
            fh.write("    patterns: [")
            fh.write(", ".join(json.dumps(p, ensure_ascii=False) for p in pats))
            fh.write("]\n")
    return target


def _write_demo_memory(demos: List[Dict[str, Any]]) -> Path:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "demo_memory.jsonl"
    text = "\n".join(json.dumps(obj, ensure_ascii=False) for obj in demos)
    target.write_text(text, encoding="utf-8")
    return target


# --------------------------- Optional artifacts ---------------------------

def _maybe_build_tfidf(demos: List[Dict[str, Any]]) -> Optional[Tuple[Path, Path]]:
    """Build TF-IDF index for demo inputs (optional; silently skips if deps missing)."""
    if not demos:
        return None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        import numpy as np  # type: ignore

        inputs = [str(d.get("input", "")) for d in demos if str(d.get("input", "")).strip()]
        if not inputs:
            return None

        vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
        matrix = vectorizer.fit_transform(inputs)

        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)

        idx_path = data_dir / "demo_index.npz"
        np.savez(
            idx_path,
            data=matrix.data,
            indices=matrix.indices,
            indptr=matrix.indptr,
            shape=matrix.shape,
        )
        voc_path = data_dir / "demo_vocab.json"
        voc_path.write_text(
            json.dumps(vectorizer.vocabulary_, ensure_ascii=False),
            encoding="utf-8",
        )
        return idx_path, voc_path
    except Exception:
        return None


def _maybe_train_intent_clf(labels: List[Dict[str, Any]]) -> Optional[Path]:
    """Train an optional intent classifier from LABEL events (skips if deps missing)."""
    if not labels:
        return None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.svm import LinearSVC  # type: ignore
        import joblib  # type: ignore

        texts = [str(item.get("input", "")) for item in labels]
        intents = [str(item.get("intent", "")) for item in labels]
        texts = [t for t in texts if t.strip()]
        intents = [y for y in intents if y.strip()]
        if not texts or not intents or len(texts) != len(intents):
            return None

        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 3))),
                ("clf", LinearSVC()),
            ]
        )
        pipe.fit(texts, intents)

        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / "intent_clf.joblib"
        joblib.dump(pipe, out_path)
        return out_path
    except Exception:
        return None


# ------------------------------ Event parsing ------------------------------

def _ensure_rule_dicts(spec: Any) -> List[Dict[str, Any]]:
    """
    Normalize RULE payload "spec" into a list of {intent, patterns}.
    Accepted forms:
      - {"rules": [ {intent, patterns}, ... ]}
      - [ {intent, patterns}, ... ]
      - {intent, patterns}
      - {"patterns_map": {"intentA": [...], "intentB": [...]}}

    Fix: handle nested {"spec": {...}} or {"payload": {"spec": {...}}} structures
    coming from teacher panel.
    """
    out: List[Dict[str, Any]] = []
    if spec is None:
        return out

    # --- FIX: handle nested {"spec": {...}} ---
    # Some RULE payloads may wrap the actual spec one level deeper.
    # This ensures we always unwrap until reaching a usable dict.
    depth_guard = 0
    while isinstance(spec, dict) and "spec" in spec and isinstance(spec["spec"], dict):
        spec = spec["spec"]
        depth_guard += 1
        if depth_guard > 3:  # avoid infinite recursion
            break

    # {"rules": [...]}
    if isinstance(spec, dict) and isinstance(spec.get("rules"), list):
        for itm in spec["rules"]:
            if isinstance(itm, dict):
                out.append({"intent": itm.get("intent"), "patterns": itm.get("patterns")})
        return out

    # {"patterns_map": {...}}
    if isinstance(spec, dict) and isinstance(spec.get("patterns_map"), dict):
        for k, v in spec["patterns_map"].items():
            out.append({"intent": k, "patterns": list(v or [])})
        return out

    # [ {...}, {...} ]
    if isinstance(spec, list):
        for itm in spec:
            if isinstance(itm, dict):
                out.append({"intent": itm.get("intent"), "patterns": itm.get("patterns")})
        return out

    # {intent, patterns}
    if isinstance(spec, dict) and {"intent", "patterns"} <= set(spec.keys()):
        out.append({"intent": spec.get("intent"), "patterns": spec.get("patterns")})
        return out

    return out



def _collect_events(events: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rules: List[Dict[str, Any]] = []
    demos: List[Dict[str, Any]] = []
    labels: List[Dict[str, Any]] = []

    for event in events:
        etype = str(event.get("type", "")).upper()
        payload = event.get("payload") or {}

        if etype == "RULE":
            spec = payload.get("spec")
            rule_dicts = _ensure_rule_dicts(spec)
            for r in rule_dicts:
                intent = (r.get("intent") or "").strip()
                patterns = [str(p).strip() for p in list(r.get("patterns") or []) if str(p).strip()]
                if intent and patterns:
                    rules.append({"intent": intent, "patterns": patterns})

        elif etype == "CLARIFY":
            intent = str(payload.get("intent") or "").strip()
            patterns = [str(p).strip() for p in list(payload.get("patterns") or []) if str(p).strip()]
            if intent and patterns:
                rules.append({"intent": intent, "patterns": patterns})

        elif etype == "DEMO":
            demos.append(
                {
                    "input": str(payload.get("input", "")),
                    "output": str(payload.get("output", "")),
                }
            )

        elif etype == "LABEL":
            labels.append(
                {
                    "input": str(payload.get("input", "")),
                    "intent": str(payload.get("intent", "")),
                }
            )

    return rules, demos, labels


# ----------------------------- Episode ingestion -----------------------------

def _iter_episode_demos(root: Path) -> Iterable[Dict[str, str]]:
    """
    Convert environment episode logs (JSONL) into demo pairs {input, output}.
    Expected fields:
      - rec["text_in"]
      - rec["outcome"]["text_out"]
    """
    if not root.exists():
        return
    for log_file in root.rglob("*.jsonl"):
        for rec in _read_jsonl(log_file):
            text_in = str(rec.get("text_in", "")).strip()
            out = rec.get("outcome") or {}
            text_out = str((out.get("text_out") or "")).strip()
            if text_in and text_out:
                yield {"input": text_in, "output": text_out}


def _dedupe_pairs(items: List[Dict[str, Any]], key: Tuple[str, str] = ("input", "output")) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    k1, k2 = key
    for it in items:
        i = str(it.get(k1, "")).strip()
        o = str(it.get(k2, "")).strip()
        if not i and not o:
            continue
        sig = (i, o)
        if sig in seen:
            continue
        seen.add(sig)
        out.append({k1: i, k2: o})
    return out


def _dedupe_rules(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in items:
        intent = (r.get("intent") or "").strip()
        pats = [str(p).strip() for p in list(r.get("patterns") or []) if str(p).strip()]
        if not intent or not pats:
            continue
        sig = (intent, tuple(sorted(set(pats))))
        if sig in seen:
            continue
        seen.add(sig)
        out.append({"intent": intent, "patterns": list(sig[1])})
    return out


# ----------------------------- Public API / Runner -----------------------------

@dataclass
class SleepCfg:
    """Configuration for the offline consolidation."""
    events_path: str = "logs/teacher_events.jsonl"
    episodes_root: str = "data/episodes"
    write_rules: bool = True
    write_demos: bool = True
    build_tfidf: bool = True
    train_intent_clf: bool = True


def run_sleep_cycle(cfg: SleepCfg, *, verbose: bool = False) -> Dict[str, Any]:
    """Run the offline consolidation according to the given config."""
    events = _read_jsonl(Path(cfg.events_path))
    rules, demos, labels = _collect_events(events)

    # Episodes → demos (fallback/augmentation)
    epi_root = Path(cfg.episodes_root)
    if epi_root.exists():
        demos.extend(list(_iter_episode_demos(epi_root)))

    # Clean & dedupe
    rules = _dedupe_rules(rules)
    demos = _dedupe_pairs(demos)

    out_paths: Dict[str, Optional[str]] = {"rules": None, "demo_memory": None, "tfidf_index": None, "tfidf_vocab": None, "intent_clf": None}

    if cfg.write_rules:
        p = _write_yaml_rules(rules)
        out_paths["rules"] = str(p)

    if cfg.write_demos:
        p = _write_demo_memory(demos)
        out_paths["demo_memory"] = str(p)

    if cfg.build_tfidf:
        tfidf = _maybe_build_tfidf(demos)
        if tfidf:
            out_paths["tfidf_index"], out_paths["tfidf_vocab"] = map(str, tfidf)

    if cfg.train_intent_clf:
        clf_path = _maybe_train_intent_clf(labels)
        if clf_path:
            out_paths["intent_clf"] = str(clf_path)

    report = {
        "counts": {
            "events_total": len(events),
            "rules": len(rules),
            "demos": len(demos),
            "labels": len(labels),
        },
        "outputs": out_paths,
    }

    if verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    return report


def run_once(**kwargs) -> Dict[str, Any]:
    """Sugar wrapper: build SleepCfg from kwargs and run one consolidation pass."""
    cfg = SleepCfg(**kwargs)
    return run_sleep_cycle(cfg, verbose=True)


# ----------------------------- CLI -----------------------------

def main() -> None:
    """CLI entrypoint: run with defaults and print a short OK message."""
    run_once()
    print("OK: learned_rules.yaml + demo_memory.jsonl (optional: tfidf, intent_clf)")


if __name__ == "__main__":
    main()
