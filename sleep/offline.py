# -*- coding: utf-8 -*-
"""NOEMA • sleep/offline.py — آفلاینِ سبک برای یادگیری قوانین و حافظهٔ DEMO."""

from pathlib import Path
import json
import os
from typing import List, Dict, Any


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[Dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _write_yaml_rules(rules: List[Dict[str, Any]]) -> None:
    config_dir = Path("config")
    config_dir.mkdir(parents=True, exist_ok=True)
    target = config_dir / "learned_rules.yaml"
    with target.open("w", encoding="utf-8") as fh:
        fh.write("rules:\n")
        for rule in rules:
            intent = rule.get("intent", "")
            pats = rule.get("patterns", []) or []
            if not intent or not pats:
                continue
            fh.write(f"  - intent: {intent}\n")
            fh.write("    patterns: [")
            fh.write(", ".join(json.dumps(str(p), ensure_ascii=False) for p in pats))
            fh.write("]\n")


def _write_demo_memory(demos: List[Dict[str, Any]]) -> None:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "demo_memory.jsonl"
    text = "\n".join(json.dumps(obj, ensure_ascii=False) for obj in demos)
    target.write_text(text, encoding="utf-8")


def _maybe_build_tfidf(demos: List[Dict[str, Any]]) -> None:
    if not demos:
        return
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        import numpy as np  # type: ignore

        inputs = [str(d.get("input", "")) for d in demos]
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
        matrix = vectorizer.fit_transform(inputs)
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            data_dir / "demo_index.npz",
            data=matrix.data,
            indices=matrix.indices,
            indptr=matrix.indptr,
            shape=matrix.shape,
        )
        (data_dir / "demo_vocab.json").write_text(
            json.dumps(vectorizer.vocabulary_, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        # وابستگی اختیاری است؛ در صورت نبود، به n-gram fallback اکتفا می‌کنیم.
        return


def _maybe_train_intent_clf(labels: List[Dict[str, Any]]) -> None:
    if not labels:
        return
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.svm import LinearSVC  # type: ignore
        import joblib  # type: ignore

        texts = [str(item.get("input", "")) for item in labels]
        intents = [str(item.get("intent", "")) for item in labels]
        if not any(intents):
            return
        pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 3))),
                ("clf", LinearSVC()),
            ]
        )
        pipe.fit(texts, intents)
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, models_dir / "intent_clf.joblib")
    except Exception:
        return


def main() -> None:
    events_path = Path("logs/teacher_events.jsonl")
    episodes_root = Path("data/episodes")

    events = _read_jsonl(events_path)
    rules: List[Dict[str, Any]] = []
    demos: List[Dict[str, Any]] = []
    labels: List[Dict[str, Any]] = []

    for event in events:
        etype = str(event.get("type", "")).upper()
        payload = event.get("payload") or {}
        if etype == "RULE":
            for spec in payload.get("spec", []) or []:
                intent = str((spec or {}).get("intent") or "")
                patterns = list((spec or {}).get("patterns", []) or [])
                if intent and patterns:
                    rules.append({"intent": intent, "patterns": patterns})
        elif etype == "CLARIFY":
            intent = str(payload.get("intent") or "")
            patterns = list(payload.get("patterns", []) or [])
            if intent and patterns:
                rules.append({"intent": intent, "patterns": patterns})
        elif etype == "DEMO":
            demos.append({
                "input": str(payload.get("input", "")),
                "output": str(payload.get("output", "")),
            })
        elif etype == "LABEL":
            labels.append({
                "input": str(payload.get("input", "")),
                "intent": str(payload.get("intent", "")),
            })

    # اپیزودها می‌توانند برای حافظهٔ نمونه استفاده شوند (fallback)
    if episodes_root.exists():
        for log_file in episodes_root.rglob("*.jsonl"):
            demos.extend(_read_jsonl(log_file))

    _write_yaml_rules(rules)
    _write_demo_memory(demos)
    _maybe_build_tfidf(demos)
    _maybe_train_intent_clf(labels)

    print("OK: learned_rules.yaml + demo_memory.jsonl (اختیاری: tfidf, intent_clf)")


if __name__ == "__main__":
    main()
