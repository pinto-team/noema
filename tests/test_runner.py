# -*- coding: utf-8 -*-
"""
NOEMA • tests/test_runner.py — اجرای سادهٔ تست‌های رگرسیونی (V0)

نحوهٔ اجرا:
    python -m tests.test_runner
    # یا:
    python tests/test_runner.py --pattern tests/regression/*.yaml
"""

from __future__ import annotations
import argparse
import glob
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# تلاش برای YAML؛ اگر نبود، JSON-شکل
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

lang_parse = importlib.import_module("lang.parse")
from skills import load_skills
from skills.invoke_calc import run as run_calc
from skills.reply_greeting import run as run_greet
from skills.reply_smalltalk import run as run_smalltalk
from skills.reply_from_memory import run as run_memory

def _load_case(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if _HAS_YAML:
        try:
            return yaml.safe_load(text) or {}
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return {}

def _assertions(expect: Dict[str, Any], intent: str, text_out: str, outcome: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    exp_int = (expect or {}).get("intent")
    if isinstance(exp_int, str) and exp_int and intent != exp_int:
        errs.append(f"intent mismatch: got={intent!r} expect={exp_int!r}")

    def _contains_all(subs: List[str], s: str) -> bool:
        return all((str(x) in s) for x in subs)

    for k, subs in (("text_contains","in"), ("not_contains","not in")):
        if isinstance(expect.get(k), list):
            for sub in expect[k]:
                has = (str(sub) in text_out)
                if k == "text_contains" and not has:
                    errs.append(f"text not contains {sub!r}")
                if k == "not_contains" and has:
                    errs.append(f"text unexpectedly contains {sub!r}")

    # outcome_equals: کلیدهای خاص در outcome باید برابر باشند
    if isinstance(expect.get("outcome_equals"), dict):
        for k, v in expect["outcome_equals"].items():
            if str(outcome.get(k)) != str(v):
                errs.append(f"outcome.{k} mismatch: got={outcome.get(k)!r} expect={v!r}")

    return errs

_ARTIFACTS = [
    Path("config/learned_rules.yaml"),
    Path("data/demo_memory.jsonl"),
    Path("data/demo_index.npz"),
    Path("data/demo_vocab.json"),
    Path("models/intent_clf.joblib"),
]


def _reset_artifacts() -> None:
    for p in _ARTIFACTS:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            continue


def _apply_setup(setup: Dict[str, Any] | None) -> None:
    _reset_artifacts()
    setup = setup or {}

    if setup.get("learned_rules"):
        Path("config").mkdir(parents=True, exist_ok=True)
        target = Path("config/learned_rules.yaml")
        data = setup["learned_rules"]
        if _HAS_YAML:
            with target.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
        else:
            target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    demos = setup.get("demos") or []
    if demos:
        Path("data").mkdir(parents=True, exist_ok=True)
        mem_path = Path("data/demo_memory.jsonl")
        text = "\n".join(json.dumps(obj, ensure_ascii=False) for obj in demos)
        mem_path.write_text(text, encoding="utf-8")

    labels = setup.get("labels") or []
    if labels:
        Path("logs").mkdir(parents=True, exist_ok=True)
        Path("logs/teacher_events.jsonl").write_text("", encoding="utf-8")

    importlib.reload(lang_parse)


def _extract_user_text(inp: Any) -> str:
    if isinstance(inp, str):
        return inp
    if isinstance(inp, dict):
        return str(inp.get("user_text", ""))
    return str(inp or "")


def run_case(case: Dict[str, Any], path: str) -> Dict[str, Any]:
    inp = case.get("input")
    plan_hint = case.get("plan_hint") or {}
    user_text = _extract_user_text(inp)

    # 1) برنامه/نیت
    plan = lang_parse.parse(user_text)
    if isinstance(plan_hint, dict) and plan_hint:
        plan.update(plan_hint)

    # 2) اجرای مهارت مناسب
    intent = plan.get("intent", "unknown")
    if intent == "greeting":
        out = run_greet(user_text=user_text, plan=plan)
    elif intent == "compute":
        out = run_calc(user_text=user_text, plan=plan)
    elif intent == "smalltalk":
        out = run_smalltalk(user_text=user_text, plan=plan)
    elif intent == "memory.reply":
        out = run_memory(user_text=user_text, plan=plan)
    else:
        out = {
            "intent": intent,
            "outcome": {},
            "text_out": "",
            "meta": {"confidence": plan.get("confidence", 0.4)},
        }

    # 3) ارزیابی
    expect = case.get("expect") or {}
    errs = _assertions(expect, out.get("intent",""), out.get("text_out",""), out.get("outcome",{}))

    return {
        "path": path,
        "ok": len(errs) == 0,
        "errors": errs,
        "intent": out.get("intent"),
        "text_out": out.get("text_out"),
        "outcome": out.get("outcome"),
    }

def main(pattern: str) -> int:
    files = sorted(glob.glob(pattern))
    if not files:
        print("No tests found for pattern:", pattern)
        return 1

    reg = load_skills()  # فقط برای اطمینان از خطایابی manifest
    _ = reg.list_all()

    results: List[Dict[str, Any]] = []
    fails = 0
    for fp in files:
        case = _load_case(fp)
        _apply_setup(case.get("setup"))
        res = run_case(case, fp)
        results.append(res)
        status = "PASS" if res["ok"] else "FAIL"
        print(f"[{status}] {fp}")
        if not res["ok"]:
            fails += 1
            for e in res["errors"]:
                print("   -", e)

    print(f"\nTotal: {len(results)}, Passed: {len(results)-fails}, Failed: {fails}")
    return 0 if fails == 0 else 2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="tests/regression/*.yaml")
    args = ap.parse_args()
    sys.exit(main(args.pattern))
