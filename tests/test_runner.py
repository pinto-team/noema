# -*- coding: utf-8 -*-
"""
NOEMA • tests/test_runner.py — اجرای سادهٔ تست‌های رگرسیونی (V0)

نحوهٔ اجرا:
    python -m tests.test_runner
    # یا:
    python tests/test_runner.py --pattern tests/regression/*.yaml
"""

from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
from typing import Any, Dict, List

# تلاش برای YAML؛ اگر نبود، JSON-شکل
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# وارد کردن اجزای سبک نوما
from lang import parse as parse_intent, format_reply, load_style
from skills import load_skills
from skills import __dict__ as _skills_pkg  # برای resolve مستقیم ماژول‌ها
from skills.invoke_calc import run as run_calc
from skills.reply_greeting import run as run_greet

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

def run_case(case: Dict[str, Any], path: str) -> Dict[str, Any]:
    inp = case.get("input") or {}
    plan_hint = case.get("plan_hint") or {}
    user_text = str(inp.get("user_text", ""))

    # 1) برنامه/نیت
    plan = parse_intent(user_text)
    if isinstance(plan_hint, dict) and plan_hint:
        plan.update(plan_hint)

    # 2) اجرای مهارت مناسب
    intent = plan.get("intent", "unknown")
    if intent == "greeting":
        out = run_greet(user_text=user_text, plan=plan)
    elif intent == "compute":
        out = run_calc(user_text=user_text, plan=plan)
    else:
        # پاسخ ناشناخته مینیمال
        txt = format_reply(intent="unknown", outcome={}, style=load_style(), meta={"confidence":0.4})
        out = {"intent":"unknown","outcome":{},"text_out":txt,"meta":{"confidence":0.4}}

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
