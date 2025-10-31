# -*- coding: utf-8 -*-
"""
NOEMA • sleep/offline.py — چرخهٔ «خواب/تثبیت» آفلاین (V0)

هدف:
  - اجرای چند کار آفلاین روی لاگ اپیزودها برای بهبود پایدار نوما:
      1) بازسازی ایندکس برداری (FAISS یا fallback)
      2) استخراج «گره‌های مفهومی» و ذخیرهٔ concepts.json
      3) ساخت گراف توالی مفاهیم و ذخیرهٔ graph.json
      4) کالیبراسیونِ اعتماد (calibration.json) با استفاده از اپیزودهای برچسب‌دار یا هیوریستیک

وابستگی‌های خارجی (اختیاری):
  - scikit-learn  → برای خوشه‌بندی MiniBatchKMeans در concept/clustering.py
  - faiss          → برای ایندکس ANN سریع (در صورت نبود، brute-force فعال می‌شود)

استفادهٔ سریع:
    from sleep.offline import SleepCfg, run_sleep_cycle
    report = run_sleep_cycle(SleepCfg())
    print(report)

قرارداد داده (انتظارات مینی‌مال از EpisodeStore):
  Episode فیلدهای زیر را «ترجیحاً» داشته باشد (عدم وجود هرکدام قابل تحمل است):
    - ts:      زمان (epoch seconds)
    - intent:  رشته (مثل "greeting"/"compute"/…)
    - action_name: نام کنش اجرا شده
    - text_in, text_out: متن ورودی/خروجی
    - conf:    اعتماد به پاسخ [0..1]
    - u:       عدم‌قطعیت [0..1]
    - r_total: پاداش نهایی در آن گام [-1..1]
    - label_ok: نتیجهٔ دودوییِ صحت (اختیاری، True/False)  ← اگر موجود باشد برای کالیبراسیون ارجح است
    - z_vec/s_vec: بردارهای نهان (در صورت وجود) برای ایندکس/مفهوم استفاده می‌شود
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json

# حافظه و شاخص
try:
    from memory import EpisodeStore, build_from_episode_store, FaissIndex, IndexConfig
except Exception as e:
    raise RuntimeError("sleep/offline.py نیازمند memory/* است. ابتدا ماژول memory را آماده کنید.") from e

# مفاهیم و گراف
try:
    from concept import cluster_run_end_to_end as concepts_build
    from concept import graph_run_end_to_end as graph_build
except Exception as e:
    raise RuntimeError("sleep/offline.py نیازمند concept/* است. ابتدا ماژول concept را آماده کنید.") from e

# مدلِ خود و کالیبره
try:
    from selfmeta import load_self_model, save_calibrator
except Exception as e:
    raise RuntimeError("sleep/offline.py نیازمند selfmeta/* است. ابتدا ماژول selfmeta را آماده کنید.") from e

# ----------------------------- پیکربندی -----------------------------

@dataclass
class SleepCfg:
    # ریشهٔ فایل‌های اپیزود
    episodes_root: str = "data/episodes"

    # ایندکس برداری
    rebuild_index: bool = True
    index_prefix: str = "data/index/faiss"
    key_mode: str = "mean"      # "mean" | "z" | "s"
    dim: int = 64
    metric: str = "ip"          # "ip" (cosine روی بردارهای نرمال) | "l2"
    kind: str = "HNSW32"
    normalize: bool = True

    # مفاهیم/گراف
    rebuild_concepts: bool = True
    rebuild_graph: bool = True
    concepts_path: str = "data/concepts/concepts.json"
    graph_path: str = "data/concepts/graph.json"
    min_conf_for_concepts: float = 0.0
    limit_for_concepts: int = 5000

    # کالیبراسیون
    calibrate: bool = True
    calibration_path: str = "data/calibration.json"
    max_calibration_pairs: int = 8000
    heuristic_label_threshold: float = 0.10   # اگر label_ok نبود: y=1 اگر r_total > +thr

# ----------------------------- ابزار کالیبراسیون -----------------------------

def _harvest_calibration_pairs(
    store: EpisodeStore,
    *,
    max_n: int = 8000,
    thr: float = 0.10,
) -> List[Tuple[float, int]]:
    """
    از اپیزودها جفت (p_raw, y) استخراج می‌کند:
      - p_raw: اعتماد خام (conf) اگر نبود: 1-u
      - y:     برچسب دودویی؛ ارجحیت با ep.label_ok؛ در غیر این صورت y = [r_total > thr]
    """
    pairs: List[Tuple[float, int]] = []
    episodes = store.tail(n=max_n)
    for ep in episodes:
        # اعتماد
        conf = getattr(ep, "conf", None)
        u = getattr(ep, "u", None)
        if isinstance(conf, (int, float)):
            p = max(0.0, min(1.0, float(conf)))
        elif isinstance(u, (int, float)):
            p = max(0.0, min(1.0, float(1.0 - u)))
        else:
            continue  # دادهٔ کافی برای p_raw نداریم

        # برچسب
        if hasattr(ep, "label_ok"):
            y = 1 if bool(getattr(ep, "label_ok")) else 0
        else:
            r_total = getattr(ep, "r_total", None)
            if isinstance(r_total, (int, float)):
                y = 1 if float(r_total) > float(thr) else 0
            else:
                # اگر هیچ نشانه‌ای نداشتیم، از intent/عمل ساده حدس می‌زنیم
                intent = (getattr(ep, "intent", "") or "").strip().lower()
                action = (getattr(ep, "action_name", "") or "").strip().lower()
                if intent == "greeting" and action.startswith("reply"):
                    y = 1
                else:
                    # بدون نشانه → صرف‌نظر از این اپیزود
                    continue
        pairs.append((p, y))

    return pairs

def _apply_calibration(calibration_path: str, pairs: List[Tuple[float, int]]) -> Dict[str, Any]:
    """کالیبراتور را بارگذاری/ایجاد، به‌روز و ذخیره می‌کند؛ خلاصه‌ای برمی‌گرداند."""
    sm, cal = load_self_model(calibration_path, attach=True, create_if_missing=True)
    if cal is None:
        return {"updated": 0, "note": "no calibrator"}

    n = 0
    for p, y in pairs:
        try:
            cal.update(float(p), int(y))
            n += 1
        except Exception:
            continue

    save_calibrator(cal, calibration_path)

    # خروجی منحنی قابلیت اطمینان برای دیباگ
    try:
        curve = cal.reliability_curve()
    except Exception:
        curve = []

    return {
        "updated": n,
        "calibration_path": calibration_path,
        "bins": len(curve) if curve else None,
    }

# ----------------------------- چرخهٔ خواب -----------------------------

def run_sleep_cycle(cfg: SleepCfg, *, verbose: bool = True) -> Dict[str, Any]:
    """
    چرخهٔ آفلاین را اجرا می‌کند. هر بخش در صورت خطا، به‌صورت «ملایم» رد می‌شود تا
    چرخهٔ کلی نشکند (گزارش خطا برمی‌گردد).
    """
    report: Dict[str, Any] = {"ok": True, "steps": []}

    # 0) منبع اپیزودها
    store = EpisodeStore(cfg.episodes_root)

    # 1) ایندکس برداری
    if cfg.rebuild_index:
        step = {"name": "index", "ok": True}
        try:
            idx = build_from_episode_store(
                store,
                key_mode=cfg.key_mode,
                dim=cfg.dim,
                metric=cfg.metric,
                kind=cfg.kind,
                normalize=cfg.normalize,
            )
            idx.save(cfg.index_prefix)
            step["ntotal"] = idx.ntotal()
            step["prefix"] = cfg.index_prefix
        except Exception as e:
            step["ok"] = False
            step["error"] = f"{type(e).__name__}: {e}"
            report["ok"] = False
        report["steps"].append(step)
        if verbose:
            print(f"• index: {step}")

    # 2) مفاهیم
    if cfg.rebuild_concepts:
        step = {"name": "concepts", "ok": True}
        try:
            concepts = concepts_build(
                EpisodeStore(cfg.episodes_root),
                key_mode=cfg.key_mode,
                dim=cfg.dim,
                normalize=cfg.normalize,
                limit=cfg.limit_for_concepts,
                min_conf=cfg.min_conf_for_concepts,
                out_path=cfg.concepts_path,
            )
            step["k"] = len(concepts)
            step["path"] = cfg.concepts_path
        except Exception as e:
            # معمولاً نبود scikit-learn
            step["ok"] = False
            step["error"] = f"{type(e).__name__}: {e}"
            report["ok"] = False
        report["steps"].append(step)
        if verbose:
            print(f"• concepts: {step}")

    # 3) گراف مفاهیم
    if cfg.rebuild_graph:
        step = {"name": "graph", "ok": True}
        try:
            G = graph_build(
                concepts_path=cfg.concepts_path,
                episodes_root=cfg.episodes_root,
                out_path=cfg.graph_path,
                assign_mode=cfg.key_mode,
                limit_tail=max(cfg.limit_for_concepts, 2000),
                min_conf=cfg.min_conf_for_concepts,
            )
            step["nodes"] = getattr(G, "number_of_nodes", lambda: 0)()
            step["edges"] = getattr(G, "number_of_edges", lambda: 0)()
            step["path"] = cfg.graph_path
        except Exception as e:
            step["ok"] = False
            step["error"] = f"{type(e).__name__}: {e}"
            report["ok"] = False
        report["steps"].append(step)
        if verbose:
            print(f"• graph: {step}")

    # 4) کالیبراسیون
    if cfg.calibrate:
        step = {"name": "calibration", "ok": True}
        try:
            pairs = _harvest_calibration_pairs(
                store,
                max_n=cfg.max_calibration_pairs,
                thr=cfg.heuristic_label_threshold,
            )
            sub = _apply_calibration(cfg.calibration_path, pairs)
            step.update(sub)
        except Exception as e:
            step["ok"] = False
            step["error"] = f"{type(e).__name__}: {e}"
            report["ok"] = False
        report["steps"].append(step)
        if verbose:
            print(f"• calibration: {step}")

    return report

# ----------------------------- اجرای مستقیم (CLI ساده) -----------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="NOEMA sleep/offline cycle")
    p.add_argument("--episodes", default="data/episodes", help="episodes root folder")
    p.add_argument("--no-index", action="store_true", help="skip index rebuild")
    p.add_argument("--no-concepts", action="store_true", help="skip concept clustering")
    p.add_argument("--no-graph", action="store_true", help="skip concept graph")
    p.add_argument("--no-calib", action="store_true", help="skip calibration update")
    p.add_argument("--dim", type=int, default=64, help="embedding dim for index/concepts")
    args = p.parse_args()

    cfg = SleepCfg(
        episodes_root=args.episodes,
        rebuild_index=not args.no_index,
        rebuild_concepts=not args.no_concepts,
        rebuild_graph=not args.no_graph,
        calibrate=not args.no_calib,
        dim=int(args.dim),
    )
    rep = run_sleep_cycle(cfg, verbose=True)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
