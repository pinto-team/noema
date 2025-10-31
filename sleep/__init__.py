# -*- coding: utf-8 -*-
"""
NOEMA • sleep package (V0)
- چرخه‌ی «خواب/تثبیت» آفلاین را یک‌جا در اختیار می‌گذارد.

استفاده‌ی سریع:
    from sleep import SleepCfg, run_sleep_cycle, run_once
    rep = run_once()               # با تنظیمات پیش‌فرض
    # یا:
    cfg = SleepCfg(dim=64, calibrate=True)
    rep = run_sleep_cycle(cfg)

یادداشت:
- این پکیج به ماژول‌های memory/*، concept/* و selfmeta/* متکی است.
- اگر هرکدام در دسترس نباشند، sleep/offline.py خطای خوانا برمی‌گرداند.
"""

from __future__ import annotations
from typing import Any, Dict
from .offline import SleepCfg, run_sleep_cycle

__all__ = [
    "SleepCfg",
    "run_sleep_cycle",
    "run_once",
]

def run_once(**kwargs) -> Dict[str, Any]:
    """
    یک اجرای سریعِ چرخه‌ی خواب با پارامترهای دلخواه.

    پارامترهای رایج (اختیاری):
      - episodes_root="data/episodes"
      - rebuild_index=True, index_prefix="data/index/faiss", dim=64, key_mode="mean",
        metric="ip", kind="HNSW32", normalize=True
      - rebuild_concepts=True, rebuild_graph=True
      - concepts_path="data/concepts/concepts.json", graph_path="data/concepts/graph.json",
        min_conf_for_concepts=0.0, limit_for_concepts=5000
      - calibrate=True, calibration_path="data/calibration.json",
        max_calibration_pairs=8000, heuristic_label_threshold=0.10
    """
    cfg = SleepCfg(**kwargs)
    return run_sleep_cycle(cfg, verbose=True)
