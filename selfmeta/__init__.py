# -*- coding: utf-8 -*-
"""
NOEMA • selfmeta package (V0)
- مدلِ خود (Self-Model) + کالیبراسیونِ احتمال (online binning).
- این فایل، سازنده‌های سطح‌بالا را فراهم می‌کند تا استفاده ساده باشد.

استفاده‌ی سریع:
    from selfmeta import load_self_model, save_calibrator
    sm, cal = load_self_model("data/calibration.json")
    snap = sm.update({"u":0.25, "r_int":0.2, "r_total":0.5, "risk":0.0, "energy":0.03})
    print(snap["confidence"])
    save_calibrator(cal, "data/calibration.json")
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

from .self_model import SelfModel, SelfModelCfg
from .calibrate import OnlineBinnedCalibrator

__all__ = [
    "SelfModel",
    "SelfModelCfg",
    "OnlineBinnedCalibrator",
    "load_self_model",
    "save_calibrator",
]

# ---------------------------------------------------------------------

def load_self_model(
    calib_path: str | Path = "data/calibration.json",
    *,
    attach: bool = True,
    cfg: Optional[SelfModelCfg] = None,
    create_if_missing: bool = True,
) -> Tuple[SelfModel, OnlineBinnedCalibrator]:
    """
    یک SelfModel می‌سازد و (اختیاری) کالیبراتور را از فایل می‌خوانَد و متصل می‌کند.
    اگر فایل وجود نداشت و create_if_missing=True باشد، یک کالیبراتور پیش‌فرض می‌سازد.

    خروجی:
        (self_model, calibrator)
    """
    sm = SelfModel(cfg or SelfModelCfg())
    p = Path(calib_path)

    if p.exists():
        cal = OnlineBinnedCalibrator.load(p)
    else:
        cal = OnlineBinnedCalibrator() if create_if_missing else None  # type: ignore

    if attach and cal is not None:
        sm.attach_calibrator(cal)

    return sm, cal  # type: ignore

def save_calibrator(calibrator: OnlineBinnedCalibrator, path: str | Path) -> Path:
    """
    کالیبراتور را در مسیر داده‌شده ذخیره می‌کند.
    """
    return calibrator.save(path)

# ---------------------------------------------------------------------

if __name__ == "__main__":
    # تست سریع: بارگذاری/ساخت کالیبراتور و به‌روزرسانی یک اسنپ‌شات
    sm, cal = load_self_model("data/calibration.json")
    snap = sm.update({
        "u": 0.30,          # عدم‌قطعیت
        "r_int": 0.15,      # پیشرفت
        "r_total": 0.40,    # پاداش شکل‌داده
        "risk": 0.0,
        "energy": 0.05,
    })
    print("conf:", round(snap["confidence"], 3), "| energy:", round(snap["energy"], 3))
    # ذخیره‌ی کالیبراتور
    if cal is not None:
        save_calibrator(cal, "data/calibration.json")
