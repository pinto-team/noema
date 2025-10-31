# -*- coding: utf-8 -*-
"""
NOEMA • scripts/init.py — آماده‌سازی پوشه‌ها/فایل‌های ضروری پروژه (V0)

اجرا:
    python scripts/init.py

کارها:
  - ساخت پوشه‌های data/{episodes,index,concepts}
  - ایجاد فایل‌های پیکربندی پیش‌فرض اگر وجود ندارند (از نمونه‌های config/* همین مخزن)
  - ساخت skills/manifest.yaml حداقلی اگر نبود
"""

from __future__ import annotations
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def ensure_dirs():
    for d in [
        ROOT / "data",
        ROOT / "data" / "episodes",
        ROOT / "data" / "index" / "faiss",
        ROOT / "data" / "concepts",
    ]:
        d.mkdir(parents=True, exist_ok=True)

def copy_if_missing(src_rel: str, dst_rel: str):
    src = ROOT / src_rel
    dst = ROOT / dst_rel
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print("• created:", dst_rel)

def main() -> int:
    ensure_dirs()
    # کپی نمونهٔ پیکربندی‌ها اگر مقصد خالی است
    copy_if_missing("config/safety.yaml", "config/safety.yaml")
    copy_if_missing("config/value.yaml", "config/value.yaml")
    copy_if_missing("config/meta.yaml", "config/meta.yaml")
    copy_if_missing("config/tools.yaml", "config/tools.yaml")
    copy_if_missing("skills/manifest.yaml", "skills/manifest.yaml")

    # ایجاد فایل episodes.jsonl اگر وجود ندارد
    ep = ROOT / "data" / "episodes" / "episodes.jsonl"
    if not ep.exists():
        ep.write_text("", encoding="utf-8")
        print("• created: data/episodes/episodes.jsonl")

    print("✅ init done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
