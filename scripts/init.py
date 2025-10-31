# -*- coding: utf-8 -*-
"""NOEMA • scripts/init.py — آماده‌سازی پوشه‌ها/فایل‌های ضروری پروژه (V0)

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


def ensure_dirs() -> None:
    """Create the directory scaffold expected by the runtime."""

    for directory in [
        ROOT / "data",
        ROOT / "data" / "episodes",
        ROOT / "data" / "index",
        ROOT / "data" / "index" / "faiss",
        ROOT / "data" / "concepts",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def _copy_if_missing(src_rel: str, dst_rel: str) -> bool:
    """Copy a template file if the destination is absent.

    Returns True when a file has been created so the caller can print feedback.
    """

    src = ROOT / src_rel
    dst = ROOT / dst_rel
    if dst.exists() or not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def main() -> int:
    """Entry-point used by `python scripts/init.py`."""

    ensure_dirs()

    created_any = False
    for path in [
        ("config/safety.yaml", "config/safety.yaml"),
        ("config/value.yaml", "config/value.yaml"),
        ("config/meta.yaml", "config/meta.yaml"),
        ("config/tools.yaml", "config/tools.yaml"),
        ("skills/manifest.yaml", "skills/manifest.yaml"),
    ]:
        if _copy_if_missing(*path):
            print(f"• created: {path[1]}")
            created_any = True

    episodes_file = ROOT / "data" / "episodes" / "episodes.jsonl"
    if not episodes_file.exists():
        episodes_file.write_text("", encoding="utf-8")
        print("• created: data/episodes/episodes.jsonl")
        created_any = True

    if not created_any:
        print("No changes; everything already in place.")
    print("✅ init done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
