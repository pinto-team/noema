# -*- coding: utf-8 -*-
"""
NOEMA • scripts/init_faiss.py — ساخت/بازسازی ایندکس FAISS (یا جایگزین ساده)

اجرا:
    python scripts/init_faiss.py --episodes data/episodes --out data/index/faiss --dim 64

یادداشت:
  - اگر ماژول memory/* یا faiss نصب نباشد، پیام راهنما چاپ می‌شود.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

def main(episodes: str, out_prefix: str, dim: int) -> int:
    try:
        from memory import EpisodeStore, build_from_episode_store  # type: ignore
    except Exception:
        print("⚠️  memory/* در دسترس نیست. ابتدا ماژول memory را اضافه/نصب کنید.")
        return 1

    try:
        store = EpisodeStore(episodes)
        idx = build_from_episode_store(store, key_mode="mean", dim=int(dim), metric="ip", kind="HNSW32", normalize=True)
        idx.save(out_prefix)
        print("✅ index saved at prefix:", out_prefix)
        return 0
    except Exception as e:
        print("❌ failed:", type(e).__name__, str(e))
        return 2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="data/episodes")
    ap.add_argument("--out", default="data/index/faiss")
    ap.add_argument("--dim", type=int, default=64)
    args = ap.parse_args()
    raise SystemExit(main(args.episodes, args.out, args.dim))
