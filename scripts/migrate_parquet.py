# -*- coding: utf-8 -*-
"""
NOEMA • scripts/migrate_parquet.py — تبدیل اپیزودهای JSONL به Parquet (اختیاری)

اجرا:
    python scripts/migrate_parquet.py --in data/episodes/episodes.jsonl --out data/episodes/episodes.parquet

پیش‌نیاز:
  - pandas + pyarrow (اختیاری). اگر موجود نباشند، اسکریپت پیام می‌دهد و خارج می‌شود.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

def main(in_path: str, out_path: str) -> int:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        print("⚠️  pandas/pyarrow در دسترس نیستند. `pip install pandas pyarrow`")
        return 1

    p_in = Path(in_path)
    if not p_in.exists():
        print("❌ ورودی وجود ندارد:", p_in)
        return 2

    rows = []
    with p_in.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if not rows:
        print("⚠️  هیچ سطری خوانده نشد.")
        return 0

    df = pd.DataFrame(rows)
    p_out = Path(out_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p_out, index=False)
    print("✅ parquet saved:", str(p_out))
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/episodes/episodes.jsonl")
    ap.add_argument("--out", dest="out_path", default="data/episodes/episodes.parquet")
    args = ap.parse_args()
    raise SystemExit(main(args.in_path, args.out_path))
