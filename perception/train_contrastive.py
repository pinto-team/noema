# -*- coding: utf-8 -*-
"""
NOEMA • perception/train_contrastive.py — minimal contrastive training (V0)

Goal
-----
Train a lightweight, learnable char n-gram encoder using an InfoNCE
objective (two augmented views). The learned model can later be loaded
by a richer encoder; this script is standalone and optional.

Usage
-----
python perception/train_contrastive.py --data data/corpus.txt --epochs 3
python perception/train_contrastive.py --data data/corpus.jsonl --jsonl

Notes
-----
- PyTorch only; no heavy dependencies beyond that.
- Simple text augmentations: drop punctuation, space jitter, random char drop.
- Deterministic hashing to bucket ids; training reproducibility via seeds.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import argparse
import json
import random
import re
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from perception.encoder import normalize_text

# ------------------------- Config -------------------------

@dataclass
class TrainCfg:
    # model
    buckets: int = 8192
    dim: int = 64
    ngrams: Tuple[int, ...] = (2, 3)
    # training
    batch_size: int = 256
    epochs: int = 3
    lr: float = 2e-3
    weight_decay: float = 1e-4
    temperature: float = 0.07
    max_len: int = 280
    # data
    data: str = "data/corpus.txt"
    jsonl: bool = False
    # output
    out_dir: str = "models"
    # misc
    seed: int = 42

# ------------------------- Hash utils -------------------------

def stable_hash_u64(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)

def char_ngrams(t: str, n_values: Tuple[int, ...]) -> List[str]:
    toks: List[str] = []
    for n in n_values:
        if len(t) < n:
            continue
        toks.extend([t[i:i+n] for i in range(len(t)-n+1)])
    return toks

# ------------------------- Augmentations -------------------------

_PUNCS = r""".,!?؛،:;…\-—()\[\]{}"\'`/\\|~*_^+=«»<>"""

def aug_drop_punct(t: str, p: float = 0.3) -> str:
    return "".join(ch for ch in t if not (ch in _PUNCS and random.random() < p))

def aug_space_jitter(t: str, p: float = 0.15) -> str:
    out = []
    for ch in t:
        if ch == " " and random.random() < p:
            if random.random() < 0.5:
                continue  # drop
            else:
                out.append("  ")  # double
        else:
            out.append(ch)
    return "".join(out)

def aug_drop_char(t: str, p: float = 0.04) -> str:
    return "".join(ch for ch in t if not (random.random() < p))

def make_two_views(text: str, cfg: TrainCfg) -> Tuple[str, str]:
    """Normalize then create two augmented views."""
    t = normalize_text(text)[: cfg.max_len]
    def pipe(x: str) -> str:
        x = aug_drop_punct(x, 0.25)
        x = aug_space_jitter(x, 0.10)
        x = aug_drop_char(x, 0.03)
        return normalize_text(x)
    return pipe(t), pipe(t)

# ------------------------- Data -------------------------

def load_lines(path: Path, jsonl: bool) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if jsonl:
                try:
                    obj = json.loads(ln)
                    t = obj.get("text", "")
                    if t:
                        texts.append(t)
                except Exception:
                    continue
            else:
                texts.append(ln)
    random.shuffle(texts)
    return texts

# ------------------------- Model -------------------------

class CharNgramEncoder(nn.Module):
    """
    Each n-gram hashes to a bucket in [0..buckets). We sum embeddings for all
    buckets in a sample (TF-like), then L2-normalize the result.
    """
    def __init__(self, buckets: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(buckets, dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, idxs: List[List[int]]) -> torch.Tensor:
        device = self.emb.weight.device
        vecs: List[torch.Tensor] = []
        for ids in idxs:
            if not ids:
                vec = torch.zeros(self.emb.embedding_dim, device=device)
            else:
                ids_t = torch.tensor(ids, dtype=torch.long, device=device)
                vec = self.emb(ids_t).sum(dim=0)
            vecs.append(vec)
        X = torch.stack(vecs, dim=0)  # [B, D]
        # Normalize; add epsilon to avoid div-by-zero
        X = X / (X.norm(p=2, dim=1, keepdim=True) + 1e-12)
        return X

# ------------------------- Collation -------------------------

def encode_to_buckets(texts: List[str], buckets: int, ngrams: Tuple[int, ...]) -> List[List[int]]:
    batch: List[List[int]] = []
    for t in texts:
        t = normalize_text(t)
        grams = char_ngrams(t, ngrams)
        ids = [(stable_hash_u64(g) % buckets) for g in grams]
        batch.append(ids)
    return batch

# ------------------------- Loss -------------------------

def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Two-view InfoNCE (SimCLR-style):
      - z1, z2: [B, D], already L2-normalized
      - positive for row i is the counterpart in the other view
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)              # [2B, D]
    sim = (z @ z.t()) / temperature             # [2B, 2B]
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    labels = torch.arange(2 * B, device=z.device)
    labels = (labels + B) % (2 * B)
    loss = F.cross_entropy(sim, labels)
    return loss

# ------------------------- Utils -------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------- Train loop -------------------------

def train(model: CharNgramEncoder, texts: List[str], cfg: TrainCfg, device: str = "cpu") -> None:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps = 0
    for ep in range(cfg.epochs):
        random.shuffle(texts)
        for i in range(0, len(texts), cfg.batch_size):
            batch_txt = texts[i : i + cfg.batch_size]
            if not batch_txt:
                continue

            v1, v2 = zip(*[make_two_views(t, cfg) for t in batch_txt])
            ids1 = encode_to_buckets(list(v1), cfg.buckets, cfg.ngrams)
            ids2 = encode_to_buckets(list(v2), cfg.buckets, cfg.ngrams)

            z1 = model(ids1)
            z2 = model(ids2)
            loss = info_nce(z1, z2, cfg.temperature)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            steps += 1
            if steps % 50 == 0:
                with torch.no_grad():
                    sim_pos = (z1 * z2).sum(dim=1).mean().item()
                print(f"[ep {ep+1}] step {steps} | loss={loss.item():.4f} | sim+={sim_pos:.3f}")

# ------------------------- Save/Load -------------------------

def save_model(model: CharNgramEncoder, cfg: TrainCfg) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "perception.pt")
    (out_dir / "perception_cfg.json").write_text(
        json.dumps(
            {"buckets": cfg.buckets, "dim": cfg.dim, "ngrams": list(cfg.ngrams)},
            ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )
    print(f"✅ saved: {out_dir/'perception.pt'} and perception_cfg.json")

# ------------------------- CLI -------------------------

def parse_args() -> TrainCfg:
    p = argparse.ArgumentParser(description="Train a minimal contrastive char n-gram encoder.")
    p.add_argument("--data", type=str, default="data/corpus.txt", help="Path to .txt or .jsonl data")
    p.add_argument("--jsonl", action="store_true", help="Treat --data as JSONL with a 'text' field")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--buckets", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--out_dir", type=str, default="models")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    return TrainCfg(
        data=args.data,
        jsonl=args.jsonl,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dim=args.dim,
        buckets=args.buckets,
        temperature=args.temperature,
        out_dir=args.out_dir,
        seed=args.seed,
    )

# ------------------------- Main -------------------------

def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)

    texts = load_lines(Path(cfg.data), cfg.jsonl)
    print(f"Loaded {len(texts)} texts from {cfg.data}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = CharNgramEncoder(cfg.buckets, cfg.dim)
    train(model, texts, cfg, device=device)
    save_model(model, cfg)

if __name__ == "__main__":
    main()
