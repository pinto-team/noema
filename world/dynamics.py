# -*- coding: utf-8 -*-
"""
NOEMA • world/dynamics.py — مدلِ جهان مینیمال (V0، بدون وابستگی سنگین)
- هدف: ساخت State پایدار از تاریخچه‌ی z و پیش‌بینی zِ گام بعد با توجه به عمل a.
- عدم‌قطعیت (u_hat) و ریسک (risk_hat) را به‌صورت اکتشافی تخمین می‌زند.
- طوری نوشته شده که با app/main.py سازگار باشد: state(z_hist) و predict(s, a).

یادداشت:
- این نسخه فقط با NumPy/stdlib کار می‌کند (بدون PyTorch). بعداً می‌توانید
  پیاده‌سازی GRU/Transformer جایگزین کنید ولی API را تغییر ندهید.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import math, hashlib, struct, numpy as np

# ------------ انواع کمینه (برای تایپ‌هینت) ------------
@dataclass
class Latent:
    z: List[float]

@dataclass
class State:
    s: List[float]
    u: float = 0.0   # uncertainty (0..1)
    conf: float = 0.0

@dataclass
class Action:
    kind: str                 # "skill" | "tool" | "policy"
    name: str                 # نام کنش
    args: Dict[str, Any]      # آرگومان‌ها

# ------------ ابزارها ------------

def _to_np(x: List[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)

def _from_np(x: np.ndarray) -> List[float]:
    return x.astype(np.float32).tolist()

def _stable_float_stream(key: str, n: int) -> np.ndarray:
    """
    از blake2b یک دنباله‌ی ثابت از float در [-1,1] می‌سازد (بدون تصادفی‌بودن محیط).
    """
    out = np.zeros((n,), dtype=np.float32)
    seed = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
    # از 4-به-4 بایت float بسازیم
    need = n
    buf = bytearray(seed)
    while len(buf) < need * 4:
        # زنجیره‌ی هش
        seed = hashlib.blake2b(seed, digest_size=16).digest()
        buf.extend(seed)
    for i in range(n):
        chunk = bytes(buf[4*i:4*i+4])
        # uint32 → float در [0,1) → سپس به [-1,1]
        v = struct.unpack(">I", chunk)[0] / 0xFFFFFFFF
        out[i] = 2.0 * v - 1.0
    # نرمال‌سازی
    norm = np.linalg.norm(out) or 1.0
    return out / norm

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    da = np.linalg.norm(a) or 1.0
    db = np.linalg.norm(b) or 1.0
    return float(np.dot(a, b) / (da * db))

# ------------ State ساز ------------

def state(z_hist: List[Latent]) -> State:
    """
    از چند z اخیر، یک حالت پایدار می‌سازد:
      - s = میانگین z ها
      - u = نرمال‌سازی‌شده‌ی پراکندگی (MAD) در [0..1]
      - conf = 1 - u
    """
    if not z_hist:
        return State(s=[], u=1.0, conf=0.0)
    Z = np.vstack([_to_np(z.z) for z in z_hist])      # [T, D]
    s_vec = Z.mean(axis=0)                            # میانگین
    # عدم‌قطعیت: میانگین فاصله‌ی L1 از میانگین، نرمال‌شده به بعد
    mad = np.mean(np.abs(Z - s_vec[None, :]))
    # نگاشت اکتشافی MAD به [0..1]
    # فرض ابعاد ~64؛ عدد 0.25 را برای مقیاس تنظیم می‌کنیم
    u = float(np.clip(mad / 0.25, 0.0, 1.0))
    conf = float(np.clip(1.0 - u, 0.0, 1.0))
    return State(s=_from_np(s_vec), u=u, conf=conf)

# ------------ پیش‌بینی پویایی ------------

# وزن نفوذ کنش روی حالت (بین 0 و 1)
_ALPHA = 0.88            # حفظ اینرسی
_BETA  = 0.12            # سهم کنش

# نام‌هایی که ریسک بسیار پایینی دارند
_LOW_RISK_ACTIONS = {"reply_greeting", "invoke_calc", "ask_clarify"}

def _action_embed(a: Action, dim: int) -> np.ndarray:
    """
    بردار تعاملی برای کنش:
      - از نام کنش یک جهت پایدار می‌سازیم.
      - اگر آرگومان‌های عددی داریم (مثل طول عبارت)، کمی تزریق می‌کنیم.
    """
    base = _stable_float_stream(f"action::{a.name}", dim)
    # سیگنال ساده از args: طول expr یا تعداد کلیدها
    bonus = 0.0
    if isinstance(a.args, dict) and a.args:
        if "expr" in a.args and isinstance(a.args["expr"], str):
            bonus = min(len(a.args["expr"]) / 64.0, 1.0)  # حداکثر 1
        else:
            bonus = min(len(a.args.keys()) / 8.0, 1.0)
    return np.clip(base + 0.05 * bonus, -1.0, 1.0)

def _uncertainty_hat(s_vec: np.ndarray, s1_vec: np.ndarray, a: Action) -> float:
    """
    عدم‌قطعیت پیش‌بینی: تابعی از میزان تغییر نسبت به s و ناشناختگی کنش.
    """
    delta = float(np.linalg.norm(s1_vec - s_vec)) / (np.linalg.norm(s_vec) + 1e-6)
    novelty = 0.35 if a.name not in _LOW_RISK_ACTIONS else 0.1
    u_hat = np.clip(0.2 * delta + novelty, 0.0, 1.0)
    return float(u_hat)

def _risk_hat(a: Action) -> float:
    return 0.0 if a.name in _LOW_RISK_ACTIONS else 0.06

def _rhat_progress(s_vec: np.ndarray, s1_vec: np.ndarray) -> float:
    """
    پاداش مورد انتظار (درونی): اگر پیش‌بینیِ آینده با جهتِ «پایدار» سازگار باشد، بیشتر.
    اینجا با cos(s, s1) تخمین می‌زنیم و به [0..1] نگاشت می‌کنیم.
    """
    c = _cosine(s_vec, s1_vec)
    return float(0.5 * (c + 1.0))   # [-1,1] → [0,1]

def predict(s: State, a: Action) -> Tuple[State, Latent, float, float, float]:
    """
    ورودی: State فعلی + یک Action
    خروجی:
      - s1 (State جدید)
      - ẑ_{t+1} (Latent پیش‌بینی‌شده)
      - r̂ (پیش‌بینی پاداش/پیشرفت)
      - risk_hat (ریسک)
      - u_hat (عدم‌قطعیت پیش‌بینی)
    """
    if not s.s:
        # حالت خالی: یک پیش‌بینی خنثی
        return State(s=[], u=1.0, conf=0.0), Latent([]), 0.0, 0.0, 1.0

    s_vec = _to_np(s.s)
    dim = s_vec.shape[0]
    a_vec = _action_embed(a, dim)

    # پویایی خطیِ ساده: s1 = α*s + β*action
    s1_vec = _ALPHA * s_vec + _BETA * a_vec
    # نرمال‌سازی نرم برای جلوگیری از انفجار
    norm = np.linalg.norm(s1_vec)
    if norm > 1.8:   # آستانه‌ی تجربی
        s1_vec = s1_vec / norm

    # عدم‌قطعیت و ریسک
    u_hat = _uncertainty_hat(s_vec, s1_vec, a)
    risk = _risk_hat(a)

    # اعتماد به s1: 1 - u_hat
    conf1 = float(np.clip(1.0 - u_hat, 0.0, 1.0))
    s1 = State(s=_from_np(s1_vec), u=u_hat, conf=conf1)

    # پیش‌بینی z گام بعد را همان s1 می‌گیریم (latent درون‌زا)
    z1_hat = Latent(_from_np(s1_vec))

    # پاداش مورد انتظار (پیشرفت)
    rhat = _rhat_progress(s_vec, s1_vec)

    return s1, z1_hat, rhat, risk, u_hat

# ------------- اجرای مستقیم برای تست دستی -------------
if __name__ == "__main__":
    # ساخت یک حالت از چند z ساختگی
    z_hist = [Latent([0.1]*64), Latent([0.12]*64), Latent([0.11]*64)]
    s0 = state(z_hist)
    print(f"s0: u={s0.u:.3f}, conf={s0.conf:.3f}, ||s||={math.sqrt(sum(x*x for x in s0.s)):.3f}")

    a = Action(kind="tool", name="invoke_calc", args={"expr": "2+2"})
    s1, z1h, rhat, risk, uhat = predict(s0, a)
    print(f"predict: rhat={rhat:.3f}, risk={risk:.3f}, uhat={uhat:.3f}, ||z1||={math.sqrt(sum(x*x for x in z1h.z)):.3f}")
