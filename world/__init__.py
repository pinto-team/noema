# -*- coding: utf-8 -*-
"""
NOEMA • world package
- رابط عمومی مدلِ جهان برای سایر ماژول‌ها.
- API ثابت:
    state(z_hist: List[Latent]) -> State
    predict(s: State, a: Action) -> Tuple[State, Latent, float, float, float]
"""

from .dynamics import state, predict, State, Latent, Action

__all__ = ["state", "predict", "State", "Latent", "Action"]
