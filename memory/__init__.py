# -*- coding: utf-8 -*-
"""
NOEMA • memory package
- حافظه‌ی کاری، اپیزودیک، و ایندکس برداری را یک‌جا اکسپورت می‌کند.
- هدف: سادگی import در سایر بلوک‌ها (control/world/sleep/…).

APIهای اصلی:
    from memory import WorkingMemory
    from memory import EpisodeStore, make_key_vector
    from memory import FaissIndex, IndexConfig, build_from_episode_store
"""

# Working Memory (کوتاه‌مدت)
from .wm import WorkingMemory, WMItem

# Episodic Store (فایل‌محور)
from .episodic import EpisodeStore, Episode, make_key_vector

# ANN Index (FAISS یا fallback)
from .index_faiss import (
    FaissIndex,
    IndexConfig,
    build_from_episode_store,
)

__all__ = [
    # wm
    "WorkingMemory", "WMItem",
    # episodic
    "EpisodeStore", "Episode", "make_key_vector",
    # index
    "FaissIndex", "IndexConfig", "build_from_episode_store",
]
