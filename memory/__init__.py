# -*- coding: utf-8 -*-
"""
NOEMA • memory package

API :
    from memory import WorkingMemory
    from memory import EpisodeStore, make_key_vector
    from memory import FaissIndex, IndexConfig, build_from_episode_store
"""

# Working Memory
from .wm import WorkingMemory, WMItem

# Episodic Store
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
