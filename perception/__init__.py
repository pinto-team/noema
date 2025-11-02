# -*- coding: utf-8 -*-
"""
NOEMA • perception package

Lightweight interface for the text encoder used across the project.
If a learned model is added later, this API can remain stable.
"""

from .encoder import (
    encode,
    encode_batch,
    normalize_text,
    set_config,
    TextEncoderV0,
    EncoderConfig,
)

__all__ = [
    "encode",
    "encode_batch",
    "normalize_text",
    "set_config",
    "TextEncoderV0",
    "EncoderConfig",
]
