# -*- coding: utf-8 -*-
"""
NOEMA • perception package
- رابط ساده برای استفاده از رمزگذار متنی در بقیه‌ی پروژه.
- اگر بعداً مدل آموختنی بارگذاری شد، همین API ثابت می‌ماند.

استفاده:
    from perception import encode, encode_batch, set_config
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
