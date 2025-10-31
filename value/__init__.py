# -*- coding: utf-8 -*-
"""
NOEMA • value package
- سیگنال‌های ارزش/پاداش و شکل‌دهی آن‌ها.

استفاده:
    from value import RewardSpec, get_default_spec
    from value import intrinsic_from_errors, intrinsic_from_features
    from value import combine_rewards, shape_bonus, safe_combine_dict
"""

from .reward import (
    RewardSpec,
    get_default_spec,
    intrinsic_from_errors,
    intrinsic_from_features,
    combine_rewards,
    shape_bonus,
    safe_combine_dict,
)

__all__ = [
    "RewardSpec",
    "get_default_spec",
    "intrinsic_from_errors",
    "intrinsic_from_features",
    "combine_rewards",
    "shape_bonus",
    "safe_combine_dict",
]
