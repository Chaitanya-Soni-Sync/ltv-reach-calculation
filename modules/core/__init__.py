#!/usr/bin/env python3
"""
Core Module - Business Logic for LTV Reach Computation
"""

from .ltv_reach import (
    compute_ltv_reach_for_campaigns,
    get_top_25_campaigns_last_28_days,
    get_real_date_range
)

__all__ = [
    # LTV reach functions
    'compute_ltv_reach_for_campaigns',
    'get_top_25_campaigns_last_28_days',
    'get_real_date_range'
]
