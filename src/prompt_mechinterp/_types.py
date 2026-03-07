"""Shared type definitions for cross-module dict structures."""

from typing import TypedDict


class RegionInfo(TypedDict, total=False):
    """Token-level region boundary from engine output.

    Always has tok_start and tok_end in practice. Optionally has n_tokens
    (sometimes derived as tok_end - tok_start instead).
    Uses total=False because n_tokens is not always present.
    """
    tok_start: int
    tok_end: int
    n_tokens: int


class CharRegionInfo(TypedDict):
    """Character-level region boundary from prep/regions.py annotate_text()."""
    char_start: int
    char_end: int


class CookingStats(TypedDict):
    """Cooking curve statistics from metrics.cooking_curve_stats().

    Returned by cooking_curve_stats() in analysis/metrics.py.
    """
    peak_layer: int
    peak_value: float
    terminal_value: float
    retention_ratio: float


class ExtendedCookingStats(CookingStats, total=False):
    """Extended stats from report.compute_cooking_table().

    Inherits all CookingStats fields and adds optional extras.
    """
    peak_terminal_ratio: float
    story: str
    n_samples: int
    trajectory: list[float]


class ContextBleedResult(TypedDict):
    """Result from report.compute_context_bleed()."""
    mean_ratio: float
    median_ratio: float
    pct_above_2x: float
    conv_turns_mean: float
    current_message_mean: float
    n_samples: int


class TokenRect(TypedDict):
    """Layout rectangle for a single token from render/_shared.layout_tokens()."""
    x: float
    y: float
    w: float
    h: float
    color: tuple[int, int, int]
    fg: str
    text: str
    token_idx: int
