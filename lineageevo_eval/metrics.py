"""Shared IC and signal utilities."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def filter_by_datetime(data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if "datetime" not in data.index.names:
        raise ValueError("Qlib data index must contain a 'datetime' level")
    dates = pd.to_datetime(data.index.get_level_values("datetime"))
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return data.loc[mask]


def clean_factor_frame(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_index().replace([float("inf"), float("-inf")], float("nan")).dropna()


def ic_metrics(data: pd.DataFrame, method: str) -> tuple[float, float]:
    if data.empty:
        raise ValueError("factor/label data is empty")
    daily_ic = data.groupby(level="datetime").apply(
        lambda frame: frame["factor"].corr(frame["label"], method=method)
    )
    daily_ic = daily_ic.dropna()
    if daily_ic.empty:
        raise ValueError("IC series is empty")
    mean_ic = float(daily_ic.mean())
    std_ic = float(daily_ic.std(ddof=1))
    if not math.isfinite(mean_ic):
        raise ValueError("IC mean is not finite")
    icir = 0.0 if std_ic == 0 or not math.isfinite(std_ic) else mean_ic / std_ic
    return mean_ic, float(icir)


def daily_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    std = frame.std(ddof=0)
    return (frame - frame.mean()) / std.replace(0, float("nan"))


def flatten_risk_analysis(analysis: Any) -> dict[str, Any]:
    raw = analysis.to_dict() if hasattr(analysis, "to_dict") else dict(analysis)
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat[str(inner_key)] = inner_value
        else:
            flat[str(key)] = value
    return flat


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.fillna(0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.min())
