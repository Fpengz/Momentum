from __future__ import annotations

import numpy as np
import pandas as pd


def deltaneutral(data: pd.DataFrame,
                 trade_percent: float = 0.2,
                 gross_target: float = 1.0,
                 hold_period: int = 2) -> pd.DataFrame:
    """
    Vectorized delta-neutral portfolio weights with holding period.

    Parameters
    ----------
    data : pd.DataFrame
        Signal matrix (index = dates, columns = instruments).
    trade_percent : float
        Fraction of instruments to long/short.
    gross_target : float
        Target gross exposure per date.
    hold_period : int
        Number of days to hold positions (default=1 → daily rebalance).
    """
    # Compute quantile thresholds
    low = data.quantile(trade_percent, axis=1)
    high = data.quantile(1 - trade_percent, axis=1)

    # Raw positions
    longs = (data.ge(high, axis=0)).astype(float)
    shorts = (data.le(low, axis=0)).astype(float) * -1
    w = longs + shorts

    # Demean
    w = w.sub(w.mean(axis=1), axis=0)

    # Scale gross exposure
    gross = w.abs().sum(axis=1)
    scale = gross_target / gross.replace(0, np.nan)
    w = w.mul(scale, axis=0).fillna(0.0)

    # Apply holding period smoothing
    if hold_period > 1:
        w = w.rolling(hold_period).mean()

    return w
