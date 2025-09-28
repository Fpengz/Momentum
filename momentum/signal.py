from __future__ import annotations
import numpy as np
import pandas as pd


def logreturns(data: pd.DataFrame, dict_parameter: dict) -> pd.DataFrame:
    """
    Time-series momentum signal per instrument.

    For date t, lookback L and skip s (default 1 day to avoid look-ahead):
        signal[t] = ln(P_{t-s}) - ln(P_{t-L-s}) = ln(P_{t-1} / P_{t-L-1})

    Parameters
    ----------
    data : pd.DataFrame
        Must contain data['price'] as a wide DataFrame:
        index = dates (ascending), columns = instruments, values > 0.
    dict_parameter : dict
        Required: 'window' (int, lookback L).
        Optional:
          - 'skip' (int, default 1)
          - 'clip' (float, winsorize symmetric bounds, optional)

    Returns
    -------
    pandas.DataFrame
        Momentum signal, same shape as price. First (L+skip) rows are NaN.
    """

    px = data.sort_index()
    if not np.isfinite(px.to_numpy()).all():
        # allow NaNs but not inf/-inf; replace non-positive with NaN
        px = px.where(px > 0)

    lookback = int(dict_parameter.get("window"))
    if lookback <= 0:
        raise ValueError("'window' must be a positive integer")
    s = int(dict_parameter.get("skip", 1))
    if s < 0:
        raise ValueError("'skip' must be >= 0")

    logp = np.log(px)
    signal = logp.shift(s) - logp.shift(lookback + s)

    return signal

