from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def logreturns(
    data: pd.DataFrame,
    dict_parameter: dict,
    mode: Literal["simple", "linear", "exponential"] = "simple"
) -> pd.DataFrame:
    """
    Flexible momentum signal generator per instrument.

    Supports:
        - unweighted log-return
        - linear-decay weighted log-return
        - exponential-decay weighted log-return

    Parameters
    ----------
    mode : Literal["simple", "linear", "exponential"]
    data : pd.DataFrame
        Price DataFrame (dates × instruments), all > 0
    dict_parameter : dict
        Required:
            'window' (int): lookback period L
        Optional:
            'skip' (int, default=1): how many days to skip to avoid lookahead
            'clip' (float, optional): symmetric winsorization bound
            'mode' (str): 'simple', 'linear', 'ewm'
            'alpha' (float): decay factor for 'ewm' mode

    Returns
    -------
    pd.DataFrame
        Momentum signal (dates × instruments), NaN for initial rows
    """
    px = data.sort_index()
    px = px.where(px > 0)  # replace non-positive with NaN

    lookback = dict_parameter.get("window", 5)
    if lookback <= 0:
        raise ValueError("'window' must be positive")
    skip = int(dict_parameter.get("skip", 1))
    if skip < 0:
        raise ValueError("'skip' must be >= 0")

    clip = dict_parameter.get("clip", None)

    log_p = np.log(px)
    signal = pd.DataFrame(0.0, index=px.index, columns=px.columns)

    if mode == "simple":
        # standard past-R log return
        signal = log_p.shift(skip) - log_p.shift(lookback + skip)

    elif mode == "linear":
        # linear decay weights
        log_return = log_p - log_p.shift(1)
        weights = np.arange(lookback, 0, -1).astype(float)
        weights /= weights.sum()

        signal = (
            log_return
            .rolling(lookback)
            .apply(lambda x: np.dot(x, weights), raw=True)
        )

    elif mode == "exponential":
        alpha = float(dict_parameter.get("alpha", 0.2))
        log_return = log_p - log_p.shift(1)
        signal = log_return.ewm(alpha=alpha, adjust=False).mean().shift(skip)

    else:
        raise ValueError("Unknown mode, choose 'simple', 'linear', or 'exponential'")

    # optional clipping
    if clip is not None:
        clip = float(clip)
        if clip <= 0:
            raise ValueError("'clip' must be positive if provided")
        signal = signal.clip(lower=-clip, upper=clip)

    return signal
