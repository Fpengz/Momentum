from typing import Any

import pandas as pd
from pandas import DataFrame, Series


def cal_bkt(data: pd.DataFrame,
            position: pd.DataFrame) -> dict:
    """
    Vectorized backtest: daily PnL, portfolio PnL, turnover, gross exposure.

    Parameters
    ----------
    data : pd.DataFrame
        Prices of assets (dates × instruments)
    position : pd.DataFrame
        Positions held per asset (dates × instruments)

    Returns
    -------
    dict:
        'pnl' : pd.DataFrame of daily PnL per asset
        'pnl_ptf' : pd.Series, total daily PnL
        'turnover' : pd.Series, daily sum of abs(position changes)
        'gross_exposure' : pd.Series, daily sum of abs(positions)
    """
    returns = data.pct_change().fillna(0)
    pos_shift = position.shift(1).fillna(0)

    pnl = pos_shift * returns
    pnl_ptf = pnl.sum(axis=1)
    turnover = (position - pos_shift).abs().sum(axis=1)
    gross_exposure = position.abs().sum(axis=1)

    return {
        'pnl': pnl,
        'pnl_ptf': pnl_ptf,
        'turnover': turnover,
        'gross_exposure': gross_exposure
    }
