import pandas as pd
import numpy as np

def cal_bkt(data, position):
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
    price_diff = data.diff(periods=1)
    pos_shift = position.shift(1).fillna(0)
    pnl = pos_shift * price_diff
    pnl_ptf = pnl.sum(axis=1)
    turnover = (position - pos_shift).abs().sum(axis=1)
    gross_exposure = position.abs().sum(axis=1)

    return {
        'pnl': pnl,
        'pnl_ptf': pnl_ptf,
        'turnover': turnover,
        'gross_exposure': gross_exposure
    }
