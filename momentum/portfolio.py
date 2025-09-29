import pandas as pd
import numpy as np


def cal_perf(bkt_result: pd.DataFrame) -> dict:
    """
    Calculate performance metrics.
    bkt_result must have column 'pnl_ptf' as daily returns.
    """
    daily_ret = bkt_result["pnl_ptf"]

    # cumulative returns
    cum = (1 + daily_ret).cumprod()

    # max drawdown
    roll_max = cum.cummax()
    dd = (cum / roll_max - 1)
    max_dd = dd.min()

    # annualized stats
    mean_daily = daily_ret.mean()
    vol_daily = daily_ret.std()
    ann_return = (1 + mean_daily) ** 252 - 1
    ann_vol = vol_daily * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "sharpe": sharpe,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_dd": max_dd,
        "calmar": calmar,
    }
