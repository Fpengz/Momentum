import numpy as np


def cal_perf(bkt_result, periods_per_year=252):
    """
    Compute performance metrics for the backtest.

    Parameters
    ----------
    bkt_result : dict
        Output from cal_bkt

    Returns
    -------
    dict:
        'sharpe' : annualized Sharpe ratio
        'pot' : PnL per turnover
    """
    daily_pnl = bkt_result['pnl_ptf']
    turnover = bkt_result['turnover']

    # Sharpe ratio using configurable annualization factor (default: daily data)
    volatility = daily_pnl.std()
    if volatility == 0:
        sharpe = 0.0
    else:
        annualization_factor = np.sqrt(periods_per_year)
        sharpe = annualization_factor * daily_pnl.mean() / volatility

    # POT = sum(PnL) / sum(Turnover) * 10000
    pot = daily_pnl.sum() / turnover.sum() * 10000

    return {
        'sharpe': sharpe,
        'pot': pot
    }
