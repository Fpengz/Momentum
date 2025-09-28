def cal_perf(bkt_result):
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

    # Sharpe ratio (assuming 16 trading periods per month ~ daily annualized factor)
    sharpe = 16 * daily_pnl.mean() / daily_pnl.std()

    # POT = sum(PnL) / sum(Turnover) * 10000
    pot = daily_pnl.sum() / turnover.sum() * 10000

    return {
        'sharpe': sharpe,
        'pot': pot
    }
