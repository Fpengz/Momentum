# Momentum Futures Strategy Report

## 1. Strategy Logic

### a) Universe

- 26 futures instruments:  
  `['hc', 'rb', 'i', 'j', 'jm', 'au', 'ag', 'v', 'ru', 'l', 'pp', 'bu', 'TA', 'FG', 'MA', 'y', 'p', 'm', 'a', 'c', 'cs', 'jd', 'RM', 'CF', 'SR', 'OI']`.
- Data range: 2018-01-01 to 2019-01-01.
- Adjusted for contract factors:  
  `adjclose = ClosePrice * factor_multiply`.

### b) Signal Generation

- **Time-series momentum** via log returns:
    $$signal_t = ln(P_{t-s}) - ln(P_{t-L-s})$$

- `L` = lookback window (5–250 days)
- `s` = skip (1–2 days to avoid look-ahead)
- Optional **winsorization** to cap extreme signals.
- Variants:
1. **Simple:** equal weighting over lookback.
2. **Linear decay:** higher weight to recent returns.
3. **Exponential decay:** exponentially higher weight to recent returns.

### c) Position Construction
- **Delta-neutral**:
- Long top `x%` instruments, short bottom `x%`.
- Demeaned to net ≈ 0.
- Scaled to **gross exposure target** (`gross_target`).
- Holding period applied (`hold_period` days).

### d) Backtest Calculation

- Daily portfolio PnL:
- Turnover
- Gross exposure
- 
```python
pnl_ptf = (position.shift(1) * data.pct_change()).sum(axis=1)
cum_ret = (1 + pnl_ptf).cumprod()
```

## 2. Backtest Result

| Logger           | Logging Level | Logging Message                                                                                                                                                          |
| ---------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| optuna_optimizer | WARNING       | No trials completed successfully. Returning best overall trial.                                                                                                          |
| main             | INFO          | === Optimization completed ===                                                                                                                                           |
| main             | INFO          | Best parameters: {'window': 82, 'skip': 2, 'clip': 2.5, 'trade_percent': 0.2176111833133985, 'gross_target': 1.3008424988171972, 'hold_period': 5, 'strategy': 'linear'} |
| main             | INFO          | Best performance: Sharpe=1.24, AnnRet=9.56%, Calmar=1.17                                                                                                                 |
| main             | INFO          | === Running best trial for detailed report ===                                                                                                                           |


## 5. Improvement Ideas

### a) Signal Enhancements

Multi-period momentum (short + medium + long windows) with weighted aggregation.

Volatility-adjusted signals to normalize across instruments.

Include cross-asset correlations to reduce crash risk.

### b) Position Sizing

- Risk parity weighting instead of uniform gross exposure.

- Smooth positions across days to limit turnover.

- Dynamic gross target based on market volatility.

### c) Backtest Enhancements

- Include transaction costs and slippage.
- Walk-forward analysis to avoid overfitting.
- Track per-instrument performance to identify weak/strong assets.

- d) Optimization

- Find the best parameters combination using grid search
- Find the best parameters combination using Optuna


## Installation

git clone https://github.com/Fpengz/Momentum.git
cd Momentum

install the relevant packages

run `uv run main.py`