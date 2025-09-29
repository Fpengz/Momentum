import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from logger import setup_logger
from research_momentum import pipeline, cal_bkt
from optimizer import optimize_optuna


def plot_performance(bkt_result, title="Strategy Performance"):
    """
    Plot cumulative PnL, rolling Sharpe, and drawdown.
    """
    pnl_ptf = bkt_result['pnl_ptf']
    cum_pnl = pnl_ptf.cumsum()

    # Rolling Sharpe (window=20 days)
    rolling_sharpe = pnl_ptf.rolling(20).mean() / pnl_ptf.rolling(20).std() * np.sqrt(252)

    # Drawdown
    roll_max = cum_pnl.cummax()
    drawdown = cum_pnl / roll_max - 1

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=16)

    axes[0].plot(cum_pnl, color="blue")
    axes[0].set_ylabel("Cumulative PnL")

    axes[1].plot(rolling_sharpe, color="green")
    axes[1].axhline(1.0, color="red", linestyle="--", label="Sharpe=1")
    axes[1].set_ylabel("Rolling Sharpe")
    axes[1].legend()

    axes[2].plot(drawdown, color="red")
    axes[2].set_ylabel("Drawdown")
    axes[2].set_xlabel("Date")

    plt.tight_layout()
    plt.show()


def main():
    logger = setup_logger(verbose=True, name="main")

    # === Configuration ===
    config_path = "config.yaml"
    instruments = [
        'hc', 'rb', 'i', 'j', 'jm', 'au', 'ag', 'v', 'ru', 'l', 'pp', 'bu', 'TA', 'FG', 'MA',
        'y', 'p', 'm', 'a', 'c', 'cs', 'jd', 'RM', 'CF', 'SR', 'OI'
    ]
    start_date = 20180101
    end_date = 20190101
    strategy = "simple"

    n_trials = 50

    # === Run baseline pipeline ===
    logger.info("=== Baseline pipeline (exponential) ===")
    baseline_perf = pipeline(
        config_path=config_path,
        instruments=instruments,
        start_date=start_date,
        end_date=end_date,
        strategy=strategy,
        plot=False  # disable plotting here
    )
    logger.info(f"Baseline performance: Sharpe={baseline_perf['sharpe']:.2f}, "
                f"AnnRet={baseline_perf['ann_return']:.2%}, "
                f"Calmar={baseline_perf['calmar']:.2f}")

    # === Run Optuna multi-criteria optimization ===
    logger.info("=== Starting multi-criteria optimization ===")
    best_params, best_perf = optimize_optuna(
        run_pipeline=pipeline,
        config_path=config_path,
        instruments=instruments,
        start_date=start_date,
        end_date=end_date,
        n_trials=n_trials,
        verbose=False
    )

    logger.info("=== Optimization completed ===")
    logger.info(f"Best parameters: {best_params}")

    if not best_perf or not best_params:
        return

    logger.info(f"Best performance: Sharpe={best_perf.get('sharpe', 0):.2f}, "
                f"AnnRet={best_perf.get('ann_return', 0):.2%}, "
                f"Calmar={best_perf.get('calmar', 0):.2f}")

    # === Run pipeline with best parameters to get full backtest ===
    trial_params = {
        "factor": {"window": best_params.get("window"),
                   "skip": best_params.get("skip"),
                   "clip": best_params.get("clip")},
        "trade": {"trade_percent": best_params.get("trade_percent"),
                  "gross_target": best_params.get("gross_target"),
                  "hold_period": best_params.get("hold_period")}
    }

    logger.info("=== Running best trial for detailed report ===")
    df = pipeline(
        config_path=config_path,
        instruments=instruments,
        strategy=best_params.get("strategy", strategy),
        start_date=start_date,
        end_date=end_date,
        plot=False,
        trial_params=trial_params
    )

    # Assuming pipeline returns performance, we recalc backtest for plotting
    # If your pipeline returns only performance, you may need to run the full pipeline steps here
    bkt_result = cal_bkt(df["adjclose"], df["position"])

    plot_performance(bkt_result, title="Best Trial Strategy Performance")


if __name__ == "__main__":
    main()
