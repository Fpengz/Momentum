from os import PathLike
from typing import Literal
import matplotlib.pyplot as plt

from config_loader import load_config_yaml
from logger import setup_logger
from momentum.backtest import cal_bkt
from momentum.data import load_data_df_from_sql
from momentum.position import deltaneutral
from momentum.signal import logreturns
from momentum.portfolio import cal_perf


def pipeline(config_path: str | PathLike,
             instruments: list[str],
             start_date: int = 20180101,
             end_date: int | None = None,
             strategy: str | Literal["simple", "linear", "exponential"] = "simple",
             table: str = "AdjustedFuturesDaily",
             plot: bool = False,
             verbose: bool = False,
             trial_params: dict | None = None) -> dict:
    """
    Run full pipeline: load data, generate signal, positions, backtest, performance.
    """

    logger = setup_logger(verbose, name="pipeline")
    logger.info(f"Pipeline started with strategy={strategy}")

    # Load config and data
    config = load_config_yaml(config_path)
    # Override config with trial_params if provided
    if trial_params is not None:
        if "factor" in trial_params:
            config["factor"].update(trial_params["factor"])
        if "trade" in trial_params:
            config["trade"].update(trial_params["trade"])

    df = load_data_df_from_sql(
        instruments=instruments,
        db_path=config["data"]["db_path"],
        table=table,
        start_date=start_date,
        end_date=end_date,
    )
    data = df["adjclose"]
    logger.info(f"Loaded data: {data.shape[0]} days × {data.shape[1]} instruments")

    # Signal generation
    signal = logreturns(data, config["factor"], mode=strategy)
    logger.info("Signal generation completed")

    # Position sizing
    position = deltaneutral(
        signal,
        config["trade"]["trade_percent"],
        config["trade"]["gross_target"],
        config["trade"]["hold_period"],
    )
    logger.info("Position construction completed")

    # Backtest
    bkt_result = cal_bkt(data, position)
    performance = cal_perf(bkt_result)

    logger.info(f"Performance: Sharpe={performance['sharpe']:.2f}, "
                f"AnnRet={performance['ann_return']:.2%}, "
                f"Calmar={performance['calmar']:.2f}")

    if plot:
        plt.plot(bkt_result["pnl_ptf"].values.cumsum())
        plt.title("Cumulative PnL")
        plt.show()

    return {"performance": performance, "position": position, "bkt_result": bkt_result}
