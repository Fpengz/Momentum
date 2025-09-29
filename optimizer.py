from os import PathLike
from typing import Callable

from logger import setup_logger

import optuna


def optimize_optuna_with_soft_penalties(
    run_pipeline,
    config_path,
    instruments,
    start_date,
    end_date=None,
    n_trials=50,
    verbose=False
):
    """
    Optimize strategy hyperparameters using Optuna with soft penalties.

    Parameters
    ----------
    verbose
    run_pipeline : callable
        Pipeline function that accepts (config_path, instruments, params, start_date, end_date)
        and returns performance metrics dict:
        {
            "sharpe": float,
            "annual_return": float,
            "max_drawdown": float,
            "turnover": float,
            ...
        }
    config_path : str
        Path to config.yaml
    instruments : list[str]
        List of instrument tickers
    start_date : int
        Backtest start date (yyyymmdd)
    end_date : int, optional
        Backtest end date (yyyymmdd)
    n_trials : int, default=50
        Number of trials to run

    Returns
    -------
    best_params : dict or None
    best_perf : dict or None
    """

    def objective(trial):
        # --- Hyperparameters to optimize ---
        params = {
            "window": trial.suggest_int("window", 5, 250),
            "skip": trial.suggest_int("skip", 1, 2),
            "clip": trial.suggest_categorical("clip", [None, 2.5, 3.0]),
            "trade_percent": trial.suggest_float("trade_percent", 0.1, 0.3),
            "gross_target": trial.suggest_float("gross_target", 1.0, 1.5),
            "hold_period": trial.suggest_int("hold_period", 5, 250),
            "strategy": trial.suggest_categorical(
                "strategy", ["simple", "linear", "exponential"]
            ),
        }
        trial_params = {
            "factor": {
                "window": params["window"],
                "skip": params["skip"],
                "clip": params["clip"],
            },
            "trade": {
                "trade_percent": params["trade_percent"],
                "gross_target": params["gross_target"],
                "hold_period": params["hold_period"],
            },
        }
        # --- Run pipeline ---
        perf = run_pipeline(
            config_path=config_path,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            strategy=params["strategy"],
            trial_params=trial_params
        )

        sharpe = perf.get("sharpe", -1e9)
        ann_return = perf.get("annual_return", -1e9)
        max_dd = perf.get("max_drawdown", 1e9)
        turnover = perf.get("turnover", 1e9)

        # --- Scoring with soft penalties ---
        score = sharpe

        if sharpe < 1.0:
            score -= (1.0 - sharpe) * 5

        if ann_return < 0.05:
            score -= (0.05 - ann_return) * 50

        if max_dd > 0.2:
            score -= (max_dd - 0.2) * 10

        if turnover > 0.5:
            score -= (turnover - 0.5) * 5

        # Save performance for inspection later
        trial.set_user_attr("performance", perf)

        return score

    logger = setup_logger(verbose=verbose, name="optuna_optimizer")

    # --- Run optimization ---
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    if len(study.trials) == 0 or all(
        t.state != optuna.trial.TrialState.COMPLETE for t in study.trials
    ):
        logger.warning("No trials completed successfully. Returning None.")
        return None, None

    # --- Best trial ---
    best_trial = study.best_trial
    best_params = best_trial.params
    best_perf = best_trial.user_attrs.get("performance", {})

    # --- Post-filter strict acceptance criteria ---
    if (
        best_perf.get("sharpe", -1e9) < 1.0
        or best_perf.get("annual_return", -1e9) < 0.05
        or best_perf.get("max_drawdown", 1e9) > 0.2
        or best_perf.get("turnover", 1e9) > 0.5
    ):
        logger.warning("Best trial did not pass acceptance criteria. Returning None.")
        return None, None

    return best_params, best_perf


def optimize_optuna(
        run_pipeline: Callable,
        config_path: str | PathLike,
        instruments: list[str],
        start_date: int | None = None,
        end_date=None,
        min_sharpe: float = 1.7,
        n_trials=50,
        verbose=False):
    """

    Parameters
    ----------
    config_path
    instruments
    start_date
    end_date
    min_sharpe
    n_trials
    verbose
    run_pipeline : Callable
    """
    logger = setup_logger(verbose=False, name="optuna_optimizer")
    logger.info(f"Starting Optuna optimization ({n_trials} trials)")
    best_perf_overall = {}

    def objective(trial):
        # --- Hyperparameters to optimize ---
        nonlocal best_perf_overall
        params = {
            "window": trial.suggest_int("window", 5, 250),
            "skip": trial.suggest_int("skip", 1, 2),
            "clip": trial.suggest_categorical("clip", [None, 2.5, 3.0]),
            "trade_percent": trial.suggest_float("trade_percent", 0.1, 0.3),
            "gross_target": trial.suggest_float("gross_target", 1.0, 1.5),
            "hold_period": trial.suggest_int("hold_period", 5, 250),
            "strategy": trial.suggest_categorical(
                "strategy", ["simple", "linear", "exponential"]
            ),
        }
        trial_params = {
            "factor": {
                "window": params["window"],
                "skip": params["skip"],
                "clip": params["clip"],
            },
            "trade": {
                "trade_percent": params["trade_percent"],
                "gross_target": params["gross_target"],
                "hold_period": params["hold_period"],
            },
        }
        # --- Run pipeline ---
        result = run_pipeline(
            config_path=config_path,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            strategy=params["strategy"],
            trial_params=trial_params
        )
        perf = result.get("performance")
        sharpe = perf.get("sharpe", -1e9)

        # Keep track of the best trial even if it fails strict criteria
        if best_perf_overall is None or sharpe > best_perf_overall.get("sharpe", -1e9):
            best_perf_overall = {"sharpe": sharpe, "params": params, "perf": perf}

        # Prune trials that cannot meet minimum Sharpe
        if sharpe < min_sharpe:
            raise optuna.exceptions.TrialPruned()

        # Maximize Sharpe
        return sharpe

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    if len(study.trials) == 0 or all(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials):
        logger.warning("No trials completed successfully. Returning best overall trial.")
        return best_perf_overall.get("params"), best_perf_overall.get("perf")

    # Otherwise return the best successful trial
    best_trial = study.best_trial
    return best_trial.params, best_trial.user_attrs.get("performance", best_trial.value)
