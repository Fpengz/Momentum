from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import pandas as pd
from sqlalchemy import create_engine, text


def load_data_df_from_sql(
        instruments: list[str],
        start_date: int,
        db_path: str | Path,
        table: str = "AdjustedFuturesDaily",
) -> pd.DataFrame:
    """
    Load a long DataFrame from SQLite for selected instruments and dates.

    Parameters
    ----------
    instruments : list[str]
        Non-empty list of instrument codes (e.g., ['rb','hc',...]).
    start_date : int
        Lower bound on TradingDay in YYYYMMDD form (e.g., 20180101).
    db_path : str | Path
        Path to the SQLite .db file.
    table : str
        Table name.

    Returns
    -------
    pandas.DataFrame
    """

    if not instruments:
        raise ValueError("instruments must be a non-empty list.")
    if not (isinstance(start_date, int) and 19000101 <= start_date <= 21001231):
        raise ValueError("start_date must be an int in YYYYMMDD form, e.g. 20180101.")

    db_path = Path(db_path).resolve()
    engine = create_engine(f"sqlite:///{db_path.as_posix()}")

    in_binds = ", ".join([f":sym{i}" for i in range(len(instruments))])
    bind_syms: Mapping[str, str] = {f"sym{i}": s for i, s in enumerate(instruments)}
    params: Mapping[str, object] = {"start": start_date, **bind_syms}
    sql = text(f"""
        SELECT *, (ClosePrice * factor_multiply) as adjclose
        FROM {table}
        WHERE TradingDay >= :start
            AND Instrument IN ({in_binds})
            AND method = 'OpenInterest'
    """)

    with engine.begin() as conn:
        df = pd.read_sql(sql, conn, params=params)

    if df.empty:
        raise RuntimeError(
            f"No rows returned. Check table='{table}', date>={start_date}, and instruments list."
        )

    return df.pivot(index="TradingDay", columns="Instrument")
