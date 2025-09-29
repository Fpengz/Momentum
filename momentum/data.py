from __future__ import annotations

from os import PathLike
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


def load_data_df_from_sql(
        instruments: list[str],
        db_path: str | PathLike,
        start_date: int,
        end_date: int | None = None,
        table: str = "AdjustedFuturesDaily",
) -> pd.DataFrame:
    """
    Load a long DataFrame from SQLite for selected instruments and dates.

    Parameters
    ----------
    end_date
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
    params: Mapping[str, object] = {**bind_syms}

    sql = text(f"""
        SELECT *, (ClosePrice * factor_multiply) as adjclose
        FROM {table}
        WHERE TradingDay >= {start_date}
            AND Instrument IN ({in_binds})
            AND method = 'OpenInterest'
    """)

    if end_date:
        sql = text(f"""
            SELECT *, (ClosePrice * factor_multiply) as adjclose
            FROM {table}
            WHERE TradingDay Between {start_date} and {end_date}
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
