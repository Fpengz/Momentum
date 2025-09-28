from config_loader import load_config_yaml
from momentum.backtest import cal_bkt
from momentum.data import load_data_df_from_sql
from momentum.position import deltaneutral
from momentum.signal import logreturns
from momentum.portfolio import cal_perf

from matplotlib import pyplot as plt


def main():
    config = load_config_yaml("config.yaml")
    instruments = ['hc', 'rb', 'i', 'j', 'jm', 'au', 'ag', 'v', 'ru', 'l', 'pp', 'bu', 'TA', 'FG', 'MA',
                   'y', 'p', 'm', 'a', 'c', 'cs', 'jd', 'RM', 'CF', 'SR', 'OI']
    start_date = 20180101
    table = "AdjustedFuturesDaily"
    df = load_data_df_from_sql(instruments, start_date, config["data"]["db_path"], table)
    data = df["adjclose"]
    signal = logreturns(data, config["factor"])
    position = deltaneutral(signal,
                            config["trade"]["trade_percent"],
                            config["trade"]["gross_target"],
                            config["trade"]["hold_period"])
    bkt_result = cal_bkt(data, position)
    performance = cal_perf(bkt_result)
    print(performance)
    plt.plot(bkt_result['pnl_ptf'].values.cumsum())
    plt.show()


if __name__ == "__main__":
    main()
