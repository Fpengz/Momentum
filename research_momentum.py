# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# momentum 
instruments = ['hc', 'rb', 'i', 'j', 'jm',
               'au', 'ag',
               'v', 'ru', 'l', 'pp', 'bu', 'TA', 'FG', 'MA',
               'y', 'p', 'm', 'a', 'c', 'cs', 'jd', 'RM', 'CF', 'SR', 'OI']

dict_parameter['window'] = 15   # lookback period for momentum
dict_parameter['trade_percent'] = 0.2  # long short percent
dict_parameter['hold_period'] = 2   # days to hold the position

# load data from sqlite
dict_data = load_data_df_from_sql(instruments, start_date=20180101)

data = {}
data['price'] = dict_data['adjclose']

# signal: use price/volume data to calculate signal
signal = logreturns(data, dict_parameter)

# strategy: calculate position based on price and signal
position = deltaneutral(data=data, 
                        signal=signal, 
                        dict_parameter=dict_parameter,                                                        
                        )


# calculate pnl, turnovers, volumes
bkt_result = cal_bkt(dict_data['adjclose'], position)

# calculate backtest performance of the whole portfolio: sharpe ratio, pot
# sharpe ratio = 16 * mean(portfolio daily pnl) / std(portfolio daily pnl)
# pot = sum(portfolio daily pnl) / sum(portfolio daily turnovers) * 10000
performance = cal_perf(bkt_result)

# plot the cumulative pnl
plt.plot(bkt_result['pnl_ptf'].values.cumsum()) 
