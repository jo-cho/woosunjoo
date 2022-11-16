import pandas as pd
import numpy as np

def get_trade_dates_bb(spread_series, upper, lower):
    position = 0
    trade_count = 0
    # Setting up columns
    columns = ['Type', "Date"]
    df = pd.DataFrame(columns=columns)

    for i in range(1, len(spread_series)):
        if spread_series[i - 1] < upper[i - 1] and spread_series[i] >= upper[i]:
            if position == 1:  # If the position is 1, then close the existing long position.
                df.loc["Trade " + str(trade_count) + ' Long Close'] = ["Long Close", spread_series.index[i]]
                position = 0

        if spread_series[i - 1] > lower[i - 1] and spread_series[i] <= lower[i]:
            if position == -1:  # If the position is -1, then close the existing short position.
                df.loc["Trade " + str(trade_count) + ' Short Close'] = ["Short Close", spread_series.index[i]]
                position = 0

        if spread_series[i - 1] >= upper[i - 1] and spread_series[i] < upper[i]:
            if position == 0:  # If the position is 0, then short a position on the spread.
                trade_count += 1
                df.loc["Trade " + str(trade_count) + ' Short Open'] = ["Short Open", spread_series.index[i]]
                position = -1

        if spread_series[i - 1] <= lower[i - 1] and spread_series[i] > lower[i]:
            if position == 0:  # If the position is 0, then long a position on the spread.
                trade_count += 1
                df.loc["Trade " + str(trade_count) + ' Long Open'] = ["Long Open", spread_series.index[i]]
                position = 1

    trades = df
    trades['position'] = 1
    trades['position'].loc[trades.Type == 'Short Open'] = -1
    trades['position'].loc[trades.Type == 'Short Close'] = -1
    open_dates = trades.Date[::2].array
    close_dates = trades.Date[1::2].array
    positions = trades.position[1::2].array
    if len(open_dates) != len(close_dates):
        open_dates = open_dates[:-1]
    out = pd.DataFrame({'entry': open_dates, 'exit': close_dates, 'position': positions})
    return out

def get_trade_dates_bb_m(spread_series, upper, lower, mean):
    position = 0
    trade_count = 0
    # Setting up columns
    columns = ['Type', "Date"]
    df = pd.DataFrame(columns=columns)

    for i in range(1, len(spread_series)):
        if spread_series[i - 1] < mean[i - 1] and spread_series[i] >= mean[i]:
            if position == 1:  # If the position is 1, then close the existing long position.
                df.loc["Trade " + str(trade_count) + ' Long Close'] = ["Long Close", spread_series.index[i]]
                position = 0

        if spread_series[i - 1] > mean[i - 1] and spread_series[i] <= mean[i]:
            if position == -1:  # If the position is -1, then close the existing short position.
                df.loc["Trade " + str(trade_count) + ' Short Close'] = ["Short Close", spread_series.index[i]]
                position = 0

        if spread_series[i - 1] <= upper[i - 1] and spread_series[i] > upper[i]:
            if position == 0:  # If the position is 0, then short a position on the spread.
                trade_count += 1
                df.loc["Trade " + str(trade_count) + ' Short Open'] = ["Short Open", spread_series.index[i]]
                position = -1

        if spread_series[i - 1] >= lower[i - 1] and spread_series[i] < lower[i]:
            if position == 0:  # If the position is 0, then long a position on the spread.
                trade_count += 1
                df.loc["Trade " + str(trade_count) + ' Long Open'] = ["Long Open", spread_series.index[i]]
                position = 1

    trades = df
    trades['position'] = 1
    trades['position'].loc[trades.Type == 'Short Open'] = -1
    trades['position'].loc[trades.Type == 'Short Close'] = -1
    open_dates = trades.Date[::2].array
    close_dates = trades.Date[1::2].array
    positions = trades.position[1::2].array
    if len(open_dates) != len(close_dates):
        open_dates = open_dates[:-1]
    out = pd.DataFrame({'entry': open_dates, 'exit': close_dates, 'position': positions})
    return out

def get_trade_dates_bbd_m(spread_series, upper, lower, upper2, lower2, mean):
    position = 0
    trade_count = 0
    # Setting up columns
    columns = ['Type', "Date"]
    df = pd.DataFrame(columns=columns)

    for i in range(1, len(spread_series)):
        if (spread_series[i - 1] < mean[i - 1] and spread_series[i] >= mean[i]) or \
                (spread_series[i - 1] > lower2[i - 1] and spread_series[i] <= lower2[i]):
            if position == 1:  # If the position is 1, then close the existing long position.
                df.loc["Trade " + str(trade_count) + ' Long Close'] = ["Long Close", spread_series.index[i]]
                position = 0

        if spread_series[i - 1] > mean[i - 1] and spread_series[i] <= mean[i] or \
                (spread_series[i - 1] < upper2[i - 1] and spread_series[i] >= upper2[i]):
            if position == -1:  # If the position is -1, then close the existing short position.
                df.loc["Trade " + str(trade_count) + ' Short Close'] = ["Short Close", spread_series.index[i]]
                position = 0

        if spread_series[i - 1] <= upper[i - 1] and spread_series[i] > upper[i]:
            if position == 0:  # If the position is 0, then short a position on the spread.
                trade_count += 1
                df.loc["Trade " + str(trade_count) + ' Short Open'] = ["Short Open", spread_series.index[i]]
                position = -1

        if spread_series[i - 1] >= lower[i - 1] and spread_series[i] < lower[i]:
            if position == 0:  # If the position is 0, then long a position on the spread.
                trade_count += 1
                df.loc["Trade " + str(trade_count) + ' Long Open'] = ["Long Open", spread_series.index[i]]
                position = 1

    trades = df
    trades['position'] = 1
    trades['position'].loc[trades.Type == 'Short Open'] = -1
    trades['position'].loc[trades.Type == 'Short Close'] = -1
    open_dates = trades.Date[::2].array
    close_dates = trades.Date[1::2].array
    positions = trades.position[1::2].array
    if len(open_dates) != len(close_dates):
        open_dates = open_dates[:-1]
    out = pd.DataFrame({'entry': open_dates, 'exit': close_dates, 'position': positions})
    return out
