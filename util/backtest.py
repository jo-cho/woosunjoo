import pandas as pd
import numpy as np


def get_trade_dates_bb(spread_series, upper, lower):
    """
    볼린저 밴드 진입, 청산 날짜
    :param spread_series: (pd.Series) 가격
    :param upper: (pd.Series) 위 밴드
    :param lower:  (pd.Series) 아래 밴드
    :return: trade_dates (pd.DataFrame)
    """
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
    """
    볼린저밴드 진입, 청산 날짜,
    평균에서 청산
    :param spread_series: (pd.Series) 가격
    :param upper: (pd.Series) 위 밴드
    :param lower: (pd.Series) 아래 밴드
    :param mean: (pd.Series) 이동평균
    :return: trade dates (pd.DataFrame)
    """
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
    """
    볼린저밴드 진입, 청산 날짜,
    평균에서 청산, 손절 밴드 추가
    :param spread_series: (pd.Series) 가격
    :param upper: (pd.Series) 위 밴드
    :param lower: (pd.Series) 아래 밴드
    :param upper2: (pd.Series) 손절 위 밴드
    :param lower2: (pd.Series) 손절 아래 밴드
    :param mean: (pd.Series) 이동평균
    :return: trade dates (pd.DataFrame)
    """
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


def get_trade_dates_bb_long(spread_series, upper, lower):
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

        if spread_series[i - 1] <= lower[i - 1] and spread_series[i] > lower[i]:
            if position == 0:  # If the position is 0, then long a position on the spread.
                trade_count += 1
                df.loc["Trade " + str(trade_count) + ' Long Open'] = ["Long Open", spread_series.index[i]]
                position = 1

    trades = df
    trades['position'] = 1
    open_dates = trades.Date[::2].array
    close_dates = trades.Date[1::2].array
    positions = trades.position[1::2].array
    if len(open_dates) != len(close_dates):
        open_dates = open_dates[:-1]
    out = pd.DataFrame({'entry': open_dates, 'exit': close_dates, 'position': positions})
    return out


def get_trade_dates_bb_m_long(spread_series, lower, mean):
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

        if spread_series[i - 1] <= lower[i - 1] and spread_series[i] > lower[i]:
            if position == 0:  # If the position is 0, then long a position on the spread.
                trade_count += 1
                df.loc["Trade " + str(trade_count) + ' Long Open'] = ["Long Open", spread_series.index[i]]
                position = 1

    trades = df
    trades['position'] = 1

    open_dates = trades.Date[::2].array
    close_dates = trades.Date[1::2].array
    positions = trades.position[1::2].array
    if len(open_dates) != len(close_dates):
        open_dates = open_dates[:-1]
    out = pd.DataFrame({'entry': open_dates, 'exit': close_dates, 'position': positions})
    return out


def get_trade_dates_bbd_long(spread_series, upper, lower, upper2, lower2):
    position = 0
    trade_count = 0
    # Setting up columns
    columns = ['Type', "Date"]
    df = pd.DataFrame(columns=columns)

    for i in range(1, len(spread_series)):
        if (spread_series[i - 1] < upper[i - 1] and spread_series[i] >= upper[i]) or \
                (spread_series[i - 1] > lower2[i - 1] and spread_series[i] <= lower2[i]):
            if position == 1:  # If the position is 1, then close the existing long position.
                df.loc["Trade " + str(trade_count) + ' Long Close'] = ["Long Close", spread_series.index[i]]
                position = 0

        if spread_series[i - 1] >= lower[i - 1] and spread_series[i] < lower[i]:
            if position == 0:  # If the position is 0, then long a position on the spread.
                trade_count += 1
                df.loc["Trade " + str(trade_count) + ' Long Open'] = ["Long Open", spread_series.index[i]]
                position = 1

    trades = df
    trades['position'] = 1
    open_dates = trades.Date[::2].array
    close_dates = trades.Date[1::2].array
    positions = trades.position[1::2].array
    if len(open_dates) != len(close_dates):
        open_dates = open_dates[:-1]
    out = pd.DataFrame({'entry': open_dates, 'exit': close_dates, 'position': positions})
    return out


def get_trade_dates_bbd_m_long(spread_series, upper, lower, upper2, lower2, mean):
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

        if spread_series[i - 1] >= lower[i - 1] and spread_series[i] < lower[i]:
            if position == 0:  # If the position is 0, then long a position on the spread.
                trade_count += 1
                df.loc["Trade " + str(trade_count) + ' Long Open'] = ["Long Open", spread_series.index[i]]
                position = 1

    trades = df
    trades['position'] = 1
    open_dates = trades.Date[::2].array
    close_dates = trades.Date[1::2].array
    positions = trades.position[1::2].array
    if len(open_dates) != len(close_dates):
        open_dates = open_dates[:-1]
    out = pd.DataFrame({'entry': open_dates, 'exit': close_dates, 'position': positions})
    return out


def get_trade_dates_sma_long(spread_series, entry_sma, exit_sma):
    position = 0
    trade_count = 0
    # Setting up columns
    columns = ['Type', "Date"]
    df = pd.DataFrame(columns=columns)

    for i in range(1, len(spread_series)):
        if exit_sma[i - 1] < spread_series[i - 1] and exit_sma[i] >= spread_series[i]:
            if position == 1:  # If the position is 1, then close the existing long position.
                df.loc["Trade " + str(trade_count) + ' Long Close'] = ["Long Close", spread_series.index[i]]
                position = 0

        if entry_sma[i - 1] >= spread_series[i - 1] and entry_sma[i] < spread_series[i]:
            if position == 0:  # If the position is 0, then long a position on the spread.
                trade_count += 1
                df.loc["Trade " + str(trade_count) + ' Long Open'] = ["Long Open", spread_series.index[i]]
                position = 1

    trades = df
    trades['position'] = 1
    open_dates = trades.Date[::2].array
    close_dates = trades.Date[1::2].array
    positions = trades.position[1::2].array
    if len(open_dates) != len(close_dates):
        open_dates = open_dates[:-1]
    out = pd.DataFrame({'entry': open_dates, 'exit': close_dates, 'position': positions})
    return out