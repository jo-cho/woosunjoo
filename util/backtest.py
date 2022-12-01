import pandas as pd
import numpy as np
from util.bands import *
from util.pairs_selection import *

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

#=======================#


def get_backtest(pairs_list, ohlcv_dict,
                 bb_window=20, bb_mult=2, bb_sl_mult=3,
                 transaction_cost=0.003, initial_invest=100_000_000,
                 start_date='2009-1-1', end_date='2023-1-1'):
    all_transactions = []
    all_inventories = []
    all_equity_curve = []
    all_pos = []

    for i in range(len(pairs_list)):
        common = pairs_list[i][0]
        preferred = pairs_list[i][1]

        common_price = ohlcv_dict[common]['종가'].rename(common)  # 수정종가 사용
        preferred_price = ohlcv_dict[preferred]['종가'].rename(preferred)
        pairs_prices = pd.concat([common_price, preferred_price], axis=1)

        # get spread
        first = pairs_prices.dropna().iloc[:, 0]  # 보통주
        second = pairs_prices.dropna().iloc[:, 1]  # 우선주
        spread = get_log_spread(first, second)  # lny-lnx (x:우선주, y:보통주)

        # 밴드 생성
        window = bb_window
        mult = bb_mult
        mult2 = bb_sl_mult
        df_bb = bollinger_bands_double(spread, window, mult, mult2).dropna()

        sp, lb, ub, ma = df_bb.price, df_bb.lb, df_bb.ub, df_bb.MA
        lb2, ub2 = df_bb.lb2, df_bb.ub2

        # 볼린저 밴드로 진입, 청산 규칙 생성
        # 볼린저 룰 선택

        # trade_dates = get_trade_dates_bbd_m(sp, ub, lb, ub2, lb2, ma)
        # trade_dates = get_trade_dates_bbd_m_long(sp, ub, lb, ub2, lb2, ma)
        trade_dates = get_trade_dates_bbd_long(sp, ub, lb, ub2, lb2)

        c = transaction_cost  # 거래비용
        transactions_list = []

        inventory_now = 0
        inventory_history = [inventory_now]

        trade_dates_is = trade_dates.loc[(trade_dates.entry > start_date) & (trade_dates.entry < end_date)]  # in-sample
        trade_dates_is = trade_dates_is.reset_index(drop=True)

        for t in trade_dates_is.index:  # t는 하나의 round-trip
            entry = pd.to_datetime(trade_dates_is.entry[t])
            exit = pd.to_datetime(trade_dates_is.exit[t])

            pos = trade_dates_is.position[t]

            p1_in = first[entry]
            p2_in = second[entry]
            p1_out = first[exit]
            p2_out = second[exit]
            q1 = int(initial_invest / (2 * p1_in))
            q2 = int(initial_invest / (2 * p2_in))

            one_transaction = [[entry, -pos * q1, p1_in, common],
                               [entry, pos * q2, p2_in, preferred],
                               [exit, pos * q1, p1_out, common],
                               [exit, -pos * q2, p2_out, preferred]]
            one_transaction = pd.DataFrame(one_transaction, columns=['index', 'amount', 'price', 'symbol']).set_index(
                'index')

            pnl1_ = (-pos) * (q1 * p1_out - q1 * p1_in)
            cost1_ = c * (q1 * p1_out + q1 * p1_in)
            pnl2_ = pos * (q2 * p2_out - q2 * p2_in)
            cost2_ = c * (q2 * p2_out + q2 * p2_in)

            pnl1 = pnl1_ - cost1_
            pnl2 = pnl2_ - cost2_

            inventory_now += (pnl1 + pnl2)
            inventory_history.append(inventory_now)
            transactions_list.append(one_transaction)

        transactions = pd.concat(transactions_list)

        # 실현 손익
        ind_ = [list(trade_dates_is.entry)[0]] + list(trade_dates_is.exit)
        inventory = pd.Series(inventory_history, index=pd.to_datetime(ind_))
        inventory = inventory.resample('D').last().ffill()
        inventory = inventory.to_frame().rename(
            columns={0: common + ' & ' + preferred})

        # 평가 손익
        df_ = (second - first).rename('price').to_frame()[trade_dates_is.entry[0]:].copy()
        df_['position'] = 0
        for k in trade_dates_is.index:
            df_.loc[trade_dates_is.entry[k]:trade_dates_is.exit[k], 'position'] = 1
        second_amounts = abs(transactions[transactions.symbol == preferred].amount)
        second_amounts = second_amounts[~second_amounts.index.duplicated(keep='last')]
        first_amounts = abs(transactions[transactions.symbol == common].amount)
        first_amounts = first_amounts[~first_amounts.index.duplicated(keep='last')]

        second_df_ = pd.concat([second[trade_dates_is.entry[0]:], second_amounts, df_['position']], axis=1).fillna(
            method='ffill')
        first_df_ = pd.concat([first[trade_dates_is.entry[0]:], first_amounts, df_['position']], axis=1).fillna(
            method='ffill')

        price_ = second_df_.prod(axis=1) - first_df_.prod(axis=1)
        equity_curve = price_.replace(0, np.nan).diff().cumsum().fillna(method='ffill')

        all_transactions.append(transactions)  # 거래기록
        all_inventories.append(inventory)  # 실현손익 누적
        all_equity_curve.append(equity_curve)  # 평가손익
        all_pos.append(df_.position)  # 매매 포지션

    return all_transactions, all_inventories, all_equity_curve, all_pos