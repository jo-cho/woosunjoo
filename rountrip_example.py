# Library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.pairs_selection import *
from util.backtest import *
from util.bands import *
from util.pf import round_trips as rt

import warnings
warnings.filterwarnings(action='ignore')


if __name__ == '__main__':
    trades_df = pd.read_csv('data/trades_df.csv', index_col=0)

    main_path = '/home/lululalamoon/CHO/chosta/data/k_stocks/daily/ohlcv'

    initial_invest = 1_000_000_000
    cash = 100_000_000
    invest_money = (initial_invest-cash)/30

    all_transactions = []

    for i in range(len(trades_df.asset.unique())):
        trades_df_i = trades_df[trades_df.asset == trades_df.asset.unique()[i]]

        price_pairs = []
        stock_names = list(eval(trades_df.asset.unique()[i]))
        for j in stock_names:
            data = pd.read_csv(main_path + f'/{j}_ohlcv.csv', index_col=0, parse_dates=True)
            price = data.종가.rename(j)
            price_pairs.append(price)

        transactions_list = []
        asset_common = price_pairs[0].name
        asset_preferred = price_pairs[1].name

        for t in trades_df_i.index:
            entry = pd.to_datetime(trades_df_i.entry[t])
            exit = pd.to_datetime(trades_df_i.exit[t])

            position = trades_df_i.position[t]

            entry_price_common = price_pairs[0][entry]
            entry_price_preferred = price_pairs[1][entry]
            exit_price_common = price_pairs[0][exit]
            exit_price_preferred = price_pairs[1][exit]

            one_transaction = [
                [entry, -position * int(invest_money / entry_price_common), entry_price_common, asset_common],
                [entry, position * int(invest_money / entry_price_preferred), entry_price_preferred, asset_preferred],
                [exit, position * int(invest_money / entry_price_common), exit_price_common, asset_common],
                [exit, -position * int(invest_money / entry_price_preferred), exit_price_preferred, asset_preferred]]
            one_transaction = pd.DataFrame(one_transaction, columns=['index', 'amount', 'price', 'symbol']).set_index(
                'index')
            transactions_list.append(one_transaction)

        transactions = pd.concat(transactions_list)
        transactions['txn_dollars'] = transactions.amount * transactions.price * -1
        all_transactions.append(transactions)

    all_transactions = pd.concat(all_transactions).sort_index()
    avail_dates = price[all_transactions.index[0]:all_transactions.index[-1]].index.rename('index')
    symbols = list(all_transactions.symbol.unique())
    positions = pd.DataFrame(index=avail_dates)
    positions['cash'] = cash
    for symb in symbols:
        txn_ = -all_transactions[all_transactions['symbol'] == symb].txn_dollars.rename(symb)
        positions = positions.join(txn_)
    positions = positions.fillna(0)
    returns = positions.sum(axis='columns').pct_change().fillna(0.0)

    rts = rt.extract_round_trips(transactions=all_transactions,
                                 portfolio_value=positions.sum(axis=1) / (returns + 1))

    rt.print_round_trip_stats(rts)