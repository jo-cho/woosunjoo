# Library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.bands import *
from util.pairs_selection import *
from util.backtest import *
import matplotlib.ticker as ticker
plt.rcParams["font.family"] = "NanumGothic"
import warnings
warnings.filterwarnings(action='ignore')

# Data load

# 페어 리스트
## 시가총액 상위 5개
pairs_list_5 = [('LG전자', 'LG전자우'),
                ('LG화학', 'LG화학우'),
                ('삼성전자', '삼성전자우'),
                ('삼성화재', '삼성화재우'),
                ('현대차', '현대차우')]

## 15개
pairs_list_15 = [('CJ제일제당', 'CJ제일제당 우'),
 ('LG', 'LG우'),
 ('LG생활건강', 'LG생활건강우'),
 ('LG전자', 'LG전자우'),
 ('LG화학', 'LG화학우'),
 ('NH투자증권', 'NH투자증권우'),
 ('SK이노베이션', 'SK이노베이션우'),
 ('S-Oil', 'S-Oil우'),
 ('미래에셋증권', '미래에셋증권우'),
 ('삼성SDI', '삼성SDI우'),
 ('삼성전자', '삼성전자우'),
 ('삼성화재', '삼성화재우'),
 ('아모레퍼시픽', '아모레퍼시픽우'),
 ('한국금융지주', '한국금융지주우'),
 ('현대차', '현대차우')]

## 30개
pairs_list_30 = [
 ('CJ', 'CJ우'),
 ('CJ제일제당', 'CJ제일제당 우'),
 ('GS', 'GS우'),
 ('JW중외제약', 'JW중외제약우'),
 ('LG', 'LG우'),
 ('LG생활건강', 'LG생활건강우'),
 ('LG전자', 'LG전자우'),
 ('LG화학', 'LG화학우'),
 ('NH투자증권', 'NH투자증권우'),
 ('SK네트웍스', 'SK네트웍스우'),
 ('SK이노베이션', 'SK이노베이션우'),
 ('S-Oil', 'S-Oil우'),
 ('금호석유', '금호석유우'),
 ('넥센타이어', '넥센타이어1우B'),
 ('대상', '대상우'),
 ('대한항공', '대한항공우'),
 ('미래에셋증권', '미래에셋증권우'),
 ('삼성SDI', '삼성SDI우'),
 ('삼성전기', '삼성전기우'),
 ('삼성전자', '삼성전자우'),
 ('삼성화재', '삼성화재우'),
 ('아모레G', '아모레G우'),
 ('아모레퍼시픽', '아모레퍼시픽우'),
 ('유한양행', '유한양행우'),
 ('한국금융지주', '한국금융지주우'),
 ('한화', '한화우'),
 ('한화솔루션', '한화솔루션우'),
 ('현대건설', '현대건설우'),
 ('현대차', '현대차우'),
 ('호텔신라', '호텔신라우')]

# 페어 리스트 정하기
pairs_list = pairs_list_30

if __name__ == "__main__":
    main_path = '/home/lululalamoon/CHO/chosta/data/k_stocks/daily/ohlcv'
    ohlcv_list = []
    price_pairs = []
    stock_names = []
    for pair in pairs_list:
        for j in pair:
            data = pd.read_csv(main_path + f'/{j}_ohlcv.csv', index_col=0, parse_dates=True)
            ohlcv_list.append(data)
            stock_names.append(j)
    ohlcv_dict = dict(zip(stock_names, ohlcv_list))

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
        window = 20
        mult = 2
        mult2 = 3
        df_bb = bollinger_bands_double(spread, window, mult, mult2).dropna()

        sp, lb, ub, ma = df_bb.price, df_bb.lb, df_bb.ub, df_bb.MA
        lb2, ub2 = df_bb.lb2, df_bb.ub2

        # 볼린저 밴드로 진입, 청산 규칙 생성
        # 볼린저 룰 선택

        # trade_dates = get_trade_dates_bbd_m(sp, ub, lb, ub2, lb2, ma)
        # trade_dates = get_trade_dates_bbd_m_long(sp, ub, lb, ub2, lb2, ma)
        trade_dates = get_trade_dates_bbd_long(sp, ub, lb, ub2, lb2)

        initial_invest = 100_000_000 #투자금
        c = 0.003 #거래비용
        transactions_list = []

        inventory_now = 0
        inventory_history = [inventory_now]

        trade_dates_is = trade_dates.loc[(trade_dates.entry > '2009-1-1') & (trade_dates.entry < '2023-1-1')] # in-sample
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

        second_df_ = pd.concat([second[trade_dates_is.entry[0]:], second_amounts, df_['position']], axis=1).fillna(method='ffill')
        first_df_ = pd.concat([first[trade_dates_is.entry[0]:], first_amounts, df_['position']], axis=1).fillna(method='ffill')

        price_ = second_df_.prod(axis=1) - first_df_.prod(axis=1)
        equity_curve = price_.replace(0, np.nan).diff().cumsum().fillna(method='ffill')

        all_transactions.append(transactions) # 거래기록
        all_inventories.append(inventory) # 실현손익 누적
        all_equity_curve.append(equity_curve) # 평가손익
        all_pos.append(df_.position) # 매매 포지션

    # 개별 자산 실현손익 누적 그림

    f, axs = plt.subplots(len(all_inventories), figsize=(10, 2 * len(all_inventories)))
    f.suptitle('실현 PNL (각 1억), cost=30bp')
    for n in range(len(all_inventories)):
        i = all_inventories[n]
        axs[n].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda i,
                                        pos: '{:,.2f}'.format(i / 100_000_000) + '억원'))
        axs[n].plot(i)
        axs[n].legend(i)
    f.tight_layout()
    plt.savefig('img/pnl_30p.png')
    plt.show()

    # 자산 전체 실현손익 누적
    inv_sum = pd.concat(all_inventories, axis=1).fillna(method='ffill').fillna(method='bfill').sum(axis=1)

    f, ax = plt.subplots(1, figsize=(10, 6))
    f.suptitle('실현 PNL (각 페어 당 매매시 1억 x 30개 페어), cost=30bp')
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda inv_sum,
                                    pos: '{:,.2f}'.format(inv_sum / 100_000_000) + '억원'))
    ax.plot(inv_sum)
    f.tight_layout()
    plt.savefig('img/pnl_sum_30p.png')
    plt.show()

    # 개별 자산 평가손익 누적
    n_pair = len(all_equity_curve)
    f, axs = plt.subplots(2*n_pair, figsize=(10, 3*n_pair), gridspec_kw={'height_ratios': [4, 1]*n_pair})
    f.suptitle('평가손익 (1억씩 매매)')
    for n in range(n_pair):
        eq = all_equity_curve[n]
        axs[2*n].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda eq,
                                        pos: f'{eq / 100_000_000}' + '억원'))
        axs[2 * n].plot(eq)
        axs[2 * n].legend([pairs_list[n]])
        axs[2 * n + 1].plot(all_pos[n])
        axs[2 * n + 1].legend(['position'])

    f.tight_layout()
    plt.savefig('img/eq_30p.png')
    plt.show()

    # 전체 자산 평가손익 누적
    eq_sum = pd.concat(all_equity_curve, axis=1).fillna(method='ffill').fillna(method='bfill').sum(axis=1)
    f, ax = plt.subplots(1, figsize=(10, 6))
    f.suptitle('평가손익 (각 페어 당 매매시 1억 x 30개 페어)')
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda eq_sum ,
                                    pos: '{:,.2f}'.format(eq_sum  / 100_000_000) + '억원'))
    ax.plot(eq_sum )
    f.tight_layout()
    plt.savefig('img/eq_sum_30p.png')
    plt.show()