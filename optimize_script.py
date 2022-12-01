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

# 페어 선정 없는 버전

# Data load
pairs_list_30 = [#('CJ', 'CJ4우(전환)'),
 ('CJ', 'CJ우'),
 ('CJ제일제당', 'CJ제일제당 우'),
 #('DB하이텍', 'DB하이텍1우'),
 #('DL이앤씨', 'DL이앤씨2우(전환)'),
 #('DL이앤씨', 'DL이앤씨우'),
 ('GS', 'GS우'),
 #('JW중외제약', 'JW중외제약2우B'),
 ('JW중외제약', 'JW중외제약우'),
 ('LG', 'LG우'),
 ('LG생활건강', 'LG생활건강우'),
 ('LG전자', 'LG전자우'),
 ('LG화학', 'LG화학우'),
 #('LX하우시스', 'LX하우시스우'),
 ('NH투자증권', 'NH투자증권우'),
 #('SK', 'SK우'),
 ('SK네트웍스', 'SK네트웍스우'),
 ('SK이노베이션', 'SK이노베이션우'),
 ('S-Oil', 'S-Oil우'),
 ('금호석유', '금호석유우'),
 ('넥센타이어', '넥센타이어1우B'),
 ('대상', '대상우'),
 ('대한항공', '대한항공우'),
 #('미래에셋증권', '미래에셋증권2우B'),
 ('미래에셋증권', '미래에셋증권우'),
 ('삼성SDI', '삼성SDI우'),
 #('삼성물산', '삼성물산우B'),
 ('삼성전기', '삼성전기우'),
 ('삼성전자', '삼성전자우'),
 ('삼성화재', '삼성화재우'),
 #('아모레G', '아모레G3우(전환)'),
 ('아모레G', '아모레G우'),
 ('아모레퍼시픽', '아모레퍼시픽우'),
 ('유한양행', '유한양행우'),
 #('하이트진로', '하이트진로2우B'),
 ('한국금융지주', '한국금융지주우'),
 #('한화', '한화3우B'),
 ('한화', '한화우'),
 ('한화솔루션', '한화솔루션우'),
 ('현대건설', '현대건설우'),
 #('현대차', '현대차2우B'),
 #('현대차', '현대차3우B'),
 ('현대차', '현대차우'),
 ('호텔신라', '호텔신라우')]

pairs_list_5 = [('LG전자', 'LG전자우'),
                ('LG화학', 'LG화학우'),
                ('삼성전자', '삼성전자우'),
                ('삼성화재', '삼성화재우'),
                ('현대차', '현대차우')]

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
    opt_window_list=[]

    for i in range(len(pairs_list)):

        common = pairs_list[i][0]
        preferred = pairs_list[i][1]

        common_price = ohlcv_dict[common]['종가'].rename(common)  # 수정종가 사용
        preferred_price = ohlcv_dict[preferred]['종가'].rename(preferred)
        pairs_prices = pd.concat([common_price, preferred_price], axis=1)

        # get spread
        first = pairs_prices.dropna().iloc[:, 0]  # 보통주
        second = pairs_prices.dropna().iloc[:, 1]  # 우선주
        spread = get_log_spread(first, second)  # lny-lnx

        # 최적 파라미터 window 정하기
        last_invens = []
        windows = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        for win in windows:
            # make bands
            window = win
            mult = 2
            mult2 = 4
            df_bb = bollinger_bands_double(spread, window, mult, mult2).dropna()

            sp, lb, ub, ma = df_bb.price, df_bb.lb, df_bb.ub, df_bb.MA
            lb2, ub2 = df_bb.lb2, df_bb.ub2

            # trade_dates = get_trade_dates_bbd_m(sp, ub, lb, ub2, lb2, ma)
            # trade_dates = get_trade_dates_bbd_m_long(sp, ub, lb, ub2, lb2, ma)
            # trade_dates = get_trade_dates_bb_m_long(sp, lb, ma)
            # trade_dates = get_trade_dates_bb_long(sp, ub, lb)
            trade_dates = get_trade_dates_bbd_long(sp, ub, lb, ub2, lb2)

            initial_invest = 100_000_000
            c = 0.002  # 거래비
            transactions_list = []

            inventory_now = 0
            inventory_history = [inventory_now]

            trade_dates_is = trade_dates.loc[(trade_dates.entry > '2009-1-1') & (trade_dates.entry < '2017-1-1')] # in-sample

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

            ind_ = [list(trade_dates_is.entry)[0]] + list(trade_dates_is.exit)
            inventory = pd.Series(inventory_history, index=pd.to_datetime(ind_))
            inventory = inventory.resample('D').last().ffill()
            inventory = inventory.to_frame().rename(
                columns={0: common + ' & ' + preferred})

            last_invens.append(inventory.iloc[-1, 0])
        opt_window = windows[np.argmax(np.array(last_invens))]
        opt_window_list.append(opt_window)
        print(common, '&', preferred, "'s Optimal window: ", opt_window)
    
    # 최적 파라미터로 인샘플 결과 보기용
    for i in range(len(pairs_list)):

        common = pairs_list[i][0]
        preferred = pairs_list[i][1]

        common_price = ohlcv_dict[common]['종가'].rename(common)  # 수정종가 사용
        preferred_price = ohlcv_dict[preferred]['종가'].rename(preferred)
        pairs_prices = pd.concat([common_price, preferred_price], axis=1)

        # get spread
        first = pairs_prices.dropna().iloc[:, 0]  # 보통주
        second = pairs_prices.dropna().iloc[:, 1]  # 우선주
        spread = get_log_spread(first, second)  # lny-lnx

        last_invens = []

        # 밴드 생성
        window = opt_window_list[i]
        mult = 2
        mult2 = 4
        df_bb = bollinger_bands_double(spread, window, mult, mult2).dropna()

        sp, lb, ub, ma = df_bb.price, df_bb.lb, df_bb.ub, df_bb.MA
        lb2, ub2 = df_bb.lb2, df_bb.ub2

        ## 볼린저 밴드로 진입, 청산 규칙 생성
        ## 볼린저 룰 선택
        # trade_dates = get_trade_dates_bbd_m(sp, ub, lb, ub2, lb2, ma)
        # trade_dates = get_trade_dates_bbd_m_long(sp, ub, lb, ub2, lb2, ma)
        # trade_dates = get_trade_dates_bb_m_long(sp, lb, ma)
        # trade_dates = get_trade_dates_bb_long(sp, ub, lb)
        trade_dates = get_trade_dates_bbd_long(sp, ub, lb, ub2, lb2)

        initial_invest = 100_000_000 #매매 투자금
        c = 0.002  # 거래비용
        transactions_list = []

        inventory_now = 0
        inventory_history = [inventory_now]

        trade_dates_is = trade_dates.loc[(trade_dates.entry > '2009-1-1') & (trade_dates.entry < '2017-1-1')] # in-sample
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

        ind_ = [list(trade_dates_is.entry)[0]] + list(trade_dates_is.exit)
        inventory = pd.Series(inventory_history, index=pd.to_datetime(ind_))
        inventory = inventory.resample('D').last().ffill()
        inventory = inventory.to_frame().rename(
            columns={0: common + ' & ' + preferred})

        all_transactions.append(transactions)
        all_inventories.append(inventory)

    f, axs = plt.subplots(len(all_inventories), figsize=(10, 2 * len(all_inventories)))
    f.suptitle('optimal params (in-sample) 실현 손익, cost=20bp')
    for n in range(len(all_inventories)):
        i = all_inventories[n]
        axs[n].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda i,
                                        pos: '{:,.2f}'.format(i / 100_000_000) + '억원'))
        axs[n].plot(i)
        axs[n].legend(i)
    f.tight_layout()
    plt.savefig('img/inven_opt_is.png')
    plt.show()

    inv_sum = pd.concat(all_inventories, axis=1).fillna(method='ffill').fillna(method='bfill').sum(axis=1)

    f, ax = plt.subplots(1, figsize=(10, 6))
    f.suptitle('optimal params (in-sample): 실현손익, cost=20bp')
    ax.set_title('각 페어 당 6억 x Top 5 시가총액 페어 (2년 주기로 재선정)')
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda inv_sum,
                                    pos: '{:,.2f}'.format(inv_sum / 100_000_000) + '억원'))
    ax.plot(inv_sum)
    f.tight_layout()
    plt.savefig('img/inven_sum_opt_is.png')
    plt.show()

    # Out-of-Sample 결과
    
    all_transactions = []
    all_inventories = []
    for i in range(len(pairs_list)):

        common = pairs_list[i][0]
        preferred = pairs_list[i][1]

        common_price = ohlcv_dict[common]['종가'].rename(common)  # 수정종가 사용
        preferred_price = ohlcv_dict[preferred]['종가'].rename(preferred)
        pairs_prices = pd.concat([common_price, preferred_price], axis=1)

        # get spread
        first = pairs_prices.dropna().iloc[:, 0]  # 보통주
        second = pairs_prices.dropna().iloc[:, 1]  # 우선주
        spread = get_log_spread(first, second)  # lny-lnx

        # make bands
        window = opt_window_list[i]
        mult = 2
        mult2 = 4
        df_bb = bollinger_bands_double(spread, window, mult, mult2).dropna()

        sp, lb, ub, ma = df_bb.price, df_bb.lb, df_bb.ub, df_bb.MA
        lb2, ub2 = df_bb.lb2, df_bb.ub2

        # trade_dates = get_trade_dates_bbd_m(sp, ub, lb, ub2, lb2, ma)
        # trade_dates = get_trade_dates_bbd_m_long(sp, ub, lb, ub2, lb2, ma)
        # trade_dates = get_trade_dates_bb_m_long(sp, lb, ma)
        # trade_dates = get_trade_dates_bb_long(sp, ub, lb)
        trade_dates = get_trade_dates_bbd_long(sp, ub, lb, ub2, lb2)

        initial_invest = 100_000_000
        c = 0.002  # 거래당
        transactions_list = []

        inventory_now = 0
        inventory_history = [inventory_now]

        trade_dates_oos = trade_dates.loc[(trade_dates.entry > '2009-1-1') & (trade_dates.entry < '2022-1-1')] # 전체

        for t in trade_dates_oos.index:  # t는 하나의 round-trip
            entry = pd.to_datetime(trade_dates_oos.entry[t])
            exit = pd.to_datetime(trade_dates_oos.exit[t])

            pos = trade_dates_oos.position[t]

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

        ind_ = [list(trade_dates_oos.entry)[0]] + list(trade_dates_oos.exit)
        inventory = pd.Series(inventory_history, index=pd.to_datetime(ind_))
        inventory = inventory.resample('D').last().ffill()
        inventory = inventory.to_frame().rename(
            columns={0: common + ' & ' + preferred})

        all_transactions.append(transactions)
        all_inventories.append(inventory)

    f, axs = plt.subplots(len(all_inventories), figsize=(10, 2 * len(all_inventories)))
    f.suptitle('optimal params: 실현손익, cost=20bp')
    ax.set_title('각 페어 당 1억 x 30개 페어')
    for n in range(len(all_inventories)):
        i = all_inventories[n]
        axs[n].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda i,
                                        pos: '{:,.2f}'.format(i / 100_000_000) + '억원'))
        axs[n].plot(i)
        axs[n].legend(i)
        axs[n].axvline(pd.DatetimeIndex(['2017-1-1']), color='red')
    f.tight_layout()
    plt.savefig('img/inven_opt_oos.png')
    plt.show()

    inv_sum = pd.concat(all_inventories, axis=1).fillna(method='ffill').fillna(method='bfill').sum(axis=1)

    f, ax = plt.subplots(1, figsize=(10, 6))
    f.suptitle('실현손익 (optimal params), cost=20bp')
    ax.set_title('각 페어 당 1억 x 30개 페어')
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda inv_sum,
                                    pos: '{:,.2f}'.format(inv_sum / 100_000_000) + '억원'))
    ax.plot(inv_sum)
    f.tight_layout()
    plt.axvline(pd.DatetimeIndex(['2017-1-1']), color='red')
    plt.savefig('img/inven_sum_opt_oos.png')
    plt.show()

    ror = (inv_sum[-1]/13)/3_000_000_000
    is_ror = (inv_sum[:'2017-1-1'][-1] / 8) / 3_000_000_000
    oos_ror = ((inv_sum['2017-1-1':][-1] - inv_sum['2017-1-1':][0])/ 6) / 3_000_000_000
    print(ror, is_ror, oos_ror)