import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "NanumGothic"
from util.pairs_selection import *
from util.backtest import *
from util.bands import *
import warnings
warnings.filterwarnings(action='ignore')
import pickle

if __name__ == '__main__':
    pairs_list = [#('CJ', 'CJ4우(전환)'),
     ('CJ', 'CJ우'),
     ('CJ제일제당', 'CJ제일제당 우'),
     ('DB하이텍', 'DB하이텍1우'),
     #('DL이앤씨', 'DL이앤씨2우(전환)'),
     #('DL이앤씨', 'DL이앤씨우'),
     ('GS', 'GS우'),
     #('JW중외제약', 'JW중외제약2우B'),
     ('JW중외제약', 'JW중외제약우'),
     ('LG', 'LG우'),
     ('LG생활건강', 'LG생활건강우'),
     ('LG전자', 'LG전자우'),
     ('LG화학', 'LG화학우'),
    # ('LX하우시스', 'LX하우시스우'),
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
    print('Get OHLCV - done')

    # Pairs selection
    selection = PairsSelection(pairs_list=pairs_list, ohlcv_dict=ohlcv_dict)

    all_pairs_df = selection.get_selected_pairs(threshold_adf=1, threshold_hurst=1)
    print('Get All Pairs - done')
    selected_pairs_df = selection.get_selected_pairs()
    print('Get Selected Pairs - done')
    print()

    pairs_selection_method = ['No Condition',
                              'ADF & Hurst']

    use = pairs_selection_method[0]

    if use == 'ADF & Hurst':
        moving_pairs = selected_pairs_df['pairs']
    elif use == 'No Condition':
        moving_pairs = all_pairs_df['pairs']

    print(f"Pairs selection method = {use}")

    trade_dates_all = []
    used_pairs_list_all = []

    for t in range(len(moving_pairs) - 1):
        start = moving_pairs.index[t]
        end = moving_pairs.index[t + 1]
        print()
        print(f"<<{start}부터 {end}까지 (진입)>>")
        trade_dates_all_pairs = []
        used_pairs_list = []
        for i in range(len(moving_pairs[t])):
            using_pairs = moving_pairs[t][i]
            common = using_pairs[0]
            preferred = using_pairs[1]
            # print()
            print(f"--- 사용페어: {common} & {preferred} ---")
            common_price = ohlcv_dict[common]['종가'].rename(common)  # 수정종가 사용
            preferred_price = ohlcv_dict[preferred]['종가'].rename(preferred)
            pairs_prices = pd.concat([common_price, preferred_price], axis=1)

            # get spread
            first = pairs_prices.dropna().iloc[:, 0]  # 보통주
            second = pairs_prices.dropna().iloc[:, 1]  # 우선주
            spread = get_log_spread(first, second)  # lny-lnx

            # make bands
            window = 20
            mult = 2
            mult2 = 2.5
            df_bb = bollinger_bands_double(spread, window, mult, mult2).dropna()

            sp, lb, ub, ma = df_bb.price, df_bb.lb, df_bb.ub, df_bb.MA
            lb2, ub2 = df_bb.lb2, df_bb.ub2
            #trade_dates = get_trade_dates_bb_m(sp, ub, lb, ma)
            #trade_dates = get_trade_dates_bbd_m(sp, ub, lb, ub2, lb2, ma)
            #trade_dates = get_trade_dates_bb_long(sp, ub, lb)
            #trade_dates = get_trade_dates_bb_m_long(sp, lb, ma)
            trade_dates = get_trade_dates_bbd_long(sp, ub, lb, ub2, lb2)
            #trade_dates = get_trade_dates_bbd_m_long(sp, ub, lb, ub2, lb2, ma)

            trade_dates = trade_dates[(trade_dates.entry >= start) & (trade_dates.entry < end)].reset_index(drop=True)

            if len(trade_dates) > 0:
                trade_dates_all_pairs.append(trade_dates)
                used_pairs_list.append(using_pairs)
        trade_dates_all.append(trade_dates_all_pairs)
        used_pairs_list_all.append(used_pairs_list)
    print()
    print('Get Trading Results - done')
    print()

    for i in range(len(trade_dates_all)):
        for j in range(len(trade_dates_all[i])):
            trade_dates_all[i][j]['asset'] = [used_pairs_list_all[i][j]] * len(trade_dates_all[i][j])
    trade_dates_all = [ele for ele in trade_dates_all if ele != []]
    list_ = []
    for i in range(len(trade_dates_all)):
        d = pd.concat(trade_dates_all[i]).sort_values(['entry', 'exit']).reset_index(drop=True)
        list_.append(d)
    all_trades_df = pd.concat(list_).reset_index(drop=True)
    all_trades_df = all_trades_df.sort_values(['entry', 'exit'])


    all_trades_df.to_csv("data/trades_df.csv")