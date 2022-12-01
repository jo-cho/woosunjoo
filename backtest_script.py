# Library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    all_transactions, all_inventories, all_equity_curve, all_pos = get_backtest(pairs_list, ohlcv_dict,
                                                                                initial_invest=100_000_000,
                                                                                transaction_cost=0.002,
                                                                                bb_sl_mult=3,
                                                                                end_date='2022-1-1')

    # 개별 자산 실현손익 누적 그림

    f, axs = plt.subplots(len(all_inventories), figsize=(10, 2 * len(all_inventories)))
    f.suptitle('실현 PNL (각 1억), cost=20bp')
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
    f.suptitle('실현 손익, cost=20bp')
    ax.set_title('각 페어 당 1억 x 30개 페어')
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
    plt.savefig('img/eq_sum_20p.png')
    plt.show()

    ror = (inv_sum[-1]/13)/3_000_000_000
    is_ror = (inv_sum[:'2017-1-1'][-1] / 8) / 3_000_000_000
    oos_ror = ((inv_sum['2017-1-1':][-1] - inv_sum['2017-1-1':][0])/ 6) / 3_000_000_000
    print(ror, is_ror, oos_ror)