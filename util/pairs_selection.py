import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression


class PairsSelection:

    def __init__(self, pairs_list, ohlcv_dict
                 ):
        """
        :param pairs_list: list of pairs(tuples)
        :param ohlcv_dict: dict {(string) stock name: (DataFrame) OHLCV}
        """
        self._pairs_list = pairs_list
        self._ohlcv_dict = ohlcv_dict
        self._threshold_adf = None
        self._threshold_hurst = None
        self._selected_pairs_df = None
        self._selected_volume_pairs_df = None
        self._selected_volatility_pairs_df = None
        self._selected_volume_volatility_pairs_df = None

        pairs_prices_list = []
        for i in range(len(pairs_list)):
            common = pairs_list[i][0]
            preferred = pairs_list[i][1]
            common_p = ohlcv_dict[common]['종가'].rename(common)  # 수정종가 사용
            preferred_p = ohlcv_dict[preferred]['종가'].rename(preferred)
            pairs_prices = pd.concat([common_p, preferred_p], axis=1)
            pairs_prices_list.append(pairs_prices)

        self.pairs_prices_list = pairs_prices_list

    def get_selected_pairs(self,
                           lookback=100, step=120,
                           threshold_adf=0.10, threshold_hurst=0.40):
        selected_pairs_list = []
        test_index_list = []
        self._threshold_adf = threshold_adf
        self._threshold_hurst = threshold_hurst
        pairs_prices_list = self.pairs_prices_list
        all_dates = pd.concat(pairs_prices_list, axis=1).dropna().index
        for n in range(lookback, len(all_dates), step):
            start = all_dates[n-lookback]
            end = all_dates[n]
            selected_pairs = get_mean_reverting_pairs(
                pairs_prices_list,
                start,
                end,
                threshold_adf,
                threshold_hurst)
            selected_pairs_list.append(selected_pairs)
            test_index_list.append(end)
        selected_pairs_df = pd.DataFrame({'pairs':selected_pairs_list}, index=test_index_list)
        self._selected_pairs_df = selected_pairs_df
        return selected_pairs_df

    def get_top_hurst_pairs(self, start_date='2009-1-1',
                            lookback=100, step=120,
                            top_n=5):
        selected_pairs_list = []
        top_pairs_list = []
        test_index_list = []
        pairs_prices_list = self.pairs_prices_list
        all_dates = pd.concat(pairs_prices_list, axis=1)[start_date:].dropna().index
        for n in range(lookback, len(all_dates), step):
            start = all_dates[n-lookback]
            end = all_dates[n]
            selected_pairs, hurst_dict = get_pairs_hurst(pairs_prices_list, start, end, threshold=1)
            top_pairs = list(pd.DataFrame(hurst_dict, index=['Hurst']).T.sort_values('Hurst').index[:top_n])
            top_pairs_list.append(top_pairs)
            test_index_list.append(end)
        selected_pairs_df = pd.DataFrame({'pairs':top_pairs_list}, index=test_index_list)
        self._selected_pairs_df = selected_pairs_df
        return selected_pairs_df
##-------------------------------##


def get_coint_spread(x, y):
    lpx = np.log(x)
    lpy = np.log(y)
    reg = LinearRegression()
    x_constant = pd.concat([lpx, pd.Series([1]*len(x), index = lpx.index)], axis=1)
    reg.fit(x_constant, lpy)
    beta = reg.coef_[0]
    alpha = reg.intercept_
    spread = lpy - lpx*beta - alpha
    return spread


def get_log_spread(x: pd.Series,
                   y: pd.Series) -> pd.Series:
    lpx = np.log(x); lpy = np.log(y)
    spread = lpy - lpx
    return spread


def adf_test_pvalue(timeseries):
    dftest = adfuller(timeseries, autolag="AIC")
    pvalue = dftest[1]
    return pvalue


def get_hurst_exponent(time_series, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]


def get_pairs_adf_test(pairs_prices_list, start, end, threshold=0.05, coint_spread=False):
    """
    :param pairs_prices_list: a list of DataFrame's of two assets' price
    :param start: Date
    :param end: Date
    :param threshold: int
    :return: selected spreads dictionary
    """
    spread_list = []
    pairs_list = []
    for i in range(len(pairs_prices_list)):
        first = np.log(pairs_prices_list[i][start:end].dropna().iloc[:,0])
        second = np.log(pairs_prices_list[i][start:end].dropna().iloc[:,1])
        spread = get_log_spread(first, second)
        if coint_spread is True:
            spread = get_coint_spread(first, second)
        spread_list.append(spread)
        c_ = pairs_prices_list[i].columns
        pairs_name = (c_[0], c_[1])
        pairs_list.append(pairs_name)
    spread_dict = dict(zip(pairs_list, spread_list))

    pv_list=[]
    for i in range(len(spread_list)):
        pv = adf_test_pvalue(spread_list[i])
        pv_list.append(pv)
    pv_dict = dict(zip(pairs_list,pv_list))
    tmp_df = pd.DataFrame(pv_dict, index=['p_value of adf']).T
    selected_pairs = tmp_df[tmp_df['p_value of adf'] <= threshold].index.to_list()

    selected_spread_list = []
    for i in range(len(selected_pairs)):
        selected_spread = spread_dict[selected_pairs[i]]
        selected_spread_list.append(selected_spread)
    selected_spread_dict = dict(zip(selected_pairs, selected_spread_list))
    return selected_spread_dict


def get_pairs_hurst(pairs_prices_list, start, end, threshold=0.4):
    """
    :param pairs_prices_list: a list of DataFrame's of two assets' price
    :param start: Date
    :param end: Date
    :param threshold: int
    :return: selected spreads dictionary
    """
    spread_list = []
    pairs_list = []
    for i in range(len(pairs_prices_list)):
        first = np.log(pairs_prices_list[i][start:end].dropna().iloc[:,0])
        second = np.log(pairs_prices_list[i][start:end].dropna().iloc[:,1])
        spread = get_log_spread(first, second)
        spread_list.append(spread)
        c_ = pairs_prices_list[i].columns
        pairs_name = (c_[0], c_[1])
        pairs_list.append(pairs_name)
    spread_dict = dict(zip(pairs_list, spread_list))

    hurst_list=[]
    for i in range(len(spread_list)):
        hurst = get_hurst_exponent(spread_list[i].values)
        hurst_list.append(hurst)
    hurst_dict = dict(zip(pairs_list, hurst_list))
    tmp_df = pd.DataFrame(hurst_dict, index=['Hurst exponent']).T
    selected_pairs = tmp_df[tmp_df['Hurst exponent'] <= threshold].index.to_list()
    selected_spread_list = []
    for i in range(len(selected_pairs)):
        selected_spread = spread_dict[selected_pairs[i]]
        selected_spread_list.append(selected_spread)
    selected_spread_dict = dict(zip(selected_pairs, selected_spread_list))
    return selected_spread_dict, hurst_dict


def get_mean_reverting_pairs(pairs_prices_list, start, end, threshold_adf, threshold_hurst):
    pairs_adf = get_pairs_adf_test(pairs_prices_list, start=start, end=end, threshold=threshold_adf)
    pairs_hurst = get_pairs_hurst(pairs_prices_list, start=start, end=end, threshold=threshold_hurst)
    selected_pairs = list(pairs_adf.keys() & pairs_hurst.keys())
    return selected_pairs