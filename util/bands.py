import pandas as pd
import numpy as np

def bollinger_bands(price, n, m):
    """
    :param price:(pd.Series)
    :param n: (int) rolling window
    :param m: (int) multiplier to standard deviation
    :return: (DataFrame) price, bollinger bands (upper, lower, mean)
    """
    df = price.rename('price').to_frame()
    B_MA = pd.Series((price.rolling(n, min_periods=n).mean()), name='B_MA')
    sigma = price.rolling(n, min_periods=n).std()

    BU = pd.Series((B_MA + m * sigma), name='ub')
    BL = pd.Series((B_MA - m * sigma), name='lb')
    BM = pd.Series(B_MA, name='MA')
    df = df.join(BU)
    df = df.join(BL)
    df = df.join(BM)
    return df

def bollinger_bands_double(price, n, m1, m2):
    """
    :param price:(pd.Series)
    :param n: (int) rolling window
    :param m: (int) multiplier to standard deviation
    :return: (DataFrame) price, bollinger bands (upper, lower, mean)
    """
    df = price.rename('price').to_frame()
    B_MA = pd.Series((price.rolling(n, min_periods=n).mean()), name='B_MA')
    sigma = price.rolling(n, min_periods=n).std()

    BU = pd.Series((B_MA + m1 * sigma), name='ub')
    BL = pd.Series((B_MA - m1 * sigma), name='lb')
    BU2 = pd.Series((B_MA + m2 * sigma), name='ub2')
    BL2 = pd.Series((B_MA - m2 * sigma), name='lb2')
    BM = pd.Series(B_MA, name='MA')
    df = df.join(BU)
    df = df.join(BL)
    df = df.join(BU2)
    df = df.join(BL2)
    df = df.join(BM)
    return df

def rsi_bands(price, period, up=70, down=30, bands=True):
    """
    :param price: (Series)
    :param period: (int) windows
    :param up: (int)
    :param down: (int)
    :param bands: (Bool)
    :return: dataframe of rsi, (upper, lower)
    """

    df = price.rename('price').to_frame()
    delta = df.price.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = pd.DataFrame.ewm(u, com=period-1, adjust=False).mean() / \
         pd.DataFrame.ewm(d, com=period-1, adjust=False).mean()
    rsi= 100 - 100 / (1 + rs)
    rsi_df = rsi.rename('rsi').to_frame()
    rsi_df['ub'] = up
    rsi_df['lb'] = down
    if bands is False:
        rsi_df = rsi_df[['rsi']]
    return rsi_df