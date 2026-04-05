import numpy as np
import pandas as pd


def _tsrank(series: pd.Series, n: int) -> pd.Series:
    # 返回当前值在过去 n 个值中的排名（1..n），用时序近似
    return series.rolling(n).apply(lambda x: pd.Series(x).rank().iloc[-1], raw=False)


def alpha1(close: pd.Series, open_: pd.Series, volume: pd.Series, window: int = 6) -> pd.Series:
    """
    Alpha1：-1 * CORR(RANK(Δlog(Volume), 1), RANK((Close-Open)/Open), 6)
    注：RANK 为截面排名，简化为时间序近似
    """
    log_vol_change = np.log(volume.replace(0, np.nan)).diff(1)
    body_ret = (close - open_) / open_.replace(0, np.nan)
    rank_vol = log_vol_change.rank()
    rank_ret = body_ret.rank()
    return -1 * rank_vol.rolling(window).corr(rank_ret).rename('alpha1')


def alpha5(high: pd.Series, volume: pd.Series,
           rank_window: int = 5, corr_window: int = 5, max_window: int = 3) -> pd.Series:
    """
    Alpha5：-1 * TSMAX(CORR(TSRANK(Volume, 5), TSRANK(High, 5), 5), 3)
    """
    ts_vol  = _tsrank(volume, rank_window)
    ts_high = _tsrank(high,   rank_window)
    corr    = ts_vol.rolling(corr_window).corr(ts_high)
    return (-1 * corr.rolling(max_window).max()).rename('alpha5')


def alpha42(high: pd.Series, volume: pd.Series, window: int = 10) -> pd.Series:
    """
    Alpha42：-1 * RANK(STD(High, 10)) * CORR(High, Volume, 10)
    注：RANK 截面排名简化为时间序百分位
    """
    std_high  = high.rolling(window).std()
    rank_std  = std_high.rank(pct=True)
    corr_hv   = high.rolling(window).corr(volume)
    return (-1 * rank_std * corr_hv).rename('alpha42')

