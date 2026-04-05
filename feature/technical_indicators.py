"""
技术指标计算模块
包含: MACD, RSI, KDJ, CCI, 布林带, ATR
纯 pandas/numpy 实现，无需 TA-Lib
"""
import numpy as np
import pandas as pd


def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    平滑异同平均线 MACD

    Returns:
        DataFrame: [macd_dif, macd_dea, macd_hist]
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = 2 * (dif - dea)

    return pd.DataFrame({
        'macd_dif':  dif,
        'macd_dea':  dea,
        'macd_hist': hist,
    }, index=close.index)


def calc_rsi(close: pd.Series, periods: list = None) -> pd.DataFrame:
    """
    相对强弱指标 RSI（多周期）

    Returns:
        DataFrame: [rsi_6, rsi_12, rsi_24]
    """
    if periods is None:
        periods = [6, 12, 24]
    result = {}
    for period in periods:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        result[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    return pd.DataFrame(result, index=close.index)


def calc_kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3
) -> pd.DataFrame:
    """
    随机指标 KDJ

    Returns:
        DataFrame: [kdj_k, kdj_d, kdj_j]
    """
    lowest_low    = low.rolling(window=n).min()
    highest_high  = high.rolling(window=n).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    rsv   = (close - lowest_low) / denom * 100

    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d

    return pd.DataFrame({'kdj_k': k, 'kdj_d': d, 'kdj_j': j}, index=close.index)


def calc_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.DataFrame:
    """
    顺势指标 CCI

    Returns:
        DataFrame: [cci]
    """
    tp   = (high + low + close) / 3
    ma   = tp.rolling(window=period).mean()
    mad  = tp.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    cci  = (tp - ma) / (0.015 * mad.replace(0, np.nan))
    return pd.DataFrame({'cci': cci}, index=close.index)


def calc_bollinger(
    close: pd.Series,
    period: int = 20,
    std_dev: int = 2
) -> pd.DataFrame:
    """
    布林带 (Bollinger Bands)

    Returns:
        DataFrame: [bb_upper, bb_middle, bb_lower, bb_width, bb_pctb]
    """
    ma  = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper  = ma + std_dev * std
    lower  = ma - std_dev * std
    width  = (upper - lower) / ma.replace(0, np.nan)
    pctb   = (close - lower) / (upper - lower).replace(0, np.nan)

    return pd.DataFrame({
        'bb_upper':  upper,
        'bb_middle': ma,
        'bb_lower':  lower,
        'bb_width':  width,
        'bb_pctb':   pctb,
    }, index=close.index)


def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.DataFrame:
    """
    平均真实波幅 ATR

    Returns:
        DataFrame: [atr]
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return pd.DataFrame({'atr': atr}, index=close.index)


# ─────────────────────────────────────────────────────────
# 线性回归类指标：LSMA 与 TSF
# ─────────────────────────────────────────────────────────
def _linear_regression_series(y: pd.Series, length: int) -> pd.Series:
    """
    对滚动窗口做一元线性回归：y ~ a + b * t
    返回各窗口末端点的拟合值 y_hat_last。
    """
    if length <= 1:
        return y.copy()
    t = np.arange(length, dtype=float)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    def reg_last(arr: np.ndarray) -> float:
        y_arr = arr.astype(float)
        y_mean = y_arr.mean()
        cov_ty = ((t - t_mean) * (y_arr - y_mean)).sum()
        b = cov_ty / t_var if t_var != 0 else 0.0
        a = y_mean - b * t_mean
        # 末端点的 t = length-1
        return a + b * (length - 1)
    return y.rolling(window=length).apply(lambda x: reg_last(x), raw=True)


def calc_lsma(close: pd.Series, length: int = 25) -> pd.Series:
    """
    LSMA (Least Squares Moving Average): 最小二乘移动平均
    计算方法：对长度为 length 的滚动窗口线性回归，取末端点拟合值形成平滑序列。
    """
    return _linear_regression_series(close, length).rename(f'lsma_{length}')


def calc_tsf(close: pd.Series, length: int = 9, forecast: int = 7) -> pd.Series:
    """
    TSF (Time Series Forecast): 时间序列线性回归外推
    计算方法：在长度为 length 的窗口上拟合 y ~ a + b*t，并对未来 forecast 步做外推。
    """
    if length <= 1:
        return close.copy().rename(f'tsf_{length}_{forecast}')
    t = np.arange(length, dtype=float)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    def reg_forecast(arr: np.ndarray) -> float:
        y_arr = arr.astype(float)
        y_mean = y_arr.mean()
        cov_ty = ((t - t_mean) * (y_arr - y_mean)).sum()
        b = cov_ty / t_var if t_var != 0 else 0.0
        a = y_mean - b * t_mean
        return a + b * (length - 1 + forecast)
    return close.rolling(window=length).apply(lambda x: reg_forecast(x), raw=True).rename(f'tsf_{length}_{forecast}')
