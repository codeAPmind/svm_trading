"""
成交量加权价格指标 (VWAP)

公式:
    VWAP_T = Σ(Ci × Vi) / Σ(Vi)   (i = t-T+1 to t)

优势:
    1. 比普通均线更早获取信号
    2. 高波动性/高流动性股票中尤为有效
    3. 机构资金分析常用基准
"""
import pandas as pd
import numpy as np


def calc_vwap(
    close: pd.Series,
    volume: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    成交量加权移动均价

    Args:
        close:  收盘价
        volume: 成交量
        period: 移动窗口

    Returns:
        VWAP 序列
    """
    cv   = close * volume
    vwap = cv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    return vwap


def calc_vwap_features(
    close: pd.Series,
    volume: pd.Series,
    periods: list = None
) -> pd.DataFrame:
    """
    多周期 VWAP 特征

    Returns:
        DataFrame:
          vwap_{p}       - 各周期 VWAP 绝对值
          vwap_ratio_{p} - 收盘价 / VWAP（>1 偏强）
          vwap_cross_{p} - 穿越信号（1=上穿, -1=下穿, 0=无）
    """
    if periods is None:
        periods = [5, 10, 20, 60]

    features = {}
    for p in periods:
        vwap = calc_vwap(close, volume, p)
        features[f'vwap_{p}']       = vwap
        features[f'vwap_ratio_{p}'] = close / vwap.replace(0, np.nan)

        above = (close > vwap).astype(int)
        cross = above.diff()
        features[f'vwap_cross_{p}'] = cross

    return pd.DataFrame(features, index=close.index)


def calc_volume_features(volume: pd.Series) -> pd.DataFrame:
    """
    成交量衍生特征

    Returns:
        DataFrame: [vol_ma5_ratio, vol_ma20_ratio, vol_change, vol_std_20]
    """
    vol_ma5  = volume.rolling(5).mean().replace(0, np.nan)
    vol_ma20 = volume.rolling(20).mean().replace(0, np.nan)

    return pd.DataFrame({
        'vol_ma5_ratio':  volume / vol_ma5,
        'vol_ma20_ratio': volume / vol_ma20,
        'vol_change':     volume.pct_change(),
        'vol_std_20':     volume.rolling(20).std() / vol_ma20,
    }, index=volume.index)
