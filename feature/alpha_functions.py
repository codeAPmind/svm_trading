"""
Alpha因子计算基础函数库（V2）
实现报告附录中的常用基础函数
"""
import pandas as pd
import numpy as np
from typing import Union


def DELAY(A: pd.Series, n: int) -> pd.Series:
    return A.shift(n)


def DELTA(A: pd.Series, n: int) -> pd.Series:
    return A - A.shift(n)


def RANK(A: pd.Series) -> pd.Series:
    return A.rank(pct=True)


def SUM(A: pd.Series, n: int) -> pd.Series:
    return A.rolling(window=n, min_periods=n).sum()


def MEAN(A: pd.Series, n: int) -> pd.Series:
    return A.rolling(window=n, min_periods=n).mean()


def STD(A: pd.Series, n: int) -> pd.Series:
    return A.rolling(window=n, min_periods=n).std()


def CORR(A: pd.Series, B: pd.Series, n: int) -> pd.Series:
    return A.rolling(window=n, min_periods=n).corr(B)


def COVIANCE(A: pd.Series, B: pd.Series, n: int) -> pd.Series:
    return A.rolling(window=n, min_periods=n).cov(B)


def TSMAX(A: pd.Series, n: int) -> pd.Series:
    return A.rolling(window=n, min_periods=n).max()


def TSMIN(A: pd.Series, n: int) -> pd.Series:
    return A.rolling(window=n, min_periods=n).min()


def TSRANK(A: pd.Series, n: int) -> pd.Series:
    def _rank_last(x):
        return pd.Series(x).rank().iloc[-1] / len(x)
    return A.rolling(window=n, min_periods=n).apply(_rank_last, raw=True)


def SIGN(A: pd.Series) -> pd.Series:
    return np.sign(A)


def LOG(A: pd.Series) -> pd.Series:
    return np.log(A.replace(0, np.nan))


def ABS(A: pd.Series) -> pd.Series:
    return A.abs()


def MAX(A: pd.Series, B: Union[pd.Series, int, float]) -> pd.Series:
    if isinstance(B, (int, float)):
        return A.clip(lower=B)
    return pd.concat([A, B], axis=1).max(axis=1)


def MIN(A: pd.Series, B: Union[pd.Series, int, float]) -> pd.Series:
    if isinstance(B, (int, float)):
        return A.clip(upper=B)
    return pd.concat([A, B], axis=1).min(axis=1)


def COUNT(condition: pd.Series, n: int) -> pd.Series:
    return condition.astype(int).rolling(window=n, min_periods=n).sum()


def SMA(A: pd.Series, n: int, m: int) -> pd.Series:
    result = pd.Series(index=A.index, dtype=float)
    if len(A) == 0:
        return result
    result.iloc[0] = A.iloc[0]
    for i in range(1, len(A)):
        if np.isnan(A.iloc[i]):
            result.iloc[i] = result.iloc[i - 1]
        else:
            result.iloc[i] = (A.iloc[i] * m + result.iloc[i - 1] * (n - m)) / n
    return result


def WMA(A: pd.Series, n: int) -> pd.Series:
    weights = np.array([0.9 ** i for i in range(n - 1, -1, -1)], dtype=float)
    weights = weights / weights.sum()
    return A.rolling(window=n, min_periods=n).apply(lambda x: np.dot(x, weights), raw=True)


def DECAYLINEAR(A: pd.Series, d: int) -> pd.Series:
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()
    return A.rolling(window=d, min_periods=d).apply(lambda x: np.dot(x, weights), raw=True)

