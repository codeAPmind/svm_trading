"""
国泰君安短周期价量Alpha因子（子集）
"""
import pandas as pd
import numpy as np
from feature.alpha_functions import *


class GTJAAlphaFactors:
    def __init__(self, df: pd.DataFrame, benchmark_df: pd.DataFrame = None):
        self.open = df['open']
        self.high = df['high']
        self.low = df['low']
        self.close = df['close']
        self.volume = df['volume']
        self.amount = df.get('turnover', df.get('amount', self.close * self.volume))
        self.vwap = (self.amount / self.volume.replace(0, np.nan)) if 'turnover' in df.columns else (self.high + self.low + self.close) / 3
        self.ret = self.close.pct_change()

    # 精选若干核心因子，后续可逐步扩展至191
    def alpha001(self):
        return -1 * CORR(
            RANK(DELTA(LOG(self.volume), 1)),
            RANK((self.close - self.open) / self.open),
            6
        )

    def alpha005(self):
        return -1 * TSMAX(
            CORR(TSRANK(self.volume, 5), TSRANK(self.high, 5), 5),
            3
        )

    def alpha042(self):
        return -1 * RANK(STD(self.high, 10)) * CORR(self.high, self.volume, 10)

    def alpha014(self):
        return self.close - DELAY(self.close, 5)

    def alpha068(self):
        mid_chg = ((self.high + self.low) / 2 - (DELAY(self.high, 1) + DELAY(self.low, 1)) / 2)
        return SMA(mid_chg * (self.high - self.low) / self.volume.replace(0, np.nan), 15, 2)

    def compute_all(self) -> pd.DataFrame:
        factors = {}
        for name in ['alpha001', 'alpha005', 'alpha042', 'alpha014', 'alpha068']:
            try:
                s = getattr(self, name)()
                if isinstance(s, pd.Series):
                    factors[name] = s
            except Exception as e:
                print(f"[GTJAAlpha] {name} 失败: {e}")
        return pd.DataFrame(factors)

