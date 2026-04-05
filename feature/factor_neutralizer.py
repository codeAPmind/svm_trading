import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class FactorNeutralizer:
    def calc_style_exposures(self, df: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame(index=df.index)
        styles['momentum'] = df['close'].pct_change(20)
        styles['volatility'] = df['close'].pct_change().rolling(20).std()
        styles['liquidity'] = df['volume'].rolling(20).mean()
        if 'pe_ratio' in df.columns:
            styles['value'] = 1 / df['pe_ratio'].replace(0, np.nan)
        else:
            styles['value'] = 0
        styles = styles.fillna(0)
        return styles

    def neutralize(self, alpha_df: pd.DataFrame, style_df: pd.DataFrame) -> pd.DataFrame:
        neutralized = pd.DataFrame(index=alpha_df.index)
        X = style_df.fillna(0).values
        for col in alpha_df.columns:
            y = alpha_df[col].values
            valid = ~(np.isnan(y) | np.isinf(y))
            if valid.sum() < 30:
                neutralized[col] = alpha_df[col]
                continue
            try:
                reg = LinearRegression()
                reg.fit(X[valid], y[valid])
                pred = reg.predict(X)
                residual = y - pred
                neutralized[col] = residual
            except Exception:
                neutralized[col] = alpha_df[col]
        return neutralized


class FactorSelector:
    def __init__(self, ic_threshold: float = 0.015, icir_threshold: float = 1.2):
        self.ic_threshold = ic_threshold
        self.icir_threshold = icir_threshold

    def calc_factor_ic(self, factor_values: pd.Series, forward_returns: pd.Series, lookback: int = 60) -> dict:
        valid = ~(factor_values.isna() | forward_returns.isna())
        ic_series = factor_values[valid].rolling(lookback).corr(forward_returns[valid]).dropna()
        if ic_series.empty:
            return {'ic_mean': 0, 'ic_std': 1, 'icir': 0}
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        return {'ic_mean': ic_mean, 'ic_std': ic_std, 'icir': (ic_mean / ic_std if ic_std > 0 else 0)}

    def select(self, alpha_df: pd.DataFrame, forward_returns: pd.Series) -> list:
        stats = []
        for col in alpha_df.columns:
            s = self.calc_factor_ic(alpha_df[col], forward_returns)
            stats.append((col, abs(s['ic_mean']), abs(s['icir'])))
        sel = [c for c, ic, ir in stats if ic >= self.ic_threshold and ir >= self.icir_threshold]
        return sel if sel else list(alpha_df.columns)

