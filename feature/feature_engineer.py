"""
特征工程主模块
整合技术指标与衍生特征，生成 SVM 训练所需的特征矩阵（30+ 维）

特征体系:
  价格类   - 多周期收益率、均线偏离度、布林带位置
  动量类   - MACD / RSI / KDJ / CCI
  成交量类 - VWAP 比率与交叉信号、量比、量变化率
  波动率类 - 历史波动率（5/20日）、ATR、布林带宽度
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature.technical_indicators import (
    calc_macd, calc_rsi, calc_kdj,
    calc_cci, calc_bollinger, calc_atr
)
from feature.vwap_indicator import calc_vwap_features, calc_volume_features
from config.settings import SVM_CFG

# 不参与 SVM 训练的列（原始数据 + 绝对值类指标）
_EXCLUDE_COLS = {
    'open', 'close', 'high', 'low', 'volume', 'turnover',
    'pe_ratio', 'turnover_rate', 'label',
    'ma_5', 'ma_10', 'ma_20', 'ma_60',
    'bb_upper', 'bb_middle', 'bb_lower',
    'vwap_5', 'vwap_10', 'vwap_20', 'vwap_60',
    'atr',          # 保留 atr_ratio 而非绝对值
    'code',         # 源数据中的字符串列，不参与训练
    'name',         # 源数据中的字符串列，不参与训练
}


class FeatureEngineer:
    """特征工程器（训练 & 预测共用同一套 Scaler）"""

    def __init__(self):
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: list = []
        self._fitted: bool = False

    # ─────────────────────────────────────────────────────────
    # 核心方法：从原始K线构建特征 DataFrame
    # ─────────────────────────────────────────────────────────
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从原始K线 DataFrame 构建完整特征集

        Args:
            df: 列至少包含 [open, close, high, low, volume]

        Returns:
            原始列 + 所有特征列 的 DataFrame
        """
        close  = df['close']
        high   = df['high']
        low    = df['low']
        volume = df['volume']

        # ── 价格衍生 ──────────────────────────────────────────
        df = df.copy()
        df['return_1d']  = close.pct_change(1)
        df['return_5d']  = close.pct_change(5)
        df['return_10d'] = close.pct_change(10)
        df['return_20d'] = close.pct_change(20)

        df['ma_5']  = close.rolling(5).mean()
        df['ma_10'] = close.rolling(10).mean()
        df['ma_20'] = close.rolling(20).mean()
        df['ma_60'] = close.rolling(60).mean()

        df['ma5_ma20_gap']   = (df['ma_5']  - df['ma_20']) / df['ma_20'].replace(0, np.nan)
        df['ma10_ma60_gap']  = (df['ma_10'] - df['ma_60']) / df['ma_60'].replace(0, np.nan)
        df['price_ma20_gap'] = (close - df['ma_20']) / df['ma_20'].replace(0, np.nan)

        # 高低价特征
        df['hl_ratio'] = (high - low) / close.replace(0, np.nan)   # 当日振幅
        df['oc_ratio'] = (close - df['open']) / df['open'].replace(0, np.nan)  # 当日涨跌

        # ── 技术指标 ─────────────────────────────────────────
        macd_df = calc_macd(close)
        rsi_df  = calc_rsi(close, periods=[6, 12, 24])
        kdj_df  = calc_kdj(high, low, close)
        cci_df  = calc_cci(high, low, close)
        bb_df   = calc_bollinger(close)
        atr_df  = calc_atr(high, low, close)

        # ATR 相对化（避免绝对值量纲问题）
        atr_ratio = atr_df['atr'] / close.replace(0, np.nan)
        atr_ratio.name = 'atr_ratio'

        # ── VWAP + 成交量特征 ─────────────────────────────────
        vwap_df = calc_vwap_features(close, volume, periods=[5, 10, 20, 60])
        vol_df  = calc_volume_features(volume)

        # ── 波动率 ────────────────────────────────────────────
        df['volatility_5']  = close.pct_change().rolling(5).std()
        df['volatility_20'] = close.pct_change().rolling(20).std()

        # ── 合并 ─────────────────────────────────────────────
        result = pd.concat([
            df, macd_df, rsi_df, kdj_df, cci_df,
            bb_df, atr_ratio, vwap_df, vol_df
        ], axis=1)

        return result

    # ─────────────────────────────────────────────────────────
    # 生成分类标签
    # ─────────────────────────────────────────────────────────
    def generate_labels(
        self,
        df: pd.DataFrame,
        horizon: int = None,
        buy_threshold: float = None,
        sell_threshold: float = None
    ) -> pd.Series:
        """
        三分类标签
          +1 : 未来 N 日涨幅 > buy_threshold
           0 : 介于 sell_threshold ~ buy_threshold
          -1 : 未来 N 日跌幅 < sell_threshold

        Args:
            df: 含 close 列的 DataFrame
        """
        horizon   = horizon       or SVM_CFG.prediction_horizon
        buy_thr   = buy_threshold  or SVM_CFG.threshold_buy
        sell_thr  = sell_threshold or SVM_CFG.threshold_sell

        future_ret = df['close'].pct_change(horizon).shift(-horizon)
        labels = pd.Series(0, index=df.index, name='label')
        labels[future_ret > buy_thr]  = 1
        labels[future_ret < sell_thr] = -1
        return labels

    # ─────────────────────────────────────────────────────────
    # 准备训练数据集
    # ─────────────────────────────────────────────────────────
    def prepare_dataset(self, df: pd.DataFrame) -> tuple:
        """
        Returns:
            (X_scaled, y, feature_columns, valid_index)
        """
        featured_df = self.build_features(df.copy())
        labels      = self.generate_labels(featured_df)
        featured_df['label'] = labels

        # 确定特征列
        self.feature_columns = [
            c for c in featured_df.columns
            if c not in _EXCLUDE_COLS
        ]

        clean_df = featured_df.dropna(subset=self.feature_columns + ['label'])

        X = clean_df[self.feature_columns].values
        y = clean_df['label'].values.astype(int)

        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True

        print(f"[FeatureEngineer] 特征维度:  {X_scaled.shape[1]}")
        print(f"[FeatureEngineer] 样本数量:  {len(X_scaled)}")
        print(f"[FeatureEngineer] 标签分布:  买入={sum(y==1)}, "
              f"卖出={sum(y==-1)}, 观望={sum(y==0)}")

        return X_scaled, y, self.feature_columns, clean_df.index

    # ─────────────────────────────────────────────────────────
    # 实时预测特征（单行）
    # ─────────────────────────────────────────────────────────
    def transform_latest(self, df: pd.DataFrame) -> np.ndarray:
        """
        从最新K线数据提取并标准化特征（用于实时预测）

        Args:
            df: 历史K线 DataFrame（至少 60+ 行以保证指标不为 NaN）

        Returns:
            shape (1, n_features) 的标准化特征数组
        """
        if not self._fitted:
            raise RuntimeError("请先调用 prepare_dataset() 拟合 Scaler")

        featured_df = self.build_features(df.copy())
        latest = featured_df[self.feature_columns].dropna().iloc[-1:]
        if latest.empty:
            raise ValueError("最新特征全为 NaN，请增加历史K线数量（建议 ≥ 120 条）")
        return self.scaler.transform(latest.values)
