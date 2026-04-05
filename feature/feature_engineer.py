"""
特征工程主模块 v2.0
整合技术指标 + Alpha 价量因子，生成 SVM 训练所需的特征矩阵（60+ 维）

特征体系 v2.0（八大类）:
  ① 价格/动量类  - 多周期收益率、均线偏离度、布林带位置         [原有，~14个]
  ② 技术指标类  - MACD / RSI / KDJ / CCI / ATR                [原有，~10个]
  ③ VWAP类     - 多周期VWAP比率与交叉信号                      [原有，~12个]
  ④ 波动率类    - 历史波动率（5/20日）、振幅                    [原有，~4个]
  ⑤ 资金流向类  - 方向性成交量、克林格流、价位资金流             [新增，~9个]
  ⑥ 开盘缺口类  - 三维缺口偏离、缺口动量、回补率                [新增，~6个]
  ⑦ 量能异常类  - 有符号累积量、量能突破、价量背离               [新增，~8个]
  ⑧ 复合统计类  - Alpha25/Alpha42/Alpha5/动量反转复合因子       [新增，~12个]

v2.0 变更:
  - 新增 feature/alpha_factors.py 四大类因子（约25个新特征）
  - 新增 return_3d / price_ma60_gap / vol_ratio_5_20 等价格特征
  - 新增 factor_report() 快速计算单因子 IC
  - save_scaler() / load_scaler() 替代 main.py 中的 joblib 裸调用
  - 最小样本要求从 60 提升到 130
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from feature.technical_indicators import (
    calc_macd, calc_rsi, calc_kdj,
    calc_cci, calc_bollinger, calc_atr
)
from feature.vwap_indicator import calc_vwap_features, calc_volume_features
from feature.alpha_factors import calc_all_alpha_factors
from config.settings import SVM_CFG

_EXCLUDE_COLS = {
    'open', 'close', 'high', 'low', 'volume', 'turnover',
    'pe_ratio', 'turnover_rate', 'label',
    'ma_5', 'ma_10', 'ma_20', 'ma_60',
    'bb_upper', 'bb_middle', 'bb_lower',
    'vwap_5', 'vwap_10', 'vwap_20', 'vwap_60',
    'atr',
    'clf_fast', 'clf_slow',
    'alpha25_vol_factor', 'alpha25_long_factor',
    'high_vol_rank', 'high_vol_corr',
    'code', 'name',
}

MIN_SAMPLES_REQUIRED = 130


class FeatureEngineer:
    """特征工程器 v2.0（训练 & 预测共用同一套 Scaler）"""

    def __init__(self):
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: list  = []
        self._fitted: bool          = False

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从原始 K 线 DataFrame 构建完整特征集（60+ 维）

        Args:
            df: 列至少包含 [open, close, high, low, volume]
        """
        df     = df.copy()
        open_  = df['open']
        close  = df['close']
        high   = df['high']
        low    = df['low']
        volume = df['volume']

        # ── ① 价格/动量特征 ──────────────────────────────────
        df['return_1d']  = close.pct_change(1)
        df['return_3d']  = close.pct_change(3)
        df['return_5d']  = close.pct_change(5)
        df['return_10d'] = close.pct_change(10)
        df['return_20d'] = close.pct_change(20)

        df['ma_5']  = close.rolling(5).mean()
        df['ma_10'] = close.rolling(10).mean()
        df['ma_20'] = close.rolling(20).mean()
        df['ma_60'] = close.rolling(60).mean()

        ma20 = df['ma_20'].replace(0, np.nan)
        ma60 = df['ma_60'].replace(0, np.nan)
        df['ma5_ma20_gap']   = (df['ma_5']  - df['ma_20']) / ma20
        df['ma10_ma60_gap']  = (df['ma_10'] - df['ma_60']) / ma60
        df['price_ma20_gap'] = (close - df['ma_20']) / ma20
        df['price_ma60_gap'] = (close - df['ma_60']) / ma60

        df['hl_ratio'] = (high - low) / close.replace(0, np.nan)
        df['oc_ratio'] = (close - open_) / open_.replace(0, np.nan)

        # ── ② 技术指标特征 ───────────────────────────────────
        macd_df   = calc_macd(close)
        rsi_df    = calc_rsi(close, periods=[6, 12, 24])
        kdj_df    = calc_kdj(high, low, close)
        cci_df    = calc_cci(high, low, close)
        bb_df     = calc_bollinger(close)
        atr_df    = calc_atr(high, low, close)
        atr_ratio = atr_df['atr'] / close.replace(0, np.nan)
        atr_ratio.name = 'atr_ratio'

        # ── ③ VWAP + 成交量基础特征 ──────────────────────────
        vwap_df = calc_vwap_features(close, volume, periods=[5, 10, 20, 60])
        vol_df  = calc_volume_features(volume)

        # ── ④ 波动率特征 ─────────────────────────────────────
        df['volatility_5']   = close.pct_change().rolling(5).std()
        df['volatility_20']  = close.pct_change().rolling(20).std()
        df['vol_ratio_5_20'] = (
            df['volatility_5'] / df['volatility_20'].replace(0, np.nan)
        )

        # ── ⑤⑥⑦⑧ 新增 Alpha 四大类因子 ─────────────────────
        alpha_df = calc_all_alpha_factors(open_, high, low, close, volume)

        # ── 合并 ─────────────────────────────────────────────
        result = pd.concat([
            df,
            macd_df, rsi_df, kdj_df, cci_df,
            bb_df, atr_ratio,
            vwap_df, vol_df,
            alpha_df,
        ], axis=1)

        return result

    def generate_labels(
        self,
        df: pd.DataFrame,
        horizon: int        = None,
        buy_threshold: float  = None,
        sell_threshold: float = None
    ) -> pd.Series:
        """三分类标签：+1 买入 / 0 观望 / -1 卖出"""
        horizon  = horizon        or SVM_CFG.prediction_horizon
        buy_thr  = buy_threshold  or SVM_CFG.threshold_buy
        sell_thr = sell_threshold or SVM_CFG.threshold_sell

        future_ret = df['close'].pct_change(horizon).shift(-horizon)
        labels = pd.Series(0, index=df.index, name='label')
        labels[future_ret > buy_thr]  = 1
        labels[future_ret < sell_thr] = -1
        return labels

    def prepare_dataset(self, df: pd.DataFrame) -> tuple:
        """
        完整的特征工程 + 标签生成 + 标准化流程

        Returns:
            (X_scaled, y, feature_columns, valid_index)
        """
        if len(df) < MIN_SAMPLES_REQUIRED:
            print(f"[FeatureEngineer] 警告: 样本数 {len(df)} 低于建议最小值 "
                  f"{MIN_SAMPLES_REQUIRED}，部分因子可能为 NaN")

        featured_df = self.build_features(df.copy())
        labels      = self.generate_labels(featured_df)
        featured_df['label'] = labels

        # 仅保留数值型特征列，且排除黑名单列
        candidate_cols = [c for c in featured_df.columns if c not in _EXCLUDE_COLS]
        numeric_cols = []
        for c in candidate_cols:
            if pd.api.types.is_numeric_dtype(featured_df[c]):
                numeric_cols.append(c)
        self.feature_columns = numeric_cols

        clean_df = featured_df.dropna(subset=self.feature_columns + ['label'])
        X = clean_df[self.feature_columns].values
        y = clean_df['label'].values.astype(int)
        X_scaled     = self.scaler.fit_transform(X)
        self._fitted  = True

        n_buy  = sum(y == 1)
        n_sell = sum(y == -1)
        n_hold = sum(y == 0)
        total  = len(y)
        print(f"\n[FeatureEngineer v2.0] 特征维度:  {X_scaled.shape[1]}")
        print(f"[FeatureEngineer v2.0] 样本数量:  {total}")
        print(f"[FeatureEngineer v2.0] 标签分布:  "
              f"买入={n_buy}({n_buy/total:.1%})  "
              f"卖出={n_sell}({n_sell/total:.1%})  "
              f"观望={n_hold}({n_hold/total:.1%})")

        return X_scaled, y, self.feature_columns, clean_df.index

    def transform_latest(self, df: pd.DataFrame) -> np.ndarray:
        """
        从最新 K 线数据提取并标准化特征（用于实时预测）

        Args:
            df: 历史K线 DataFrame（建议 ≥ 130 行）
        """
        if not self._fitted:
            raise RuntimeError("请先调用 prepare_dataset() 拟合 Scaler")

        featured_df = self.build_features(df.copy())
        latest = featured_df[self.feature_columns].dropna().iloc[-1:]
        if latest.empty:
            raise ValueError(
                f"最新特征全为 NaN，请增加历史K线数量（建议 ≥ {MIN_SAMPLES_REQUIRED} 条）"
            )
        return self.scaler.transform(latest.values)

    def factor_report(self, df: pd.DataFrame, horizon: int = 2) -> pd.DataFrame:
        """
        快速计算各因子单期 IC（与未来 horizon 日收益的相关系数）

        Returns:
            DataFrame，index=因子名，columns=[ic, |ic|]，按 |ic| 降序
        """
        featured_df = self.build_features(df.copy())
        future_ret  = df['close'].pct_change(horizon).shift(-horizon)
        featured_df['__fr__'] = future_ret
        clean = featured_df.dropna(subset=['__fr__'])

        cols = [c for c in self.feature_columns if c in clean.columns]
        records = []
        for col in cols:
            s = clean[col].dropna()
            idx = s.index.intersection(clean.index)
            if len(idx) < 30:
                continue
            ic = s[idx].corr(clean.loc[idx, '__fr__'])
            records.append({'factor': col, 'ic': round(ic, 4)})

        if not records:
            return pd.DataFrame()

        result = (
            pd.DataFrame(records)
            .set_index('factor')
            .assign(**{'|ic|': lambda x: x['ic'].abs()})
            .sort_values('|ic|', ascending=False)
        )
        return result

    def save_scaler(self, path: str):
        """保存 Scaler 和特征列名到磁盘"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'scaler':          self.scaler,
            'feature_columns': self.feature_columns,
            'fitted':          self._fitted,
            'version':         'v2.0',
        }, p)
        print(f"[FeatureEngineer] Scaler 已保存至 {p}  "
              f"（特征数={len(self.feature_columns)}）")

    def load_scaler(self, path: str):
        """从磁盘加载 Scaler 和特征列名"""
        data = joblib.load(path)
        self.scaler          = data['scaler']
        self.feature_columns = data['feature_columns']
        self._fitted         = data.get('fitted', True)
        print(f"[FeatureEngineer] Scaler 已加载  "
              f"version={data.get('version','?')}  "
              f"特征数={len(self.feature_columns)}")
