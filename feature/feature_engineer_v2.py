"""
特征工程V2：GTJA Alpha 因子 + 中性化 + 筛选
"""
import pandas as pd
from sklearn.preprocessing import RobustScaler
from feature.gtja_alpha_factors import GTJAAlphaFactors
from feature.factor_neutralizer import FactorNeutralizer, FactorSelector
from config.settings import SVM_CFG


class FeatureEngineerV2:
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_columns = []
        self.selected_factors = []
        self._fitted = False

    def prepare_dataset(self, df: pd.DataFrame) -> tuple:
        # 1) 计算子集Alpha因子
        alpha_df = GTJAAlphaFactors(df).compute_all()
        # 2) 中性化
        neutral = FactorNeutralizer().calc_style_exposures(df)
        alpha_df = FactorNeutralizer().neutralize(alpha_df, neutral)
        # 3) 筛选
        fwd_ret = df['close'].pct_change(2).shift(-2)
        selected = FactorSelector().select(alpha_df, fwd_ret)
        if len(selected) >= 5:
            alpha_df = alpha_df[selected]
        # 缺失处理以保证最少样本
        alpha_df = alpha_df.replace([float('inf'), float('-inf')], pd.NA)
        alpha_df = alpha_df.ffill().bfill().fillna(0)
        # 4) 标签
        future_ret = df['close'].pct_change(SVM_CFG.prediction_horizon).shift(-SVM_CFG.prediction_horizon)
        labels = pd.Series(0, index=df.index, name='label')
        labels[future_ret > SVM_CFG.threshold_buy] = 1
        labels[future_ret < SVM_CFG.threshold_sell] = -1
        # 5) 标准化
        merged = alpha_df.copy()
        merged['label'] = labels
        self.feature_columns = [c for c in merged.columns if c != 'label']
        clean = merged.dropna(subset=['label'])
        X = clean[self.feature_columns].fillna(0).values
        y = clean['label'].values
        Xs = self.scaler.fit_transform(X)
        self._fitted = True
        return Xs, y, self.feature_columns, clean.index

