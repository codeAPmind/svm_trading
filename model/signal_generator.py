"""
买卖信号生成模块
将 SVM 预测结果转化为可执行的交易信号，并过滤低置信度与连续重复信号
"""
from typing import Optional

import numpy as np
import pandas as pd


def confidence_thresholds_from_atr(
    base: float,
    atr: np.ndarray,
    ref: float,
    *,
    scale: float,
    max_boost: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    按相对波动率抬高置信度门槛：atr/ref > 1 时逐步增加，上限 base+max_boost。

    Args:
        base: CLI 传入的基础 conf_threshold
        atr:  各样本的 ATR/close（与训练 ref 同量纲）
        ref:  参考 ATR/价（回测用训练段中位数，避免前视）
    """
    a = np.asarray(atr, dtype=float)
    ref_f = float(ref) if np.isfinite(ref) and ref > 0 else np.nan
    if not np.isfinite(ref_f) or ref_f <= 0:
        ref_f = float(np.nanmedian(a))
    if not np.isfinite(ref_f) or ref_f <= 0:
        ref_f = 1e-6
    ratio = a / ref_f
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
    excess = np.maximum(0.0, ratio - 1.0)
    boost = np.minimum(max_boost, scale * excess)
    return np.clip(base + boost, lo, hi)


class SignalGenerator:
    """交易信号生成器"""

    def __init__(self, confidence_threshold: float = 0.55):
        """
        Args:
            confidence_threshold: 最低置信度，低于此值时信号降为 HOLD
        """
        self.confidence_threshold = confidence_threshold

    def generate_signals(
        self,
        dates: pd.DatetimeIndex,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        classes: np.ndarray,
        confidence_thresholds: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Args:
            dates:         日期索引
            predictions:   模型预测标签 (-1 / 0 / 1)
            probabilities: 预测概率矩阵 (n_samples, n_classes)
            classes:       模型类别数组 (e.g. [-1, 0, 1])
            confidence_thresholds: 可选，与样本等长的逐日最低置信度；缺省用构造时的标量

        Returns:
            DataFrame index=date, columns=[signal, confidence, action,
                                            prob_buy, prob_sell, required_conf]
        """
        action_map = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
        signals = []

        # 预计算各类别在 classes 中的位置
        idx_buy  = int(np.where(classes == 1)[0][0])  if 1  in classes else None
        idx_sell = int(np.where(classes == -1)[0][0]) if -1 in classes else None

        if confidence_thresholds is not None:
            thr_arr = np.asarray(confidence_thresholds, dtype=float)
            if thr_arr.shape[0] != len(predictions):
                raise ValueError(
                    "confidence_thresholds 长度必须与 predictions 一致"
                )
        else:
            thr_arr = None

        for i, pred in enumerate(predictions):
            prob        = probabilities[i]
            class_idx   = int(np.where(classes == pred)[0][0])
            confidence  = float(prob[class_idx])
            thr_row     = float(thr_arr[i]) if thr_arr is not None else self.confidence_threshold

            # 低置信度 → 观望
            signal = int(pred) if confidence >= thr_row else 0

            signals.append({
                'date':       dates[i],
                'signal':     signal,
                'confidence': round(confidence, 4),
                'required_conf': round(thr_row, 4),
                'action':     action_map[signal],
                'prob_buy':   round(float(prob[idx_buy]),  4) if idx_buy  is not None else 0.0,
                'prob_sell':  round(float(prob[idx_sell]), 4) if idx_sell is not None else 0.0,
            })

        return pd.DataFrame(signals).set_index('date')

    def filter_consecutive_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤连续重复信号（避免重复开/平仓）
        仅保留信号发生变化时的点位，中间相同信号改为 HOLD
        """
        result = signals_df.copy()
        change = result['signal'].diff().fillna(result['signal'])
        mask_no_change = change == 0
        result.loc[mask_no_change, 'action'] = 'HOLD'
        result.loc[mask_no_change, 'signal'] = 0
        return result
