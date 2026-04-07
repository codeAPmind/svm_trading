"""
模型评估辅助模块
提供特征重要性分析（线性核）、学习曲线绘制等诊断工具
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import learning_curve, TimeSeriesSplit


class ModelEvaluator:
    """SVM 模型诊断与评估"""

    def __init__(self, model, feature_columns: list, output_dir: str = 'output'):
        self.model = model
        self.feature_columns = feature_columns
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    @staticmethod
    def _suffix_text(file_suffix: str | None) -> str:
        return f"_{file_suffix}" if file_suffix else ""

    # ─────────────────────────────────────────────────────────
    def feature_importance(self) -> pd.Series:
        """
        线性核 SVM 的特征权重（绝对值）
        RBF 核不直接支持，此处打印提示
        """
        svc = self.model.model
        if svc.kernel != 'linear':
            print("[ModelEvaluator] RBF 核不支持直接特征重要性，建议用置换重要性")
            return pd.Series(dtype=float)

        # 对于多分类 OvR，取各类权重的 L2 norm
        coef = np.linalg.norm(svc.coef_, axis=0)
        importance = pd.Series(coef, index=self.feature_columns).sort_values(ascending=False)
        print("\n[ModelEvaluator] 特征重要性（线性核 L2-norm）:")
        print(importance.head(15).to_string())
        return importance

    # ─────────────────────────────────────────────────────────
    def plot_learning_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save: bool = True,
        file_suffix: str | None = None,
    ):
        """绘制学习曲线以诊断过拟合/欠拟合"""
        tscv = TimeSeriesSplit(n_splits=5)
        train_sizes, train_scores, val_scores = learning_curve(
            self.model.model, X, y,
            cv=tscv,
            scoring='f1_weighted',
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 8)
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_sizes, train_mean, 'o-', color='#4CAF50', label='训练集')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#4CAF50')
        ax.plot(train_sizes, val_mean, 'o-', color='#2196F3', label='验证集')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#2196F3')
        ax.set_title('SVM 学习曲线')
        ax.set_xlabel('训练样本数')
        ax.set_ylabel('F1 (weighted)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            suffix = self._suffix_text(file_suffix)
            path = self.output_dir / f'learning_curve{suffix}.png'
            plt.savefig(path, dpi=120, bbox_inches='tight')
            print(f"[ModelEvaluator] 学习曲线已保存至 {path}")
        plt.close()

    # ─────────────────────────────────────────────────────────
    def plot_confusion_matrix(self, cm: np.ndarray, save: bool = True, file_suffix: str | None = None):
        """可视化混淆矩阵"""
        import matplotlib.patches as mpatches

        labels = ['卖出(-1)', '观望(0)', '买入(1)']
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)

        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels)

        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

        ax.set_ylabel('真实标签')
        ax.set_xlabel('预测标签')
        ax.set_title('混淆矩阵')
        plt.tight_layout()

        if save:
            suffix = self._suffix_text(file_suffix)
            path = self.output_dir / f'confusion_matrix{suffix}.png'
            plt.savefig(path, dpi=120, bbox_inches='tight')
            print(f"[ModelEvaluator] 混淆矩阵已保存至 {path}")
        plt.close()
