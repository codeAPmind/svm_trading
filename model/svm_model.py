"""
SVM 模型训练、预测与优化

支持:
  - GridSearchCV + TimeSeriesSplit 避免前视偏差
  - RBF / Linear / Poly 多核函数对比
  - 模型持久化 (joblib)
"""
import numpy as np
import joblib
from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

from config.settings import SVM_CFG

MODEL_DIR = Path(__file__).parent.parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL_PATH = MODEL_DIR / 'svm_model.pkl'


class SVMTradingModel:
    """SVM 买卖点判断模型（三分类: -1/0/+1）"""

    def __init__(self):
        self.model = None
        self.best_params: dict = {}
        self.train_score: float = 0.0
        self.test_score: float = 0.0

    # ─────────────────────────────────────────────────────────
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimize: bool = True
    ):
        """
        训练 SVM 模型

        Args:
            X_train:  标准化后的训练特征矩阵
            y_train:  训练标签（-1 / 0 / 1）
            optimize: True → GridSearchCV 调参；False → 使用固定参数快速训练
        """
        if optimize:
            param_grid = {
                'C':            SVM_CFG.C_range,
                'gamma':        SVM_CFG.gamma_range,
                'kernel':       ['rbf', 'linear'],
                'class_weight': ['balanced', None],
            }
            tscv = TimeSeriesSplit(n_splits=5)
            gs = GridSearchCV(
                SVC(probability=True, random_state=42),
                param_grid,
                cv=tscv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            gs.fit(X_train, y_train)
            self.model       = gs.best_estimator_
            self.best_params = gs.best_params_
            self.train_score = gs.best_score_
            print(f"[SVMModel] 最优参数:   {self.best_params}")
            print(f"[SVMModel] 训练CV得分: {self.train_score:.4f}")
        else:
            self.model = SVC(
                kernel=SVM_CFG.kernel,
                C=10,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            self.train_score = self.model.score(X_train, y_train)
            print(f"[SVMModel] 快速训练完成，训练集准确率: {self.train_score:.4f}")

    # ─────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    # ─────────────────────────────────────────────────────────
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """返回完整评估指标字典"""
        y_pred = self.predict(X_test)
        self.test_score = accuracy_score(y_test, y_pred)

        label_names = ['卖出(-1)', '观望(0)', '买入(1)']
        report = classification_report(
            y_test, y_pred,
            target_names=label_names,
            output_dict=True
        )
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n[SVMModel] 测试集准确率: {self.test_score:.4f}")
        print(f"[SVMModel] F1 (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"\n混淆矩阵:\n{cm}")
        print(f"\n{classification_report(y_test, y_pred, target_names=label_names)}")

        return {
            'accuracy':              self.test_score,
            'f1_weighted':           f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix':      cm,
            'classification_report': report,
            'y_pred':                y_pred,
            'y_test':                y_test,
        }

    # ─────────────────────────────────────────────────────────
    def save_model(self, path: str = None):
        p = Path(path) if path else DEFAULT_MODEL_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model':        self.model,
            'best_params':  self.best_params,
            'train_score':  self.train_score,
            'test_score':   self.test_score,
        }, p)
        print(f"[SVMModel] 模型已保存至 {p}")

    def load_model(self, path: str = None):
        p = Path(path) if path else DEFAULT_MODEL_PATH
        if not p.exists():
            raise FileNotFoundError(f"模型文件不存在: {p}")
        data = joblib.load(p)
        self.model       = data['model']
        self.best_params = data.get('best_params', {})
        self.train_score = data.get('train_score', 0)
        self.test_score  = data.get('test_score', 0)
        print(f"[SVMModel] 模型已加载，测试得分: {self.test_score:.4f}")
