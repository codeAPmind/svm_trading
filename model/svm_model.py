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
        optimize: bool = True,
        fast_search: bool = False,
    ):
        """
        训练 SVM 模型

        Args:
            X_train:     标准化后的训练特征矩阵
            y_train:     训练标签（-1 / 0 / 1）
            optimize:    True  → 网格/随机搜索调参
                         False → 使用固定参数快速训练（适合调试）
            fast_search: True  → RandomizedSearchCV（快，约1/3时间，轻微精度损失）
                         False → GridSearchCV（慢，穷举所有组合，精度最高）

        参数选择指南:
            调试阶段  → optimize=False（秒级完成）
            快速验证  → optimize=True, fast_search=True（分钟级）
            正式训练  → optimize=True, fast_search=False（小时级，生产用）
        """
        if optimize:
            param_grid = {
                'C':            SVM_CFG.C_range,      # [0.1,0.5,1,5,10,50,100]
                'gamma':        SVM_CFG.gamma_range,   # ['scale','auto',0.001,0.01,0.1]
                'kernel':       ['rbf', 'linear'],
                'class_weight': ['balanced', None],
            }
            tscv = TimeSeriesSplit(n_splits=5)
            base_svc = SVC(probability=True, random_state=42)

            n_combos = (len(SVM_CFG.C_range)
                        * len(SVM_CFG.gamma_range) * 2 * 2)

            if fast_search:
                # RandomizedSearchCV：随机抽取 n_iter 组参数
                # 速度约为 GridSearch 的 1/3，适合快速探索
                from sklearn.model_selection import RandomizedSearchCV
                n_iter = min(40, n_combos)
                print(f"[SVMModel] RandomizedSearchCV: "
                      f"从 {n_combos} 组中随机抽取 {n_iter} 组...")
                gs = RandomizedSearchCV(
                    base_svc, param_grid,
                    n_iter=n_iter,
                    cv=tscv,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1,
                )
            else:
                print(f"[SVMModel] GridSearchCV: 穷举 {n_combos} 组参数组合...")
                gs = GridSearchCV(
                    base_svc, param_grid,
                    cv=tscv,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=1,
                )

            gs.fit(X_train, y_train)
            self.model       = gs.best_estimator_
            self.best_params = gs.best_params_
            self.train_score = gs.best_score_

            # 打印最优参数及其含义
            bp = self.best_params
            print(f"\n[SVMModel] 最优参数:   {bp}")
            print(f"[SVMModel] 训练CV得分: {self.train_score:.4f}")
            self._explain_params(bp)

        else:
            # 固定参数快速训练（调试用）
            self.model = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42,
            )
            self.model.fit(X_train, y_train)
            self.train_score = self.model.score(X_train, y_train)
            print(f"[SVMModel] 快速训练完成  "
                  f"C=10  gamma=scale  "
                  f"训练准确率={self.train_score:.4f}")

    def _explain_params(self, params: dict):
        """打印最优参数的直觉解释，帮助理解模型状态"""
        C     = params.get('C', '?')
        gamma = params.get('gamma', '?')
        kern  = params.get('kernel', '?')
        cw    = params.get('class_weight', '?')

        # C 的解释
        if isinstance(C, (int, float)):
            if C <= 1:
                c_note = "宽松（容忍误分类，间隔宽，泛化好）"
            elif C <= 10:
                c_note = "中等（平衡间隔与误分类）"
            else:
                c_note = "严格（不容忍误分类，注意过拟合风险）"
        else:
            c_note = ""

        # gamma 的解释
        if gamma == 'scale':
            g_note = "自动缩放（推荐，随特征数自适应）"
        elif gamma == 'auto':
            g_note = "按特征数缩放（不考虑特征方差）"
        elif isinstance(gamma, float):
            if gamma <= 0.001:
                g_note = "很小（影响范围大，决策面平滑）"
            elif gamma <= 0.01:
                g_note = "适中（平衡复杂度）"
            else:
                g_note = "较大（影响范围小，决策面复杂）"
        else:
            g_note = ""

        print(f"  C={C}     → {c_note}")
        print(f"  gamma={gamma} → {g_note}")
        print(f"  kernel={kern}  class_weight={cw}")

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
