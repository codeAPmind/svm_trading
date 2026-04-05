import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


class SVMTradingModelV2:
    """简化版多周期融合（此处先训练单周期以跑通流程）"""
    def __init__(self):
        self.model = None
        self.best_params = {}
        self.classes_ = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        tscv = TimeSeriesSplit(n_splits=5)
        param_grid = {
            'C': [1, 10, 50],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf'],
            'class_weight': ['balanced', None],
        }
        base = SVC(probability=True, random_state=42)
        grid = GridSearchCV(base, param_grid, cv=tscv, scoring='f1_weighted', n_jobs=-1)
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.classes_ = self.model.classes_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

