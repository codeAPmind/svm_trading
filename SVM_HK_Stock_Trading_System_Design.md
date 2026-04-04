# 基于 SVM 的港股买卖点判断交易系统设计文档

## — 数据对接 Futu OpenD · 含回测引擎 · AI 多空分析

---

## 1. 系统概览

### 1.1 目标

构建一套端到端的港股量化交易系统，核心以 **SVM（支持向量机）** 算法判断股票买卖点，数据通过 **Futu OpenD** 实时获取，具备历史回测、实时信号生成、以及结合新闻/基本面/宏观环境的 **AI 多空分析** 能力。

### 1.2 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                     主控模块 main.py                            │
├──────────┬──────────┬──────────┬──────────┬─────────────────────┤
│ 数据采集 │ 特征工程 │ SVM模型  │ 回测引擎 │  AI多空分析         │
│ data/    │ feature/ │ model/   │ backtest/│  analysis/          │
│          │          │          │          │                     │
│ Futu API │ 技术指标 │ 训练/预测│ 收益统计 │  新闻爬取           │
│ 行情数据 │ 成交量加 │ 参数优化 │ 夏普比率 │  基本面分析         │
│ K线/成交 │ 权价格   │ 信号生成 │ 最大回撤 │  宏观环境           │
│ 量/财报  │ MACD/RSI │ 多模型   │ 可视化   │  Claude API总结     │
│          │ KDJ/CCI  │ 集成     │          │                     │
└──────────┴──────────┴──────────┴──────────┴─────────────────────┘
```

### 1.3 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 数据源 | Futu OpenD + futu-api | 港股行情、K线、财务数据 |
| 核心算法 | scikit-learn SVM | SVC/SVR 多核函数支持 |
| 特征工程 | TA-Lib / pandas_ta | 技术指标计算 |
| 回测框架 | 自研 + matplotlib | 轻量级事件驱动回测 |
| AI分析 | Anthropic Claude API | 多空综合研判 |
| 新闻数据 | Web Scraping / RSS | 财经新闻聚合 |
| 可视化 | matplotlib + plotly | 图表展示 |
| 数据存储 | SQLite / CSV | 本地持久化 |

---

## 2. 项目结构

```
svm_hk_trading/
├── config/
│   ├── settings.py            # 全局配置（Futu连接、API Key等）
│   └── symbols.py             # 港股标的列表
├── data/
│   ├── futu_client.py         # Futu OpenD 数据采集封装
│   ├── data_store.py          # 本地数据存储管理
│   └── news_fetcher.py        # 新闻数据爬取
├── feature/
│   ├── technical_indicators.py # 技术指标计算（MACD/RSI/KDJ/CCI）
│   ├── vwap_indicator.py      # 成交量加权价格指标
│   └── feature_engineer.py    # 特征工程主模块
├── model/
│   ├── svm_model.py           # SVM 模型训练与预测
│   ├── signal_generator.py    # 买卖信号生成
│   └── model_evaluator.py     # 模型评估指标
├── backtest/
│   ├── backtest_engine.py     # 回测引擎
│   ├── portfolio.py           # 组合管理与资金管理
│   └── metrics.py             # 绩效指标计算
├── analysis/
│   ├── ai_analyst.py          # AI 多空分析（Claude API）
│   ├── fundamental.py         # 基本面分析
│   └── macro_analysis.py      # 宏观环境分析
├── visualization/
│   └── charts.py              # 可视化模块
├── main.py                    # 主入口
├── requirements.txt           # 依赖
└── README.md
```

---

## 3. 模块详细设计

---

### 3.1 配置模块 `config/settings.py`

```python
"""全局配置"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class FutuConfig:
    """Futu OpenD 连接配置"""
    host: str = "127.0.0.1"
    port: int = 11111
    security_firm: str = "FUTUINC"  # 券商标识
    market: str = "HK"

@dataclass
class SVMConfig:
    """SVM 模型配置"""
    kernel: str = "rbf"                  # 核函数: rbf / linear / poly
    C_range: List[float] = field(default_factory=lambda: [0.1, 1, 10, 100])
    gamma_range: List[str] = field(default_factory=lambda: ["scale", "auto"])
    test_size: float = 0.2              # 测试集比例
    lookback_window: int = 20           # 回看窗口期
    prediction_horizon: int = 5         # 预测周期（交易日）
    threshold_buy: float = 0.02         # 涨幅阈值 -> 买入标签
    threshold_sell: float = -0.02       # 跌幅阈值 -> 卖出标签

@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1_000_000  # 初始资金（HKD）
    commission_rate: float = 0.0005     # 佣金费率（万五）
    stamp_duty: float = 0.001           # 印花税（千一）
    min_commission: float = 5.0         # 最低佣金（HKD）
    position_size: float = 0.3          # 单次仓位比例
    max_positions: int = 3              # 最大持仓数

@dataclass
class ClaudeConfig:
    """Claude API 配置"""
    api_key: str = ""                   # 在环境变量中设置
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2000

# 全局配置实例
FUTU_CFG = FutuConfig()
SVM_CFG = SVMConfig()
BACKTEST_CFG = BacktestConfig()
CLAUDE_CFG = ClaudeConfig()
```

---

### 3.2 数据采集模块 `data/futu_client.py`

```python
"""
Futu OpenD 数据采集封装
- 日K线数据
- 分钟级数据（可选）
- 实时报价
- 财务数据
"""
import pandas as pd
from futu import (
    OpenQuoteContext, OpenSecurityTradeContext,
    KLType, KL_FIELD, AuType, SubType,
    RET_OK, Market, SecurityFirm
)
from config.settings import FUTU_CFG


class FutuDataClient:
    """Futu OpenD 数据客户端"""

    def __init__(self):
        self.quote_ctx = None

    def connect(self):
        """建立连接"""
        self.quote_ctx = OpenQuoteContext(
            host=FUTU_CFG.host,
            port=FUTU_CFG.port
        )
        print("[FutuClient] 已连接 OpenD")

    def disconnect(self):
        """断开连接"""
        if self.quote_ctx:
            self.quote_ctx.close()
            print("[FutuClient] 已断开")

    def get_history_kline(
        self,
        code: str,
        start: str,
        end: str,
        ktype: KLType = KLType.K_DAY,
        autype: AuType = AuType.QFQ
    ) -> pd.DataFrame:
        """
        获取历史K线数据

        Args:
            code: 股票代码，如 "HK.00700"（腾讯）
            start: 起始日期 "YYYY-MM-DD"
            end: 结束日期 "YYYY-MM-DD"
            ktype: K线类型（日K、周K等）
            autype: 复权类型（前复权）

        Returns:
            DataFrame with columns:
            [time_key, open, close, high, low, volume, turnover, pe_ratio, turnover_rate]
        """
        ret, data, page_req_key = self.quote_ctx.request_history_kline(
            code=code,
            start=start,
            end=end,
            ktype=ktype,
            autype=autype,
            fields=[
                KL_FIELD.DATE_TIME,
                KL_FIELD.OPEN,
                KL_FIELD.CLOSE,
                KL_FIELD.HIGH,
                KL_FIELD.LOW,
                KL_FIELD.VOLUME,
                KL_FIELD.TURNOVER,
                KL_FIELD.PE_RATIO,
                KL_FIELD.TURNOVER_RATE
            ],
            max_count=1000
        )

        if ret != RET_OK:
            raise ConnectionError(f"获取K线失败: {data}")

        all_data = [data]

        # 分页获取全部数据
        while page_req_key is not None:
            ret, data, page_req_key = self.quote_ctx.request_history_kline(
                code=code,
                start=start,
                end=end,
                ktype=ktype,
                autype=autype,
                max_count=1000,
                page_req_key=page_req_key
            )
            if ret == RET_OK:
                all_data.append(data)

        df = pd.concat(all_data, ignore_index=True)
        df['time_key'] = pd.to_datetime(df['time_key'])
        df.set_index('time_key', inplace=True)
        df.sort_index(inplace=True)

        print(f"[FutuClient] {code}: 获取 {len(df)} 条K线数据 ({start} ~ {end})")
        return df

    def get_realtime_quote(self, codes: list) -> pd.DataFrame:
        """获取实时报价"""
        ret, data = self.quote_ctx.get_market_snapshot(codes)
        if ret != RET_OK:
            raise ConnectionError(f"获取实时报价失败: {data}")
        return data

    def get_financial_data(self, code: str) -> dict:
        """
        获取基本面财务数据（最近一期）

        Returns:
            dict: {pe, pb, market_cap, dividend_yield, roe, ...}
        """
        ret, data = self.quote_ctx.get_market_snapshot([code])
        if ret != RET_OK:
            raise ConnectionError(f"获取财务数据失败: {data}")

        row = data.iloc[0]
        return {
            "code": code,
            "name": row.get("name", ""),
            "last_price": row.get("last_price", 0),
            "pe_ratio": row.get("pe_ratio", 0),
            "pb_ratio": row.get("pb_ratio", 0),
            "market_cap": row.get("market_cap", 0),
            "turnover_rate": row.get("turnover_rate", 0),
            "volume": row.get("volume", 0),
            "amplitude": row.get("amplitude", 0),
            "high52w": row.get("high_price", 0),
            "low52w": row.get("low_price", 0),
        }
```

---

### 3.3 特征工程模块

#### 3.3.1 技术指标 `feature/technical_indicators.py`

```python
"""
技术指标计算模块
包含：MACD, RSI, KDJ, CCI, 布林带, ATR 等
"""
import pandas as pd
import numpy as np


def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    平滑异同平均线 MACD

    Returns:
        DataFrame: [macd_dif, macd_dea, macd_hist]
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = 2 * (dif - dea)

    return pd.DataFrame({
        'macd_dif': dif,
        'macd_dea': dea,
        'macd_hist': hist
    })


def calc_rsi(close: pd.Series, periods: list = [6, 12, 24]) -> pd.DataFrame:
    """
    相对强弱指标 RSI（多周期）

    Returns:
        DataFrame: [rsi_6, rsi_12, rsi_24]
    """
    result = {}
    for period in periods:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        result[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    return pd.DataFrame(result)


def calc_kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3
) -> pd.DataFrame:
    """
    随机指标 KDJ

    Returns:
        DataFrame: [kdj_k, kdj_d, kdj_j]
    """
    lowest_low = low.rolling(window=n).min()
    highest_high = high.rolling(window=n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100

    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d

    return pd.DataFrame({
        'kdj_k': k,
        'kdj_d': d,
        'kdj_j': j
    })


def calc_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.DataFrame:
    """
    顺势指标 CCI

    Returns:
        DataFrame: [cci]
    """
    tp = (high + low + close) / 3
    ma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - ma_tp) / (0.015 * mad)

    return pd.DataFrame({'cci': cci})


def calc_bollinger(
    close: pd.Series,
    period: int = 20,
    std_dev: int = 2
) -> pd.DataFrame:
    """布林带"""
    ma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    return pd.DataFrame({
        'bb_upper': ma + std_dev * std,
        'bb_middle': ma,
        'bb_lower': ma - std_dev * std,
        'bb_width': (ma + std_dev * std - (ma - std_dev * std)) / ma,
        'bb_pctb': (close - (ma - std_dev * std)) / ((ma + std_dev * std) - (ma - std_dev * std))
    })


def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.DataFrame:
    """平均真实波幅 ATR"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return pd.DataFrame({'atr': atr})
```

#### 3.3.2 成交量加权价格指标 `feature/vwap_indicator.py`

```python
"""
成交量加权价格指标 (Volume Weighted Average Price)

根据技术分析理论，成交量领先于价格并证实价格变化趋势的有效性。
成交量加权移动平均值将每个交易日的收盘价用该日的成交量占
给定期间总成交量的比例进行加权。

公式:
    VWAP_T = Σ(Ci × Vi) / Σ(Vi)   (i = t-T+1 to t)

优势:
    1. 更早获取交易信号
    2. 提高投资决策正确率
    3. 在高波动性/高流动性股票中尤为突出
"""
import pandas as pd
import numpy as np


def calc_vwap(
    close: pd.Series,
    volume: pd.Series,
    period: int = 20
) -> pd.Series:
    """
    成交量加权移动平均价格

    Args:
        close: 收盘价序列
        volume: 成交量序列
        period: 移动周期 T

    Returns:
        VWAP 序列
    """
    cv = close * volume
    vwap = cv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    return vwap


def calc_vwap_features(
    close: pd.Series,
    volume: pd.Series,
    periods: list = [5, 10, 20, 60]
) -> pd.DataFrame:
    """
    多周期 VWAP 特征

    Returns:
        DataFrame 包含:
        - vwap_{period}: 各周期VWAP值
        - vwap_ratio_{period}: 收盘价与VWAP的比值（>1 表示价格高于VWAP，偏强）
        - vwap_cross_{period}: VWAP交叉信号（1=上穿, -1=下穿, 0=无）
    """
    features = {}

    for p in periods:
        vwap = calc_vwap(close, volume, p)
        features[f'vwap_{p}'] = vwap
        features[f'vwap_ratio_{p}'] = close / vwap

        # 交叉信号
        above = (close > vwap).astype(int)
        cross = above.diff()
        features[f'vwap_cross_{p}'] = cross  # 1=上穿, -1=下穿

    return pd.DataFrame(features)


def calc_volume_features(volume: pd.Series) -> pd.DataFrame:
    """
    成交量衍生特征

    Returns:
        DataFrame: [vol_ma5_ratio, vol_ma20_ratio, vol_change, vol_std_20]
    """
    return pd.DataFrame({
        'vol_ma5_ratio': volume / volume.rolling(5).mean(),
        'vol_ma20_ratio': volume / volume.rolling(20).mean(),
        'vol_change': volume.pct_change(),
        'vol_std_20': volume.rolling(20).std() / volume.rolling(20).mean()
    })
```

#### 3.3.3 特征工程主模块 `feature/feature_engineer.py`

```python
"""
特征工程主模块
整合所有技术指标与衍生特征，生成 SVM 训练所需的特征矩阵
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from feature.technical_indicators import (
    calc_macd, calc_rsi, calc_kdj, calc_cci,
    calc_bollinger, calc_atr
)
from feature.vwap_indicator import calc_vwap_features, calc_volume_features
from config.settings import SVM_CFG


class FeatureEngineer:
    """特征工程器"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从原始K线数据构建完整特征集

        Args:
            df: 原始K线 DataFrame (columns: open, close, high, low, volume, turnover)

        Returns:
            带有所有特征列的 DataFrame
        """
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # ---- 价格衍生特征 ----
        df['return_1d'] = close.pct_change(1)
        df['return_5d'] = close.pct_change(5)
        df['return_10d'] = close.pct_change(10)
        df['return_20d'] = close.pct_change(20)

        df['ma_5'] = close.rolling(5).mean()
        df['ma_10'] = close.rolling(10).mean()
        df['ma_20'] = close.rolling(20).mean()
        df['ma_60'] = close.rolling(60).mean()

        # 均线交叉距离
        df['ma5_ma20_gap'] = (df['ma_5'] - df['ma_20']) / df['ma_20']
        df['ma10_ma60_gap'] = (df['ma_10'] - df['ma_60']) / df['ma_60']
        df['price_ma20_gap'] = (close - df['ma_20']) / df['ma_20']

        # ---- 技术指标 ----
        macd_df = calc_macd(close)
        rsi_df = calc_rsi(close, periods=[6, 12, 24])
        kdj_df = calc_kdj(high, low, close)
        cci_df = calc_cci(high, low, close)
        bb_df = calc_bollinger(close)
        atr_df = calc_atr(high, low, close)

        # ---- 成交量加权指标 ----
        vwap_df = calc_vwap_features(close, volume, periods=[5, 10, 20, 60])
        vol_df = calc_volume_features(volume)

        # ---- 波动率特征 ----
        df['volatility_5'] = close.pct_change().rolling(5).std()
        df['volatility_20'] = close.pct_change().rolling(20).std()

        # ---- 合并所有特征 ----
        result = pd.concat([
            df, macd_df, rsi_df, kdj_df, cci_df,
            bb_df, atr_df, vwap_df, vol_df
        ], axis=1)

        return result

    def generate_labels(
        self,
        df: pd.DataFrame,
        horizon: int = None,
        buy_threshold: float = None,
        sell_threshold: float = None
    ) -> pd.Series:
        """
        生成分类标签

        规则:
            - 未来 horizon 个交易日的收益率 > buy_threshold  -> 1 (买入)
            - 未来 horizon 个交易日的收益率 < sell_threshold -> -1 (卖出)
            - 其他 -> 0 (持有/观望)

        Returns:
            pd.Series: 标签 {-1, 0, 1}
        """
        horizon = horizon or SVM_CFG.prediction_horizon
        buy_thr = buy_threshold or SVM_CFG.threshold_buy
        sell_thr = sell_threshold or SVM_CFG.threshold_sell

        future_return = df['close'].pct_change(horizon).shift(-horizon)

        labels = pd.Series(0, index=df.index, name='label')
        labels[future_return > buy_thr] = 1
        labels[future_return < sell_thr] = -1

        return labels

    def prepare_dataset(self, df: pd.DataFrame) -> tuple:
        """
        准备 SVM 训练数据集

        Returns:
            (X, y, feature_columns, scaler)
            X: 标准化后的特征矩阵
            y: 标签
        """
        featured_df = self.build_features(df.copy())
        labels = self.generate_labels(featured_df)
        featured_df['label'] = labels

        # 选定特征列（排除原始 OHLCV 和标签）
        exclude_cols = [
            'open', 'close', 'high', 'low', 'volume', 'turnover',
            'pe_ratio', 'turnover_rate', 'label',
            'ma_5', 'ma_10', 'ma_20', 'ma_60',  # 保留 gap 而非绝对值
            'vwap_5', 'vwap_10', 'vwap_20', 'vwap_60'
        ]
        self.feature_columns = [
            c for c in featured_df.columns if c not in exclude_cols
        ]

        # 删除 NaN
        clean_df = featured_df.dropna(subset=self.feature_columns + ['label'])

        X = clean_df[self.feature_columns].values
        y = clean_df['label'].values.astype(int)

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        print(f"[FeatureEngineer] 特征维度: {X_scaled.shape[1]}")
        print(f"[FeatureEngineer] 样本数: {len(X_scaled)}")
        print(f"[FeatureEngineer] 标签分布: 买入={sum(y==1)}, "
              f"卖出={sum(y==-1)}, 观望={sum(y==0)}")

        return X_scaled, y, self.feature_columns, clean_df.index
```

---

### 3.4 SVM 模型模块 `model/svm_model.py`

```python
"""
SVM 模型训练、预测与优化模块

支持:
- GridSearchCV 参数调优
- 多核函数对比 (RBF / Linear / Poly)
- 时间序列交叉验证 (TimeSeriesSplit)
- 模型持久化
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import (
    GridSearchCV, TimeSeriesSplit, cross_val_score
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from config.settings import SVM_CFG


class SVMTradingModel:
    """SVM 买卖点判断模型"""

    def __init__(self):
        self.model = None
        self.best_params = {}
        self.train_score = 0
        self.test_score = 0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimize: bool = True
    ):
        """
        训练 SVM 模型

        Args:
            X_train: 训练特征矩阵
            y_train: 训练标签
            optimize: 是否进行网格搜索调优
        """
        if optimize:
            param_grid = {
                'C': SVM_CFG.C_range,
                'gamma': SVM_CFG.gamma_range,
                'kernel': ['rbf', 'linear'],
                'class_weight': ['balanced', None]
            }

            # 时间序列交叉验证（不打乱顺序）
            tscv = TimeSeriesSplit(n_splits=5)

            grid_search = GridSearchCV(
                SVC(probability=True, random_state=42),
                param_grid,
                cv=tscv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.train_score = grid_search.best_score_

            print(f"[SVMModel] 最优参数: {self.best_params}")
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测标签"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        模型评估

        Returns:
            dict: {accuracy, f1, confusion_matrix, classification_report}
        """
        y_pred = self.predict(X_test)
        self.test_score = accuracy_score(y_test, y_pred)

        report = classification_report(
            y_test, y_pred,
            target_names=['卖出(-1)', '观望(0)', '买入(1)'],
            output_dict=True
        )
        cm = confusion_matrix(y_test, y_pred)

        eval_result = {
            'accuracy': self.test_score,
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': cm,
            'classification_report': report,
            'y_pred': y_pred,
            'y_test': y_test
        }

        print(f"\n[SVMModel] 测试集准确率: {self.test_score:.4f}")
        print(f"[SVMModel] F1 加权: {eval_result['f1_weighted']:.4f}")
        print(f"\n混淆矩阵:\n{cm}")
        print(f"\n分类报告:\n{classification_report(y_test, y_pred, target_names=['卖出(-1)', '观望(0)', '买入(1)'])}")

        return eval_result

    def save_model(self, path: str = "model/svm_model.pkl"):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params,
            'train_score': self.train_score,
            'test_score': self.test_score
        }, path)
        print(f"[SVMModel] 模型已保存至 {path}")

    def load_model(self, path: str = "model/svm_model.pkl"):
        """加载模型"""
        data = joblib.load(path)
        self.model = data['model']
        self.best_params = data['best_params']
        self.train_score = data['train_score']
        self.test_score = data['test_score']
        print(f"[SVMModel] 模型已加载，测试得分: {self.test_score:.4f}")
```

#### 3.4.1 信号生成器 `model/signal_generator.py`

```python
"""
买卖信号生成模块
将 SVM 预测结果转化为可执行的交易信号
"""
import pandas as pd
import numpy as np
from typing import Tuple


class SignalGenerator:
    """交易信号生成器"""

    def __init__(self, confidence_threshold: float = 0.6):
        """
        Args:
            confidence_threshold: 置信度阈值，仅当预测概率 > 阈值时才生成信号
        """
        self.confidence_threshold = confidence_threshold

    def generate_signals(
        self,
        dates: pd.DatetimeIndex,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        classes: np.ndarray
    ) -> pd.DataFrame:
        """
        生成交易信号

        Args:
            dates: 日期索引
            predictions: 模型预测标签数组
            probabilities: 模型预测概率矩阵 (n_samples, n_classes)
            classes: 模型的类别数组 [-1, 0, 1]

        Returns:
            DataFrame: [date, signal, confidence, action]
                signal:     -1=卖出, 0=观望, 1=买入
                confidence: 预测的置信度
                action:     'BUY' / 'SELL' / 'HOLD'
        """
        signals = []

        for i in range(len(predictions)):
            pred = predictions[i]
            prob = probabilities[i]

            # 获取预测类别对应的概率
            class_idx = np.where(classes == pred)[0][0]
            confidence = prob[class_idx]

            # 仅在置信度足够高时生成有效信号
            if confidence >= self.confidence_threshold:
                signal = int(pred)
            else:
                signal = 0  # 不确定时观望

            action_map = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
            signals.append({
                'date': dates[i],
                'signal': signal,
                'confidence': round(confidence, 4),
                'action': action_map[signal],
                'prob_buy': round(prob[np.where(classes == 1)[0][0]], 4) if 1 in classes else 0,
                'prob_sell': round(prob[np.where(classes == -1)[0][0]], 4) if -1 in classes else 0,
            })

        return pd.DataFrame(signals).set_index('date')

    def filter_consecutive_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤连续重复信号（避免重复开仓/平仓）
        只保留信号变化时的点位
        """
        filtered = signals_df.copy()
        filtered['signal_change'] = filtered['signal'].diff().fillna(filtered['signal'])
        filtered.loc[filtered['signal_change'] == 0, 'action'] = 'HOLD'
        filtered.loc[filtered['signal_change'] == 0, 'signal'] = 0
        return filtered.drop(columns=['signal_change'])
```

---

### 3.5 回测引擎 `backtest/backtest_engine.py`

```python
"""
回测引擎
事件驱动式回测框架，支持:
- 多标的回测
- 佣金与印花税（港股特有费率）
- 资金管理
- 绩效指标全面计算
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from config.settings import BACKTEST_CFG


@dataclass
class Trade:
    """单笔交易记录"""
    date: pd.Timestamp
    code: str
    action: str          # 'BUY' or 'SELL'
    price: float
    quantity: int
    commission: float
    stamp_duty: float    # 港股卖出时收取
    signal_confidence: float = 0.0


@dataclass
class Position:
    """持仓信息"""
    code: str
    quantity: int
    avg_cost: float
    entry_date: pd.Timestamp


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config: Optional[object] = None):
        self.cfg = config or BACKTEST_CFG
        self.cash = self.cfg.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[dict] = []

    def reset(self):
        """重置回测状态"""
        self.cash = self.cfg.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []

    def _calc_commission(self, amount: float) -> float:
        """计算佣金（港股）"""
        commission = amount * self.cfg.commission_rate
        return max(commission, self.cfg.min_commission)

    def _calc_stamp_duty(self, amount: float) -> float:
        """计算印花税（港股卖出时收取）"""
        return amount * self.cfg.stamp_duty

    def _get_lot_size(self, code: str) -> int:
        """获取每手股数（港股默认100股/手，实际应查询）"""
        # 实际使用时应通过 Futu API 查询每手股数
        return 100

    def execute_buy(
        self,
        date: pd.Timestamp,
        code: str,
        price: float,
        confidence: float = 0.0
    ):
        """执行买入"""
        if len(self.positions) >= self.cfg.max_positions:
            return  # 已满仓

        # 计算可买入金额
        available = self.cash * self.cfg.position_size
        lot_size = self._get_lot_size(code)
        quantity = int(available / (price * lot_size)) * lot_size

        if quantity <= 0:
            return

        amount = price * quantity
        commission = self._calc_commission(amount)
        total_cost = amount + commission

        if total_cost > self.cash:
            return

        self.cash -= total_cost

        # 更新持仓
        if code in self.positions:
            pos = self.positions[code]
            total_qty = pos.quantity + quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + price * quantity) / total_qty
            pos.quantity = total_qty
        else:
            self.positions[code] = Position(
                code=code,
                quantity=quantity,
                avg_cost=price,
                entry_date=date
            )

        self.trades.append(Trade(
            date=date, code=code, action='BUY',
            price=price, quantity=quantity,
            commission=commission, stamp_duty=0,
            signal_confidence=confidence
        ))

    def execute_sell(
        self,
        date: pd.Timestamp,
        code: str,
        price: float,
        confidence: float = 0.0
    ):
        """执行卖出（全仓卖出）"""
        if code not in self.positions:
            return

        pos = self.positions[code]
        amount = price * pos.quantity
        commission = self._calc_commission(amount)
        stamp_duty = self._calc_stamp_duty(amount)
        net_amount = amount - commission - stamp_duty

        self.cash += net_amount

        self.trades.append(Trade(
            date=date, code=code, action='SELL',
            price=price, quantity=pos.quantity,
            commission=commission, stamp_duty=stamp_duty,
            signal_confidence=confidence
        ))

        del self.positions[code]

    def run(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        code: str
    ) -> pd.DataFrame:
        """
        运行回测

        Args:
            price_data: 价格数据 (index=date, columns=[open, close, high, low])
            signals: 信号数据 (index=date, columns=[signal, confidence, action])
            code: 股票代码

        Returns:
            DataFrame: 每日组合净值
        """
        self.reset()

        for date in price_data.index:
            price = price_data.loc[date, 'close']

            # 执行信号
            if date in signals.index:
                sig = signals.loc[date]
                action = sig['action'] if isinstance(sig, pd.Series) else sig.iloc[0]['action']
                conf = sig['confidence'] if isinstance(sig, pd.Series) else sig.iloc[0]['confidence']

                if action == 'BUY':
                    self.execute_buy(date, code, price, conf)
                elif action == 'SELL':
                    self.execute_sell(date, code, price, conf)

            # 记录每日组合价值
            position_value = sum(
                pos.quantity * price_data.loc[date, 'close']
                for pos in self.positions.values()
                if date in price_data.index
            )
            total_value = self.cash + position_value

            self.portfolio_history.append({
                'date': date,
                'cash': self.cash,
                'position_value': position_value,
                'total_value': total_value,
                'return': total_value / self.cfg.initial_capital - 1
            })

        return pd.DataFrame(self.portfolio_history).set_index('date')
```

#### 3.5.1 绩效指标 `backtest/metrics.py`

```python
"""
回测绩效指标计算
"""
import pandas as pd
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class BacktestMetrics:
    """回测绩效指标"""
    total_return: float          # 总收益率
    annual_return: float         # 年化收益率
    sharpe_ratio: float          # 夏普比率
    max_drawdown: float          # 最大回撤
    max_drawdown_duration: int   # 最大回撤持续天数
    win_rate: float              # 胜率
    profit_loss_ratio: float     # 盈亏比
    total_trades: int            # 总交易次数
    avg_holding_days: float      # 平均持仓天数
    annual_volatility: float     # 年化波动率
    calmar_ratio: float          # 卡尔玛比率


def calculate_metrics(
    portfolio_df: pd.DataFrame,
    trades: list,
    risk_free_rate: float = 0.02
) -> BacktestMetrics:
    """
    计算全面的回测绩效指标

    Args:
        portfolio_df: 组合净值 DataFrame (columns: [total_value, return])
        trades: 交易记录列表
        risk_free_rate: 无风险利率（年化）

    Returns:
        BacktestMetrics
    """
    returns = portfolio_df['total_value'].pct_change().dropna()
    total_days = len(portfolio_df)
    annual_factor = 252  # 港股年交易日

    # 总收益率
    total_return = portfolio_df['return'].iloc[-1]

    # 年化收益率
    years = total_days / annual_factor
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 年化波动率
    annual_volatility = returns.std() * np.sqrt(annual_factor)

    # 夏普比率
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0

    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # 最大回撤持续时间
    dd_start = None
    dd_duration = 0
    max_dd_duration = 0
    for i, dd in enumerate(drawdown):
        if dd < 0:
            if dd_start is None:
                dd_start = i
            dd_duration = i - dd_start
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_start = None
            dd_duration = 0

    # 卡尔玛比率
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # 交易统计
    buy_trades = [t for t in trades if t.action == 'BUY']
    sell_trades = [t for t in trades if t.action == 'SELL']
    total_trades = len(buy_trades)

    # 盈亏统计（配对买卖）
    profits = []
    holding_days = []
    for i, sell in enumerate(sell_trades):
        if i < len(buy_trades):
            buy = buy_trades[i]
            pnl = (sell.price - buy.price) / buy.price
            profits.append(pnl)
            days = (sell.date - buy.date).days
            holding_days.append(days)

    win_rate = sum(1 for p in profits if p > 0) / len(profits) if profits else 0
    avg_win = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
    avg_loss = abs(np.mean([p for p in profits if p < 0])) if any(p < 0 for p in profits) else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    return BacktestMetrics(
        total_return=round(total_return, 4),
        annual_return=round(annual_return, 4),
        sharpe_ratio=round(sharpe_ratio, 4),
        max_drawdown=round(max_drawdown, 4),
        max_drawdown_duration=max_dd_duration,
        win_rate=round(win_rate, 4),
        profit_loss_ratio=round(profit_loss_ratio, 4),
        total_trades=total_trades,
        avg_holding_days=round(np.mean(holding_days), 1) if holding_days else 0,
        annual_volatility=round(annual_volatility, 4),
        calmar_ratio=round(calmar_ratio, 4)
    )


def print_metrics_report(metrics: BacktestMetrics):
    """打印绩效报告"""
    print("\n" + "=" * 60)
    print("            回测绩效报告")
    print("=" * 60)
    print(f"  总收益率:          {metrics.total_return:>10.2%}")
    print(f"  年化收益率:        {metrics.annual_return:>10.2%}")
    print(f"  年化波动率:        {metrics.annual_volatility:>10.2%}")
    print(f"  夏普比率:          {metrics.sharpe_ratio:>10.4f}")
    print(f"  最大回撤:          {metrics.max_drawdown:>10.2%}")
    print(f"  最大回撤持续(天):  {metrics.max_drawdown_duration:>10d}")
    print(f"  卡尔玛比率:        {metrics.calmar_ratio:>10.4f}")
    print("-" * 60)
    print(f"  总交易次数:        {metrics.total_trades:>10d}")
    print(f"  胜率:              {metrics.win_rate:>10.2%}")
    print(f"  盈亏比:            {metrics.profit_loss_ratio:>10.4f}")
    print(f"  平均持仓天数:      {metrics.avg_holding_days:>10.1f}")
    print("=" * 60)
```

---

### 3.6 AI 多空分析模块 `analysis/ai_analyst.py`

```python
"""
AI 多空分析模块
整合新闻、基本面、宏观环境，调用 Claude API 进行综合研判
"""
import json
import requests
import os
from datetime import datetime
from typing import Dict, Optional

from config.settings import CLAUDE_CFG


class AIAnalyst:
    """AI 多空综合分析师"""

    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", CLAUDE_CFG.api_key)
        self.model = CLAUDE_CFG.model
        self.base_url = "https://api.anthropic.com/v1/messages"

    def _call_claude(self, prompt: str) -> str:
        """调用 Claude API"""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": self.model,
            "max_tokens": CLAUDE_CFG.max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data['content'][0]['text']
        except Exception as e:
            return f"[AI分析错误] {str(e)}"

    def build_analysis_prompt(
        self,
        code: str,
        name: str,
        svm_signal: dict,
        technical_summary: dict,
        fundamental_data: dict,
        news_summary: str,
        macro_context: str
    ) -> str:
        """
        构建分析提示词

        Args:
            code: 股票代码
            name: 股票名称
            svm_signal: SVM 模型输出 {signal, confidence, prob_buy, prob_sell}
            technical_summary: 技术指标摘要
            fundamental_data: 基本面数据
            news_summary: 近期新闻摘要
            macro_context: 宏观环境描述

        Returns:
            完整的分析提示词
        """
        prompt = f"""你是一位专业的港股量化交易分析师。请根据以下数据，对 {name}（{code}）进行综合多空分析。

## 一、SVM 模型信号
- 当前信号: {'买入' if svm_signal['signal']==1 else '卖出' if svm_signal['signal']==-1 else '观望'}
- 置信度: {svm_signal['confidence']:.1%}
- 买入概率: {svm_signal.get('prob_buy', 0):.1%}
- 卖出概率: {svm_signal.get('prob_sell', 0):.1%}

## 二、技术指标概况
- MACD DIF: {technical_summary.get('macd_dif', 'N/A')}
- MACD 柱状图: {technical_summary.get('macd_hist', 'N/A')}（正=多头动能, 负=空头动能）
- RSI(14): {technical_summary.get('rsi_12', 'N/A')}（>70 超买, <30 超卖）
- KDJ-J: {technical_summary.get('kdj_j', 'N/A')}（>80 超买, <20 超卖）
- CCI: {technical_summary.get('cci', 'N/A')}（>100 强势, <-100 弱势）
- 布林带位置: {technical_summary.get('bb_pctb', 'N/A')}（>1 超上轨, <0 超下轨）
- 价格相对MA20: {technical_summary.get('price_ma20_gap', 'N/A')}
- VWAP比率(20日): {technical_summary.get('vwap_ratio_20', 'N/A')}（>1 强势）

## 三、基本面数据
- 市盈率(PE): {fundamental_data.get('pe_ratio', 'N/A')}
- 市净率(PB): {fundamental_data.get('pb_ratio', 'N/A')}
- 市值: {fundamental_data.get('market_cap', 'N/A')}
- 换手率: {fundamental_data.get('turnover_rate', 'N/A')}
- 52周高点: {fundamental_data.get('high52w', 'N/A')}
- 52周低点: {fundamental_data.get('low52w', 'N/A')}

## 四、近期新闻与事件
{news_summary}

## 五、宏观环境
{macro_context}

---

请按照以下结构输出分析报告：

### 1. 综合评级
给出明确的评级：【强烈买入】/【买入】/【中性】/【卖出】/【强烈卖出】

### 2. 多方因素（利好）
列出支持买入的因素

### 3. 空方因素（利空）
列出支持卖出的因素

### 4. 关键风险
当前需要关注的主要风险

### 5. 操作建议
- 建议仓位比例
- 建议入场/离场价位
- 止损位
- 目标价位

### 6. 置信度评估
综合所有信息，你对本次判断的置信度（0-100%）及理由
"""
        return prompt

    def analyze(
        self,
        code: str,
        name: str,
        svm_signal: dict,
        technical_summary: dict,
        fundamental_data: dict,
        news_summary: str = "暂无近期新闻",
        macro_context: str = "暂无宏观数据"
    ) -> str:
        """
        执行 AI 多空分析

        Returns:
            分析报告文本
        """
        prompt = self.build_analysis_prompt(
            code, name, svm_signal, technical_summary,
            fundamental_data, news_summary, macro_context
        )

        print(f"\n[AIAnalyst] 正在分析 {name}({code})...")
        report = self._call_claude(prompt)
        print(f"[AIAnalyst] 分析完成")

        return report


class NewsFetcher:
    """新闻数据获取（简化版，实际可对接多数据源）"""

    @staticmethod
    def fetch_hk_news(code: str, days: int = 7) -> str:
        """
        获取港股相关新闻摘要
        实际生产环境中可对接:
        - 富途新闻 API
        - 财联社/东方财富 RSS
        - Google News RSS
        - 彭博 API

        此处为框架示意，需要根据实际数据源实现
        """
        # 框架实现 - 对接实际新闻 API
        try:
            # 方案1: 利用 Claude API + Web Search
            # 方案2: 对接财经新闻 RSS
            # 方案3: Futu 的新闻推送
            return f"[新闻模块] 请配置实际新闻数据源，当前使用占位数据。\n需要为 {code} 获取最近 {days} 天新闻。"
        except Exception as e:
            return f"新闻获取失败: {e}"


class MacroAnalyzer:
    """宏观环境分析"""

    @staticmethod
    def get_macro_context() -> str:
        """
        获取当前宏观环境概要
        可对接的数据:
        - 美联储利率决议
        - 中国PMI/CPI
        - 恒生指数走势
        - 港元利率 HIBOR
        - 南向资金流向
        - VIX 恐慌指数
        """
        # 框架实现
        context = """
宏观环境数据（请对接实际数据源）:
- 美联储政策: [待获取]
- 中国经济数据: [待获取]
- 恒生指数趋势: [待获取]
- 南向资金: [待获取]
- 港元 HIBOR: [待获取]
- 全球风险偏好(VIX): [待获取]
"""
        return context
```

---

### 3.7 可视化模块 `visualization/charts.py`

```python
"""
可视化模块
生成回测报告图表
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_backtest_report(
    price_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    code: str,
    save_path: str = "output/backtest_report.png"
):
    """
    生成完整的回测报告图表（4子图）

    子图1: 股价走势 + 买卖信号标注
    子图2: 组合净值曲线 vs 基准
    子图3: 回撤曲线
    子图4: 成交量

    Args:
        price_df: 价格数据
        signals_df: 信号数据
        portfolio_df: 组合净值数据
        code: 股票代码
        save_path: 图表保存路径
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), gridspec_kw={'height_ratios': [3, 2, 1, 1]})
    fig.suptitle(f'SVM 交易策略回测报告 - {code}', fontsize=16, fontweight='bold')

    # ---- 子图1: 股价 + 买卖点 ----
    ax1 = axes[0]
    ax1.plot(price_df.index, price_df['close'], color='#333333', linewidth=0.8, label='收盘价')
    ax1.plot(price_df.index, price_df['close'].rolling(20).mean(),
             color='#2196F3', linewidth=0.6, alpha=0.7, label='MA20')
    ax1.plot(price_df.index, price_df['close'].rolling(60).mean(),
             color='#FF9800', linewidth=0.6, alpha=0.7, label='MA60')

    # 标注买卖点
    buy_signals = signals_df[signals_df['signal'] == 1]
    sell_signals = signals_df[signals_df['signal'] == -1]

    buy_dates = buy_signals.index.intersection(price_df.index)
    sell_dates = sell_signals.index.intersection(price_df.index)

    ax1.scatter(buy_dates, price_df.loc[buy_dates, 'close'],
                marker='^', color='#4CAF50', s=100, zorder=5, label='买入信号')
    ax1.scatter(sell_dates, price_df.loc[sell_dates, 'close'],
                marker='v', color='#F44336', s=100, zorder=5, label='卖出信号')

    ax1.set_title('股价走势与买卖信号')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ---- 子图2: 组合净值 vs 基准 ----
    ax2 = axes[1]
    portfolio_return = portfolio_df['total_value'] / portfolio_df['total_value'].iloc[0]
    benchmark_return = price_df['close'] / price_df['close'].iloc[0]

    ax2.plot(portfolio_df.index, portfolio_return, color='#4CAF50', linewidth=1.2, label='策略净值')
    ax2.plot(price_df.index, benchmark_return, color='#9E9E9E', linewidth=0.8, label='基准（买入持有）')
    ax2.fill_between(portfolio_df.index, portfolio_return, benchmark_return,
                     where=portfolio_return >= benchmark_return,
                     alpha=0.1, color='green', label='超额收益')
    ax2.fill_between(portfolio_df.index, portfolio_return, benchmark_return,
                     where=portfolio_return < benchmark_return,
                     alpha=0.1, color='red', label='跑输基准')

    ax2.set_title('策略净值 vs 基准（买入持有）')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # ---- 子图3: 回撤曲线 ----
    ax3 = axes[2]
    cum_return = portfolio_return
    running_max = cum_return.cummax()
    drawdown = (cum_return - running_max) / running_max
    ax3.fill_between(drawdown.index, drawdown, 0, color='#F44336', alpha=0.3)
    ax3.plot(drawdown.index, drawdown, color='#F44336', linewidth=0.8)
    ax3.set_title('策略回撤')
    ax3.set_ylabel('回撤幅度')
    ax3.grid(True, alpha=0.3)

    # ---- 子图4: 成交量 ----
    ax4 = axes[3]
    colors = ['#4CAF50' if c >= o else '#F44336'
              for c, o in zip(price_df['close'], price_df['open'])]
    ax4.bar(price_df.index, price_df['volume'], color=colors, alpha=0.6, width=0.8)
    ax4.set_title('成交量')
    ax4.set_ylabel('成交量')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Charts] 回测报告已保存至 {save_path}")
```

---

### 3.8 主控模块 `main.py`

```python
"""
主控模块 - 端到端运行流程
支持三种模式:
1. 回测模式: 历史数据训练 + 回测验证
2. 实时模式: 生成当日买卖信号
3. 全量模式: 回测 + 实时 + AI分析
"""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

from config.settings import SVM_CFG, BACKTEST_CFG
from data.futu_client import FutuDataClient
from feature.feature_engineer import FeatureEngineer
from model.svm_model import SVMTradingModel
from model.signal_generator import SignalGenerator
from backtest.backtest_engine import BacktestEngine
from backtest.metrics import calculate_metrics, print_metrics_report
from analysis.ai_analyst import AIAnalyst, NewsFetcher, MacroAnalyzer
from visualization.charts import plot_backtest_report


def run_backtest(code: str, start: str, end: str, optimize: bool = True):
    """
    运行完整回测流程

    Args:
        code: 港股代码（如 HK.00700）
        start: 回测起始日期
        end: 回测结束日期
        optimize: 是否优化 SVM 参数
    """
    print("=" * 60)
    print(f"  SVM 港股交易策略回测")
    print(f"  标的: {code}")
    print(f"  区间: {start} ~ {end}")
    print("=" * 60)

    # ---- Step 1: 数据获取 ----
    print("\n[Step 1] 获取历史数据...")
    client = FutuDataClient()
    client.connect()
    try:
        df = client.get_history_kline(code, start, end)
    finally:
        client.disconnect()

    # ---- Step 2: 特征工程 ----
    print("\n[Step 2] 特征工程...")
    fe = FeatureEngineer()
    X, y, feature_cols, valid_index = fe.prepare_dataset(df)

    # ---- Step 3: 时间序列划分（不能随机划分） ----
    print("\n[Step 3] 数据划分...")
    split_idx = int(len(X) * (1 - SVM_CFG.test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = valid_index[split_idx:]

    print(f"  训练集: {split_idx} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    # ---- Step 4: 模型训练 ----
    print("\n[Step 4] SVM 模型训练...")
    model = SVMTradingModel()
    model.train(X_train, y_train, optimize=optimize)

    # ---- Step 5: 模型评估 ----
    print("\n[Step 5] 模型评估...")
    eval_result = model.evaluate(X_test, y_test)

    # ---- Step 6: 生成交易信号 ----
    print("\n[Step 6] 生成交易信号...")
    sig_gen = SignalGenerator(confidence_threshold=0.55)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    signals_df = sig_gen.generate_signals(
        dates_test, predictions, probabilities, model.model.classes_
    )
    signals_df = sig_gen.filter_consecutive_signals(signals_df)

    buy_count = len(signals_df[signals_df['signal'] == 1])
    sell_count = len(signals_df[signals_df['signal'] == -1])
    print(f"  买入信号: {buy_count} 次")
    print(f"  卖出信号: {sell_count} 次")

    # ---- Step 7: 回测执行 ----
    print("\n[Step 7] 运行回测...")
    engine = BacktestEngine()
    test_prices = df.loc[dates_test]
    portfolio_df = engine.run(test_prices, signals_df, code)

    # ---- Step 8: 绩效统计 ----
    print("\n[Step 8] 绩效统计...")
    metrics = calculate_metrics(portfolio_df, engine.trades)
    print_metrics_report(metrics)

    # ---- Step 9: 可视化 ----
    print("\n[Step 9] 生成可视化报告...")
    plot_backtest_report(test_prices, signals_df, portfolio_df, code)

    # ---- Step 10: 保存模型 ----
    model.save_model()

    return model, fe, signals_df, metrics


def run_realtime_signal(code: str, model: SVMTradingModel, fe: FeatureEngineer):
    """
    生成实时买卖信号

    Args:
        code: 港股代码
        model: 已训练的SVM模型
        fe: 已拟合的特征工程器
    """
    print("\n" + "=" * 60)
    print(f"  实时信号生成 - {code}")
    print("=" * 60)

    client = FutuDataClient()
    client.connect()
    try:
        # 获取最近120个交易日数据（用于计算技术指标）
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        df = client.get_history_kline(code, start_date, end_date)
        fundamental = client.get_financial_data(code)
    finally:
        client.disconnect()

    # 构建特征
    featured_df = fe.build_features(df.copy())
    latest = featured_df[fe.feature_columns].iloc[-1:].values
    latest_scaled = fe.scaler.transform(latest)

    # 预测
    pred = model.predict(latest_scaled)[0]
    prob = model.predict_proba(latest_scaled)[0]

    signal_map = {1: '买入', -1: '卖出', 0: '观望'}
    class_idx = np.where(model.model.classes_ == pred)[0][0]

    signal_info = {
        'signal': int(pred),
        'confidence': float(prob[class_idx]),
        'prob_buy': float(prob[np.where(model.model.classes_ == 1)[0][0]]) if 1 in model.model.classes_ else 0,
        'prob_sell': float(prob[np.where(model.model.classes_ == -1)[0][0]]) if -1 in model.model.classes_ else 0,
    }

    print(f"\n  当前信号: {signal_map[pred]}")
    print(f"  置信度:   {signal_info['confidence']:.1%}")
    print(f"  买入概率: {signal_info['prob_buy']:.1%}")
    print(f"  卖出概率: {signal_info['prob_sell']:.1%}")
    print(f"  最新收盘: {df['close'].iloc[-1]}")

    return signal_info, featured_df, fundamental


def run_ai_analysis(
    code: str,
    name: str,
    signal_info: dict,
    featured_df: pd.DataFrame,
    fundamental: dict
):
    """运行 AI 多空分析"""
    print("\n" + "=" * 60)
    print(f"  AI 多空分析 - {name}({code})")
    print("=" * 60)

    # 技术指标摘要（取最新一行）
    latest = featured_df.iloc[-1]
    tech_summary = {
        'macd_dif': round(latest.get('macd_dif', 0), 4),
        'macd_hist': round(latest.get('macd_hist', 0), 4),
        'rsi_12': round(latest.get('rsi_12', 0), 2),
        'kdj_j': round(latest.get('kdj_j', 0), 2),
        'cci': round(latest.get('cci', 0), 2),
        'bb_pctb': round(latest.get('bb_pctb', 0), 4),
        'price_ma20_gap': round(latest.get('price_ma20_gap', 0), 4),
        'vwap_ratio_20': round(latest.get('vwap_ratio_20', 0), 4),
    }

    # 获取新闻和宏观
    news = NewsFetcher.fetch_hk_news(code)
    macro = MacroAnalyzer.get_macro_context()

    # AI 分析
    analyst = AIAnalyst()
    report = analyst.analyze(
        code=code,
        name=name,
        svm_signal=signal_info,
        technical_summary=tech_summary,
        fundamental_data=fundamental,
        news_summary=news,
        macro_context=macro
    )

    print("\n" + report)
    return report


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='SVM 港股交易策略系统')
    parser.add_argument('--code', type=str, default='HK.00700', help='港股代码')
    parser.add_argument('--name', type=str, default='腾讯控股', help='股票名称')
    parser.add_argument('--start', type=str, default='2018-01-01', help='回测起始日期')
    parser.add_argument('--end', type=str, default='2025-12-31', help='回测结束日期')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['backtest', 'realtime', 'full'],
                        help='运行模式: backtest/realtime/full')
    parser.add_argument('--no-optimize', action='store_true', help='跳过参数优化')

    args = parser.parse_args()

    if args.mode in ['backtest', 'full']:
        model, fe, signals, metrics = run_backtest(
            args.code, args.start, args.end,
            optimize=not args.no_optimize
        )

    if args.mode in ['realtime', 'full']:
        if args.mode == 'realtime':
            # 单独实时模式需要先加载模型
            model = SVMTradingModel()
            model.load_model()
            fe = FeatureEngineer()  # 需要保存/加载 scaler

        signal_info, featured_df, fundamental = run_realtime_signal(
            args.code, model, fe
        )

        # AI 分析
        report = run_ai_analysis(
            args.code, args.name,
            signal_info, featured_df, fundamental
        )


if __name__ == '__main__':
    main()
```

---

## 4. 依赖清单 `requirements.txt`

```
futu-api>=9.1
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
plotly>=5.15
joblib>=1.3
requests>=2.31
ta-lib  # 可选，需要系统级安装
```

---

## 5. 使用说明

### 5.1 环境准备

```bash
# 1. 安装 Futu OpenD 并启动
# 下载: https://www.futunn.com/download/openAPI
# 启动 OpenD 并登录账户

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 设置 Claude API Key（用于 AI 分析）
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 5.2 运行方式

```bash
# 完整模式（回测 + 实时信号 + AI分析）
python main.py --code HK.00700 --name 腾讯控股 --start 2019-01-01 --end 2025-12-31 --mode full

# 仅回测
python main.py --code HK.09988 --name 阿里巴巴 --start 2020-01-01 --end 2025-06-30 --mode backtest

# 仅生成实时信号（需已有训练好的模型）
python main.py --code HK.00700 --name 腾讯控股 --mode realtime

# 快速回测（跳过参数优化）
python main.py --code HK.00700 --start 2022-01-01 --end 2025-12-31 --mode backtest --no-optimize
```

---

## 6. 核心算法说明

### 6.1 SVM 分类逻辑

本系统采用三分类 SVM：

| 标签 | 含义 | 判定规则 |
|------|------|----------|
| +1 | 买入 | 未来 N 日收益 > +2% |
| 0 | 观望 | 未来 N 日收益在 ±2% 之间 |
| -1 | 卖出 | 未来 N 日收益 < -2% |

采用 RBF 核函数将非线性特征映射至高维空间，通过 `GridSearchCV` + `TimeSeriesSplit` 确保参数调优不引入前视偏差。

### 6.2 特征体系（共 30+ 维度）

**价格类**: 多周期收益率、均线偏离度、布林带位置

**动量类**: MACD（DIF/DEA/柱状图）、RSI（多周期）、KDJ、CCI

**成交量类**: VWAP 多周期比率、VWAP 交叉信号、量比、量变化率

**波动率类**: 历史波动率（5/20日）、ATR、布林带宽度

### 6.3 港股特殊处理

- 佣金结构: 万五佣金 + 千一印花税（仅卖出）+ 最低佣金 5 HKD
- 每手交易: 港股每手股数因标的而异（非统一100股），需查询
- 交易时段: 09:30-12:00, 13:00-16:00（港股通有差异）
- T+0 交易: 港股支持日内回转，但本策略以日线级别为主

---

## 7. 风险提示

1. **模型风险**: SVM 基于历史数据训练，不保证未来表现，市场结构性变化可能导致模型失效。
2. **过拟合风险**: 尽管使用 `TimeSeriesSplit` 交叉验证，复杂的特征组合仍可能过拟合噪声。
3. **流动性风险**: 中小盘港股可能存在流动性不足问题，实际成交价与模型假设可能有偏差。
4. **政策风险**: 港股受到中国大陆与香港两地监管政策影响。
5. **汇率风险**: 港元与美元挂钩，人民币计价投资者需关注汇率波动。

**本系统仅供学习研究，不构成任何投资建议。**

---

## 8. 扩展方向

- **集成学习**: 将 SVM 与 Random Forest、XGBoost 结合，投票表决提高稳定性
- **深度学习**: 引入 LSTM/Transformer 捕捉时序依赖
- **实盘对接**: 通过 Futu `OpenSecurityTradeContext` 实现自动下单
- **多标的轮动**: 在港股通标的池中进行行业轮动
- **实时监控**: WebSocket 行情推送 + 信号即时通知（Telegram/微信）
- **因子挖掘**: 引入另类数据（南向资金、港股通持股变动、期权隐含波动率）

---

*文档版本: v1.0 | 更新日期: 2026-04-04*
