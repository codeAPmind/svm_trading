# 基于 SVM 的港股买卖点判断交易系统设计文档 V2.0

## — 融合国泰君安191短周期价量因子 · Futu OpenD · 回测引擎 · AI多空分析

---

## 升级说明

V2.0 核心升级点：
1. **特征工程全面重构**：融合国泰君安《基于短周期价量特征的多因子选股体系》中的191个Alpha因子，替代原有简单技术指标
2. **因子正交化处理**：新增风格因子中性化模块，剔除行业/市值等风格影响，提取纯Alpha
3. **因子筛选机制**：基于IC/ICIR的因子有效性检验与动态筛选
4. **换手率-成本平衡**：引入交易成本罚函数优化目标
5. **多周期预测融合**：T+1至T+3多周期信号加权

---

## 1. 系统架构（升级版）

```
┌────────────────────────────────────────────────────────────────────────┐
│                        主控模块 main.py                                │
├──────────┬───────────────┬──────────┬──────────┬──────────────────────┤
│ 数据采集 │  特征工程V2    │ SVM模型  │ 回测引擎 │  AI多空分析          │
│ data/    │  feature/     │ model/   │ backtest/│  analysis/           │
│          │               │          │          │                      │
│ Futu API │ 191个Alpha因子│ 训练/预测│ 收益统计 │  新闻爬取            │
│ 行情数据 │ 因子正交化     │ 参数优化 │ 夏普比率 │  基本面分析          │
│ K线/成交 │ 风格中性化     │ 信号生成 │ 最大回撤 │  宏观环境            │
│ 量/财报  │ IC/ICIR筛选   │ 多周期   │ 可视化   │  Claude API总结      │
│          │ 动态因子权重   │ 集成     │          │                      │
└──────────┴───────────────┴──────────┴──────────┴──────────────────────┘
```

---

## 2. 特征工程V2：191个短周期价量Alpha因子

### 2.1 设计理念

根据国泰君安研报的核心发现：
- 短周期因子体系对T+2日收益预测效果最佳（年化因子收益率均值8.08%）
- 因子体系呈现**高显著、低相关**特征（94%的因子收益率相关系数绝对值<0.1）
- 因子数量增加可单调提升预测能力（20→191个因子，模型IC从0.034→0.057）
- 因子有效预测周期极限为T+4日

### 2.2 因子分类体系

我们将191个因子按底层逻辑归纳为**8大类**：

| 类别 | 因子数量 | 代表因子 | 核心逻辑 |
|------|--------|--------|---------|
| 价量背离 | ~25个 | Alpha1, Alpha11, Alpha60 | 价格与成交量的相关性/背离度 |
| 动量/反转 | ~35个 | Alpha14, Alpha18, Alpha20 | 短周期价格变化率及趋势延续 |
| 开盘缺口 | ~10个 | Alpha15, Alpha107 | 隔夜跳空信号 |
| 量能异常 | ~20个 | Alpha40, Alpha43, Alpha80 | 成交量相对变化与趋势确认 |
| 波动特征 | ~25个 | Alpha42, Alpha70, Alpha76 | 价格波动率的变化模式 |
| 资金流向 | ~30个 | Alpha9, Alpha55, Alpha128 | 基于分价成交的资金推断 |
| 技术形态 | ~25个 | Alpha28, Alpha47, Alpha57 | KDJ/RSI/CCI等经典指标变体 |
| 复合统计 | ~21个 | Alpha25, Alpha30, Alpha92 | 多维度复合统计特征 |

### 2.3 基础函数库 `feature/alpha_functions.py`

```python
"""
Alpha因子计算基础函数库
实现报告附录2中定义的所有基础函数
"""
import pandas as pd
import numpy as np
from typing import Union


def DELAY(A: pd.Series, n: int) -> pd.Series:
    """延迟函数: A_{i-n}"""
    return A.shift(n)


def DELTA(A: pd.Series, n: int) -> pd.Series:
    """差分函数: A_i - A_{i-n}"""
    return A - A.shift(n)


def RANK(A: pd.Series) -> pd.Series:
    """截面排序（百分位排名）"""
    return A.rank(pct=True)


def SUM(A: pd.Series, n: int) -> pd.Series:
    """过去n天求和"""
    return A.rolling(window=n, min_periods=n).sum()


def MEAN(A: pd.Series, n: int) -> pd.Series:
    """过去n天均值"""
    return A.rolling(window=n, min_periods=n).mean()


def STD(A: pd.Series, n: int) -> pd.Series:
    """过去n天标准差"""
    return A.rolling(window=n, min_periods=n).std()


def CORR(A: pd.Series, B: pd.Series, n: int) -> pd.Series:
    """过去n天相关系数"""
    return A.rolling(window=n, min_periods=n).corr(B)


def COVIANCE(A: pd.Series, B: pd.Series, n: int) -> pd.Series:
    """过去n天协方差"""
    return A.rolling(window=n, min_periods=n).cov(B)


def TSMAX(A: pd.Series, n: int) -> pd.Series:
    """过去n天最大值"""
    return A.rolling(window=n, min_periods=n).max()


def TSMIN(A: pd.Series, n: int) -> pd.Series:
    """过去n天最小值"""
    return A.rolling(window=n, min_periods=n).min()


def TSRANK(A: pd.Series, n: int) -> pd.Series:
    """末位值在过去n天的顺序排位"""
    def _rank_last(x):
        return pd.Series(x).rank().iloc[-1] / len(x)
    return A.rolling(window=n, min_periods=n).apply(_rank_last, raw=True)


def SIGN(A: pd.Series) -> pd.Series:
    """符号函数"""
    return np.sign(A)


def LOG(A: pd.Series) -> pd.Series:
    """自然对数"""
    return np.log(A.replace(0, np.nan))


def ABS(A: pd.Series) -> pd.Series:
    """绝对值"""
    return A.abs()


def MAX(A: pd.Series, B: Union[pd.Series, int, float]) -> pd.Series:
    """取最大值"""
    if isinstance(B, (int, float)):
        return A.clip(lower=B)
    return pd.concat([A, B], axis=1).max(axis=1)


def MIN(A: pd.Series, B: Union[pd.Series, int, float]) -> pd.Series:
    """取最小值"""
    if isinstance(B, (int, float)):
        return A.clip(upper=B)
    return pd.concat([A, B], axis=1).min(axis=1)


def COUNT(condition: pd.Series, n: int) -> pd.Series:
    """过去n天满足条件的次数"""
    return condition.astype(int).rolling(window=n, min_periods=n).sum()


def SMA(A: pd.Series, n: int, m: int) -> pd.Series:
    """
    递推移动平均: Y_i = (A_i * m + Y_{i-1} * (n - m)) / n
    """
    result = pd.Series(index=A.index, dtype=float)
    result.iloc[0] = A.iloc[0]
    for i in range(1, len(A)):
        if np.isnan(A.iloc[i]):
            result.iloc[i] = result.iloc[i - 1]
        else:
            result.iloc[i] = (A.iloc[i] * m + result.iloc[i - 1] * (n - m)) / n
    return result


def WMA(A: pd.Series, n: int) -> pd.Series:
    """加权移动平均，权重为 0.9^i"""
    weights = np.array([0.9 ** i for i in range(n - 1, -1, -1)])
    weights = weights / weights.sum()
    return A.rolling(window=n, min_periods=n).apply(
        lambda x: np.dot(x, weights), raw=True
    )


def DECAYLINEAR(A: pd.Series, d: int) -> pd.Series:
    """线性衰减加权移动平均，权重 d, d-1, ..., 1"""
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()
    return A.rolling(window=d, min_periods=d).apply(
        lambda x: np.dot(x, weights), raw=True
    )


def REGBETA(A: pd.Series, B: pd.Series, n: int) -> pd.Series:
    """回归系数"""
    def _beta(xy):
        mid = len(xy) // 2
        x = xy[:mid]
        y = xy[mid:]
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 3:
            return np.nan
        x, y = x[valid], y[valid]
        cov = np.cov(x, y)
        if cov[0, 0] == 0:
            return np.nan
        return cov[0, 1] / cov[0, 0]

    combined = pd.concat([B, A], axis=1)
    return combined.rolling(window=n).apply(
        lambda x: _beta(x.values.flatten()), raw=False
    ).iloc[:, -1]


def SEQUENCE(n: int) -> pd.Series:
    """生成1~n的等差序列"""
    return pd.Series(range(1, n + 1))


def HIGHDAY(A: pd.Series, n: int) -> pd.Series:
    """最大值距离当前时点的间隔"""
    return A.rolling(window=n, min_periods=n).apply(
        lambda x: n - 1 - np.argmax(x), raw=True
    )


def LOWDAY(A: pd.Series, n: int) -> pd.Series:
    """最小值距离当前时点的间隔"""
    return A.rolling(window=n, min_periods=n).apply(
        lambda x: n - 1 - np.argmin(x), raw=True
    )


def PROD(A: pd.Series, n: int) -> pd.Series:
    """过去n天累乘"""
    return A.rolling(window=n, min_periods=n).apply(np.prod, raw=True)


def SUMIF(A: pd.Series, n: int, condition: pd.Series) -> pd.Series:
    """条件求和"""
    masked = A.where(condition, 0)
    return masked.rolling(window=n, min_periods=n).sum()
```

### 2.4 191个Alpha因子实现 `feature/gtja_alpha_factors.py`

```python
"""
国泰君安191个短周期价量Alpha因子
数据来源：个股日频OHLCV数据

港股适配说明:
- VWAP: 通过 Futu API 获取均价字段，或用 (HIGH+LOW+CLOSE)/3 近似
- BANCHMARKINDEXCLOSE/OPEN: 使用恒生指数或恒生科技指数
- RET: 收盘价/前收盘价 - 1
- 涨跌停处理: 港股无涨跌停限制，但需处理停牌
"""
import pandas as pd
import numpy as np
from feature.alpha_functions import *


class GTJAAlphaFactors:
    """
    国泰君安191个短周期Alpha因子计算器

    港股适配:
    - 港股无涨跌停，无需过滤涨跌停导致的成交量异常
    - VWAP 使用 turnover/volume 计算（Futu提供turnover字段）
    - 基准指数使用恒生指数（HSI）
    """

    def __init__(self, df: pd.DataFrame, benchmark_df: pd.DataFrame = None):
        """
        Args:
            df: 个股日频数据 DataFrame
                必须包含: open, high, low, close, volume, turnover
            benchmark_df: 基准指数日频数据（可选）
                必须包含: open, close
        """
        self.open = df['open']
        self.high = df['high']
        self.low = df['low']
        self.close = df['close']
        self.volume = df['volume']
        self.amount = df.get('turnover', df.get('amount', self.close * self.volume))

        # VWAP: 成交额/成交量，若无则用 (H+L+C)/3
        if 'turnover' in df.columns and df['turnover'].sum() > 0:
            self.vwap = df['turnover'] / df['volume'].replace(0, np.nan)
        else:
            self.vwap = (self.high + self.low + self.close) / 3

        self.ret = self.close / self.close.shift(1) - 1

        # 基准指数
        if benchmark_df is not None:
            self.bench_close = benchmark_df['close']
            self.bench_open = benchmark_df['open']
        else:
            self.bench_close = None
            self.bench_open = None

        # DTM / DBM 辅助变量
        self.dtm = pd.Series(np.where(
            self.open <= DELAY(self.open, 1),
            0,
            np.maximum(self.high - self.open, self.open - DELAY(self.open, 1))
        ), index=df.index)

        self.dbm = pd.Series(np.where(
            self.open >= DELAY(self.open, 1),
            0,
            np.maximum(self.open - self.low, self.open - DELAY(self.open, 1))
        ), index=df.index)

        # TR / HD / LD 辅助变量
        self.tr = np.maximum(
            np.maximum(self.high - self.low,
                       ABS(self.high - DELAY(self.close, 1))),
            ABS(self.low - DELAY(self.close, 1))
        )
        self.hd = self.high - DELAY(self.high, 1)
        self.ld = DELAY(self.low, 1) - self.low

    # ===== 第一批核心因子（精选高ICIR因子优先实现）=====

    def alpha001(self):
        """价量背离：成交量变化与日内收益的相关系数"""
        return -1 * CORR(
            RANK(DELTA(LOG(self.volume), 1)),
            RANK((self.close - self.open) / self.open),
            6
        )

    def alpha002(self):
        """CLV指标变化"""
        clv = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)
        return -1 * DELTA(clv, 1)

    def alpha003(self):
        """True Strength累积"""
        cond_eq = self.close == DELAY(self.close, 1)
        cond_gt = self.close > DELAY(self.close, 1)
        val = pd.Series(np.where(
            cond_eq, 0,
            np.where(
                cond_gt,
                self.close - np.minimum(self.low, DELAY(self.close, 1)),
                self.close - np.maximum(self.high, DELAY(self.close, 1))
            )
        ), index=self.close.index)
        return SUM(val, 6)

    def alpha005(self):
        """量价排名相关性"""
        return -1 * TSMAX(
            CORR(TSRANK(self.volume, 5), TSRANK(self.high, 5), 5),
            3
        )

    def alpha006(self):
        """加权开盘-最高价变化方向"""
        return RANK(SIGN(DELTA(self.open * 0.85 + self.high * 0.15, 4))) * -1

    def alpha007(self):
        """VWAP偏离与成交量变化的复合"""
        return (RANK(np.maximum(self.vwap - self.close, 3)) +
                RANK(np.minimum(self.vwap - self.close, 3))) * \
               RANK(DELTA(self.volume, 3))

    def alpha008(self):
        """加权中间价-VWAP变化"""
        return RANK(DELTA(
            ((self.high + self.low) / 2) * 0.2 + self.vwap * 0.8,
            4
        ) * -1)

    def alpha009(self):
        """中间价变化与量幅的SMA"""
        mid_change = ((self.high + self.low) / 2 -
                      (DELAY(self.high, 1) + DELAY(self.low, 1)) / 2)
        return SMA(mid_change * (self.high - self.low) / self.volume, 7, 2)

    def alpha011(self):
        """CLV * Volume 累积（资金流指标）"""
        clv = ((self.close - self.low) - (self.high - self.close)) / \
              (self.high - self.low)
        return SUM(clv * self.volume, 6)

    def alpha012(self):
        """开盘价与VWAP偏离的复合"""
        return RANK(self.open - SUM(self.vwap, 10) / 10) * \
               (-1 * RANK(ABS(self.close - self.vwap)))

    def alpha014(self):
        """5日价格动量"""
        return self.close - DELAY(self.close, 5)

    def alpha015(self):
        """开盘缺口"""
        return self.open / DELAY(self.close, 1) - 1

    def alpha016(self):
        """量-VWAP排名相关性"""
        return -1 * TSMAX(
            RANK(CORR(RANK(self.volume), RANK(self.vwap), 5)),
            5
        )

    def alpha018(self):
        """5日收益率"""
        return self.close / DELAY(self.close, 5)

    def alpha020(self):
        """6日涨跌幅"""
        return (self.close - DELAY(self.close, 6)) / DELAY(self.close, 6) * 100

    def alpha021(self):
        """6日均价线性回归斜率"""
        close_ma6 = MEAN(self.close, 6)
        seq = pd.Series(range(1, 7))
        return close_ma6.rolling(6).apply(
            lambda x: np.polyfit(range(6), x, 1)[0] if len(x) == 6 else np.nan,
            raw=True
        )

    def alpha024(self):
        """5日价差SMA"""
        return SMA(self.close - DELAY(self.close, 5), 5, 1)

    def alpha026(self):
        """7日均线偏离 + VWAP长期相关"""
        return (SUM(self.close, 7) / 7 - self.close) + \
               CORR(self.vwap, DELAY(self.close, 5), 230)

    def alpha032(self):
        """最高价-成交量排名相关性（高ICIR因子）"""
        return -1 * SUM(
            RANK(CORR(RANK(self.high), RANK(self.volume), 3)),
            3
        )

    def alpha034(self):
        """12日均线/收盘价比率"""
        return MEAN(self.close, 12) / self.close

    def alpha038(self):
        """20日最高价突破"""
        cond = SUM(self.high, 20) / 20 < self.high
        return pd.Series(np.where(cond, -1 * DELTA(self.high, 2), 0),
                         index=self.close.index)

    def alpha040(self):
        """上涨成交量/下跌成交量比率（26日）"""
        up_vol = pd.Series(np.where(
            self.close > DELAY(self.close, 1), self.volume, 0
        ), index=self.close.index)
        down_vol = pd.Series(np.where(
            self.close <= DELAY(self.close, 1), self.volume, 0
        ), index=self.close.index)
        return SUM(up_vol, 26) / SUM(down_vol, 26).replace(0, np.nan) * 100

    def alpha041(self):
        """VWAP变化排名"""
        return RANK(np.maximum(DELTA(self.vwap, 3), 5)) * -1

    def alpha042(self):
        """高价波动与量价相关（高ICIR因子）"""
        return -1 * RANK(STD(self.high, 10)) * CORR(self.high, self.volume, 10)

    def alpha043(self):
        """OBV变体（6日）"""
        obv = pd.Series(np.where(
            self.close > DELAY(self.close, 1), self.volume,
            np.where(self.close < DELAY(self.close, 1), -self.volume, 0)
        ), index=self.close.index)
        return SUM(obv, 6)

    def alpha044(self):
        """低价-均量相关 + VWAP变化（复合因子）"""
        return (TSRANK(DECAYLINEAR(
            CORR(self.low, MEAN(self.volume, 10), 7), 6), 4) +
                TSRANK(DECAYLINEAR(DELTA(self.vwap, 3), 10), 15))

    def alpha046(self):
        """多周期均线复合/收盘价"""
        return (MEAN(self.close, 3) + MEAN(self.close, 6) +
                MEAN(self.close, 12) + MEAN(self.close, 24)) / (4 * self.close)

    def alpha047(self):
        """威廉指标变体（6日）"""
        return SMA(
            (TSMAX(self.high, 6) - self.close) /
            (TSMAX(self.high, 6) - TSMIN(self.low, 6)) * 100,
            9, 1
        )

    def alpha053(self):
        """上涨天数比例（12日）"""
        return COUNT(self.close > DELAY(self.close, 1), 12) / 12 * 100

    def alpha054(self):
        """波动-日内收益-量价相关复合"""
        return -1 * RANK(
            STD(ABS(self.close - self.open)) +
            (self.close - self.open) +
            CORR(self.close, self.open, 10)
        )

    def alpha057(self):
        """KDJ_K变体"""
        return SMA(
            (self.close - TSMIN(self.low, 9)) /
            (TSMAX(self.high, 9) - TSMIN(self.low, 9)) * 100,
            3, 1
        )

    def alpha060(self):
        """CLV * Volume 累积（20日加强版）"""
        clv = ((self.close - self.low) - (self.high - self.close)) / \
              (self.high - self.low)
        return SUM(clv * self.volume, 20)

    def alpha062(self):
        """高价-成交量排名相关"""
        return -1 * CORR(self.high, RANK(self.volume), 5)

    def alpha063(self):
        """RSI变体（6日）"""
        return SMA(
            np.maximum(self.close - DELAY(self.close, 1), 0), 6, 1
        ) / SMA(ABS(self.close - DELAY(self.close, 1)), 6, 1) * 100

    def alpha065(self):
        """6日均线/收盘价比率"""
        return MEAN(self.close, 6) / self.close

    def alpha068(self):
        """中间价变化量幅比的SMA（15日）"""
        mid_chg = ((self.high + self.low) / 2 -
                   (DELAY(self.high, 1) + DELAY(self.low, 1)) / 2)
        return SMA(mid_chg * (self.high - self.low) / self.volume, 15, 2)

    def alpha078(self):
        """CCI变体"""
        tp = (self.high + self.low + self.close) / 3
        ma_tp = MEAN(tp, 12)
        mad = MEAN(ABS(self.close - ma_tp), 12)
        return (tp - ma_tp) / (0.015 * mad)

    def alpha079(self):
        """RSI变体（12日）"""
        return SMA(
            np.maximum(self.close - DELAY(self.close, 1), 0), 12, 1
        ) / SMA(ABS(self.close - DELAY(self.close, 1)), 12, 1) * 100

    def alpha084(self):
        """OBV变体（20日）"""
        obv = pd.Series(np.where(
            self.close > DELAY(self.close, 1), self.volume,
            np.where(self.close < DELAY(self.close, 1), -self.volume, 0)
        ), index=self.close.index)
        return SUM(obv, 20)

    def alpha085(self):
        """量价排名复合"""
        return TSRANK(self.volume / MEAN(self.volume, 20), 20) * \
               TSRANK(-1 * DELTA(self.close, 7), 8)

    def alpha088(self):
        """20日涨跌幅"""
        return (self.close - DELAY(self.close, 20)) / DELAY(self.close, 20) * 100

    def alpha090(self):
        """VWAP-Volume排名相关"""
        return RANK(CORR(RANK(self.vwap), RANK(self.volume), 5)) * -1

    def alpha096(self):
        """KDJ双重平滑"""
        rsv = (self.close - TSMIN(self.low, 9)) / \
              (TSMAX(self.high, 9) - TSMIN(self.low, 9)) * 100
        return SMA(SMA(rsv, 3, 1), 3, 1)

    def alpha099(self):
        """收盘价-成交量排名协方差"""
        return -1 * RANK(COVIANCE(RANK(self.close), RANK(self.volume), 5))

    def alpha101(self):
        """收盘价-均量相关 vs 加权VWAP-量排名相关"""
        part1 = RANK(CORR(self.close, SUM(MEAN(self.volume, 30), 37), 15))
        part2 = RANK(CORR(
            RANK(self.high * 0.1 + self.vwap * 0.9),
            RANK(self.volume), 11
        ))
        return pd.Series(np.where(part1 < part2, -1, 0), index=self.close.index)

    def alpha104(self):
        """高价-成交量相关变化 × 波动"""
        return -1 * DELTA(CORR(self.high, self.volume, 5), 5) * \
               RANK(STD(self.close, 20))

    def alpha105(self):
        """开盘价-成交量排名相关"""
        return -1 * CORR(RANK(self.open), RANK(self.volume), 10)

    def alpha107(self):
        """开盘价三重缺口"""
        return ((-1 * RANK(self.open - DELAY(self.high, 1))) *
                RANK(self.open - DELAY(self.close, 1)) *
                RANK(self.open - DELAY(self.low, 1)))

    def alpha109(self):
        """真实波幅SMA比率"""
        return SMA(self.high - self.low, 10, 2) / \
               SMA(SMA(self.high - self.low, 10, 2), 10, 2)

    def alpha110(self):
        """上涨幅度/下跌幅度比（20日）"""
        up = SUM(np.maximum(0, self.high - DELAY(self.close, 1)), 20)
        down = SUM(np.maximum(0, DELAY(self.close, 1) - self.low), 20)
        return up / down.replace(0, np.nan) * 100

    def alpha116(self):
        """收盘价20日线性回归斜率"""
        return self.close.rolling(20).apply(
            lambda x: np.polyfit(range(20), x, 1)[0] if len(x) == 20 else np.nan,
            raw=True
        )

    def alpha120(self):
        """VWAP-收盘价偏离度"""
        return RANK(self.vwap - self.close) / RANK(self.vwap + self.close)

    def alpha168(self):
        """成交量/20日均量"""
        return -1 * self.volume / MEAN(self.volume, 20)

    # ========================================
    # 批量计算所有因子
    # ========================================

    def compute_all(self) -> pd.DataFrame:
        """
        计算所有已实现的Alpha因子

        Returns:
            DataFrame，每列为一个Alpha因子
        """
        factors = {}
        factor_methods = [m for m in dir(self) if m.startswith('alpha') and callable(getattr(self, m))]

        for method_name in sorted(factor_methods):
            try:
                method = getattr(self, method_name)
                result = method()
                if isinstance(result, pd.Series):
                    factors[method_name] = result
            except Exception as e:
                print(f"[Warning] {method_name} 计算失败: {e}")
                continue

        df = pd.DataFrame(factors)
        print(f"[GTJAAlpha] 成功计算 {len(df.columns)} 个因子")
        return df
```

### 2.5 因子正交化与风格中性化 `feature/factor_neutralizer.py`

```python
"""
因子正交化与风格中性化模块

根据国泰君安报告的方法论:
X_k = β_industry * X_industry + β_style * X_style + ε_k
取残差 ε_k 作为中性化后的因子值

港股适配:
- 行业分类: 使用恒生行业分类或GICS
- 风格因子: Beta, Momentum, Size, Volatility, Value, Liquidity
- 市值因子: 港股自由流通市值
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class FactorNeutralizer:
    """因子正交化处理器"""

    def __init__(self):
        self.style_factors = [
            'beta', 'momentum', 'size', 'volatility',
            'value', 'liquidity'
        ]

    def calc_style_exposures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算风格因子暴露

        Args:
            df: 包含 OHLCV + market_cap 的 DataFrame

        Returns:
            风格因子暴露矩阵
        """
        styles = pd.DataFrame(index=df.index)

        # Size: 总市值对数
        if 'market_cap' in df.columns:
            styles['size'] = np.log(df['market_cap'])
        else:
            styles['size'] = np.log(df['close'] * df.get('total_shares', 1))

        # Momentum: 过去20日收益率
        styles['momentum'] = df['close'].pct_change(20)

        # Volatility: 20日收益率标准差
        styles['volatility'] = df['close'].pct_change().rolling(20).std()

        # Liquidity: 20日平均换手率
        if 'turnover_rate' in df.columns:
            styles['liquidity'] = df['turnover_rate'].rolling(20).mean()
        else:
            styles['liquidity'] = df['volume'].rolling(20).mean()

        # Value: 简化为 1/PE (如有PE数据)
        if 'pe_ratio' in df.columns:
            styles['value'] = 1 / df['pe_ratio'].replace(0, np.nan)
        else:
            styles['value'] = 0

        # Beta: 与基准的相关性（简化计算）
        bench_ret = df['close'].pct_change()  # 如无基准则自身
        styles['beta'] = bench_ret.rolling(60).std() / bench_ret.rolling(250).std()

        return styles.fillna(0)

    def neutralize(
        self,
        alpha_df: pd.DataFrame,
        style_df: pd.DataFrame,
        industry_dummies: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        对Alpha因子进行风格中性化处理

        对每个因子，在截面上回归行业+风格因子，取残差

        Args:
            alpha_df: Alpha因子矩阵 (T × K)
            style_df: 风格因子暴露矩阵 (T × S)
            industry_dummies: 行业哑变量矩阵 (T × I)，可选

        Returns:
            中性化后的Alpha因子矩阵
        """
        neutralized = pd.DataFrame(index=alpha_df.index)

        # 构建回归自变量
        X_parts = [style_df]
        if industry_dummies is not None:
            X_parts.append(industry_dummies)
        X = pd.concat(X_parts, axis=1).fillna(0)

        for col in alpha_df.columns:
            y = alpha_df[col].values
            valid = ~(np.isnan(y) | np.isinf(y))

            if valid.sum() < 30:
                neutralized[col] = alpha_df[col]
                continue

            try:
                reg = LinearRegression()
                reg.fit(X.values[valid], y[valid])
                pred = reg.predict(X.values)
                residual = y - pred
                neutralized[col] = residual
            except Exception:
                neutralized[col] = alpha_df[col]

        return neutralized


class FactorSelector:
    """
    因子有效性检验与动态筛选

    基于IC/ICIR筛选有效因子:
    - IC > 0.02 且 ICIR > 1.5 为有效因子
    - 因子收益率相关性 > 0.5 时保留IR更高的
    """

    def __init__(self, ic_threshold: float = 0.02, icir_threshold: float = 1.5):
        self.ic_threshold = ic_threshold
        self.icir_threshold = icir_threshold

    def calc_factor_ic(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        periods: list = [1, 2, 3]
    ) -> dict:
        """
        计算单因子IC值

        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益率序列
            periods: 预测周期列表

        Returns:
            {period: {'ic_mean': float, 'ic_std': float, 'icir': float}}
        """
        results = {}
        for d in periods:
            fwd_ret = forward_returns.shift(-d)
            valid = ~(factor_values.isna() | fwd_ret.isna())

            if valid.sum() < 60:
                results[d] = {'ic_mean': 0, 'ic_std': 1, 'icir': 0}
                continue

            # 滚动IC
            ic_series = factor_values[valid].rolling(60).corr(fwd_ret[valid])
            ic_series = ic_series.dropna()

            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0

            results[d] = {
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'icir': icir,
                'ic_positive_rate': (ic_series > 0).mean()
            }

        return results

    def select_factors(
        self,
        alpha_df: pd.DataFrame,
        forward_returns: pd.Series,
        prediction_horizon: int = 2
    ) -> list:
        """
        筛选有效因子

        Returns:
            有效因子列名列表（按ICIR降序排列）
        """
        factor_stats = []

        for col in alpha_df.columns:
            ic_result = self.calc_factor_ic(
                alpha_df[col], forward_returns, [prediction_horizon]
            )
            stats = ic_result[prediction_horizon]
            factor_stats.append({
                'factor': col,
                'ic_mean': abs(stats['ic_mean']),
                'icir': abs(stats['icir']),
                'ic_positive_rate': stats.get('ic_positive_rate', 0.5)
            })

        stats_df = pd.DataFrame(factor_stats)

        # 筛选条件
        valid = stats_df[
            (stats_df['ic_mean'] >= self.ic_threshold) &
            (stats_df['icir'] >= self.icir_threshold)
        ]

        # 按ICIR降序排列
        valid = valid.sort_values('icir', ascending=False)

        selected = valid['factor'].tolist()
        print(f"[FactorSelector] 有效因子: {len(selected)}/{len(alpha_df.columns)}")
        print(f"[FactorSelector] 平均IC: {valid['ic_mean'].mean():.4f}")
        print(f"[FactorSelector] 平均ICIR: {valid['icir'].mean():.4f}")

        return selected

    def remove_correlated(
        self,
        alpha_df: pd.DataFrame,
        selected_factors: list,
        max_corr: float = 0.5
    ) -> list:
        """
        剔除高相关因子，保留ICIR更高的

        Args:
            alpha_df: 因子矩阵
            selected_factors: 已选因子列表
            max_corr: 最大允许相关系数

        Returns:
            去相关后的因子列表
        """
        if len(selected_factors) <= 1:
            return selected_factors

        sub_df = alpha_df[selected_factors]
        corr_matrix = sub_df.corr().abs()

        to_remove = set()
        for i in range(len(selected_factors)):
            if selected_factors[i] in to_remove:
                continue
            for j in range(i + 1, len(selected_factors)):
                if selected_factors[j] in to_remove:
                    continue
                if corr_matrix.iloc[i, j] > max_corr:
                    # 移除排在后面的（ICIR更低的）
                    to_remove.add(selected_factors[j])

        final = [f for f in selected_factors if f not in to_remove]
        print(f"[FactorSelector] 去相关后因子数: {len(final)}")
        return final
```

### 2.6 升级版特征工程主模块 `feature/feature_engineer_v2.py`

```python
"""
特征工程V2主模块
整合191个Alpha因子、风格中性化、因子筛选
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

from feature.gtja_alpha_factors import GTJAAlphaFactors
from feature.factor_neutralizer import FactorNeutralizer, FactorSelector
from config.settings import SVM_CFG


class FeatureEngineerV2:
    """
    V2版特征工程器

    工作流程:
    1. 计算191个Alpha因子
    2. 因子正交化（风格中性化）
    3. IC/ICIR检验筛选有效因子
    4. 去除高相关因子
    5. 标准化（使用RobustScaler抗异常值）
    6. 生成SVM训练标签
    """

    def __init__(self, use_robust_scaler: bool = True):
        self.scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        self.neutralizer = FactorNeutralizer()
        self.selector = FactorSelector(ic_threshold=0.015, icir_threshold=1.2)
        self.selected_factors = []
        self.feature_columns = []

    def build_alpha_features(
        self,
        df: pd.DataFrame,
        benchmark_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Step 1: 计算所有Alpha因子

        Args:
            df: 个股日频K线数据
            benchmark_df: 基准指数数据

        Returns:
            Alpha因子矩阵
        """
        alpha_calc = GTJAAlphaFactors(df, benchmark_df)
        alpha_df = alpha_calc.compute_all()

        # 异常值处理：MAD方法
        for col in alpha_df.columns:
            median = alpha_df[col].median()
            mad = (alpha_df[col] - median).abs().median()
            upper = median + 5 * 1.4826 * mad
            lower = median - 5 * 1.4826 * mad
            alpha_df[col] = alpha_df[col].clip(lower, upper)

        return alpha_df

    def neutralize_factors(
        self,
        alpha_df: pd.DataFrame,
        raw_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Step 2: 因子风格中性化

        Args:
            alpha_df: Alpha因子矩阵
            raw_df: 原始K线数据（用于计算风格暴露）

        Returns:
            中性化后的因子矩阵
        """
        style_df = self.neutralizer.calc_style_exposures(raw_df)
        neutralized = self.neutralizer.neutralize(alpha_df, style_df)
        return neutralized

    def select_effective_factors(
        self,
        alpha_df: pd.DataFrame,
        df: pd.DataFrame,
        horizon: int = 2
    ) -> list:
        """
        Step 3 & 4: 因子筛选 + 去相关

        Args:
            alpha_df: 因子矩阵
            df: 原始数据（用于计算未来收益率）
            horizon: 预测周期

        Returns:
            有效因子列表
        """
        forward_ret = df['close'].pct_change(horizon).shift(-horizon)

        selected = self.selector.select_factors(alpha_df, forward_ret, horizon)
        selected = self.selector.remove_correlated(alpha_df, selected, max_corr=0.5)

        self.selected_factors = selected
        return selected

    def generate_labels(
        self,
        df: pd.DataFrame,
        horizon: int = None,
        buy_threshold: float = None,
        sell_threshold: float = None
    ) -> pd.Series:
        """
        生成三分类标签

        标签定义:
            +1 (买入): 未来horizon日收益 > buy_threshold
            -1 (卖出): 未来horizon日收益 < sell_threshold
             0 (观望): 其他
        """
        horizon = horizon or SVM_CFG.prediction_horizon
        buy_thr = buy_threshold or SVM_CFG.threshold_buy
        sell_thr = sell_threshold or SVM_CFG.threshold_sell

        future_ret = df['close'].pct_change(horizon).shift(-horizon)

        labels = pd.Series(0, index=df.index, name='label')
        labels[future_ret > buy_thr] = 1
        labels[future_ret < sell_thr] = -1

        return labels

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        benchmark_df: pd.DataFrame = None,
        do_neutralize: bool = True,
        do_select: bool = True
    ) -> tuple:
        """
        完整的数据准备流程

        Args:
            df: 原始K线数据
            benchmark_df: 基准指数数据
            do_neutralize: 是否进行风格中性化
            do_select: 是否进行因子筛选

        Returns:
            (X_scaled, y, feature_columns, valid_index)
        """
        # Step 1: 计算Alpha因子
        print("[V2] Step 1: 计算191个Alpha因子...")
        alpha_df = self.build_alpha_features(df, benchmark_df)

        # Step 2: 风格中性化（可选）
        if do_neutralize:
            print("[V2] Step 2: 因子风格中性化...")
            alpha_df = self.neutralize_factors(alpha_df, df)

        # Step 3: 因子筛选（可选）
        if do_select:
            print("[V2] Step 3: IC/ICIR因子筛选...")
            selected = self.select_effective_factors(alpha_df, df, horizon=2)
            if len(selected) > 10:
                alpha_df = alpha_df[selected]

        # Step 4: 生成标签
        labels = self.generate_labels(df)

        # Step 5: 清洗 & 标准化
        merged = alpha_df.copy()
        merged['label'] = labels

        self.feature_columns = [c for c in merged.columns if c != 'label']
        clean_df = merged.dropna(subset=self.feature_columns + ['label'])

        X = clean_df[self.feature_columns].values
        y = clean_df['label'].values.astype(int)

        X_scaled = self.scaler.fit_transform(X)

        print(f"\n[V2] === 数据集统计 ===")
        print(f"  特征维度: {X_scaled.shape[1]}")
        print(f"  样本数量: {len(X_scaled)}")
        print(f"  标签分布: 买入={sum(y==1)}, 卖出={sum(y==-1)}, 观望={sum(y==0)}")
        print(f"  有效因子: {self.feature_columns[:10]}... (共{len(self.feature_columns)}个)")

        return X_scaled, y, self.feature_columns, clean_df.index
```

---

## 3. SVM模型升级

### 3.1 多周期预测融合

```python
"""
model/svm_model_v2.py

升级点:
- 多预测周期融合 (T+1, T+2, T+3 加权)
- Platt缩放概率校准
- 样本权重（近期样本更重要）
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


class SVMTradingModelV2:
    """V2版SVM模型，支持多周期融合"""

    def __init__(self):
        self.models = {}  # {horizon: model}
        self.weights = {1: 0.3, 2: 0.5, 3: 0.2}  # 周期权重

    def train_multi_horizon(
        self,
        X_train: np.ndarray,
        y_trains: dict,  # {horizon: y_labels}
        sample_weights: np.ndarray = None
    ):
        """
        多周期模型训练

        Args:
            X_train: 特征矩阵
            y_trains: {1: y_t1, 2: y_t2, 3: y_t3}
            sample_weights: 样本权重（时间衰减）
        """
        for horizon, y in y_trains.items():
            print(f"\n[SVMv2] 训练 T+{horizon} 模型...")

            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=5)

            param_grid = {
                'C': [1, 10, 50],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf'],
                'class_weight': ['balanced']
            }

            base_svm = SVC(probability=True, random_state=42)
            grid = GridSearchCV(
                base_svm, param_grid,
                cv=tscv, scoring='f1_weighted',
                n_jobs=-1
            )
            grid.fit(X_train, y, sample_weight=sample_weights)

            # Platt概率校准
            calibrated = CalibratedClassifierCV(
                grid.best_estimator_, cv=3, method='isotonic'
            )
            calibrated.fit(X_train, y)

            self.models[horizon] = calibrated
            print(f"  T+{horizon} 最优参数: {grid.best_params_}")
            print(f"  T+{horizon} CV得分: {grid.best_score_:.4f}")

    def predict_fused(self, X: np.ndarray) -> tuple:
        """
        多周期加权融合预测

        Returns:
            (predictions, probabilities)
        """
        prob_accum = None

        for horizon, model in self.models.items():
            prob = model.predict_proba(X)  # (n, 3)
            weight = self.weights.get(horizon, 0.33)

            if prob_accum is None:
                prob_accum = prob * weight
            else:
                prob_accum += prob * weight

        # 归一化
        prob_accum /= prob_accum.sum(axis=1, keepdims=True)
        predictions = model.classes_[np.argmax(prob_accum, axis=1)]

        return predictions, prob_accum

    @staticmethod
    def calc_time_decay_weights(n_samples: int, half_life: int = 120) -> np.ndarray:
        """
        计算时间衰减权重

        近期样本权重更大，半衰期为 half_life 个交易日

        Args:
            n_samples: 样本数量
            half_life: 半衰期

        Returns:
            权重数组 (归一化到均值为1)
        """
        decay = np.exp(-np.log(2) / half_life * np.arange(n_samples - 1, -1, -1))
        return decay / decay.mean()
```

---

## 4. 回测引擎升级

### 4.1 换手率-交易成本平衡优化

```python
"""
backtest/turnover_optimizer.py

根据国泰君安报告的目标函数:
Max  w' · E(ε) - Tc · Σ|w_t - w_{t-1}| / 2

通过交易成本罚函数自动平衡换手率与Alpha收益
"""
import numpy as np
from scipy.optimize import minimize


class TurnoverOptimizer:
    """换手率-交易成本平衡优化器"""

    def __init__(self, tc: float = 0.003):
        """
        Args:
            tc: 客观交易成本（港股约0.15%佣金+0.1%印花税+滑点）
                建议设置为0.3%-0.5%
        """
        self.tc = tc

    def optimize_weights(
        self,
        alpha_scores: np.ndarray,
        prev_weights: np.ndarray,
        industry_matrix: np.ndarray = None,
        benchmark_industry_weights: np.ndarray = None,
        n_stocks: int = 50
    ) -> np.ndarray:
        """
        求解最优组合权重

        目标: Max w'·E(ε) - Tc·Σ|w-w_{t-1}|/2
        约束: 行业中性、风格中性、做多约束、权重之和为1

        Args:
            alpha_scores: Alpha预测得分向量
            prev_weights: 上期权重向量
            industry_matrix: 行业哑变量矩阵
            benchmark_industry_weights: 基准行业权重
            n_stocks: 目标持仓股票数

        Returns:
            最优权重向量
        """
        n = len(alpha_scores)

        def objective(w):
            # Alpha收益 - 换手成本
            alpha_return = np.dot(w, alpha_scores)
            turnover_cost = self.tc * np.sum(np.abs(w - prev_weights)) / 2
            return -(alpha_return - turnover_cost)  # 最小化负值

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
        ]

        # 行业中性约束
        if industry_matrix is not None and benchmark_industry_weights is not None:
            for j in range(industry_matrix.shape[1]):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, j=j: (
                        0.05 - abs(np.dot(w, industry_matrix[:, j]) -
                                   benchmark_industry_weights[j])
                    )
                })

        # 边界: 做多约束，单只不超过5%
        bounds = [(0, 0.05)] * n

        result = minimize(
            objective,
            x0=prev_weights if prev_weights.sum() > 0 else np.ones(n) / n,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500}
        )

        if result.success:
            return result.x
        else:
            # 退化为等权
            top_n = np.argsort(alpha_scores)[-n_stocks:]
            w = np.zeros(n)
            w[top_n] = 1.0 / n_stocks
            return w
```

---

## 5. 配置更新 `config/settings.py`

```python
"""V2版配置"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class SVMConfigV2:
    """V2 SVM模型配置"""
    # SVM参数
    kernel: str = "rbf"
    C_range: List[float] = field(default_factory=lambda: [1, 10, 50])
    gamma_range: List[str] = field(default_factory=lambda: ["scale", "auto"])
    test_size: float = 0.2
    class_weight: str = "balanced"

    # 预测配置
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 2, 3])
    horizon_weights: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.2])
    threshold_buy: float = 0.015         # 港股波动较大，阈值适当调低
    threshold_sell: float = -0.015

    # 因子配置
    ic_threshold: float = 0.015          # IC筛选阈值
    icir_threshold: float = 1.2          # ICIR筛选阈值
    max_factor_corr: float = 0.5         # 最大因子相关性
    neutralize_styles: bool = True       # 是否做风格中性化
    factor_lookback: int = 250           # 因子收益率回看窗口

    # 样本权重
    use_time_decay: bool = True          # 使用时间衰减权重
    time_decay_halflife: int = 120       # 半衰期

@dataclass
class BacktestConfigV2:
    """V2 回测配置"""
    initial_capital: float = 1_000_000
    # 港股交易成本明细
    commission_rate: float = 0.0003      # 佣金费率（万三）
    platform_fee: float = 15.0           # 平台使用费（HKD/笔）
    stamp_duty: float = 0.0013           # 印花税（千一点三，卖出）
    trading_levy: float = 0.00005        # 交易征费
    sfc_levy: float = 0.000027           # 证监会征费
    tc_total: float = 0.004              # 综合交易成本（用于优化目标函数）

    # 组合配置
    position_size: float = 0.3
    max_positions: int = 50
    target_turnover: float = 0.38        # 目标单次换仓比率
```

---

## 6. 升级后运行流程 `main_v2.py`

```python
"""V2主入口 - 融合191个Alpha因子"""

def run_v2(code: str, start: str, end: str):
    """
    V2版完整流程:
    1. Futu获取数据
    2. 191个Alpha因子计算
    3. 风格中性化
    4. IC/ICIR因子筛选
    5. 多周期SVM训练
    6. 加权融合预测
    7. 换手率-成本平衡优化
    8. 回测 + 可视化
    9. AI多空分析
    """

    # Step 1: 数据
    client = FutuDataClient()
    client.connect()
    df = client.get_history_kline(code, start, end)
    # 获取恒生指数作为基准
    bench_df = client.get_history_kline("HK.800000", start, end)
    client.disconnect()

    # Step 2-4: 特征工程V2
    fe = FeatureEngineerV2()
    X, y, feature_cols, valid_idx = fe.prepare_dataset(
        df, bench_df,
        do_neutralize=True,
        do_select=True
    )

    # Step 5: 时间序列划分
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 生成多周期标签
    y_trains = {}
    for h in [1, 2, 3]:
        labels_h = fe.generate_labels(df, horizon=h)
        y_h = labels_h.loc[valid_idx].values[:split]
        y_trains[h] = y_h

    # Step 6: 训练多周期模型
    model = SVMTradingModelV2()
    weights = model.calc_time_decay_weights(len(X_train), half_life=120)
    model.train_multi_horizon(X_train, y_trains, sample_weights=weights)

    # Step 7: 融合预测
    predictions, probabilities = model.predict_fused(X_test)

    # Step 8: 回测...
    # Step 9: AI分析...

    print("\n[V2] 完整流程执行完毕")
```

---

## 7. V1 vs V2 对比

| 维度 | V1（原版） | V2（升级版） |
|------|----------|-----------|
| 因子数量 | ~30个（MACD/RSI/KDJ等） | **191个短周期Alpha因子** |
| 因子来源 | 传统技术指标 | 国泰君安价量因子体系 |
| 风格处理 | 无 | **行业/市值/风格中性化** |
| 因子筛选 | 无 | **IC/ICIR动态筛选+去相关** |
| 预测周期 | 单一T+5 | **T+1/T+2/T+3多周期融合** |
| 标准化 | StandardScaler | **RobustScaler（抗异常值）** |
| 样本权重 | 等权 | **时间衰减权重（半衰期120日）** |
| 概率校准 | SVC probability | **Platt/Isotonic校准** |
| 换手率控制 | 无 | **交易成本罚函数优化** |
| 交易成本 | 简化佣金+印花税 | **港股完整费率结构** |

---

## 8. 风险提示

1. **因子衰减风险**：191个因子源自A股2010-2017年数据，港股市场结构不同，部分因子可能失效
2. **适配风险**：港股无涨跌停、T+0交易、每手股数不同，需充分测试
3. **容量限制**：短周期高换手策略资金容纳规模有限（参考报告估算2-3亿HKD）
4. **过拟合风险**：191个因子在样本内可能过拟合，需严格的样本外验证
5. **执行风险**：高频调仓对交易执行质量要求极高

**本系统仅供学习研究，不构成任何投资建议。**

---

*文档版本: v2.0 | 更新日期: 2026-04-05*
*参考文献: 国泰君安《基于短周期价量特征的多因子选股体系》(2017.06.15)*
