"""
alpha_factors.py — 基于国泰君安《短周期价量特征》研报的 Alpha 因子库
v2.0

新增四大类因子，填补原特征体系的空白：
  1. 资金流向类  — Alpha9/Alpha11/Alpha43/Alpha55/Alpha60 变体
  2. 开盘缺口类  — Alpha15/Alpha107 变体
  3. 量能异常类  — Alpha40/Alpha43/Alpha80 变体
  4. 复合统计类  — Alpha25/Alpha1/Alpha5/Alpha42 变体

设计原则：
  - 全部纯 pandas/numpy，无外部依赖
  - 返回值已相对化（无量纲），可直接与现有特征合并
  - 每个函数独立，方便单独调用或批量禁用
  - 港股适配：窗口参数默认偏短（港股流动性波动更快）

参考：
  国泰君安证券研究《基于短周期价量特征的多因子选股体系》2017.06.15
"""
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """安全除法，分母为0时返回NaN"""
    return a / b.replace(0, np.nan)


def _ts_rank(series: pd.Series, window: int) -> pd.Series:
    """
    时序排名（TSRANK）：当前值在过去 window 天中的升序排位（1~window）
    与截面排名不同，这是单只股票自身的时间序列排名
    """
    return series.rolling(window).apply(
        lambda x: pd.Series(x).rank(ascending=True).iloc[-1],
        raw=False
    )


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window).corr(b)


# ══════════════════════════════════════════════════════════════════════
# 第一类：资金流向因子
# 核心逻辑：收盘价靠近最高价 + 放量 = 主动买入；靠近最低价 + 放量 = 主动卖出
# 港股适配：机构持仓高，盘口信息更干净，此类因子信噪比高于A股
# ══════════════════════════════════════════════════════════════════════

def calc_money_flow_strength(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 6
) -> pd.DataFrame:
    """
    资金流向强度（Alpha11/Alpha60 变体）
    
    原始公式：SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW) * VOLUME, n)
    
    每根K线赋予一个"方向权重"：
      收盘靠近最高价（多方主导）→ 权重为正
      收盘靠近最低价（空方主导）→ 权重为负
    再乘以成交量累加，得到带方向的累积资金流
    
    Returns:
        mf_strength_{window}: 资金流向强度（相对化）
        mf_ratio_{window}:    多空资金比（>1 多方占优）
    """
    hl = (high - low).replace(0, np.nan)
    # 克林格方向权重：[-1, +1] 之间
    direction = _safe_div((close - low) - (high - close), hl)
    signed_vol = direction * volume

    result = {}

    # 累积资金流（用均值相对化避免量纲问题）
    raw_flow = signed_vol.rolling(window).sum()
    avg_vol  = volume.rolling(window).mean().replace(0, np.nan)
    result[f'mf_strength_{window}'] = _safe_div(raw_flow, avg_vol * window)

    # 多空分拆：上涨日成交量 vs 下跌日成交量
    up_vol   = volume.where(close >= close.shift(1), 0.0).rolling(window).sum()
    down_vol = volume.where(close <  close.shift(1), 0.0).rolling(window).sum()
    result[f'mf_ratio_{window}'] = _safe_div(up_vol, down_vol.replace(0, np.nan))

    return pd.DataFrame(result, index=close.index)


def calc_clinger_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10
) -> pd.DataFrame:
    """
    克林格资金流指标（Alpha9 变体）
    
    原始公式：SMA(((H+L)/2 - (prev_H+prev_L)/2) * (H-L)/V, 7, 2)
    
    衡量价格中枢移动幅度与成交量的比值。
    价格中枢大幅上移但成交量不足 → 弱势上涨信号
    价格中枢小幅移动但成交量巨大 → 主力建仓信号
    
    Returns:
        clf_fast: 快线（3日EMA）
        clf_slow: 慢线（10日EMA）
        clf_signal: 快慢线差值（动量）
    """
    mid       = (high + low) / 2
    mid_shift = (high.shift(1) + low.shift(1)) / 2
    # 每根K线的"价量效率"：中枢移动 / (量/振幅)
    hl_vol    = _safe_div(high - low, volume.replace(0, np.nan))
    raw       = (mid - mid_shift) * hl_vol

    clf_fast   = raw.ewm(span=fast,  adjust=False).mean()
    clf_slow   = raw.ewm(span=slow,  adjust=False).mean()
    clf_signal = clf_fast - clf_slow

    return pd.DataFrame({
        'clf_fast':   clf_fast,
        'clf_slow':   clf_slow,
        'clf_signal': clf_signal,
    }, index=close.index)


def calc_price_position_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14
) -> pd.DataFrame:
    """
    价位资金流（Alpha128/Alpha52 变体）
    
    思路：用典型价（(H+L+C)/3）作为每日价格锚点，
    典型价上升且放量 → 买盘主导；典型价下降且放量 → 卖盘主导
    
    Returns:
        ppf_{window}: 多空资金流比值（>50 多方占优，<50 空方占优）
    """
    tp = (high + low + close) / 3
    tp_prev = tp.shift(1)

    # 多方资金流：典型价高于前日时的 tp×volume
    bull_flow = (tp * volume).where(tp > tp_prev, 0.0).rolling(window).sum()
    bear_flow = (tp * volume).where(tp < tp_prev, 0.0).rolling(window).sum()

    total = (bull_flow + bear_flow).replace(0, np.nan)
    ppf   = _safe_div(bull_flow, total) * 100  # [0, 100]

    return pd.DataFrame({f'ppf_{window}': ppf}, index=close.index)


# ══════════════════════════════════════════════════════════════════════
# 第二类：开盘缺口因子
# 核心逻辑：隔夜信息差被集中定价在开盘瞬间，缺口方向预示当日趋势
# 港股适配：受美股ADR、隔夜期货影响大，开盘缺口信息含量高于A股
# ══════════════════════════════════════════════════════════════════════

def calc_gap_features(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """
    开盘缺口综合特征（Alpha15/Alpha107 变体）
    
    Alpha15（原版）: OPEN/DELAY(CLOSE,1) - 1
    Alpha107（原版）: -1 × RANK(OPEN - DELAY(HIGH,1)) 
                      × RANK(OPEN - DELAY(CLOSE,1))
                      × RANK(OPEN - DELAY(LOW,1))
    
    三维缺口分析：
      gap_pct:        开盘缺口幅度（相对于昨收）
      gap_vs_high:    开盘相对昨日最高价的偏离（高开强度）
      gap_vs_low:     开盘相对昨日最低价的偏离（低开强度）
      gap_fill_ratio: 缺口回补程度（当日收盘是否向昨收靠拢）
      gap_alpha107:   三维缺口乘积（Alpha107 简化版）
    
    Returns:
        DataFrame with 5 gap feature columns
    """
    prev_close = close.shift(1)
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)

    result = {}

    # 基础缺口幅度（Alpha15）
    result['gap_pct'] = _safe_div(open_ - prev_close, prev_close)

    # 三维缺口偏离（Alpha107 拆解）
    result['gap_vs_high']  = _safe_div(open_ - prev_high, prev_close)
    result['gap_vs_low']   = _safe_div(open_ - prev_low,  prev_close)

    # 缺口回补率：当日收盘是否向昨收方向运动
    # 正值 = 缺口被强化（顺方向），负值 = 缺口被回补（反转）
    gap_dir  = np.sign(open_ - prev_close)
    fill_dist = (close - open_) * gap_dir
    result['gap_fill_ratio'] = _safe_div(fill_dist, (open_ - prev_close).abs().replace(0, np.nan))

    # Alpha107 变体：三个偏离量的乘积符号（-1 取负后为看涨信号）
    d1 = open_ - prev_high
    d2 = open_ - prev_close
    d3 = open_ - prev_low
    # 都为负（低开低于所有前日价位）→ 乘积为负 → 取负后为正（反转买入）
    result['gap_alpha107'] = -1 * np.sign(d1 * d2 * d3) * (
        d1.abs() * d2.abs() * d3.abs()
    ) ** (1/3)  # 几何均值保留量纲

    return pd.DataFrame(result, index=close.index)


def calc_gap_momentum(
    open_: pd.Series,
    close: pd.Series,
    window: int = 5
) -> pd.DataFrame:
    """
    开盘缺口动量：衡量近期缺口的方向一致性
    
    如果过去 window 天持续高开，说明隔夜买盘持续强劲；
    持续低开则是隔夜抛压持续释放。
    
    Returns:
        gap_momentum_{window}: 缺口方向一致性得分
    """
    gap = _safe_div(open_ - close.shift(1), close.shift(1))
    gap_momentum = gap.rolling(window).mean()  # 平均缺口幅度

    return pd.DataFrame({
        f'gap_momentum_{window}': gap_momentum
    }, index=close.index)


# ══════════════════════════════════════════════════════════════════════
# 第三类：量能异常因子
# 核心逻辑：异常成交量是市场参与者结构变化的信号
# 港股适配：港股蓝筹有大量机构持仓，量能异常更能体现机构行为
# ══════════════════════════════════════════════════════════════════════

def calc_volume_anomaly(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20
) -> pd.DataFrame:
    """
    量能异常综合指标（Alpha40/Alpha43/Alpha80 变体）
    
    Alpha40: SUM(CLOSE>PREV_CLOSE ? VOLUME : 0, 26) /
             SUM(CLOSE<=PREV_CLOSE ? VOLUME : 0, 26) × 100
    Alpha43: SUM(CLOSE>PREV ? VOL : CLOSE<PREV ? -VOL : 0, 6)  有符号成交量
    Alpha80: (VOLUME - DELAY(VOLUME,5)) / DELAY(VOLUME,5) × 100  量变化率
    
    Returns:
        vol_updown_ratio: 上涨日成交量 / 下跌日成交量（OBV变体）
        signed_vol_{w}:   有符号累积成交量（区分方向的量能）
        vol_surge_{w}:    量能突破（标准差倍数）
        vol_pct_5d:       5日量变化率
    """
    prev_close = close.shift(1)
    result = {}

    # Alpha40 变体：上涨/下跌日成交量比值（OBV思路）
    up_vol   = volume.where(close > prev_close,  0.0).rolling(window).sum()
    dn_vol   = volume.where(close <= prev_close, 0.0).rolling(window).sum()
    result['vol_updown_ratio'] = _safe_div(up_vol, dn_vol.replace(0, np.nan))

    # Alpha43 变体：有符号累积成交量（短周期）
    signed = pd.Series(np.where(
        close > prev_close,  volume,
        np.where(close < prev_close, -volume, 0.0)
    ), index=close.index)
    for w in [6, 12]:
        result[f'signed_vol_{w}'] = signed.rolling(w).sum() / (
            volume.rolling(w).mean().replace(0, np.nan) * w
        )

    # 量能突破：当日量相对于历史均值偏离几个标准差
    vol_mean = volume.rolling(window).mean()
    vol_std  = volume.rolling(window).std().replace(0, np.nan)
    result['vol_surge_20'] = _safe_div(volume - vol_mean, vol_std)

    # Alpha80 变体：5日量变化率
    vol_5d_ago = volume.shift(5).replace(0, np.nan)
    result['vol_pct_5d'] = _safe_div(volume - vol_5d_ago, vol_5d_ago) * 100

    return pd.DataFrame(result, index=close.index)


def calc_volume_price_divergence(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 6
) -> pd.DataFrame:
    """
    价量背离指标（Alpha1/量幅背离 变体）
    
    原版价量背离：-1 × CORR(VWAP, VOLUME, n)
    量幅背离：    -1 × CORR(HIGH/LOW, VOLUME, n)  ← 报告中最强单因子(IR=8.39)
    
    背离越强（相关系数越负），反转概率越高
    
    Returns:
        vpd_price_vol_{w}: 均价-量相关系数（取负）
        vpd_range_vol_{w}: 振幅-量相关系数（取负，量幅背离）
    """
    vwap = _safe_div(close * volume, volume)  # 简化 VWAP
    hl_ratio = _safe_div(high, low)           # 振幅替代

    result = {}
    result[f'vpd_price_vol_{window}']  = -1 * _rolling_corr(vwap,     volume, window)
    result[f'vpd_range_vol_{window}']  = -1 * _rolling_corr(hl_ratio, volume, window)

    return pd.DataFrame(result, index=close.index)


# ══════════════════════════════════════════════════════════════════════
# 第四类：复合统计因子
# 核心逻辑：单因子IC极低，多维度复合可显著提升信噪比
# 港股适配：港股有250天基准足够长，复合统计统计显著性可保证
# ══════════════════════════════════════════════════════════════════════

def calc_alpha25_composite(
    close: pd.Series,
    volume: pd.Series,
    short_window: int = 7,
    long_window: int = 250,
    vol_window: int = 20
) -> pd.DataFrame:
    """
    Alpha25 复合因子：短期动量 × 量能衰减 × 长期动量修正
    
    原始公式：
    -1 × RANK(DELTA(CLOSE, 7) × (1 - RANK(DECAYLINEAR(VOL/MEAN_VOL, 9))))
        × (1 + RANK(SUM(RET, 250)))
    
    三个维度含义：
      短期动量：7日价格变化（近期走势方向）
      量能权重：成交量相对均值的线性衰减权重（量大的日子权重更高）
      长期修正：过去一年累积收益的排名（控制长期趋势方向）
    
    结合逻辑：
      短期下跌 + 量能在衰减 + 长期强势 → 做空被压制，反转买入
      短期上涨 + 量能在衰减 + 长期弱势 → 虚涨即将结束，做空
    
    Returns:
        alpha25: 复合信号强度
        alpha25_momentum_only: 仅短期动量部分（对照组）
    """
    ret = close.pct_change(1)

    # 短期动量：7日价格变化率
    delta_7 = close.pct_change(7)

    # 量能线性衰减权重（DECAYLINEAR 近似：用指数加权替代）
    vol_ratio = _safe_div(volume, volume.rolling(vol_window).mean())
    vol_decay = vol_ratio.ewm(span=9, adjust=False).mean()

    # 量能衰减因子：量比越高，(1 - 量能排名) 越低，对动量的压制越小
    # 这里用滚动百分位近似截面 RANK
    vol_decay_rank = vol_decay.rolling(252).rank(pct=True).fillna(0.5)
    vol_factor = 1 - vol_decay_rank  # [0,1]，量能低时接近1（放大动量）

    # 长期动量修正
    long_ret_rank = ret.rolling(long_window).sum().rolling(252).rank(pct=True).fillna(0.5)
    long_factor = 1 + long_ret_rank  # [1,2]，长期强势时接近2

    # 合成：短期动量 × 量能调整 × 长期修正，取负（做反转）
    raw = delta_7 * vol_factor * long_factor
    alpha25 = -1 * raw.rolling(252).rank(pct=True).fillna(0.5)

    return pd.DataFrame({
        'alpha25':              alpha25,
        'alpha25_vol_factor':   vol_factor,
        'alpha25_long_factor':  long_ret_rank,
    }, index=close.index)


def calc_alpha42_volatility_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 10
) -> pd.DataFrame:
    """
    Alpha42 波动率×量价协同因子
    
    原始公式：-1 × RANK(STD(HIGH, 10)) × CORR(HIGH, VOLUME, 10)
    
    两个维度：
      高价波动率：最高价的10日标准差排名（波动大 = 主力活跃）
      量价协同度：最高价与成交量的相关系数（同向 = 拉高出货）
    
    乘积取负：波动大且量价正相关 → 拉高出货形态 → 卖出信号
    
    Returns:
        alpha42:        合成因子
        high_vol_rank:  高价波动率百分位
        high_vol_corr:  高价量价相关系数
    """
    high_std  = high.rolling(window).std()
    high_corr = _rolling_corr(high, volume, window)

    # 截面排名用时序百分位近似
    high_vol_rank = high_std.rolling(252).rank(pct=True).fillna(0.5)

    alpha42 = -1 * high_vol_rank * high_corr

    return pd.DataFrame({
        'alpha42':         alpha42,
        'high_vol_rank':   high_vol_rank,
        'high_vol_corr':   high_corr,
    }, index=close.index)


def calc_alpha5_tsrank_flow(
    high: pd.Series,
    volume: pd.Series,
    rank_window: int = 5,
    corr_window: int = 5,
    max_window:  int = 3
) -> pd.DataFrame:
    """
    Alpha5 VWAP成交量时序排名相关性
    
    原始公式：-1 × TSMAX(CORR(TSRANK(VOLUME,5), TSRANK(HIGH,5), 5), 3)
    
    含义：过去5天，量大的那天是不是也是高点高的那天？
    如果量和价的时序排名高度正相关（量堆价），预示见顶
    取过去3天的最大值，捕捉"量堆价"最严重的时刻
    
    Returns:
        alpha5: 量堆价信号（越高越看空）
    """
    ts_vol  = _ts_rank(volume, rank_window)
    ts_high = _ts_rank(high,   rank_window)

    corr   = _rolling_corr(ts_vol, ts_high, corr_window)
    alpha5 = -1 * corr.rolling(max_window).max()

    return pd.DataFrame({'alpha5': alpha5}, index=high.index)


def calc_momentum_reversal_composite(
    close: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """
    动量-反转复合因子
    
    在不同时间尺度上动量与反转的切换：
      1-5日：反转效应为主（追涨杀跌被修正）
      5-20日：动量效应为主（趋势延续）
      20日+：均值回归
    
    综合三个尺度，生成更稳定的复合信号
    
    Returns:
        mr_ultra_short: 超短期反转信号（1-3日）
        mr_short:       短期反转信号（5日）
        mr_medium:      中期动量信号（10-20日）
        mr_composite:   综合复合信号
    """
    ret1  = close.pct_change(1)
    ret3  = close.pct_change(3)
    ret5  = close.pct_change(5)
    ret10 = close.pct_change(10)
    ret20 = close.pct_change(20)

    # 量能加权（成交量大时信号权重更高）
    vol_weight = _safe_div(volume, volume.rolling(20).mean()).clip(0.2, 5)

    # 超短期反转（1-3日）：取负代表反转倾向
    mr_ultra = -1 * (ret1 * 0.6 + ret3 * 0.4) * vol_weight

    # 短期反转（5日）
    mr_short = -1 * ret5 * vol_weight

    # 中期动量（10-20日）：正号代表趋势延续
    mr_medium = (ret10 * 0.4 + ret20 * 0.6)

    # 综合：超短期反转 + 中期动量（两者通常互补）
    mr_composite = mr_ultra * 0.3 + mr_short * 0.3 + mr_medium * 0.4

    return pd.DataFrame({
        'mr_ultra_short': mr_ultra,
        'mr_short':       mr_short,
        'mr_medium':      mr_medium,
        'mr_composite':   mr_composite,
    }, index=close.index)


# ══════════════════════════════════════════════════════════════════════
# 统一入口：一次性计算所有新增因子
# ══════════════════════════════════════════════════════════════════════

def calc_all_alpha_factors(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """
    计算所有新增 Alpha 因子，返回合并后的 DataFrame
    
    调用方只需传入基础 OHLCV 序列，即可获取全部新增特征。
    
    新增特征数量：约 25 个
    总特征维度（含原有36个）：约 61 个
    
    Args:
        open_:  开盘价序列
        high:   最高价序列
        low:    最低价序列
        close:  收盘价序列
        volume: 成交量序列
    
    Returns:
        DataFrame，index 与输入序列一致，列为所有新增因子
    """
    parts = []

    # ── 资金流向（4类，约9个特征）──────────────────────────
    parts.append(calc_money_flow_strength(high, low, close, volume, window=6))
    parts.append(calc_money_flow_strength(high, low, close, volume, window=12))
    parts.append(calc_clinger_flow(high, low, close, volume))
    parts.append(calc_price_position_flow(high, low, close, volume, window=14))

    # ── 开盘缺口（2类，约6个特征）──────────────────────────
    parts.append(calc_gap_features(open_, high, low, close))
    parts.append(calc_gap_momentum(open_, close, window=5))

    # ── 量能异常（2类，约8个特征）──────────────────────────
    parts.append(calc_volume_anomaly(close, volume, window=20))
    parts.append(calc_volume_price_divergence(high, low, close, volume, window=6))

    # ── 复合统计（4类，约12个特征）─────────────────────────
    parts.append(calc_alpha25_composite(close, volume))
    parts.append(calc_alpha42_volatility_flow(high, low, close, volume))
    parts.append(calc_alpha5_tsrank_flow(high, volume))
    parts.append(calc_momentum_reversal_composite(close, volume))

    result = pd.concat(parts, axis=1)

    # 去重列（多次调用相同窗口时可能出现）
    result = result.loc[:, ~result.columns.duplicated()]

    return result
