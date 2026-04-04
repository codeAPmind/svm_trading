"""
回测绩效指标计算
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

from backtest.backtest_engine import Trade


@dataclass
class BacktestMetrics:
    total_return:          float   # 总收益率
    annual_return:         float   # 年化收益率
    sharpe_ratio:          float   # 夏普比率
    max_drawdown:          float   # 最大回撤
    max_drawdown_duration: int     # 最大回撤持续交易日数
    win_rate:              float   # 胜率
    profit_loss_ratio:     float   # 盈亏比
    total_trades:          int     # 买入次数
    avg_holding_days:      float   # 平均持仓天数
    annual_volatility:     float   # 年化波动率
    calmar_ratio:          float   # 卡尔玛比率


def calculate_metrics(
    portfolio_df: pd.DataFrame,
    trades: List[Trade],
    risk_free_rate: float = 0.02
) -> BacktestMetrics:
    """
    Args:
        portfolio_df:    BacktestEngine.run() 返回的每日净值 DataFrame
        trades:          Trade 列表
        risk_free_rate:  无风险年化利率
    """
    ANNUAL_FACTOR = 252  # 港股年交易日

    returns     = portfolio_df['total_value'].pct_change().dropna()
    total_days  = len(portfolio_df)
    years       = total_days / ANNUAL_FACTOR

    # 总收益
    total_return = float(portfolio_df['return'].iloc[-1])

    # 年化收益
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # 年化波动率
    annual_vol = float(returns.std() * np.sqrt(ANNUAL_FACTOR))

    # 夏普比率
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0

    # 最大回撤 & 持续时间
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd   = float(drawdown.min())

    dd_start = None
    dd_dur   = 0
    max_dur  = 0
    for dd_val in drawdown:
        if dd_val < 0:
            if dd_start is None:
                dd_start = 0
            dd_dur += 1
            max_dur = max(max_dur, dd_dur)
        else:
            dd_start = None
            dd_dur   = 0

    # 卡尔玛比率
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

    # 交易统计（配对买卖）
    buys  = [t for t in trades if t.action == 'BUY']
    sells = [t for t in trades if t.action == 'SELL']

    profits, holding_days = [], []
    for i, sell in enumerate(sells):
        if i < len(buys):
            buy = buys[i]
            profits.append((sell.price - buy.price) / buy.price)
            holding_days.append((sell.date - buy.date).days)

    win_rate = sum(1 for p in profits if p > 0) / len(profits) if profits else 0.0
    avg_win  = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0.0
    avg_loss = abs(np.mean([p for p in profits if p < 0])) if any(p < 0 for p in profits) else 1.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    return BacktestMetrics(
        total_return          = round(total_return, 4),
        annual_return         = round(annual_return, 4),
        sharpe_ratio          = round(sharpe, 4),
        max_drawdown          = round(max_dd, 4),
        max_drawdown_duration = max_dur,
        win_rate              = round(win_rate, 4),
        profit_loss_ratio     = round(pl_ratio, 4),
        total_trades          = len(buys),
        avg_holding_days      = round(float(np.mean(holding_days)), 1) if holding_days else 0.0,
        annual_volatility     = round(annual_vol, 4),
        calmar_ratio          = round(calmar, 4),
    )


def print_metrics_report(metrics: BacktestMetrics):
    """格式化输出回测绩效报告"""
    print("\n" + "=" * 60)
    print("               回 测 绩 效 报 告")
    print("=" * 60)
    print(f"  总收益率:          {metrics.total_return:>10.2%}")
    print(f"  年化收益率:        {metrics.annual_return:>10.2%}")
    print(f"  年化波动率:        {metrics.annual_volatility:>10.2%}")
    print(f"  夏普比率:          {metrics.sharpe_ratio:>10.4f}")
    print(f"  最大回撤:          {metrics.max_drawdown:>10.2%}")
    print(f"  最大回撤持续(天):  {metrics.max_drawdown_duration:>10d}")
    print(f"  卡尔玛比率:        {metrics.calmar_ratio:>10.4f}")
    print("-" * 60)
    print(f"  总买入次数:        {metrics.total_trades:>10d}")
    print(f"  胜率:              {metrics.win_rate:>10.2%}")
    print(f"  盈亏比:            {metrics.profit_loss_ratio:>10.4f}")
    print(f"  平均持仓天数:      {metrics.avg_holding_days:>10.1f}")
    print("=" * 60)


def metrics_to_dict(metrics: BacktestMetrics) -> dict:
    """转换为字典，方便序列化"""
    return {
        'total_return':          f"{metrics.total_return:.2%}",
        'annual_return':         f"{metrics.annual_return:.2%}",
        'annual_volatility':     f"{metrics.annual_volatility:.2%}",
        'sharpe_ratio':          metrics.sharpe_ratio,
        'max_drawdown':          f"{metrics.max_drawdown:.2%}",
        'max_drawdown_duration': metrics.max_drawdown_duration,
        'calmar_ratio':          metrics.calmar_ratio,
        'total_trades':          metrics.total_trades,
        'win_rate':              f"{metrics.win_rate:.2%}",
        'profit_loss_ratio':     metrics.profit_loss_ratio,
        'avg_holding_days':      metrics.avg_holding_days,
    }
