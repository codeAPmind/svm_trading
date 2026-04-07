"""
可视化模块
生成回测报告图表（4 子图布局）及技术指标图
"""
import matplotlib
matplotlib.use('Agg')   # 非交互后端，适合无 GUI 环境
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path

# 中文字体支持（Windows: SimHei，macOS: PingFang SC，Linux: DejaVu）
plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).parent.parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def _suffix_text(file_suffix: str | None) -> str:
    return f"_{file_suffix}" if file_suffix else ""


def plot_backtest_report(
    price_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    code: str,
    save_path: str = None,
    file_suffix: str | None = None,
):
    """
    生成完整的回测报告图表（4 子图）

    子图1: 股价走势 + MA20/MA60 + 买卖信号标注
    子图2: 策略净值 vs 基准（买入持有）
    子图3: 回撤曲线
    子图4: 成交量（涨跌染色）

    Args:
        price_df:     价格数据（index=DatetimeIndex, columns含 close/open/volume）
        signals_df:   信号数据（index=date, signal / action 列）
        portfolio_df: 组合净值（total_value 列）
        code:         股票代码（图表标题用）
        save_path:    保存路径（None 则自动命名至 output/）
    """
    if save_path is None:
        suffix = _suffix_text(file_suffix)
        save_path = str(OUTPUT_DIR / f"backtest_{code.replace('.', '_')}{suffix}.png")

    fig, axes = plt.subplots(
        4, 1, figsize=(16, 22),
        gridspec_kw={'height_ratios': [3, 2, 1, 1]},
        sharex=True
    )
    fig.suptitle(f'SVM 交易策略回测报告  —  {code}', fontsize=15, fontweight='bold', y=0.98)

    # ── 子图 1: 股价 + 均线 + 买卖点 ─────────────────────────
    ax1 = axes[0]
    ax1.plot(price_df.index, price_df['close'], color='#333333', linewidth=0.9, label='收盘价', zorder=2)
    if len(price_df) >= 20:
        ma20 = price_df['close'].rolling(20).mean()
        ax1.plot(price_df.index, ma20, color='#2196F3', linewidth=0.7, alpha=0.8, label='MA20')
    if len(price_df) >= 60:
        ma60 = price_df['close'].rolling(60).mean()
        ax1.plot(price_df.index, ma60, color='#FF9800', linewidth=0.7, alpha=0.8, label='MA60')

    # 买卖点
    buy_dates  = signals_df[signals_df['signal'] == 1].index.intersection(price_df.index)
    sell_dates = signals_df[signals_df['signal'] == -1].index.intersection(price_df.index)

    if len(buy_dates) > 0:
        ax1.scatter(buy_dates, price_df.loc[buy_dates, 'close'],
                    marker='^', color='#4CAF50', s=120, zorder=5, label=f'买入 ({len(buy_dates)})')
    if len(sell_dates) > 0:
        ax1.scatter(sell_dates, price_df.loc[sell_dates, 'close'],
                    marker='v', color='#F44336', s=120, zorder=5, label=f'卖出 ({len(sell_dates)})')

    ax1.set_title('股价走势与买卖信号', fontsize=11)
    ax1.set_ylabel('价格 (HKD)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.25)

    # ── 子图 2: 策略净值 vs 基准 ──────────────────────────────
    ax2 = axes[1]
    strat_ret  = portfolio_df['total_value'] / portfolio_df['total_value'].iloc[0]
    bench_ret  = price_df['close'].reindex(portfolio_df.index).ffill()
    bench_ret  = bench_ret / bench_ret.iloc[0]

    ax2.plot(portfolio_df.index, strat_ret, color='#4CAF50', linewidth=1.2, label='策略净值')
    ax2.plot(portfolio_df.index, bench_ret,  color='#9E9E9E', linewidth=0.8, alpha=0.8, label='基准（买入持有）')
    ax2.fill_between(portfolio_df.index, strat_ret, bench_ret,
                     where=strat_ret >= bench_ret, alpha=0.12, color='#4CAF50', label='超额')
    ax2.fill_between(portfolio_df.index, strat_ret, bench_ret,
                     where=strat_ret < bench_ret,  alpha=0.12, color='#F44336', label='跑输')
    ax2.axhline(1.0, color='#BDBDBD', linewidth=0.5, linestyle='--')
    ax2.set_title('策略净值 vs 基准（买入持有）', fontsize=11)
    ax2.set_ylabel('净值（初始=1）')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.25)

    # ── 子图 3: 回撤曲线 ──────────────────────────────────────
    ax3 = axes[2]
    cum    = strat_ret
    runmax = cum.cummax()
    dd     = (cum - runmax) / runmax
    ax3.fill_between(dd.index, dd, 0, color='#F44336', alpha=0.35)
    ax3.plot(dd.index, dd, color='#E53935', linewidth=0.7)
    ax3.set_title('策略回撤', fontsize=11)
    ax3.set_ylabel('回撤幅度')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax3.grid(True, alpha=0.25)

    # ── 子图 4: 成交量（涨跌染色）────────────────────────────
    ax4 = axes[3]
    if 'volume' in price_df.columns and 'open' in price_df.columns:
        colors = [
            '#EF5350' if c >= o else '#26A69A'
            for c, o in zip(price_df['close'], price_df['open'])
        ]
        ax4.bar(price_df.index, price_df['volume'], color=colors, alpha=0.65, width=1.0)
    ax4.set_title('成交量', fontsize=11)
    ax4.set_ylabel('成交量')
    ax4.grid(True, alpha=0.25)

    # 日期格式
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Charts] 回测报告已保存至 {save_path}")
    return save_path


def plot_signal_distribution(
    signals_df: pd.DataFrame,
    code: str,
    save_path: str = None,
    file_suffix: str | None = None,
):
    """
    绘制信号置信度分布图
    """
    if save_path is None:
        suffix = _suffix_text(file_suffix)
        save_path = str(OUTPUT_DIR / f"signal_dist_{code.replace('.', '_')}{suffix}.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'信号分布分析 — {code}', fontsize=13)

    # 信号类别分布
    ax1 = axes[0]
    counts = signals_df['signal'].value_counts().sort_index()
    labels = {-1: '卖出', 0: '观望', 1: '买入'}
    colors = ['#F44336', '#9E9E9E', '#4CAF50']
    bars = ax1.bar(
        [labels.get(k, str(k)) for k in counts.index],
        counts.values,
        color=[colors[i % 3] for i in range(len(counts))]
    )
    ax1.set_title('信号类别分布')
    ax1.set_ylabel('次数')
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha='center', va='bottom', fontsize=9)

    # 置信度直方图
    ax2 = axes[1]
    ax2.hist(signals_df['confidence'], bins=20, color='#2196F3', alpha=0.7, edgecolor='white')
    ax2.axvline(signals_df['confidence'].mean(), color='#FF9800', linewidth=1.5,
                linestyle='--', label=f"均值={signals_df['confidence'].mean():.2f}")
    ax2.set_title('预测置信度分布')
    ax2.set_xlabel('置信度')
    ax2.set_ylabel('频次')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"[Charts] 信号分布图已保存至 {save_path}")
    return save_path


def plot_technical_indicators(
    df: pd.DataFrame,
    code: str,
    days: int = 120,
    save_path: str = None,
    file_suffix: str | None = None,
):
    """
    绘制技术指标面板（K线 + MACD + RSI + KDJ）
    """
    if save_path is None:
        suffix = _suffix_text(file_suffix)
        save_path = str(OUTPUT_DIR / f"tech_{code.replace('.', '_')}{suffix}.png")

    df = df.tail(days).copy()

    fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                              gridspec_kw={'height_ratios': [3, 1, 1, 1]}, sharex=True)
    fig.suptitle(f'技术指标面板 — {code}（最近 {days} 日）', fontsize=13)

    # K线（简化版：只画收盘价 + 布林带）
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], color='#333', linewidth=1.0, label='Close')
    if 'bb_upper' in df.columns:
        ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'],
                         alpha=0.1, color='#2196F3', label='布林带')
        ax1.plot(df.index, df['bb_upper'],  color='#2196F3', linewidth=0.5, alpha=0.6)
        ax1.plot(df.index, df['bb_middle'], color='#2196F3', linewidth=0.7, linestyle='--', alpha=0.5)
        ax1.plot(df.index, df['bb_lower'],  color='#2196F3', linewidth=0.5, alpha=0.6)
    ax1.set_ylabel('价格')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)

    # MACD
    ax2 = axes[1]
    if 'macd_dif' in df.columns:
        ax2.plot(df.index, df['macd_dif'],  color='#1565C0', linewidth=0.9, label='DIF')
        ax2.plot(df.index, df['macd_dea'],  color='#E53935', linewidth=0.9, label='DEA')
        colors_macd = ['#EF5350' if v < 0 else '#4CAF50' for v in df['macd_hist']]
        ax2.bar(df.index, df['macd_hist'], color=colors_macd, alpha=0.6, width=0.8)
        ax2.axhline(0, color='gray', linewidth=0.5)
        ax2.legend(fontsize=8)
    ax2.set_ylabel('MACD')
    ax2.grid(True, alpha=0.25)

    # RSI
    ax3 = axes[2]
    if 'rsi_12' in df.columns:
        ax3.plot(df.index, df['rsi_12'], color='#7B1FA2', linewidth=0.9, label='RSI(12)')
        ax3.axhline(70, color='#E53935', linewidth=0.5, linestyle='--', alpha=0.7)
        ax3.axhline(30, color='#4CAF50', linewidth=0.5, linestyle='--', alpha=0.7)
        ax3.fill_between(df.index, df['rsi_12'], 70,
                         where=df['rsi_12'] > 70, alpha=0.15, color='#E53935')
        ax3.fill_between(df.index, df['rsi_12'], 30,
                         where=df['rsi_12'] < 30, alpha=0.15, color='#4CAF50')
        ax3.set_ylim(0, 100)
        ax3.legend(fontsize=8)
    ax3.set_ylabel('RSI')
    ax3.grid(True, alpha=0.25)

    # KDJ
    ax4 = axes[3]
    if 'kdj_k' in df.columns:
        ax4.plot(df.index, df['kdj_k'], color='#1565C0', linewidth=0.8, label='K')
        ax4.plot(df.index, df['kdj_d'], color='#E53935', linewidth=0.8, label='D')
        ax4.plot(df.index, df['kdj_j'], color='#4CAF50', linewidth=0.8, label='J')
        ax4.axhline(80, color='#E53935', linewidth=0.4, linestyle='--', alpha=0.6)
        ax4.axhline(20, color='#4CAF50', linewidth=0.4, linestyle='--', alpha=0.6)
        ax4.legend(fontsize=8)
    ax4.set_ylabel('KDJ')
    ax4.grid(True, alpha=0.25)

    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax4.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"[Charts] 技术指标图已保存至 {save_path}")
    return save_path
