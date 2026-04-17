"""
主控模块 - SVM 港股交易策略系统

支持三种运行模式:
  backtest  - 历史数据训练 + 回测验证 + 图表输出
  realtime  - 加载已训练模型，生成当日买卖信号 + AI 多空分析
  full      - 回测 + 实时信号 + AI 分析（完整流程）

用法示例:
  python main.py --code HK.00700 --name 腾讯控股 --start 2019-01-01 --end 2025-12-31 --mode full
  python main.py --code HK.09988 --mode backtest --no-optimize
  python main.py --code HK.00700 --mode realtime
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── 项目内部模块 ───────────────────────────────────────────────
from config.settings import SVM_CFG, BACKTEST_CFG
from config.symbols import DEFAULT_CODE, DEFAULT_NAME
from data.fmp_client import FMPDataClient
from data.symbol_mapper import SymbolMapper
from feature.feature_engineer import FeatureEngineer
from model.svm_model import SVMTradingModel
from model.signal_generator import SignalGenerator
from model.model_evaluator import ModelEvaluator
from backtest.backtest_engine import BacktestEngine
from backtest.metrics import calculate_metrics, print_metrics_report, metrics_to_dict
from analysis.ai_analyst import AIAnalyst
from analysis.macro_analysis import MacroAnalyzer
from notification.feishu_notifier import push_run_report
from visualization.charts import plot_backtest_report, plot_signal_distribution

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

SCALER_PATH = Path(__file__).parent / 'models' / 'feature_scaler.pkl'
FEATURES_PATH = Path(__file__).parent / 'models' / 'feature_columns.json'


def _market_currency(code: str) -> str:
    market = SymbolMapper.detect_market(code)
    if market == "HK":
        return "HKD"
    if market == "CN":
        return "CNY"
    return "USD"


# ══════════════════════════════════════════════════════════════
# 回测流程
# ══════════════════════════════════════════════════════════════
def run_backtest(
    code: str,
    start: str,
    end: str,
    optimize: bool = True,
    conf_threshold: float = 0.7,
    fast_search: bool = False,
    file_suffix: str | None = None,
) -> tuple:
    """
    完整回测流程（步骤 1~9）

    Returns:
        (model, fe, signals_df, metrics)
    """
    print("\n" + "=" * 65)
    print(f"  SVM 多市场交易策略(FMP) — 回测模式")
    print(f"  标的: {code}   区间: {start} ~ {end}")
    print("=" * 65)

    # ── Step 1: 获取历史K线 ────────────────────────────────────
    print("\n[Step 1] 获取历史K线数据...")
    client = FMPDataClient()
    client.connect()
    try:
        df = client.get_history_kline(code, start, end)
    finally:
        client.disconnect()

    print(f"  获取 {len(df)} 条K线 ({df.index[0].date()} ~ {df.index[-1].date()})")

    # ── Step 2: 特征工程 ───────────────────────────────────────
    print("\n[Step 2] 特征工程...")
    fe = FeatureEngineer()
    X, y, feature_cols, valid_index = fe.prepare_dataset(df)

    # ── Step 3: 时间序列划分 ───────────────────────────────────
    print("\n[Step 3] 时间序列划分...")
    split_idx = int(len(X) * (1 - SVM_CFG.test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = valid_index[split_idx:]

    print(f"  训练集: {split_idx} 样本  |  测试集: {len(X_test)} 样本")
    print(f"  测试区间: {dates_test[0].date()} ~ {dates_test[-1].date()}")

    # ── Step 4: SVM 训练 ───────────────────────────────────────
    print("\n[Step 4] SVM 模型训练" + ("（含 GridSearchCV）..." if optimize else "（快速模式）..."))
    model = SVMTradingModel()
    model.train(X_train, y_train, optimize=optimize, fast_search=fast_search)

    # ── Step 5: 模型评估 ───────────────────────────────────────
    print("\n[Step 5] 模型评估...")
    eval_result = model.evaluate(X_test, y_test)

    # 混淆矩阵可视化
    evaluator = ModelEvaluator(model, feature_cols)
    evaluator.plot_confusion_matrix(eval_result['confusion_matrix'], file_suffix=file_suffix)

    # ── Step 6: 生成交易信号 ───────────────────────────────────
    print("\n[Step 6] 生成交易信号...")
    sig_gen = SignalGenerator(confidence_threshold=conf_threshold)
    predictions  = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    signals_df = sig_gen.generate_signals(
        dates_test, predictions, probabilities, model.model.classes_
    )
    signals_df = sig_gen.filter_consecutive_signals(signals_df)

    buy_cnt  = (signals_df['signal'] == 1).sum()
    sell_cnt = (signals_df['signal'] == -1).sum()
    print(f"  买入信号: {buy_cnt} 次   卖出信号: {sell_cnt} 次")

    # ── Step 7: 回测执行 ───────────────────────────────────────
    print("\n[Step 7] 运行回测引擎...")
    engine = BacktestEngine()
    test_prices = df.loc[dates_test]
    portfolio_df = engine.run(test_prices, signals_df, code)

    # ── Step 8: 绩效指标 ───────────────────────────────────────
    print("\n[Step 8] 计算绩效指标...")
    metrics = calculate_metrics(portfolio_df, engine.trades)
    print_metrics_report(metrics)

    # 保存绩效到 JSON
    suffix = f"_{file_suffix}" if file_suffix else ""
    metrics_path = OUTPUT_DIR / f"metrics_{code.replace('.', '_')}{suffix}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'code': code, 'start': start, 'end': end,
            'eval': {
                'accuracy':   round(eval_result['accuracy'], 4),
                'f1_weighted': round(eval_result['f1_weighted'], 4),
            },
            'metrics': metrics_to_dict(metrics),
        }, f, ensure_ascii=False, indent=2)
    print(f"  绩效指标已保存至 {metrics_path}")

    # ── Step 9: 可视化 ────────────────────────────────────────
    print("\n[Step 9] 生成可视化报告...")
    plot_backtest_report(test_prices, signals_df, portfolio_df, code, file_suffix=file_suffix)
    plot_signal_distribution(signals_df, code, file_suffix=file_suffix)

    # ── Step 10: 保存模型 ─────────────────────────────────────
    model.save_model()
    SCALER_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(fe.scaler, SCALER_PATH)
    with open(FEATURES_PATH, 'w', encoding='utf-8') as f:
        json.dump(fe.feature_columns, f)
    print(f"  Scaler 已保存至 {SCALER_PATH}")

    return model, fe, signals_df, metrics


# ══════════════════════════════════════════════════════════════
# 实时信号生成
# ══════════════════════════════════════════════════════════════
def run_realtime_signal(
    code: str,
    model: SVMTradingModel,
    fe: FeatureEngineer,
    conf_threshold: float = 0.7
) -> tuple:
    """
    生成实时买卖信号

    Returns:
        (signal_info, featured_df, fundamental)
    """
    print("\n" + "=" * 65)
    print(f"  实时信号生成 — {code}")
    print("=" * 65)

    end_date   = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')

    client = FMPDataClient()
    client.connect()
    try:
        df         = client.get_history_kline(code, start_date, end_date)
        fundamental = client.get_financial_data(code)
    finally:
        client.disconnect()

    # 构建特征
    featured_df  = fe.build_features(df.copy())
    latest_scaled = fe.transform_latest(df)

    # 预测
    pred = model.predict(latest_scaled)[0]
    prob = model.predict_proba(latest_scaled)[0]

    classes   = model.model.classes_
    class_idx = int(np.where(classes == pred)[0][0])

    signal_info = {
        'signal':     int(pred),
        'confidence': float(prob[class_idx]),
        'prob_buy':   float(prob[np.where(classes == 1)[0][0]]) if 1 in classes else 0.0,
        'prob_sell':  float(prob[np.where(classes == -1)[0][0]]) if -1 in classes else 0.0,
    }

    # 避免在 GBK 控制台输出 emoji 导致编码错误
    signal_text = {1: '买入', -1: '卖出', 0: '观望'}.get(pred, '观望')
    print(f"\n  当前信号:  {signal_text}")
    print(f"  置信度:    {signal_info['confidence']:.1%}")
    print(f"  买入概率:  {signal_info['prob_buy']:.1%}")
    print(f"  卖出概率:  {signal_info['prob_sell']:.1%}")
    print(f"  最新收盘:  {df['close'].iloc[-1]:.2f} {_market_currency(code)}")

    return signal_info, featured_df, fundamental


# ══════════════════════════════════════════════════════════════
# AI 多空分析
# ══════════════════════════════════════════════════════════════
def run_ai_analysis(
    code: str,
    name: str,
    signal_info: dict,
    featured_df: pd.DataFrame,
    fundamental: dict,
    file_suffix: str | None = None,
) -> str:
    """调用 DeepSeek AI 进行综合多空分析，返回报告字符串"""
    print("\n" + "=" * 65)
    print(f"  AI 多空分析 — {name}({code})")
    print("=" * 65)

    latest = featured_df.iloc[-1]

    tech_summary = {
        'macd_dif':       round(float(latest.get('macd_dif',       0)), 4),
        'macd_hist':      round(float(latest.get('macd_hist',      0)), 4),
        'rsi_12':         round(float(latest.get('rsi_12',         50)), 2),
        'kdj_j':          round(float(latest.get('kdj_j',          50)), 2),
        'cci':            round(float(latest.get('cci',             0)), 2),
        'bb_pctb':        round(float(latest.get('bb_pctb',        0.5)), 4),
        'price_ma20_gap': round(float(latest.get('price_ma20_gap', 0)), 4),
        'vwap_ratio_20':  round(float(latest.get('vwap_ratio_20',  1)), 4),
    }

    # 选取一组核心 Alpha 因子做摘要（存在即输出）
    alpha_keys = [
        'alpha25', 'alpha5', 'alpha42',
        'mf_strength_6', 'mf_strength_12', 'mf_ratio_6', 'mf_ratio_12',
        'signed_vol_6', 'signed_vol_12', 'vol_surge_20',
        'gap_pct', 'gap_alpha107', 'gap_fill_ratio',
        'vpd_price_vol_6', 'vpd_range_vol_6',
    ]
    alpha_summary = {}
    for k in alpha_keys:
        if k in latest.index:
            try:
                alpha_summary[k] = round(float(latest.get(k, 0)), 4)
            except Exception:
                continue

    # 获取新闻与宏观数据
    news_summary  = FMPDataClient().get_stock_news(code, limit=10)
    macro_context = MacroAnalyzer.get_macro_context()

    # AI 分析
    analyst = AIAnalyst()
    report  = analyst.analyze(
        code=code,
        name=name,
        svm_signal=signal_info,
        technical_summary=tech_summary,
        fundamental_data=fundamental,
        news_summary=news_summary,
        macro_context=macro_context,
    )

    # 保存报告
    if file_suffix is None:
        file_suffix = datetime.now().strftime('%Y%m%d_%H%M')
    report_path = OUTPUT_DIR / f"ai_report_{code.replace('.', '_')}_{file_suffix}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# AI 多空分析报告 — {name}({code})\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## SVM 信号\n")
        f.write(f"- 信号: {signal_info['signal']}\n")
        f.write(f"- 置信度: {signal_info['confidence']:.1%}\n\n")
        f.write(f"## 技术指标\n")
        for k, v in tech_summary.items():
            f.write(f"- {k}: {v}\n")
        if alpha_summary:
            f.write(f"\n## Alpha 因子摘要\n")
            for k, v in alpha_summary.items():
                f.write(f"- {k}: {v}\n")
        f.write(f"\n## DeepSeek 分析报告\n\n{report}\n")

    print(f"\n  AI 分析报告已保存至 {report_path}")
    print("\n" + report)
    return report


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='SVM 多市场交易策略系统（FMP 数据源）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --code HK.00700 --name 腾讯控股 --start 2019-01-01 --end 2025-12-31 --mode full
  python main.py --code AAPL --name Apple --start 2019-01-01 --end 2025-12-31 --mode full
  python main.py --code SH.600519 --name 贵州茅台 --mode backtest --no-optimize
        """
    )
    parser.add_argument('--code',        type=str, default=DEFAULT_CODE,  help='股票代码（如 HK.00700 / AAPL / SH.600519）')
    parser.add_argument('--name',        type=str, default=DEFAULT_NAME,  help='股票名称（用于报告）')
    parser.add_argument('--start',       type=str, default='2019-01-01',  help='回测起始日期')
    parser.add_argument('--end',         type=str, default='2025-12-31',  help='回测结束日期')
    parser.add_argument('--mode',        type=str, default='full',
                        choices=['backtest', 'realtime', 'full'],
                        help='运行模式: backtest(仅回测) / realtime(仅实时) / full(完整)')
    parser.add_argument('--no-optimize', action='store_true',             help='跳过 GridSearchCV 参数优化（快速）')
    parser.add_argument('--conf-threshold', type=float, default=0.7,
                        help='信号最小置信度阈值，低于此值的买卖信号将视为观望')
    parser.add_argument('--fast-search', action='store_true',
                        help='启用 RandomizedSearchCV（更快的随机搜索）')
    parser.add_argument('--feishu-webhook', type=str, default=os.environ.get("FEISHU_WEBHOOK", ""),
                        help='飞书机器人 webhook，配置后运行结束自动推送结果')

    args = parser.parse_args()
    run_suffix = datetime.now().strftime('%Y%m%d_%H%M')

    print("\n" + "█" * 65)
    print("  SVM 多市场买卖点判断交易系统(FMP)  v1.1")
    print(f"  模式: {args.mode.upper()}   标的: {args.name}({args.code})")
    print("█" * 65)

    model, fe = None, None
    metrics_path = None
    backtest_png_path = None
    ai_report_path = None

    # ── 回测模式 ───────────────────────────────────────────────
    if args.mode in ('backtest', 'full'):
        model, fe, signals, metrics = run_backtest(
            args.code, args.start, args.end,
            optimize=not args.no_optimize,
            conf_threshold=args.conf_threshold,
            fast_search=args.fast_search,
            file_suffix=run_suffix,
        )
        metrics_path = OUTPUT_DIR / f"metrics_{args.code.replace('.', '_')}_{run_suffix}.json"
        backtest_png_path = OUTPUT_DIR / f"backtest_{args.code.replace('.', '_')}_{run_suffix}.png"

    # ── 实时信号模式 ───────────────────────────────────────────
    if args.mode in ('realtime', 'full'):
        if args.mode == 'realtime':
            # 单独实时模式：从磁盘加载已训练的模型和 scaler
            print("\n[Realtime] 加载已训练模型...")
            model = SVMTradingModel()
            model.load_model()

            if not SCALER_PATH.exists() or not FEATURES_PATH.exists():
                print("[错误] 未找到 Scaler/Feature 文件，请先运行 backtest 模式训练模型")
                sys.exit(1)

            fe = FeatureEngineer()
            fe.scaler = joblib.load(SCALER_PATH)
            with open(FEATURES_PATH, encoding='utf-8') as f:
                fe.feature_columns = json.load(f)
            fe._fitted = True

        signal_info, featured_df, fundamental = run_realtime_signal(
            args.code, model, fe, conf_threshold=args.conf_threshold
        )

        # AI 分析
        report = run_ai_analysis(
            args.code, args.name,
            signal_info, featured_df, fundamental,
            file_suffix=run_suffix,
        )
        ai_report_path = OUTPUT_DIR / f"ai_report_{args.code.replace('.', '_')}_{run_suffix}.md"

    if args.feishu_webhook:
        try:
            push_run_report(
                webhook_url=args.feishu_webhook,
                code=args.code,
                name=args.name,
                mode=args.mode,
                run_suffix=run_suffix,
                metrics_path=metrics_path,
                backtest_png_path=backtest_png_path,
                ai_report_md_path=ai_report_path,
            )
            print("[Feishu] 已推送运行结果")
        except Exception as e:
            print(f"[Feishu] 推送失败: {e}")

    print("\n" + "=" * 65)
    print("  系统运行完毕。输出文件保存在 output/ 目录。")
    print("=" * 65)


if __name__ == '__main__':
    main()
