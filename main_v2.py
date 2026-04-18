"""
SVM 交易管线 V2：国君 Alpha 因子 + 中性化 + SVMTradingModelV2

数据源可选：FMP / Futu / Tushare；--source auto 时：港股→Futu，A股→Tushare，美股等→FMP。

示例:
  python main_v2.py --code CRML --start 2020-01-01 --end 2025-12-31 --source auto
  python main_v2.py --code HK.00700 --start 2019-01-01 --end 2025-12-31 --source auto
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config.settings  # noqa: F401 — 加载 .env，供 FMP 等使用

from data.fmp_client import FMPDataClient
from data.symbol_mapper import SymbolMapper
from feature.feature_engineer_v2 import FeatureEngineerV2
from model.svm_model_v2 import SVMTradingModelV2
from backtest.backtest_engine import BacktestEngine
from backtest.metrics import calculate_metrics, print_metrics_report, metrics_to_dict
from visualization.charts import plot_backtest_report, plot_signal_distribution
from model.signal_generator import SignalGenerator

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def _create_data_client(source: str, code: str) -> tuple[str, object]:
    s = (source or 'auto').strip().lower()
    if s == 'fmp':
        return 'FMP', FMPDataClient()
    if s == 'futu':
        from data.futu_client import FutuDataClient

        return 'Futu', FutuDataClient()
    if s == 'tushare':
        from data.tushare_client import TushareDataClient

        return 'Tushare', TushareDataClient()
    if s == 'auto':
        q = SymbolMapper.infer_quote_source(code)
        if q == 'futu':
            from data.futu_client import FutuDataClient

            return 'Futu(auto)', FutuDataClient()
        if q == 'tushare':
            from data.tushare_client import TushareDataClient

            return 'Tushare(auto)', TushareDataClient()
        return 'FMP(auto)', FMPDataClient()
    raise ValueError(f"未知数据源 {source!r}，请使用 fmp / futu / tushare / auto")


def run_v2(
    code: str,
    start: str,
    end: str,
    conf_threshold: float = 0.6,
    source: str = 'auto',
) -> None:
    src_label, client = _create_data_client(source, code)
    print(f"\n[V2] Step 1: 获取历史K线数据... ({src_label})")
    client.connect()
    try:
        df = client.get_history_kline(code, start, end)
    finally:
        client.disconnect()

    print(f"  共 {len(df)} 条 ({df.index[0].date()} ~ {df.index[-1].date()})")

    print("\n[V2] Step 2-3: 特征工程V2 准备数据")
    fe = FeatureEngineerV2()
    X, y, feat_cols, valid_idx = fe.prepare_dataset(df)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = valid_idx[split:]

    print("\n[V2] Step 4: 训练 SVM v2")
    model = SVMTradingModelV2()
    model.train(X_train, y_train)

    print("\n[V2] Step 5: 评估与回测")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    sig_gen = SignalGenerator(confidence_threshold=conf_threshold)
    signals_df = sig_gen.generate_signals(
        dates_test, preds, probs, model.classes_
    )
    signals_df = sig_gen.filter_consecutive_signals(signals_df)

    engine = BacktestEngine()
    test_prices = df.loc[dates_test]
    portfolio_df = engine.run(test_prices, signals_df, code)
    metrics = calculate_metrics(portfolio_df, engine.trades)
    print_metrics_report(metrics)

    safe_code = code.replace('.', '_')
    metrics_path = OUTPUT_DIR / f"metrics_v2_{safe_code}_{source}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'code': code,
                'source': source,
                'start': start,
                'end': end,
                'metrics': metrics_to_dict(metrics),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[V2] 指标已保存: {metrics_path}")

    run_suffix = datetime.now().strftime('%Y%m%d_%H%M')
    plot_backtest_report(
        test_prices, signals_df, portfolio_df, code, file_suffix=run_suffix
    )
    plot_signal_distribution(signals_df, code, file_suffix=run_suffix)


def main():
    parser = argparse.ArgumentParser(
        description='SVM Trading System V2 (Alpha factors + FMP/Futu/Tushare)'
    )
    parser.add_argument('--code', type=str, required=True, help='标的代码')
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, required=True)
    parser.add_argument('--conf-threshold', type=float, default=0.6)
    parser.add_argument(
        '--source',
        type=str,
        default='auto',
        choices=['fmp', 'futu', 'tushare', 'auto'],
        help='数据源: auto=港股Futu、A股Tushare、美股等FMP；亦可显式 fmp/futu/tushare',
    )
    args = parser.parse_args()
    run_v2(
        args.code,
        args.start,
        args.end,
        conf_threshold=args.conf_threshold,
        source=args.source,
    )


if __name__ == '__main__':
    main()
