import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from data.tushare_client import TushareDataClient
from data.futu_client import FutuDataClient
from feature.feature_engineer_v2 import FeatureEngineerV2
from model.svm_model_v2 import SVMTradingModelV2
from backtest.backtest_engine import BacktestEngine
from backtest.metrics import calculate_metrics, print_metrics_report, metrics_to_dict
from visualization.charts import plot_backtest_report, plot_signal_distribution
from model.signal_generator import SignalGenerator

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def _pick_market_client(code: str):
    cu = code.upper().strip()
    # HK./US. 或带点的代码 → Futu；否则默认 A 股 → Tushare
    if cu.startswith('HK.') or cu.startswith('US.') or ('.' in cu):
        return 'futu'
    return 'tushare'


def run_v2(code: str, start: str, end: str, conf_threshold: float = 0.6):
    market = _pick_market_client(code)
    print(f"\n[V2] Step 1: 获取历史K线数据... ({'Futu' if market=='futu' else 'Tushare'})")
    if market == 'futu':
        client = FutuDataClient()
    else:
        client = TushareDataClient()
    client.connect()
    try:
        df = client.get_history_kline(code, start, end)
    finally:
        client.disconnect()

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
    signals_df = sig_gen.generate_signals(dates_test, preds, probs, model.classes_)
    signals_df = sig_gen.filter_consecutive_signals(signals_df)

    engine = BacktestEngine()
    test_prices = df.loc[dates_test]
    portfolio_df = engine.run(test_prices, signals_df, code)
    metrics = calculate_metrics(portfolio_df, engine.trades)
    print_metrics_report(metrics)

    metrics_path = OUTPUT_DIR / f"metrics_v2_{code}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({'code': code, 'start': start, 'end': end, 'metrics': metrics_to_dict(metrics)}, f, ensure_ascii=False, indent=2)
    print(f"[V2] 指标已保存: {metrics_path}")

    plot_backtest_report(test_prices, signals_df, portfolio_df, code)
    plot_signal_distribution(signals_df, code)


def main():
    parser = argparse.ArgumentParser(description="SVM Trading System V2 (Multi-Market)")
    parser.add_argument('--code', type=str, required=True)
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, required=True)
    parser.add_argument('--conf-threshold', type=float, default=0.6)
    args = parser.parse_args()
    run_v2(args.code, args.start, args.end, conf_threshold=args.conf_threshold)


if __name__ == '__main__':
    main()
