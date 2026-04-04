"""
组合管理模块
扩展回测引擎以支持多标的组合管理、仓位权重调整与止损机制
"""
import pandas as pd
import numpy as np
from typing import Dict, List

from backtest.backtest_engine import BacktestEngine, Trade, Position
from config.settings import BACKTEST_CFG


class PortfolioManager:
    """
    多标的组合管理器
    基于等权或信号强度加权进行仓位分配
    """

    def __init__(self, config=None):
        self.cfg = config or BACKTEST_CFG
        self.engine = BacktestEngine(config)

        # 止损参数
        self.stop_loss_pct: float = -0.08    # -8% 触发止损
        self.trailing_stop: float = -0.05    # 移动止损

        # 各标的高水位（用于移动止损）
        self._high_water: Dict[str, float] = {}

    # ─────────────────────────────────────────────────────────
    def check_stop_loss(
        self,
        date: pd.Timestamp,
        current_prices: Dict[str, float]
    ):
        """检查并执行止损（硬止损 + 移动止损）"""
        to_sell = []
        for code, pos in self.engine.positions.items():
            price = current_prices.get(code, pos.avg_cost)

            # 更新高水位
            if code not in self._high_water or price > self._high_water[code]:
                self._high_water[code] = price

            hard_loss   = (price - pos.avg_cost) / pos.avg_cost
            trail_loss  = (price - self._high_water[code]) / self._high_water[code]

            if hard_loss <= self.stop_loss_pct:
                print(f"[Portfolio] {code} 触发硬止损: {hard_loss:.2%}")
                to_sell.append((code, price, '硬止损'))
            elif trail_loss <= self.trailing_stop:
                print(f"[Portfolio] {code} 触发移动止损: {trail_loss:.2%}")
                to_sell.append((code, price, '移动止损'))

        for code, price, reason in to_sell:
            self.engine.execute_sell(date, code, price, confidence=1.0)
            self._high_water.pop(code, None)
            print(f"  → 已止损 {code} @ {price:.2f} ({reason})")

    # ─────────────────────────────────────────────────────────
    def run_portfolio(
        self,
        price_data_dict: Dict[str, pd.DataFrame],
        signals_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        多标的组合回测

        Args:
            price_data_dict: {code: price_df}
            signals_dict:    {code: signals_df}

        Returns:
            每日组合净值 DataFrame
        """
        self.engine.reset()
        self._high_water.clear()

        # 合并所有日期
        all_dates = sorted(set(
            dt for df in price_data_dict.values() for dt in df.index
        ))

        for date in all_dates:
            # 当日价格快照
            current_prices = {
                code: float(df.loc[date, 'close'])
                for code, df in price_data_dict.items()
                if date in df.index
            }

            # 止损检查（优先于信号执行）
            self.check_stop_loss(date, current_prices)

            # 执行各标的信号
            for code, signals in signals_dict.items():
                if date not in signals.index:
                    continue
                price = current_prices.get(code)
                if price is None:
                    continue

                sig_row = signals.loc[date]
                if isinstance(sig_row, pd.DataFrame):
                    sig_row = sig_row.iloc[0]

                action = sig_row['action']
                conf   = float(sig_row['confidence'])

                if action == 'BUY':
                    self.engine.execute_buy(date, code, price, conf)
                elif action == 'SELL':
                    self.engine.execute_sell(date, code, price, conf)

            # 记录组合净值
            pos_value = sum(
                pos.quantity * current_prices.get(pos.code, pos.avg_cost)
                for pos in self.engine.positions.values()
            )
            total = self.engine.cash + pos_value
            self.engine.portfolio_history.append({
                'date':           date,
                'cash':           round(self.engine.cash, 2),
                'position_value': round(pos_value, 2),
                'total_value':    round(total, 2),
                'return':         total / self.cfg.initial_capital - 1,
                'n_positions':    len(self.engine.positions),
            })

        return pd.DataFrame(self.engine.portfolio_history).set_index('date')

    # ─────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        """输出所有已平仓交易的 PnL 汇总"""
        buys  = [t for t in self.engine.trades if t.action == 'BUY']
        sells = [t for t in self.engine.trades if t.action == 'SELL']

        records = []
        for i, sell in enumerate(sells):
            if i < len(buys):
                buy = buys[i]
                pnl     = (sell.price - buy.price) * sell.quantity
                pnl_pct = (sell.price - buy.price) / buy.price
                records.append({
                    'code':        sell.code,
                    'entry_date':  buy.date,
                    'exit_date':   sell.date,
                    'entry_price': buy.price,
                    'exit_price':  sell.price,
                    'quantity':    sell.quantity,
                    'pnl':         round(pnl, 2),
                    'pnl_pct':     round(pnl_pct, 4),
                    'holding_days': (sell.date - buy.date).days,
                })

        return pd.DataFrame(records)
