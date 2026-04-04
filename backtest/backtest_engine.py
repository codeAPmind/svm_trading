"""
回测引擎 - 事件驱动式，支持港股真实费率结构

费率结构（港股）:
  佣金:   成交金额 × 0.05%，最低 HKD 5
  印花税: 成交金额 × 0.1%（仅卖出方收取）
  平台费: 本系统暂不模拟
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.settings import BACKTEST_CFG


@dataclass
class Trade:
    """单笔交易记录"""
    date:               pd.Timestamp
    code:               str
    action:             str           # 'BUY' | 'SELL'
    price:              float
    quantity:           int
    commission:         float
    stamp_duty:         float         # 卖出时收取
    signal_confidence:  float = 0.0

    @property
    def amount(self) -> float:
        return self.price * self.quantity

    @property
    def total_cost(self) -> float:
        return self.amount + self.commission + self.stamp_duty


@dataclass
class Position:
    """持仓信息"""
    code:       str
    quantity:   int
    avg_cost:   float
    entry_date: pd.Timestamp


class BacktestEngine:
    """港股回测引擎"""

    def __init__(self, config=None):
        self.cfg = config or BACKTEST_CFG
        self.cash: float = self.cfg.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[dict] = []

    # ── 重置 ──────────────────────────────────────────────────
    def reset(self):
        self.cash = self.cfg.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []

    # ── 费用计算 ───────────────────────────────────────────────
    def _commission(self, amount: float) -> float:
        return max(amount * self.cfg.commission_rate, self.cfg.min_commission)

    def _stamp_duty(self, amount: float) -> float:
        return amount * self.cfg.stamp_duty

    def _lot_size(self, code: str) -> int:
        """
        港股每手股数因标的而异。
        实际生产应通过 Futu API 查询 lot_size，此处统一 100 作为默认。
        常见例外: 00700(腾讯)=100, 09988(阿里)=1, 03690(美团)=100 ...
        """
        _lot_map = {
            'HK.09988': 1,
            'HK.09618': 1,
        }
        return _lot_map.get(code, 100)

    # ── 买入 ───────────────────────────────────────────────────
    def execute_buy(
        self,
        date: pd.Timestamp,
        code: str,
        price: float,
        confidence: float = 0.0
    ):
        if len(self.positions) >= self.cfg.max_positions:
            return  # 持仓已满

        available  = self.cash * self.cfg.position_size
        lot_size   = self._lot_size(code)
        quantity   = int(available / (price * lot_size)) * lot_size

        if quantity <= 0:
            return

        amount     = price * quantity
        commission = self._commission(amount)
        total_cost = amount + commission

        if total_cost > self.cash:
            return

        self.cash -= total_cost

        if code in self.positions:
            pos = self.positions[code]
            new_qty  = pos.quantity + quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + price * quantity) / new_qty
            pos.quantity = new_qty
        else:
            self.positions[code] = Position(
                code=code, quantity=quantity,
                avg_cost=price, entry_date=date
            )

        self.trades.append(Trade(
            date=date, code=code, action='BUY',
            price=price, quantity=quantity,
            commission=commission, stamp_duty=0.0,
            signal_confidence=confidence
        ))

    # ── 卖出（全仓）────────────────────────────────────────────
    def execute_sell(
        self,
        date: pd.Timestamp,
        code: str,
        price: float,
        confidence: float = 0.0
    ):
        if code not in self.positions:
            return

        pos        = self.positions[code]
        amount     = price * pos.quantity
        commission = self._commission(amount)
        stamp_duty = self._stamp_duty(amount)
        net        = amount - commission - stamp_duty

        self.cash += net

        self.trades.append(Trade(
            date=date, code=code, action='SELL',
            price=price, quantity=pos.quantity,
            commission=commission, stamp_duty=stamp_duty,
            signal_confidence=confidence
        ))
        del self.positions[code]

    # ── 主回测循环 ─────────────────────────────────────────────
    def run(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        code: str
    ) -> pd.DataFrame:
        """
        运行回测

        Args:
            price_data: 价格数据 (index=DatetimeIndex, 含 open/close/high/low/volume)
            signals:    信号 DataFrame (index=date, 含 action / confidence)
            code:       股票代码

        Returns:
            portfolio_df: 每日净值记录
        """
        self.reset()

        for date in price_data.index:
            price = float(price_data.loc[date, 'close'])

            # 执行信号
            if date in signals.index:
                sig_row = signals.loc[date]
                # 防止同一日期出现多行（取第一条）
                if isinstance(sig_row, pd.DataFrame):
                    sig_row = sig_row.iloc[0]
                action = sig_row['action']
                conf   = float(sig_row['confidence'])

                if action == 'BUY':
                    self.execute_buy(date, code, price, conf)
                elif action == 'SELL':
                    self.execute_sell(date, code, price, conf)

            # 计算每日组合市值
            pos_value = 0.0
            for pos in self.positions.values():
                if pos.code == code:
                    pos_value += pos.quantity * price
                # 多标的时此处需扩展

            total_value = self.cash + pos_value

            self.portfolio_history.append({
                'date':           date,
                'cash':           round(self.cash, 2),
                'position_value': round(pos_value, 2),
                'total_value':    round(total_value, 2),
                'return':         total_value / self.cfg.initial_capital - 1,
            })

        return pd.DataFrame(self.portfolio_history).set_index('date')

    # ── 未实现盈亏汇总 ─────────────────────────────────────────
    def unrealized_pnl(self, current_prices: dict) -> dict:
        """
        计算当前持仓的未实现盈亏

        Args:
            current_prices: {code: price}
        """
        result = {}
        for code, pos in self.positions.items():
            cur = current_prices.get(code, pos.avg_cost)
            pnl = (cur - pos.avg_cost) * pos.quantity
            pct = (cur - pos.avg_cost) / pos.avg_cost
            result[code] = {
                'quantity': pos.quantity,
                'avg_cost': pos.avg_cost,
                'current':  cur,
                'pnl':      round(pnl, 2),
                'pnl_pct':  round(pct, 4),
            }
        return result
