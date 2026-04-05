"""
Tushare 数据客户端，用于 A 股（日线）数据抓取
"""
import os
import pandas as pd
from datetime import datetime

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False


class TushareDataClient:
    """A 股日线行情抓取（前复权）"""

    def __init__(self):
        self._available = TUSHARE_AVAILABLE
        self._pro = None

    def connect(self):
        if not self._available:
            print("[TushareClient] 警告: tushare 未安装，A股数据不可用")
            return
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        if not token:
            raise RuntimeError("未配置 TUSHARE_TOKEN 环境变量，请在 .env 中设置")
        ts.set_token(token)
        self._pro = ts.pro_api()
        print("[TushareClient] 已连接 Tushare")

    def disconnect(self):
        self._pro = None
        print("[TushareClient] 已断开")

    @staticmethod
    def _to_ts_code(code: str) -> str:
        """
        将 '002353' 等转换为 Tushare ts_code: '002353.SZ'（简单规则）
        - 以 '6' 开头 → 上证 'SH'
        - 否则 → 深市 'SZ'
        """
        code = code.strip()
        if code.upper().endswith((".SZ", ".SH")):
            return code.upper()
        suffix = "SH" if code.startswith("6") else "SZ"
        return f"{code}.{suffix}"

    def get_history_kline(
        self,
        code: str,
        start: str,
        end: str
    ) -> pd.DataFrame:
        """
        返回 DataFrame，包含列: [open, close, high, low, volume]
        index: DatetimeIndex
        """
        if self._pro is None:
            raise RuntimeError("Tushare 未连接，请先调用 connect()")

        ts_code = self._to_ts_code(code)
        start_dt = start.replace("-", "")
        end_dt   = end.replace("-", "")

        # 前复权日线
        df = self._pro.daily(ts_code=ts_code, start_date=start_dt, end_date=end_dt)
        if df is None or df.empty:
            raise ValueError(f"Tushare 无数据: {code} {start}~{end}")

        # 统一字段命名
        df = df[['trade_date', 'open', 'close', 'high', 'low', 'vol']]
        df.rename(columns={'vol': 'volume'}, inplace=True)
        # Tushare 的 vol 单位通常为手，这里按手直接使用，后续特征均为相对量纲

        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.sort_values('trade_date', inplace=True)
        df.set_index('trade_date', inplace=True)
        return df

    def get_financial_data(self, code: str) -> dict:
        """
        简化的财务/快照占位，Tushare 可通过 daily_basic 等接口扩展。
        先返回最关键的 last_price 与占位字段，避免上层依赖出错。
        """
        if self._pro is None:
            raise RuntimeError("Tushare 未连接，请先调用 connect()")
        ts_code = self._to_ts_code(code)
        today = datetime.now().strftime('%Y%m%d')
        df = self._pro.daily(ts_code=ts_code, start_date=today, end_date=today)
        last_price = float(df['close'].iloc[0]) if df is not None and not df.empty else None
        return {
            'code': ts_code,
            'name': '',
            'last_price': last_price,
            'pe_ratio': None,
            'pb_ratio': None,
            'market_cap': None,
            'turnover_rate': None,
            'volume': None,
            'amplitude': None,
            'high52w': None,
            'low52w': None,
        }

