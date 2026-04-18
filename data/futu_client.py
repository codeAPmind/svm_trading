"""
Futu OpenD 数据采集封装
- 日K线 / 分钟级数据
- 实时报价
- 财务快照数据
"""
import pandas as pd
from datetime import datetime, timedelta
from config.settings import FUTU_CFG
from data.symbol_mapper import SymbolMapper

try:
    from futu import (
        OpenQuoteContext, KLType, KL_FIELD, AuType, RET_OK
    )
    FUTU_AVAILABLE = True
except ImportError:
    FUTU_AVAILABLE = False
    print("[FutuClient] 警告: futu-api 未安装，Futu 数据功能不可用")


class FutuDataClient:
    """Futu OpenD 数据客户端"""

    def __init__(self):
        self.quote_ctx = None
        self._available = FUTU_AVAILABLE

    def connect(self):
        if not self._available:
            print("[FutuClient] futu-api 未安装，跳过连接")
            return
        self.quote_ctx = OpenQuoteContext(
            host=FUTU_CFG.host,
            port=FUTU_CFG.port
        )
        print(f"[FutuClient] 已连接 OpenD ({FUTU_CFG.host}:{FUTU_CFG.port})")

    def disconnect(self):
        if self.quote_ctx:
            self.quote_ctx.close()
            self.quote_ctx = None
            print("[FutuClient] 已断开连接")

    def get_history_kline(
        self,
        code: str,
        start: str,
        end: str,
        ktype=None,
        autype=None
    ) -> pd.DataFrame:
        """
        获取历史K线数据

        Args:
            code: 股票代码，如 "HK.00700"
            start: 起始日期 "YYYY-MM-DD"
            end:   结束日期 "YYYY-MM-DD"

        Returns:
            DataFrame columns: [open, close, high, low, volume, turnover,
                                pe_ratio, turnover_rate]
            index: DatetimeIndex (time_key)
        """
        if not self._available or self.quote_ctx is None:
            raise RuntimeError("Futu OpenD 未连接，请先调用 connect()")

        code = SymbolMapper.to_futu(code)

        if ktype is None:
            from futu import KLType
            ktype = KLType.K_DAY
        if autype is None:
            from futu import AuType
            autype = AuType.QFQ

        ret, data, page_req_key = self.quote_ctx.request_history_kline(
            code=code,
            start=start,
            end=end,
            ktype=ktype,
            autype=autype,
            max_count=1000
        )

        if ret != RET_OK:
            raise ConnectionError(f"获取K线失败: {data}")

        all_data = [data]

        while page_req_key is not None:
            ret, data, page_req_key = self.quote_ctx.request_history_kline(
                code=code, start=start, end=end,
                ktype=ktype, autype=autype,
                max_count=1000, page_req_key=page_req_key
            )
            if ret == RET_OK:
                all_data.append(data)

        df = pd.concat(all_data, ignore_index=True)
        df['time_key'] = pd.to_datetime(df['time_key'])
        df.set_index('time_key', inplace=True)
        df.sort_index(inplace=True)

        print(f"[FutuClient] {code}: 获取 {len(df)} 条K线 ({start} ~ {end})")
        return df

    def get_realtime_quote(self, codes: list) -> pd.DataFrame:
        """获取实时快照报价"""
        if not self._available or self.quote_ctx is None:
            raise RuntimeError("Futu OpenD 未连接")
        ret, data = self.quote_ctx.get_market_snapshot(codes)
        if ret != RET_OK:
            raise ConnectionError(f"获取实时报价失败: {data}")
        return data

    def get_financial_data(self, code: str) -> dict:
        """
        获取基本面财务快照

        Returns:
            dict: pe_ratio, pb_ratio, market_cap, turnover_rate,
                  last_price, high52w, low52w, volume, amplitude
        """
        if not self._available or self.quote_ctx is None:
            raise RuntimeError("Futu OpenD 未连接")
        ret, data = self.quote_ctx.get_market_snapshot([code])
        if ret != RET_OK:
            raise ConnectionError(f"获取财务数据失败: {data}")

        row = data.iloc[0]
        # 数值清洗与合理性约束，避免异常数据干扰 AI 分析
        def _to_float(value, default=0.0):
            try:
                return float(value)
            except Exception:
                return default

        last_price   = _to_float(row.get('last_price', 0))
        pe_ratio_raw = _to_float(row.get('pe_ratio', 0))
        pb_ratio     = _to_float(row.get('pb_ratio', 0))
        market_cap   = _to_float(row.get('market_cap', 0))
        turnover_rt  = _to_float(row.get('turnover_rate', 0))
        volume       = _to_float(row.get('volume', 0))
        amplitude    = _to_float(row.get('amplitude', 0))
        high52w      = _to_float(row.get('high_price', 0))
        low52w       = _to_float(row.get('low_price', 0))

        # 规则：
        # - PE 若 <=0 或 >200 视为无效，置为 None
        pe_ratio = None if (pe_ratio_raw <= 0 or pe_ratio_raw > 200) else pe_ratio_raw
        # - 市值若 <=0 视为缺失，置为 None
        market_cap = None if market_cap <= 0 else market_cap

        return {
            'code':          code,
            'name':          row.get('name', ''),
            'last_price':    last_price if last_price > 0 else None,
            'pe_ratio':      pe_ratio,
            'pb_ratio':      pb_ratio if pb_ratio > 0 else None,
            'market_cap':    market_cap,
            'turnover_rate': turnover_rt if turnover_rt >= 0 else None,
            'volume':        volume if volume >= 0 else None,
            'amplitude':     amplitude if amplitude >= 0 else None,
            'high52w':       high52w if high52w > 0 else None,
            'low52w':        low52w if low52w > 0 else None,
        }
