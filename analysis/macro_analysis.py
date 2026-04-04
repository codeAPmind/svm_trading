"""
宏观环境分析模块
通过 AkShare 或公开 API 获取宏观指标数据，生成港股宏观环境摘要

数据来源（均为公开/免费）:
  - 恒生指数走势:  AkShare stock_hk_index_daily_em
  - 南向资金:      AkShare stock_em_hsgt_north_money
  - VIX 恐慌指数:  AkShare index_vix
  - 美元指数:      AkShare currency_boc_em
"""
import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_akshare(func_name: str, **kwargs):
    """安全调用 AkShare，失败返回空 DataFrame"""
    try:
        import akshare as ak
        fn = getattr(ak, func_name, None)
        if fn is None:
            return pd.DataFrame()
        return fn(**kwargs)
    except Exception as e:
        logger.debug(f"AkShare {func_name} 失败: {e}")
        return pd.DataFrame()


class MacroAnalyzer:
    """宏观环境分析器"""

    @staticmethod
    def get_hsi_trend(days: int = 20) -> str:
        """获取恒生指数近期走势描述"""
        df = _safe_akshare('stock_hk_index_daily_em', symbol='恒生指数')
        if df.empty:
            return "恒生指数数据获取失败"
        try:
            df = df.tail(days)
            latest = df.iloc[-1]
            oldest = df.iloc[0]
            chg = (float(latest.get('收盘', 0)) - float(oldest.get('收盘', 0))) / float(oldest.get('收盘', 1)) * 100
            trend = '上涨' if chg > 0 else '下跌'
            return f"恒生指数近 {days} 日{trend} {abs(chg):.1f}%，最新收盘: {latest.get('收盘', 'N/A')}"
        except Exception:
            return "恒生指数数据解析失败"

    @staticmethod
    def get_south_money_flow() -> str:
        """南向资金净流入（港股通）"""
        try:
            import akshare as ak
            # 南向资金（沪深港通南向）
            df = ak.stock_em_hsgt_north_money(indicator='南向资金')
            if df is None or df.empty:
                return "南向资金数据暂不可用"
            latest = df.tail(3)
            lines = []
            for _, row in latest.iterrows():
                date = str(row.get('日期', ''))[:10]
                net  = row.get('当日净买额', row.get('净买额', 0))
                direction = '净流入' if float(net) > 0 else '净流出'
                lines.append(f"  {date}: {direction} {abs(float(net)):.1f} 亿")
            return "南向资金近3日:\n" + '\n'.join(lines)
        except Exception as e:
            logger.debug(f"南向资金获取失败: {e}")
            return "南向资金数据暂不可用"

    @staticmethod
    def get_vix() -> str:
        """VIX 恐慌指数"""
        df = _safe_akshare('index_vix')
        if df.empty:
            return "VIX 数据暂不可用"
        try:
            latest = df.iloc[-1]
            vix = float(latest.get('close', latest.get('收盘', 0)))
            if vix < 15:
                level = '低（市场情绪乐观）'
            elif vix < 25:
                level = '正常（市场波动中性）'
            elif vix < 35:
                level = '偏高（市场情绪紧张）'
            else:
                level = '极高（恐慌状态）'
            return f"VIX 恐慌指数: {vix:.1f}，处于 {level}"
        except Exception:
            return "VIX 数据解析失败"

    @staticmethod
    def get_macro_context() -> str:
        """
        生成综合宏观环境文本摘要，供 AI 分析使用

        此方法会尝试获取多个指标，失败则使用占位描述
        """
        today = datetime.now().strftime('%Y-%m-%d')

        sections = [f"宏观环境摘要（{today}）:"]

        # 恒生指数
        try:
            hsi = MacroAnalyzer.get_hsi_trend()
            sections.append(f"  ✦ {hsi}")
        except Exception:
            sections.append("  ✦ 恒生指数: [数据获取中]")

        # 南向资金
        try:
            south = MacroAnalyzer.get_south_money_flow()
            sections.append(f"  ✦ {south}")
        except Exception:
            sections.append("  ✦ 南向资金: [数据获取中]")

        # VIX
        try:
            vix = MacroAnalyzer.get_vix()
            sections.append(f"  ✦ {vix}")
        except Exception:
            sections.append("  ✦ VIX: [数据获取中]")

        # 补充静态提示（实际可接入更多数据源）
        sections.append("  ✦ 美联储政策: 关注 FOMC 声明与通胀数据（请自行补充最新信息）")
        sections.append("  ✦ 中国经济: 关注 PMI、CPI、PPI 及政策动向（请自行补充最新信息）")

        return '\n'.join(sections)
