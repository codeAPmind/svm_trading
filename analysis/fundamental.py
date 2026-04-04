"""
基本面分析模块
从 Futu 快照数据提取基本面指标，生成评分与解读
"""
import pandas as pd
from typing import Dict, Optional


class FundamentalAnalyzer:
    """基本面分析器"""

    # PE 历史分位数参考（港股蓝筹，粗略参考值）
    PE_CHEAP_THRESHOLD  = 15.0
    PE_FAIR_THRESHOLD   = 25.0
    PE_EXPENSIVE_THRESHOLD = 40.0

    def __init__(self, financial_data: dict):
        """
        Args:
            financial_data: FutuDataClient.get_financial_data() 返回的字典
        """
        self.data = financial_data

    # ─────────────────────────────────────────────────────────
    def pe_assessment(self) -> dict:
        pe = self.data.get('pe_ratio', 0)
        if pe <= 0:
            return {'level': '无法评估', 'score': 50, 'note': 'PE 为负或无数据（亏损股）'}
        if pe < self.PE_CHEAP_THRESHOLD:
            level, score = '低估', 80
        elif pe < self.PE_FAIR_THRESHOLD:
            level, score = '合理', 60
        elif pe < self.PE_EXPENSIVE_THRESHOLD:
            level, score = '偏贵', 40
        else:
            level, score = '高估', 20
        return {'level': level, 'score': score, 'pe': pe,
                'note': f"PE={pe:.1f}，处于{level}区间"}

    def pb_assessment(self) -> dict:
        pb = self.data.get('pb_ratio', 0)
        if pb <= 0:
            return {'level': '无法评估', 'score': 50, 'pb': pb}
        if pb < 1.0:
            level, score = '破净', 70
        elif pb < 2.0:
            level, score = '低估', 65
        elif pb < 4.0:
            level, score = '合理', 50
        else:
            level, score = '高估', 30
        return {'level': level, 'score': score, 'pb': pb,
                'note': f"PB={pb:.2f}，{level}"}

    def position_in_52w(self) -> dict:
        """当前价格在 52 周高低区间的位置（0%~100%）"""
        high52 = self.data.get('high52w', 0)
        low52  = self.data.get('low52w', 0)
        price  = self.data.get('last_price', 0)
        if high52 <= low52 or price <= 0:
            return {'pct': None, 'note': '数据不足'}
        pct = (price - low52) / (high52 - low52) * 100
        if pct < 20:
            note = f"靠近 52 周低点（{pct:.0f}%），存在反弹机会"
        elif pct > 80:
            note = f"靠近 52 周高点（{pct:.0f}%），注意回调风险"
        else:
            note = f"处于 52 周区间中部（{pct:.0f}%）"
        return {'pct': round(pct, 1), 'high52': high52, 'low52': low52, 'note': note}

    # ─────────────────────────────────────────────────────────
    def comprehensive_score(self) -> dict:
        """综合基本面评分（0-100）"""
        pe_res = self.pe_assessment()
        pb_res = self.pb_assessment()
        pos_res = self.position_in_52w()

        # 简单加权
        score = (pe_res['score'] * 0.5 + pb_res['score'] * 0.3 +
                 (100 - (pos_res.get('pct') or 50)) * 0.2)
        score = max(0, min(100, score))

        return {
            'total_score':  round(score, 1),
            'pe':           pe_res,
            'pb':           pb_res,
            'position_52w': pos_res,
        }

    def to_summary_string(self) -> str:
        """生成供 AI 分析使用的基本面摘要"""
        res = self.comprehensive_score()
        lines = [
            f"基本面综合评分: {res['total_score']}/100",
            f"PE评估: {res['pe'].get('note', 'N/A')}",
            f"PB评估: {res['pb'].get('note', 'N/A')}",
            f"52周位置: {res['position_52w'].get('note', 'N/A')}",
        ]
        return '\n'.join(lines)
