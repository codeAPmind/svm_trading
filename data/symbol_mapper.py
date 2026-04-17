"""
股票代码格式转换

不同系统的代码格式:
  Futu: US.AAPL / HK.00700 / SH.600036 / SZ.000001
  FMP:  AAPL    / 0700.HK  / 600036.SS / 000001.SZ
"""


class SymbolMapper:
    """股票代码格式转换器"""

    @staticmethod
    def to_fmp(code: str) -> str:
        """任意常见格式 -> FMP 格式"""
        if not code:
            return code

        c = code.strip().upper()

        if c.startswith("US."):
            return c[3:]
        if c.startswith("HK."):
            return f"{c[3:]}.HK"
        if c.startswith("SH."):
            return f"{c[3:]}.SS"
        if c.startswith("SZ."):
            return f"{c[3:]}.SZ"

        return c

    @staticmethod
    def to_futu(code: str) -> str:
        """FMP 格式 -> Futu 风格（用于兼容展示）"""
        if not code:
            return code

        c = code.strip().upper()
        if c.endswith(".HK"):
            return f"HK.{c.replace('.HK', '').zfill(5)}"
        if c.endswith(".SS"):
            return f"SH.{c.replace('.SS', '')}"
        if c.endswith(".SZ"):
            return f"SZ.{c.replace('.SZ', '')}"
        return f"US.{c}"

    @staticmethod
    def detect_market(code: str) -> str:
        """从代码格式判断市场: US / HK / CN"""
        c = (code or "").strip().upper()
        if c.startswith("HK.") or c.endswith(".HK"):
            return "HK"
        if c.startswith("SH.") or c.startswith("SZ.") or c.endswith(".SS") or c.endswith(".SZ"):
            return "CN"
        return "US"
