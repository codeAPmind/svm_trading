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
        if c.isdigit() and len(c) == 5:
            return f"{c.zfill(5)}.HK"

        return c

    @staticmethod
    def to_futu(code: str) -> str:
        """常见输入 / FMP 格式 -> Futu 风格（用于展示或请求）"""
        if not code:
            return code

        c = code.strip().upper()
        if c.startswith("HK."):
            return f"HK.{c[3:].zfill(5)}"
        if c.startswith("US.") or c.startswith("SH.") or c.startswith("SZ."):
            return c
        if c.endswith(".HK"):
            return f"HK.{c.replace('.HK', '').zfill(5)}"
        if c.endswith(".SS"):
            return f"SH.{c.replace('.SS', '')}"
        if c.endswith(".SZ"):
            return f"SZ.{c.replace('.SZ', '')}"
        if c.isdigit() and len(c) == 5:
            return f"HK.{c.zfill(5)}"
        return f"US.{c}"

    @staticmethod
    def detect_market(code: str) -> str:
        """从代码格式判断市场: US / HK / CN"""
        c = (code or "").strip().upper()
        if c.startswith("HK.") or c.endswith(".HK"):
            return "HK"
        if c.isdigit() and len(c) == 5:
            return "HK"
        if c.startswith("SH.") or c.startswith("SZ.") or c.endswith(".SS") or c.endswith(".SZ"):
            return "CN"
        if c.isdigit() and len(c) == 6:
            return "CN"
        return "US"

    @staticmethod
    def infer_quote_source(code: str) -> str:
        """
        按代码格式推断行情数据源（与 main_v2 --source auto 一致）:
        - 港股（HK. / .HK / 纯 5 位数字）→ futu
        - A 股（沪深 / 常见 6 位数字代码）→ tushare
        - 其余（美股 ticker 等）→ fmp
        """
        c = (code or "").strip().upper()
        if c.startswith("HK.") or c.endswith(".HK"):
            return "futu"
        if c.isdigit() and len(c) == 5:
            return "futu"
        if (
            c.startswith("SH.")
            or c.startswith("SZ.")
            or c.endswith(".SS")
            or c.endswith(".SZ")
        ):
            return "tushare"
        if c.startswith("US."):
            return "fmp"
        if c.isdigit() and len(c) == 6:
            return "tushare"
        return "fmp"
