"""
FMP (Financial Modeling Prep) 数据客户端
用于替代 Futu/Tushare 的行情与基本面拉取。
"""
import os
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List

import pandas as pd
import requests

from data.symbol_mapper import SymbolMapper


class FMPDataClient:
    """统一行情客户端（US/HK/CN）"""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    class _RateLimiter:
        """简单速率限制器，默认留足余量避免触发套餐限流。"""

        def __init__(self, max_calls_per_minute: int = 280):
            self.max_calls_per_minute = max_calls_per_minute
            self.min_interval = 60.0 / max_calls_per_minute
            self.last_call_time = 0.0
            self.call_count = 0
            self.window_start = time.time()
            self.lock = Lock()

        def wait_if_needed(self):
            with self.lock:
                now = time.time()
                if now - self.window_start >= 60:
                    self.call_count = 0
                    self.window_start = now
                if self.call_count >= self.max_calls_per_minute:
                    sleep_time = 60 - (now - self.window_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    self.call_count = 0
                    self.window_start = time.time()
                elapsed = now - self.last_call_time
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
                self.last_call_time = time.time()
                self.call_count += 1

    @staticmethod
    def _clean_value(val: str) -> str:
        # 去掉行内注释，并剥离引号
        pure = val.split("#", 1)[0].strip()
        return pure.strip('"').strip("'")

    @staticmethod
    def _load_api_key_from_env_file() -> str:
        """兼容两种格式:
        1) FMP_API_KEY=xxxx
        2) fmp:\n     api_key: "xxxx"
        """
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if not env_path.exists():
            return ""

        current_section = None
        with open(env_path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.endswith(":") and "=" not in line:
                    current_section = line[:-1].strip().lower()
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    if key.strip().upper() == "FMP_API_KEY":
                        return FMPDataClient._clean_value(val)
                if ":" in line and "=" not in line:
                    key, _, val = line.partition(":")
                    if current_section == "fmp" and key.strip().lower() == "api_key":
                        return FMPDataClient._clean_value(val)
        return ""

    @staticmethod
    def _load_fmp_settings_from_env_file() -> dict:
        """读取 YAML 风格 .env 中的 fmp/proxy 配置。"""
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if not env_path.exists():
            return {}

        settings = {}
        current_section = None
        with open(env_path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                if line.endswith(":") and "=" not in line:
                    current_section = line[:-1].strip().lower()
                    continue

                # key: value 风格
                is_yaml_kv = ":" in line and ("=" not in line or line.index(":") < line.index("="))
                if is_yaml_kv:
                    key, _, val = line.partition(":")
                    key = key.strip().lower()
                    val = FMPDataClient._clean_value(val)
                    if key == "proxy":
                        settings["proxy"] = val
                        continue
                    if current_section == "fmp":
                        settings[f"fmp.{key}"] = val
                    elif current_section is None:
                        settings[key] = val
                    continue

                # KEY=VALUE 风格
                if "=" in line:
                    key, _, val = line.partition("=")
                    settings[key.strip().lower()] = FMPDataClient._clean_value(val)
                    continue
        return settings

    def __init__(self, api_key: str | None = None, timeout: int = 30):
        file_settings = self._load_fmp_settings_from_env_file()
        self.api_key = (
            api_key
            or os.environ.get("FMP_API_KEY", "")
            or self._load_api_key_from_env_file()
        ).strip()
        if not self.api_key:
            raise RuntimeError("未配置 FMP_API_KEY，请在 .env 或环境变量中设置")
        self.timeout = int(
            os.environ.get("FMP_TIMEOUT")
            or file_settings.get("fmp.request_timeout")
            or str(timeout)
        )
        self.base_url = (
            os.environ.get("FMP_BASE_URL")
            or file_settings.get("fmp.base_url")
            or self.BASE_URL
        ).rstrip("/")
        self.retry_max = int(
            os.environ.get("FMP_RETRY_MAX")
            or file_settings.get("fmp.retry_max")
            or "3"
        )
        self.retry_backoff = float(
            os.environ.get("FMP_RETRY_BACKOFF")
            or file_settings.get("fmp.retry_backoff")
            or "2.0"
        )
        self.proxy = (
            os.environ.get("PROXY")
            or os.environ.get("proxy")
            or file_settings.get("proxy")
        )
        self._session = requests.Session()
        if self.proxy:
            self._session.proxies.update({"http": self.proxy, "https": self.proxy})
        self._request_count = 0
        self._rate_limiter = self._RateLimiter(
            max_calls_per_minute=int(
                os.environ.get("FMP_RATE_LIMIT_PER_MINUTE")
                or file_settings.get("fmp.rate_limit_per_minute")
                or "280"
            )
        )

    def connect(self):
        """保持与旧 client 一致的生命周期接口。"""
        return None

    def disconnect(self):
        """保持与旧 client 一致的生命周期接口。"""
        return None

    def _get(self, endpoint: str, params: dict | None = None):
        query = params.copy() if params else {}
        query["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        last_err = None
        for i in range(self.retry_max):
            try:
                self._rate_limiter.wait_if_needed()
                resp = self._session.get(url, params=query, timeout=self.timeout)
                if resp.status_code in (401, 403):
                    raise PermissionError(
                        "FMP 鉴权失败(401/403)：请检查 FMP_API_KEY 是否有效、套餐权限或当日额度。"
                    )
                resp.raise_for_status()
                self._request_count += 1
                data = resp.json()
                if isinstance(data, dict) and "Error Message" in data:
                    raise ValueError(f"FMP API 错误: {data['Error Message']}")
                return data
            except PermissionError:
                raise
            except Exception as e:
                last_err = e
                if i < self.retry_max - 1:
                    time.sleep(self.retry_backoff ** i)
        raise ConnectionError(f"FMP 请求失败: {last_err}")

    def _get_stable(self, endpoint: str, params: dict | None = None):
        """走 FMP stable 新版端点，规避 legacy /api/v3 限制。"""
        stable_base = self.base_url.replace("/api/v3", "/stable").replace("/api", "/stable")
        query = params.copy() if params else {}
        query["apikey"] = self.api_key
        url = f"{stable_base.rstrip('/')}/{endpoint.lstrip('/')}"
        last_err = None
        for i in range(self.retry_max):
            try:
                self._rate_limiter.wait_if_needed()
                resp = self._session.get(url, params=query, timeout=self.timeout)
                resp.raise_for_status()
                self._request_count += 1
                data = resp.json()
                if isinstance(data, dict) and "Error Message" in data:
                    raise ValueError(f"FMP API 错误: {data['Error Message']}")
                return data
            except Exception as e:
                last_err = e
                if i < self.retry_max - 1:
                    time.sleep(self.retry_backoff ** i)
        raise ConnectionError(f"FMP stable 请求失败: {last_err}")

    def get_history_kline(self, code: str, start: str, end: str) -> pd.DataFrame:
        """返回与现有特征工程兼容的 OHLCV DataFrame。"""
        symbol = SymbolMapper.to_fmp(code)
        data = self._get_stable(
            "historical-price-eod/full",
            {"symbol": symbol, "from": start, "to": end},
        )
        records = data.get("historical", []) if isinstance(data, dict) else data
        if not records:
            raise ValueError(f"FMP 无历史数据: {symbol} {start}~{end}")

        df = pd.DataFrame(records)
        df.rename(
            columns={
                "date": "time_key",
                "changePercent": "change_pct",
                "adjClose": "adj_close",
            },
            inplace=True,
        )
        df["time_key"] = pd.to_datetime(df["time_key"])
        df.set_index("time_key", inplace=True)
        df.sort_index(inplace=True)

        # turnover 兜底，保证与原系统字段兼容
        if "turnover" not in df.columns:
            if "vwap" in df.columns:
                df["turnover"] = df["vwap"] * df["volume"]
            else:
                df["turnover"] = df["close"] * df["volume"]

        # 使用复权价更接近原先 QFQ 效果
        if "adj_close" in df.columns:
            close = df["close"].replace(0, pd.NA)
            adj_factor = (df["adj_close"] / close).fillna(1.0)
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col] * adj_factor

        return df

    def get_realtime_quote(self, codes: List[str]) -> pd.DataFrame:
        symbols = [SymbolMapper.to_fmp(c) for c in codes]
        payload = self._get_stable("quote", {"symbol": ",".join(symbols)})
        if not payload:
            raise ValueError(f"无报价数据: {symbols}")
        return pd.DataFrame(payload)

    def get_financial_data(self, code: str) -> Dict:
        symbol = SymbolMapper.to_fmp(code)
        try:
            quote = self._get_stable("quote", {"symbol": symbol})
        except Exception:
            quote = []
        q = quote[0] if isinstance(quote, list) and quote else {}

        try:
            metrics = self._get_stable("key-metrics-ttm", {"symbol": symbol})
        except Exception:
            metrics = []
        m = metrics[0] if isinstance(metrics, list) and metrics else {}

        try:
            profile = self._get_stable("profile", {"symbol": symbol})
        except Exception:
            profile = []
        p = profile[0] if isinstance(profile, list) and profile else {}

        return {
            "code": code,
            "name": q.get("name", p.get("companyName", "")),
            "last_price": q.get("price"),
            "pe_ratio": q.get("pe", m.get("peRatioTTM")),
            "pb_ratio": m.get("pbRatioTTM"),
            "market_cap": q.get("marketCap"),
            "dividend_yield": m.get("dividendYieldTTM"),
            "roe": m.get("roeTTM"),
            "eps": q.get("eps"),
            "volume": q.get("volume"),
            "avg_volume": q.get("avgVolume"),
            "high52w": q.get("yearHigh"),
            "low52w": q.get("yearLow"),
            "beta": p.get("beta"),
            "sector": p.get("sector", ""),
            "industry": p.get("industry", ""),
        }

    def get_stock_news(self, code: str, limit: int = 10) -> str:
        symbol = SymbolMapper.to_fmp(code)
        try:
            data = self._get_stable("news/stock", {"symbols": symbol, "limit": limit})
        except Exception:
            return f"[{code}] 暂无近期新闻"
        if not data:
            return f"[{code}] 暂无近期新闻"
        lines = []
        for item in data[:10]:
            dt = str(item.get("publishedDate", ""))[:10]
            title = item.get("title", "")
            lines.append(f"[{dt}] {title}")
        return "\n".join(lines)

    @property
    def request_count(self) -> int:
        return self._request_count
