"""
本地数据存储管理
支持 CSV 缓存与 SQLite 持久化，减少重复调用 Futu API
"""
import pandas as pd
import sqlite3
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / 'data_cache'
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / 'trading.db'


class DataStore:
    """本地数据存储（CSV + SQLite）"""

    def __init__(self):
        self._init_db()

    # ── SQLite 初始化 ──────────────────────────────────────────
    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS kline_cache (
                    code TEXT,
                    date TEXT,
                    open REAL, close REAL, high REAL, low REAL,
                    volume REAL, turnover REAL,
                    pe_ratio REAL, turnover_rate REAL,
                    PRIMARY KEY (code, date)
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS news_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT,
                    title TEXT,
                    pub_time TEXT,
                    source TEXT,
                    content TEXT,
                    fetched_at TEXT
                )
            ''')
            conn.commit()

    # ── K线缓存 ────────────────────────────────────────────────
    def save_kline(self, code: str, df: pd.DataFrame):
        """保存K线数据到 SQLite"""
        records = []
        for dt, row in df.iterrows():
            records.append((
                code, str(dt.date()),
                float(row.get('open', 0)),
                float(row.get('close', 0)),
                float(row.get('high', 0)),
                float(row.get('low', 0)),
                float(row.get('volume', 0)),
                float(row.get('turnover', 0)),
                float(row.get('pe_ratio', 0)),
                float(row.get('turnover_rate', 0)),
            ))
        with sqlite3.connect(DB_PATH) as conn:
            conn.executemany('''
                INSERT OR REPLACE INTO kline_cache
                (code, date, open, close, high, low, volume, turnover,
                 pe_ratio, turnover_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            conn.commit()
        print(f"[DataStore] 已缓存 {len(records)} 条K线 ({code})")

    def load_kline(self, code: str, start: str, end: str) -> pd.DataFrame:
        """从 SQLite 加载K线缓存"""
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                '''SELECT * FROM kline_cache
                   WHERE code=? AND date BETWEEN ? AND ?
                   ORDER BY date''',
                conn,
                params=(code, start, end)
            )
        if df.empty:
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.index.name = 'time_key'
        return df.drop(columns=['code'])

    def has_kline(self, code: str, start: str, end: str) -> bool:
        """检查本地是否已有足够缓存"""
        df = self.load_kline(code, start, end)
        return len(df) > 10

    # ── 新闻缓存 ───────────────────────────────────────────────
    def save_news(self, code: str, news_list: list):
        """保存新闻到 SQLite"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        records = [
            (code, n.get('title', ''), n.get('time', ''),
             n.get('source', ''), n.get('content', '')[:500], now)
            for n in news_list
        ]
        with sqlite3.connect(DB_PATH) as conn:
            conn.executemany('''
                INSERT INTO news_cache (code, title, pub_time, source, content, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', records)
            conn.commit()

    def load_news(self, code: str, days: int = 3) -> list:
        """加载最近 N 天新闻"""
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                '''SELECT title, pub_time, source, content FROM news_cache
                   WHERE code=?
                   ORDER BY fetched_at DESC LIMIT 20''',
                (code,)
            ).fetchall()
        return [
            {'title': r[0], 'time': r[1], 'source': r[2], 'content': r[3]}
            for r in rows
        ]

    # ── CSV 导出 ───────────────────────────────────────────────
    def export_kline_csv(self, code: str, start: str, end: str) -> Path:
        df = self.load_kline(code, start, end)
        path = DATA_DIR / f"{code.replace('.', '_')}_{start}_{end}.csv"
        df.to_csv(path)
        return path
