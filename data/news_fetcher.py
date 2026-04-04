"""
新闻数据获取模块
基于 AkShare 获取财联社电报与个股新闻，供 AI 分析使用
参考: data_fetcher_news.py（附件）

数据来源优先级:
  1. 财联社电报（多接口自动探测）
  2. 东方财富个股新闻
  3. Google News RSS（港股英文新闻）
"""
import hashlib
import logging
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# ── 财联社电报接口候选列表（兼容不同 AkShare 版本） ─────────────
_CLS_INTERFACES = [
    'stock_telegraph_cls_em',
    'news_cls_telegraph_em',
    'stock_news_cls_telegraph',
]


def _try_cls_df() -> pd.DataFrame:
    """自动探测可用的财联社电报接口"""
    try:
        import akshare as ak
    except ImportError:
        logger.warning("akshare 未安装，跳过财联社新闻")
        return pd.DataFrame()

    for iface in _CLS_INTERFACES:
        fn = getattr(ak, iface, None)
        if fn is None:
            continue
        try:
            df = fn()
            if df is not None and not df.empty:
                logger.info(f"财联社电报使用接口: {iface}")
                return df
        except Exception as e:
            logger.debug(f"接口 {iface} 失败: {e}")

    # 降级：东方财富市场新闻
    try:
        df = ak.stock_news_em(symbol='000001')
        if df is not None and not df.empty:
            logger.info("降级使用东方财富市场新闻")
            return df
    except Exception as e:
        logger.debug(f"东方财富新闻也失败: {e}")

    return pd.DataFrame()


def _try_stock_news(code: str, max_count: int = 5) -> List[Dict]:
    """获取个股东方财富新闻"""
    try:
        import akshare as ak
    except ImportError:
        return []

    # 港股代码格式转换 HK.00700 -> 00700
    ak_code = code.replace('HK.', '').lstrip('0') or '700'
    try:
        df = ak.stock_news_em(symbol=ak_code)
        if df is None or df.empty:
            return []
    except Exception:
        # 使用原始代码重试
        try:
            df = ak.stock_news_em(symbol=code.replace('HK.', ''))
            if df is None or df.empty:
                return []
        except Exception as e:
            logger.debug(f"个股新闻获取失败 {code}: {e}")
            return []

    news = []
    for _, row in df.head(max_count).iterrows():
        news.append({
            'title':   str(row.get('新闻标题', row.get('title', ''))),
            'time':    str(row.get('发布时间', row.get('time', ''))),
            'source':  str(row.get('文章来源', row.get('source', '东方财富'))),
            'content': str(row.get('新闻内容', row.get('content', '')))[:400],
        })
    return news


# ─────────────────────────────────────────────────────────────
# 策略关键词配置（政策级别判断）
# ─────────────────────────────────────────────────────────────
POLICY_KEYWORDS = {
    'level_3': ['国务院', '央行', '中央', '政治局', '习近平', '重大利好', '重磅'],
    'level_2': ['监管', '政策', '支持', '鼓励', '财政部', '证监会', '港交所'],
    'level_1': ['公司', '业绩', '分红', '回购', '增持'],
}

INDUSTRY_KEYWORDS = {
    '科技':   ['互联网', '人工智能', 'AI', '芯片', '半导体', '云计算'],
    '金融':   ['银行', '保险', '券商', '基金', '利率'],
    '能源':   ['石油', '天然气', '新能源', '光伏', '风电'],
    '消费':   ['零售', '餐饮', '旅游', '电商', '消费'],
    '医药':   ['医疗', '药品', '生物', '疫苗', '医院'],
    '地产':   ['房地产', '物业', '建筑', '楼市'],
    '电信':   ['电信', '移动', '联通', '5G', '通信'],
}


def _judge_policy_level(text: str) -> int:
    for kw in POLICY_KEYWORDS['level_3']:
        if kw in text:
            return 3
    for kw in POLICY_KEYWORDS['level_2']:
        if kw in text:
            return 2
    return 1


def _extract_industries(text: str) -> List[str]:
    matched = []
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            matched.append(industry)
    return list(set(matched))


# ─────────────────────────────────────────────────────────────
# 主要接口类
# ─────────────────────────────────────────────────────────────
class NewsFetcher:
    """新闻数据获取器"""

    def __init__(self):
        self._cache: Dict[str, List[Dict]] = {}

    # ── 财联社电报 ──────────────────────────────────────────────
    def fetch_cls_telegraph(self, count: int = 30) -> List[Dict]:
        """获取财联社电报，返回结构化新闻列表"""
        logger.info("正在获取财联社电报...")
        df = _try_cls_df()
        if df.empty:
            logger.warning("财联社电报获取失败，返回空列表")
            return []

        news_list = []
        for _, row in df.head(count).iterrows():
            title   = str(row.get('title') or row.get('新闻标题') or row.get('标题') or row.get('content') or '')
            content = str(row.get('content') or row.get('新闻内容') or row.get('内容') or '')
            pub_time = str(row.get('time') or row.get('发布时间') or row.get('pub_time') or datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            level      = _judge_policy_level(title + content)
            industries = _extract_industries(title + content)
            news_id    = hashlib.md5(f"{title}{pub_time}".encode()).hexdigest()
            pub_date   = pub_time[:10] if len(pub_time) >= 10 else datetime.now().strftime('%Y-%m-%d')

            news_list.append({
                'news_id':      news_id,
                'publish_date': pub_date,
                'publish_time': pub_time,
                'source':       '财联社',
                'level':        level,
                'title':        title,
                'content':      content,
                'industries':   ','.join(industries),
            })

        logger.info(f"财联社电报获取 {len(news_list)} 条")
        return news_list

    # ── 个股新闻 ────────────────────────────────────────────────
    def fetch_stock_news(self, code: str, max_count: int = 5) -> List[Dict]:
        """获取个股新闻，结果写入内存缓存"""
        news = _try_stock_news(code, max_count)
        if news:
            logger.info(f"  {code} ← {len(news)} 条新闻")
            self._cache[code] = news
        else:
            logger.debug(f"  {code}: 无新闻")
        return news

    # ── 批量获取 ────────────────────────────────────────────────
    def fetch_batch(self, codes: List[str], delay: float = 0.5) -> Dict[str, List[Dict]]:
        """批量获取多只股票新闻"""
        result = {}
        total = len(codes)
        for i, code in enumerate(codes):
            result[code] = self.fetch_stock_news(code)
            if (i + 1) % 10 == 0:
                logger.info(f"  ── 已处理 {i+1}/{total} ──")
            time.sleep(delay)
        has = sum(1 for v in result.values() if v)
        logger.info(f"批量新闻完成：有新闻 {has} 只 / 无新闻 {total-has} 只")
        return result

    # ── 港股新闻摘要（供 AI Analyst 调用）──────────────────────
    @staticmethod
    def fetch_hk_news(code: str, days: int = 7) -> str:
        """
        获取港股相关新闻，返回格式化字符串供 AI 分析使用

        数据源优先级：
          1. AkShare 东方财富个股新闻
          2. Google News RSS（简单 HTTP 请求）
        """
        fetcher = NewsFetcher()
        news_list = fetcher.fetch_stock_news(code, max_count=6)

        if not news_list:
            # 尝试 Google News RSS
            news_list = _fetch_google_news_rss(code, days)

        if not news_list:
            return f"[{code}] 暂无最新新闻数据（请配置实际数据源）"

        lines = [f"近期相关新闻（{code}）:"]
        for i, n in enumerate(news_list[:5], 1):
            t = n.get('time', '')[:16]
            s = n.get('source', '')
            title = n.get('title', '')
            lines.append(f"  {i}. [{t}] {title}  (来源: {s})")

        return '\n'.join(lines)

    def get_cache(self) -> Dict[str, List[Dict]]:
        return self._cache


def _fetch_google_news_rss(code: str, days: int = 7) -> List[Dict]:
    """
    尝试从 Google News RSS 获取港股英文新闻（无需 API Key）
    code 格式: HK.00700
    """
    try:
        import urllib.request
        import xml.etree.ElementTree as ET

        # 构建查询词
        ticker = code.replace('HK.', '').lstrip('0') or '700'
        query = f"HK:{ticker} stock"
        url = f"https://news.google.com/rss/search?q={query}&hl=zh-CN&gl=HK&ceid=HK:zh-Hant"

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_data = resp.read()

        root = ET.fromstring(xml_data)
        news = []
        for item in root.findall('.//item')[:5]:
            title = item.findtext('title', '')
            pub_date = item.findtext('pubDate', '')
            source_el = item.find('{https://news.google.com/rss}source')
            source = source_el.text if source_el is not None else 'Google News'
            news.append({'title': title, 'time': pub_date, 'source': source, 'content': ''})

        return news
    except Exception as e:
        logger.debug(f"Google News RSS 获取失败: {e}")
        return []
