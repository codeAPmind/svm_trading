"""
AI 多空分析模块
调用 DeepSeek API 对港股进行综合研判

DeepSeek API 兼容 OpenAI 接口规范，直接使用 HTTP POST 调用。
"""
import os
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Optional

from config.settings import DEEPSEEK_CFG

logger = logging.getLogger(__name__)


class AIAnalyst:
    """AI 多空综合分析师（DeepSeek 驱动）"""

    def __init__(self):
        self.api_key  = DEEPSEEK_CFG.api_key or os.environ.get('DEEPSEEK_API_KEY', '')
        self.model    = DEEPSEEK_CFG.model
        self.base_url = DEEPSEEK_CFG.base_url
        self.max_tokens = DEEPSEEK_CFG.max_tokens

        if not self.api_key:
            logger.warning("[AIAnalyst] DEEPSEEK_API_KEY 未配置，AI 分析将返回占位信息")

    # ─────────────────────────────────────────────────────────
    def _call_deepseek(self, prompt: str, system: str = None) -> str:
        """
        调用 DeepSeek Chat API

        DeepSeek 兼容 OpenAI /chat/completions 接口
        """
        if not self.api_key:
            return "[AI分析] DeepSeek API Key 未配置，请在 .env 中设置 DEEPSEEK_API_KEY"

        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": DEEPSEEK_CFG.temperature,
            "stream":      False,
        }

        try:
            url = self.base_url.rstrip('/') + '/chat/completions'
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.Timeout:
            return "[AI分析错误] 请求超时，请检查网络连接"
        except requests.exceptions.HTTPError as e:
            return f"[AI分析错误] HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"[AI分析错误] {str(e)}"

    # ─────────────────────────────────────────────────────────
    def build_analysis_prompt(
        self,
        code: str,
        name: str,
        svm_signal: dict,
        technical_summary: dict,
        fundamental_data: dict,
        news_summary: str,
        macro_context: str
    ) -> str:
        """构建结构化分析 Prompt"""
        signal_text = {1: '买入 🟢', -1: '卖出 🔴', 0: '观望 ⚪'}.get(
            svm_signal.get('signal', 0), '观望 ⚪'
        )

        prompt = f"""你是一位专业的港股量化交易分析师，请根据以下数据对 {name}（{code}）进行综合多空分析。

## 一、SVM 模型量化信号
| 指标 | 数值 |
|------|------|
| 当前信号 | {signal_text} |
| 置信度 | {svm_signal.get('confidence', 0):.1%} |
| 买入概率 | {svm_signal.get('prob_buy', 0):.1%} |
| 卖出概率 | {svm_signal.get('prob_sell', 0):.1%} |

## 二、技术指标快照
| 指标 | 数值 | 解读参考 |
|------|------|----------|
| MACD DIF | {technical_summary.get('macd_dif', 'N/A')} | 正值偏多 |
| MACD 柱 | {technical_summary.get('macd_hist', 'N/A')} | 正=多头加速 |
| RSI(12) | {technical_summary.get('rsi_12', 'N/A')} | >70超买/<30超卖 |
| KDJ-J | {technical_summary.get('kdj_j', 'N/A')} | >80超买/<20超卖 |
| CCI | {technical_summary.get('cci', 'N/A')} | >100强势/<-100弱势 |
| 布林带%B | {technical_summary.get('bb_pctb', 'N/A')} | >1超上轨/<0超下轨 |
| 价格偏MA20 | {technical_summary.get('price_ma20_gap', 'N/A')} | 正=价格在均线上 |
| VWAP比率(20) | {technical_summary.get('vwap_ratio_20', 'N/A')} | >1偏强势 |

## 三、基本面数据
| 指标 | 数值 |
|------|------|
| 市盈率(PE) | {fundamental_data.get('pe_ratio', 'N/A')} |
| 市净率(PB) | {fundamental_data.get('pb_ratio', 'N/A')} |
| 市值(HKD) | {fundamental_data.get('market_cap', 'N/A')} |
| 换手率 | {fundamental_data.get('turnover_rate', 'N/A')} |
| 52周高点 | {fundamental_data.get('high52w', 'N/A')} |
| 52周低点 | {fundamental_data.get('low52w', 'N/A')} |
| 最新收盘 | {fundamental_data.get('last_price', 'N/A')} |

## 四、近期新闻与事件
{news_summary}

## 五、宏观环境
{macro_context}

---

请严格按以下结构输出分析报告（使用中文，简洁专业）：

### 1. 综合评级
【强烈买入 / 买入 / 中性 / 卖出 / 强烈卖出】—— 一句话核心判断

### 2. 多方因素（利好）
逐条列出支持上涨的因素（技术面 + 基本面 + 消息面）

### 3. 空方因素（利空）
逐条列出压制上涨的因素

### 4. 关键风险
当前最需关注的 2-3 个风险点

### 5. 操作建议
- 建议仓位: xx%
- 入场价区间: xxx ~ xxx
- 止损位: xxx（跌破此位离场）
- 目标价: xxx（第一目标） / xxx（第二目标）
- 持仓周期建议: x 周左右

### 6. 置信度评估
综合置信度: xx%
主要不确定因素: ...
"""
        return prompt

    # ─────────────────────────────────────────────────────────
    def analyze(
        self,
        code: str,
        name: str,
        svm_signal: dict,
        technical_summary: dict,
        fundamental_data: dict,
        news_summary: str = "暂无近期新闻",
        macro_context: str = "暂无宏观数据"
    ) -> str:
        """
        执行 AI 多空分析，返回格式化报告字符串
        """
        prompt = self.build_analysis_prompt(
            code, name, svm_signal, technical_summary,
            fundamental_data, news_summary, macro_context
        )

        system_prompt = (
            "你是一位专业的港股量化交易分析师，擅长结合技术面、基本面和消息面进行综合研判。"
            "你的分析客观、简洁、有据可查，不做无根据的推测。"
            "你清楚地认识到量化模型的局限性，始终在分析中提示风险。"
        )

        print(f"\n[AIAnalyst] 正在调用 DeepSeek 分析 {name}({code})...")
        report = self._call_deepseek(prompt, system=system_prompt)
        print(f"[AIAnalyst] 分析完成（{len(report)} 字符）")
        return report

    # ─────────────────────────────────────────────────────────
    def quick_sentiment(self, news_text: str, code: str) -> dict:
        """
        快速新闻情绪分析

        Returns:
            {'sentiment': 'positive/negative/neutral', 'score': float, 'reason': str}
        """
        prompt = f"""分析以下关于 {code} 的新闻文本，判断市场情绪。
仅返回 JSON 格式，不要其他文字：
{{"sentiment": "positive/negative/neutral", "score": 0.0~1.0, "reason": "一句话理由"}}

新闻内容:
{news_text[:800]}"""

        raw = self._call_deepseek(prompt)
        try:
            # 提取 JSON 部分
            start = raw.find('{')
            end   = raw.rfind('}') + 1
            return json.loads(raw[start:end])
        except Exception:
            return {'sentiment': 'neutral', 'score': 0.5, 'reason': '解析失败'}
