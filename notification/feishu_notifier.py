"""
飞书群机器人 webhook 推送（仅 webhook，不依赖 App 凭证）。

发送:
- 运行摘要 + metrics
- AI 报告正文预览（截断避免超长）
- 本地回测图等产物路径（便于本机/GitHub Actions 查看）

说明: 自定义机器人 webhook 无法直接上传本地二进制图片，图片请见 output/ 或 CI Artifact。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import requests

# 单条 text 不宜过长，报告预览上限
_REPORT_PREVIEW_CHARS = 3500


def _post_webhook(webhook_url: str, payload: dict) -> dict:
    resp = requests.post(webhook_url, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code", 0) != 0:
        raise RuntimeError(f"飞书 webhook 返回错误: {data}")
    return data


def push_run_report(
    webhook_url: str,
    code: str,
    name: str,
    mode: str,
    run_suffix: str,
    metrics_path: Optional[Path] = None,
    backtest_png_path: Optional[Path] = None,
    ai_report_md_path: Optional[Path] = None,
) -> None:
    """推送本次运行结果到飞书群机器人（仅 msg_type=text）。"""
    title = f"SVM策略运行结果 | {name}({code}) | {mode.upper()} | {run_suffix}"
    lines = [title]

    if metrics_path and metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            eval_data = metrics.get("eval", {})
            lines.append(
                f"accuracy={eval_data.get('accuracy')} | f1_weighted={eval_data.get('f1_weighted')}"
            )
        except Exception:
            lines.append(f"metrics: {metrics_path}")

    if backtest_png_path and backtest_png_path.exists():
        lines.append(f"回测图(本地): {backtest_png_path.resolve()}")

    if ai_report_md_path and ai_report_md_path.exists():
        report_text = ai_report_md_path.read_text(encoding="utf-8")
        preview = report_text[:_REPORT_PREVIEW_CHARS]
        if len(report_text) > _REPORT_PREVIEW_CHARS:
            preview += "\n\n...(报告已截断，完整见 output/ai_report_*.md)"
        lines.append("\n【AI报告】")
        lines.append(preview)

    text_payload = {
        "msg_type": "text",
        "content": {"text": "\n".join(lines)},
    }
    _post_webhook(webhook_url, text_payload)
