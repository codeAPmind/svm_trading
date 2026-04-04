# SVM 港股买卖点判断交易系统

> 基于支持向量机（SVM）的港股量化交易系统，数据通过 **Futu OpenD** 获取，AI 分析由 **DeepSeek** 驱动。

---

## 功能特性

| 模块 | 功能 |
|------|------|
| 数据采集 | Futu OpenD 实时/历史K线、财务快照 |
| 新闻数据 | AkShare 财联社电报 + 东方财富个股新闻 |
| 特征工程 | 30+ 维：MACD / RSI / KDJ / CCI / 布林带 / VWAP / 波动率 |
| SVM 模型 | RBF/Linear 多核对比，GridSearchCV + TimeSeriesSplit 调参 |
| 回测引擎 | 事件驱动，港股真实费率（万五佣金 + 千一印花税） |
| 绩效指标 | 总收益率 / 夏普 / 最大回撤 / 卡尔玛 / 胜率 / 盈亏比 |
| AI 分析 | DeepSeek Chat 综合多空研判（信号+技术+基本面+新闻+宏观） |
| 可视化 | 4 子图回测报告 + 技术指标面板 + 混淆矩阵 |

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo> SVM_trading
cd SVM_trading

# 安装依赖
pip install -r requirements.txt

# 配置密钥（复制并编辑 .env）
cp .env .env.local   # 重命名后编辑
```

### 2. 配置 `.env`

```ini
# Futu OpenD 连接（本机需启动 OpenD 并登录）
FUTU_HOST=127.0.0.1
FUTU_PORT=11111

# DeepSeek AI API（在 https://platform.deepseek.com/ 申请）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

### 3. 启动 Futu OpenD

从 [富途官网](https://www.futunn.com/download/openAPI) 下载并启动 OpenD，登录后保持运行。

### 4. 运行

```bash
# 完整流程（回测 + 实时信号 + AI 分析）
python main.py --code HK.00700 --name 腾讯控股 --start 2019-01-01 --end 2025-12-31 --mode full

# 仅回测（快速，跳过参数优化）
python main.py --code HK.09988 --name 阿里巴巴 --start 2020-01-01 --end 2025-06-30 --mode backtest --no-optimize

# 仅生成实时信号（需要已训练好的模型）
python main.py --code HK.00700 --name 腾讯控股 --mode realtime
```

---

## 项目结构

```
SVM_trading/
├── .env                          # 环境变量（API Key 等，不提交 Git）
├── main.py                       # 主入口（三种模式）
├── requirements.txt
├── config/
│   ├── settings.py               # 全局配置（从 .env 读取）
│   └── symbols.py                # 港股标的列表
├── data/
│   ├── futu_client.py            # Futu OpenD 数据封装
│   ├── data_store.py             # 本地 SQLite/CSV 缓存
│   └── news_fetcher.py           # 财联社/东方财富新闻获取
├── feature/
│   ├── technical_indicators.py   # MACD/RSI/KDJ/CCI/布林带/ATR
│   ├── vwap_indicator.py         # VWAP 多周期特征
│   └── feature_engineer.py       # 特征工程主模块
├── model/
│   ├── svm_model.py              # SVM 训练/预测/持久化
│   ├── signal_generator.py       # 买卖信号生成与过滤
│   └── model_evaluator.py        # 学习曲线/混淆矩阵诊断
├── backtest/
│   ├── backtest_engine.py        # 事件驱动回测引擎
│   ├── portfolio.py              # 多标的组合管理 + 止损
│   └── metrics.py                # 绩效指标计算
├── analysis/
│   ├── ai_analyst.py             # DeepSeek API 多空分析
│   ├── fundamental.py            # PE/PB/52周位置评估
│   └── macro_analysis.py         # 宏观环境摘要（恒指/南向/VIX）
├── visualization/
│   └── charts.py                 # 4 子图回测报告 + 技术指标面板
├── models/                       # 训练好的模型文件（自动生成）
│   ├── svm_model.pkl
│   ├── feature_scaler.pkl
│   └── feature_columns.json
└── output/                       # 输出文件（图表/报告/绩效）
    ├── backtest_HK_00700.png
    ├── ai_report_HK_00700_20260404_1200.md
    └── metrics_HK_00700.json
```

---

## SVM 分类逻辑

| 标签 | 含义 | 判定规则 |
|------|------|----------|
| +1   | 买入 | 未来 5 日收益 > +2% |
| 0    | 观望 | 未来 5 日收益在 ±2% |
| -1   | 卖出 | 未来 5 日收益 < -2% |

- 核函数：RBF（默认）+ Linear（对比）
- 调参：`GridSearchCV` + `TimeSeriesSplit` 防止前视偏差
- 信号过滤：置信度 < 55% 时降为观望；连续重复信号仅保留第一个

---

## 港股费率说明

| 费用 | 规则 |
|------|------|
| 佣金 | 成交额 × 0.05%，最低 HKD 5 |
| 印花税 | 成交额 × 0.1%（仅卖出方） |
| 交收费 | 本系统暂不模拟 |

---

## 风险提示

> **本系统仅供学习研究，不构成任何投资建议。**

1. SVM 基于历史数据，不保证未来有效
2. GridSearchCV 可能过拟合，建议 Walk-Forward 验证
3. 港股流动性不均，中小盘实际成交可能有滑点
4. 港股受中港两地监管政策影响，政策风险较大
5. AI 分析结论受模型限制，请结合自身判断

---

*版本: v1.0 | 日期: 2026-04-04*


对当前文件夹，使用conda 创建环境svm_traind， 安装必要的requirements，然后运行python main.py --code HK.01797 --name 东方甄选 --start 2019-01-01 --end 2025-12-31 --mode full