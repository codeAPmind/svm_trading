"""
全局配置模块
从 .env 文件读取敏感配置，其余参数使用默认值
"""
import os
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

# 自动加载 .env 文件（不依赖 python-dotenv，手动解析）
_env_path = Path(__file__).parent.parent / '.env'
if _env_path.exists():
    with open(_env_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, val = line.partition('=')
                key = key.strip()
                val = val.strip()
                if key and val and key not in os.environ:
                    os.environ[key] = val


@dataclass
class FutuConfig:
    """Futu OpenD 连接配置"""
    host: str = os.environ.get('FUTU_HOST', '127.0.0.1')
    port: int = int(os.environ.get('FUTU_PORT', '11111'))
    market: str = 'HK'


@dataclass
class SVMConfig:
    """SVM 模型配置"""
    kernel: str = 'rbf'
    C_range: List[float] = field(default_factory=lambda: [0.1, 1, 10, 100])
    gamma_range: List[str] = field(default_factory=lambda: ['scale', 'auto'])
    test_size: float = 0.2
    lookback_window: int = 20
    prediction_horizon: int = 5
    threshold_buy: float = 0.02
    threshold_sell: float = -0.02


@dataclass
class BacktestConfig:
    """回测配置（支持 .env 覆盖）"""
    initial_capital: float = float(os.environ.get('INITIAL_CAPITAL', '1000000'))
    commission_rate: float = float(os.environ.get('COMMISSION_RATE', '0.0005'))
    stamp_duty: float = 0.001           # 港股印花税（卖出时收取）
    min_commission: float = 5.0         # 最低佣金（HKD）
    position_size: float = float(os.environ.get('POSITION_SIZE', '0.3'))
    max_positions: int = int(os.environ.get('MAX_POSITIONS', '3'))


@dataclass
class DeepSeekConfig:
    """DeepSeek AI API 配置（用于多空分析）"""
    api_key: str = os.environ.get('DEEPSEEK_API_KEY', '')
    model: str = os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')
    base_url: str = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
    max_tokens: int = 2000
    temperature: float = 0.3


# ── 全局配置实例 ──────────────────────────────────────────────
FUTU_CFG = FutuConfig()
SVM_CFG = SVMConfig()
BACKTEST_CFG = BacktestConfig()
DEEPSEEK_CFG = DeepSeekConfig()

# 向后兼容：部分旧代码引用 CLAUDE_CFG
CLAUDE_CFG = DEEPSEEK_CFG
