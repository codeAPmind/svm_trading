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
    """SVM 模型配置

    C（惩罚系数）：控制对误分类的容忍度
      - 小C → 宽间隔，容忍误分类，泛化更好，不易过拟合
      - 大C → 窄间隔，尽量不误分类，容易过拟合
      - 搜索范围用对数均匀分布，避免跨度过大时遗漏最优值

    gamma（RBF核带宽）：控制单个样本的影响范围
      - 小gamma → 影响范围大，决策面平滑，欠拟合风险
      - 大gamma → 影响范围小，决策面复杂，过拟合风险
      - 'scale' = 1/(n_features × var(X))，随特征数自动缩放
      - 'auto'  = 1/n_features，不考虑特征方差
      - 加入具体数值候选，在 scale/auto 之外再精细搜索

    v2.0 变更：
      - C_range 从 [0.1,1,10,100]（4个，10倍间距）
              → [0.1,0.5,1,5,10,50,100]（7个，更均匀）
      - gamma_range 从 ['scale','auto']（2个）
                  → ['scale','auto',0.001,0.01,0.1]（5个，加具体值）
      - 搜索组合数：32 → 7×5×2×2 = 140（约4倍，结果更精确）
    """
    kernel: str = 'rbf'
    # C 搜索空间：对数均匀，覆盖从宽松到严格的完整范围
    C_range: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 1, 5, 10, 50, 100]
    )
    # gamma 搜索空间：自动值 + 具体数值候选
    # 具体值参考：特征数65时 scale ≈ 1/(65×var)，通常在 0.001~0.05 之间
    gamma_range: List = field(
        default_factory=lambda: ['scale', 'auto', 0.001, 0.01, 0.1]
    )
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
