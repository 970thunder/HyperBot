"""
心流机制配置文件
支持灵活配置各种参数，适应不同群聊环境
"""

import os
from dataclasses import dataclass
from typing import Optional

# 添加dotenv支持
try:
    from dotenv import load_dotenv
    load_dotenv()  # 显式加载.env文件
except ImportError:
    pass

# 添加NoneBot配置支持
try:
    from nonebot import get_driver
    nonebot_driver = get_driver()
    nonebot_config = nonebot_driver.config
except:
    nonebot_driver = None
    nonebot_config = None

@dataclass
class HeartflowConfig:
    """心流机制配置类"""
    
    # === 核心回复参数 ===
    reply_threshold: float = 0.6  # 回复阈值 (0-1，越高越难触发回复)
    
    # === 精力系统参数 ===
    energy_decay_rate: float = 0.15  # 精力衰减速度 (每次回复消耗的精力)
    energy_recovery_rate: float = 0.02  # 精力恢复速度 (每分钟恢复的精力)
    daily_energy_bonus: float = 0.3  # 每日精力恢复奖励
    min_energy_threshold: float = 0.3  # 最低精力阈值 (低于此值不会回复)
    
    # === 频率控制参数 ===
    max_replies_per_hour: int = 10  # 每小时最大回复数
    min_reply_interval: int = 30  # 最小回复间隔(秒)
    context_messages_count: int = 5  # 上下文消息数量
    
    # === AI 分析参数 ===
    api_timeout: int = 15  # API调用超时时间从60秒减少到15秒
    analysis_temperature: float = 0.3  # AI分析温度 (0-1，越高越随机)
    max_analysis_tokens: int = 200  # AI分析最大token数从300减少到200
    
    # === 评分权重 ===
    content_relevance_weight: float = 0.3  # 内容相关度权重
    reply_willingness_weight: float = 0.25  # 回复意愿权重
    social_appropriateness_weight: float = 0.25  # 社交适宜性权重
    timing_appropriateness_weight: float = 0.2  # 时机恰当性权重
    
    # === 环境变量配置 ===
    @classmethod
    def from_env(cls) -> 'HeartflowConfig':
        """从环境变量创建配置"""
        return cls(
            reply_threshold=float(os.getenv("HEARTFLOW_REPLY_THRESHOLD", "0.6")),
            energy_decay_rate=float(os.getenv("HEARTFLOW_ENERGY_DECAY_RATE", "0.15")),
            energy_recovery_rate=float(os.getenv("HEARTFLOW_ENERGY_RECOVERY_RATE", "0.02")),
            daily_energy_bonus=float(os.getenv("HEARTFLOW_DAILY_ENERGY_BONUS", "0.3")),
            min_energy_threshold=float(os.getenv("HEARTFLOW_MIN_ENERGY_THRESHOLD", "0.3")),
            
            max_replies_per_hour=int(os.getenv("HEARTFLOW_MAX_REPLIES_PER_HOUR", "10")),
            min_reply_interval=int(os.getenv("HEARTFLOW_MIN_REPLY_INTERVAL", "30")),
            context_messages_count=int(os.getenv("HEARTFLOW_CONTEXT_MESSAGES_COUNT", "5")),
            
            api_timeout=int(os.getenv("HEARTFLOW_API_TIMEOUT", "10")),
            analysis_temperature=float(os.getenv("HEARTFLOW_ANALYSIS_TEMPERATURE", "0.3")),
            max_analysis_tokens=int(os.getenv("HEARTFLOW_MAX_ANALYSIS_TOKENS", "300")),
        )
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        if not (0 <= self.reply_threshold <= 1):
            return False
        if not (0 < self.energy_decay_rate <= 1):
            return False
        if not (0 < self.energy_recovery_rate <= 1):
            return False
        if not (0 <= self.daily_energy_bonus <= 1):
            return False
        if not (0 < self.min_energy_threshold <= 1):
            return False
        if self.max_replies_per_hour <= 0:
            return False
        if self.min_reply_interval < 0:
            return False
        if self.context_messages_count < 1:
            return False
        if self.api_timeout <= 0:
            return False
        if not (0 <= self.analysis_temperature <= 2):
            return False
        if self.max_analysis_tokens <= 0:
            return False
        
        # 检查权重总和是否接近1
        total_weight = (
            self.content_relevance_weight + 
            self.reply_willingness_weight + 
            self.social_appropriateness_weight + 
            self.timing_appropriateness_weight
        )
        if not (0.9 <= total_weight <= 1.1):
            return False
        
        return True
    
    def get_weights(self) -> dict:
        """获取评分权重字典"""
        return {
            "content_relevance": self.content_relevance_weight,
            "reply_willingness": self.reply_willingness_weight,
            "social_appropriateness": self.social_appropriateness_weight,
            "timing_appropriateness": self.timing_appropriateness_weight
        }

# === 预设配置模板 ===

def get_active_config() -> HeartflowConfig:
    """活跃群聊配置 - 更频繁的回复"""
    return HeartflowConfig(
        reply_threshold=0.5,
        energy_decay_rate=0.12,
        energy_recovery_rate=0.03,
        max_replies_per_hour=15,
        min_reply_interval=20,
        content_relevance_weight=0.35,
        reply_willingness_weight=0.3,
        social_appropriateness_weight=0.2,
        timing_appropriateness_weight=0.15
    )

def get_conservative_config() -> HeartflowConfig:
    """保守群聊配置 - 更谨慎的回复"""
    return HeartflowConfig(
        reply_threshold=0.7,
        energy_decay_rate=0.2,
        energy_recovery_rate=0.015,
        max_replies_per_hour=6,
        min_reply_interval=60,
        min_energy_threshold=0.4,
        content_relevance_weight=0.4,
        reply_willingness_weight=0.2,
        social_appropriateness_weight=0.3,
        timing_appropriateness_weight=0.1
    )

def get_balanced_config() -> HeartflowConfig:
    """平衡配置 - 默认推荐"""
    return HeartflowConfig()  # 使用默认值

# 延迟初始化全局配置实例
def get_heartflow_config() -> HeartflowConfig:
    """获取心流配置实例（延迟初始化）"""
    try:
        config = HeartflowConfig.from_env()
        if config.validate():
            return config
        else:
            print("警告：心流配置验证失败，使用默认配置")
            return get_balanced_config()
    except Exception as e:
        print(f"警告：心流配置加载失败 ({e})，使用默认配置")
        return get_balanced_config()

# 延迟初始化，避免在模块导入时立即执行
heartflow_config = None 