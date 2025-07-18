# 记忆系统配置文件
import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv  # 添加这行

logger = logging.getLogger(__name__)

class MemoryConfig:
    """记忆系统配置类"""
    
    def __init__(self):
        load_dotenv()  # 添加这行

        # 临时记录配置信息（不要在生产环境中记录密码）
        logger.info("加载Neo4j配置...")
        logger.info(f"NEO4J_URI: {os.getenv('NEO4J_URI')}")
        logger.info(f"NEO4J_USER: {os.getenv('NEO4J_USER')}")
        logger.info("NEO4J_PASSWORD is set: " + ("Yes" if os.getenv('NEO4J_PASSWORD') else "No"))

        # Neo4j数据库配置
        self.NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
        self.NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
        
        # 嵌入模型配置
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        
        # 记忆检索配置
        self.MEMORY_RETRIEVAL_LIMIT = int(os.getenv('MEMORY_RETRIEVAL_LIMIT', '5'))
        self.SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
        
        # 记忆维护配置
        self.MEMORY_CLEANUP_DAYS = int(os.getenv('MEMORY_CLEANUP_DAYS', '90'))
        # 启用自动清理
        self.AUTO_CLEANUP_ENABLED = os.getenv('AUTO_CLEANUP_ENABLED', 'true').lower() == 'true'
        self.ENTITY_MERGE_THRESHOLD = float(os.getenv('ENTITY_MERGE_THRESHOLD', '0.8'))
        
        # 功能开关
        # 启用记忆系统
        self.ENABLE_MEMORY_SYSTEM = os.getenv('ENABLE_MEMORY_SYSTEM', 'true').lower() == 'true'
        # 启用情感分析
        self.ENABLE_EMOTION_ANALYSIS = os.getenv('ENABLE_EMOTION_ANALYSIS', 'true').lower() == 'true'
        # 启用偏好提取
        self.ENABLE_PREFERENCE_EXTRACTION = os.getenv('ENABLE_PREFERENCE_EXTRACTION', 'true').lower() == 'true'
        # 启用实体提取
        self.ENABLE_ENTITY_EXTRACTION = os.getenv('ENABLE_ENTITY_EXTRACTION', 'true').lower() == 'true'
        
        # 性能配置
        self.BATCH_SIZE = int(os.getenv('MEMORY_BATCH_SIZE', '10'))
        self.CACHE_SIZE = int(os.getenv('MEMORY_CACHE_SIZE', '100'))
        
        # 日志配置
        self.LOG_LEVEL = os.getenv('MEMORY_LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.getenv('MEMORY_LOG_FILE', 'memory_system.log')
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'neo4j': {
                'uri': self.NEO4J_URI,
                'user': self.NEO4J_USER,
                'password': self.NEO4J_PASSWORD
            },
            'embedding': {
                'model': self.EMBEDDING_MODEL
            },
            'retrieval': {
                'limit': self.MEMORY_RETRIEVAL_LIMIT,
                'similarity_threshold': self.SIMILARITY_THRESHOLD
            },
            'maintenance': {
                'cleanup_days': self.MEMORY_CLEANUP_DAYS,
                'auto_cleanup': self.AUTO_CLEANUP_ENABLED,
                'entity_merge_threshold': self.ENTITY_MERGE_THRESHOLD
            },
            'features': {
                'memory_system': self.ENABLE_MEMORY_SYSTEM,
                'emotion_analysis': self.ENABLE_EMOTION_ANALYSIS,
                'preference_extraction': self.ENABLE_PREFERENCE_EXTRACTION,
                'entity_extraction': self.ENABLE_ENTITY_EXTRACTION
            },
            'performance': {
                'batch_size': self.BATCH_SIZE,
                'cache_size': self.CACHE_SIZE
            },
            'logging': {
                'level': self.LOG_LEVEL,
                'file': self.LOG_FILE
            }
        }
    
    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            # 检查必要的配置项
            if not self.NEO4J_URI or not self.NEO4J_USER or not self.NEO4J_PASSWORD:
                print("错误: Neo4j数据库配置不完整")
                return False
            
            if self.MEMORY_RETRIEVAL_LIMIT <= 0:
                print("错误: 记忆检索限制必须大于0")
                return False
            
            if not (0.0 <= self.SIMILARITY_THRESHOLD <= 1.0):
                print("错误: 相似度阈值必须在0.0到1.0之间")
                return False
            
            if self.MEMORY_CLEANUP_DAYS <= 0:
                print("错误: 记忆清理天数必须大于0")
                return False
            
            if not (0.0 <= self.ENTITY_MERGE_THRESHOLD <= 1.0):
                print("错误: 实体合并阈值必须在0.0到1.0之间")
                return False
            
            return True
            
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False

# 全局配置实例
memory_config = MemoryConfig()

