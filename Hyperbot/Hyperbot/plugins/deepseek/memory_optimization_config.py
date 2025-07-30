# -*- coding: utf-8 -*-
"""
记忆系统优化配置文件
为了提升机器人的知识查阅、记忆回忆和回复效率
"""

# 缓存配置
CACHE_CONFIG = {
    "knowledge_cache": {
        "max_size": 500,
        "ttl": 1800  # 30分钟，知识查询结果相对稳定
    },
    "embedding_cache": {
        "max_size": 1000,
        "ttl": 3600  # 1小时，向量嵌入计算成本高
    },
    "memory_cache": {
        "max_size": 200,
        "ttl": 600   # 10分钟，记忆查询结果需要较新
    }
}

# 知识检索配置
KNOWLEDGE_CONFIG = {
    "similarity_threshold": 0.7,  # 记忆相似度阈值
    "max_memory_days": 30,        # 最大记忆检索天数
    "max_knowledge_triples": 10,  # 最大知识三元组数量
    "max_context_length": 5000,   # 最大上下文长度
    "enable_external_search": True  # 是否启用外部知识搜索
}

# 外部知识源配置
EXTERNAL_KNOWLEDGE_CONFIG = {
    "wikipedia": {
        "enabled": True,
        "language": "zh",  # 中文维基百科
        "timeout": 10.0,
        "max_content_length": 500
    },
    "web_search": {
        "enabled": False,  # 需要配置搜索API
        "api_key": "",     # 搜索引擎API密钥
        "max_results": 3
    }
}

# 性能优化配置
PERFORMANCE_CONFIG = {
    "max_concurrent_tasks": 5,     # 最大并发任务数
    "api_timeout": 45.0,           # API超时时间
    "batch_similarity_calc": True, # 批量计算相似度
    "preload_embedding_model": False,  # 是否预加载嵌入模型
    "async_conversation_storage": True  # 异步存储对话
}

# 回复优化配置
RESPONSE_CONFIG = {
    "enable_smart_context_selection": True,  # 智能上下文选择
    "context_priority": [
        "external_knowledge",  # 最高优先级：外部专业知识
        "internal_knowledge",  # 内部知识图谱
        "fact_checking",       # 事实核查
        "recent_context",      # 近期对话
        "historical_memory"    # 历史记忆
    ],
    "max_response_tokens": 1000,
    "temperature": 0.8
}

# 知识图谱优化配置
GRAPH_CONFIG = {
    "enable_multi_layer_query": True,  # 启用多层关系查询
    "max_query_depth": 2,              # 最大查询深度
    "enable_entity_linking": True,     # 启用实体链接
    "auto_knowledge_extraction": True  # 自动知识提取
}

# 监控和日志配置
MONITORING_CONFIG = {
    "log_cache_performance": True,     # 记录缓存性能
    "log_query_time": True,            # 记录查询时间
    "log_knowledge_sources": True,     # 记录知识来源
    "performance_alerts": {
        "slow_query_threshold": 5.0,   # 慢查询阈值（秒）
        "cache_hit_rate_threshold": 0.3  # 缓存命中率阈值
    }
}

# 使用示例
def get_optimized_memory_config():
    """获取优化后的记忆系统配置"""
    return {
        "cache": CACHE_CONFIG,
        "knowledge": KNOWLEDGE_CONFIG,
        "external": EXTERNAL_KNOWLEDGE_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "response": RESPONSE_CONFIG,
        "graph": GRAPH_CONFIG,
        "monitoring": MONITORING_CONFIG
    }

# 性能调优建议
TUNING_RECOMMENDATIONS = {
    "high_traffic_scenario": {
        "description": "高流量场景优化",
        "adjustments": {
            "cache.knowledge_cache.max_size": 3000,
            "cache.memory_cache.ttl": 300,
            "performance.max_concurrent_tasks": 8,
            "knowledge.similarity_threshold": 0.75
        }
    },
    "low_latency_scenario": {
        "description": "低延迟场景优化",
        "adjustments": {
            "external.wikipedia.enabled": False,
            "knowledge.max_memory_days": 7,
            "performance.preload_embedding_model": True,
            "response.max_response_tokens": 1000
        }
    },
    "knowledge_intensive_scenario": {
        "description": "知识密集型场景优化",
        "adjustments": {
            "external.wikipedia.enabled": True,
            "graph.max_query_depth": 3,
            "knowledge.max_knowledge_triples": 15,
            "response.context_priority": ["external_knowledge", "internal_knowledge"]
        }
    }
} 