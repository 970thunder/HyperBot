# -*- coding: utf-8 -*-
"""
优化后的记忆系统使用示例
展示如何利用新功能提升机器人回复效率和准确性
"""

import asyncio
from memory_graph import MemoryGraph
from memory_optimization_config import get_optimized_memory_config

class OptimizedChatBot:
    """优化后的聊天机器人示例"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, deepseek_api_key: str):
        self.memory_graph = MemoryGraph(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user, 
            neo4j_password=neo4j_password,
            deepseek_api_key=deepseek_api_key
        )
        self.config = get_optimized_memory_config()
        
    async def enhanced_chat(self, user_id: str, session_id: str, message: str, 
                           persona_content: str, user_nickname: str = None,
                           group_members: list = None) -> str:
        """
        增强版聊天函数，整合所有优化功能
        """
        try:
            # 使用优化后的上下文回复生成
            response = await self.memory_graph.generate_contextual_response(
                user_id=user_id,
                session_id=session_id,
                current_message=message,
                persona_content=persona_content,
                user_nickname=user_nickname,
                use_memory=True,
                group_members=group_members
            )
            
            return response
            
        except Exception as e:
            print(f"聊天过程中出现错误: {e}")
            return "抱歉，我现在有点忙，稍后再聊吧"

    async def add_knowledge_to_graph(self, knowledge_data: dict) -> bool:
        """
        添加知识到图数据库
        """
        try:
            success = await self.memory_graph.store_knowledge_graph(knowledge_data)
            if success:
                print(f"成功添加知识: {len(knowledge_data.get('nodes', []))} 个节点")
            return success
        except Exception as e:
            print(f"添加知识失败: {e}")
            return False

    async def search_knowledge(self, query: str, source: str = "wikipedia") -> str:
        """
        搜索外部知识
        """
        try:
            result = await self.memory_graph.search_external_knowledge(query, source)
            if result:
                print(f"从 {source} 检索到知识: {query}")
            return result
        except Exception as e:
            print(f"知识搜索失败: {e}")
            return ""

    def get_cache_stats(self) -> dict:
        """
        获取缓存统计信息
        """
        return {
            "knowledge_cache_size": len(self.memory_graph.knowledge_cache.cache),
            "embedding_cache_size": len(self.memory_graph.embedding_cache.cache),
            "memory_cache_size": len(self.memory_graph.memory_cache.cache)
        }

# 使用示例
async def demo_usage():
    """演示优化后的功能"""
    
    # 初始化机器人（请替换为实际的配置信息）
    bot = OptimizedChatBot(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j", 
        neo4j_password="your_password",
        deepseek_api_key="your_deepseek_api_key"
    )
    
    # 示例人设
    persona = """
    你是一个博学的AI助手，擅长回答各种问题。你的回复应该：
    1. 简洁明了
    2. 准确可靠
    3. 富有人情味
    """
    
    # 示例1：普通对话（会使用记忆）
    print("=== 示例1：普通对话 ===")
    response1 = await bot.enhanced_chat(
        user_id="12345",
        session_id="session_001", 
        message="你好，我想了解一下人工智能",
        persona_content=persona,
        user_nickname="小明"
    )
    print(f"用户: 你好，我想了解一下人工智能")
    print(f"机器人: {response1}")
    
    # 示例2：知识密集型问题（会触发外部知识搜索）
    print("\n=== 示例2：知识密集型问题 ===")
    response2 = await bot.enhanced_chat(
        user_id="12345",
        session_id="session_001",
        message="什么是量子计算的基本原理？",
        persona_content=persona,
        user_nickname="小明"
    )
    print(f"用户: 什么是量子计算的基本原理？")
    print(f"机器人: {response2}")
    
    # 示例3：添加自定义知识
    print("\n=== 示例3：添加自定义知识 ===")
    knowledge_data = {
        "nodes": [
            {
                "id": "量子计算",
                "type": "Technology",
                "properties": {"description": "利用量子力学现象进行计算的技术"}
            },
            {
                "id": "量子比特",
                "type": "Concept", 
                "properties": {"description": "量子计算的基本单位"}
            }
        ],
        "relations": [
            {
                "source": "量子计算",
                "target": "量子比特",
                "type": "USES",
                "properties": {"strength": 0.9}
            }
        ]
    }
    
    success = await bot.add_knowledge_to_graph(knowledge_data)
    print(f"知识添加结果: {'成功' if success else '失败'}")
    
    # 示例4：外部知识搜索
    print("\n=== 示例4：外部知识搜索 ===")
    wiki_result = await bot.search_knowledge("机器学习", "wikipedia")
    if wiki_result:
        print(f"维基百科搜索结果:\n{wiki_result[:200]}...")
    
    # 示例5：查看缓存统计
    print("\n=== 示例5：缓存统计 ===")
    cache_stats = bot.get_cache_stats()
    print(f"缓存统计: {cache_stats}")

# 性能测试示例
async def performance_test():
    """性能测试示例"""
    import time
    
    bot = OptimizedChatBot(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password", 
        deepseek_api_key="your_deepseek_api_key"
    )
    
    persona = "你是一个helpful的AI助手"
    test_messages = [
        "你好",
        "今天天气怎么样？",
        "什么是深度学习？",
        "能帮我解释一下区块链吗？",
        "谢谢你的帮助"
    ]
    
    print("=== 性能测试 ===")
    total_time = 0
    
    for i, message in enumerate(test_messages):
        start_time = time.time()
        
        response = await bot.enhanced_chat(
            user_id="test_user",
            session_id="performance_test",
            message=message,
            persona_content=persona
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        total_time += response_time
        
        print(f"消息 {i+1}: {message}")
        print(f"回复时间: {response_time:.2f}秒")
        print(f"回复: {response[:100]}...")
        print("-" * 50)
    
    print(f"平均回复时间: {total_time / len(test_messages):.2f}秒")
    print(f"总时间: {total_time:.2f}秒")

if __name__ == "__main__":
    # 运行演示
    print("开始演示优化后的记忆系统...")
    asyncio.run(demo_usage())
    
    # 可选：运行性能测试
    # print("\n开始性能测试...")
    # asyncio.run(performance_test()) 