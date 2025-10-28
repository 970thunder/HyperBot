import asyncio
import json
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import httpx
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import logging
import hashlib
from collections import defaultdict
from asyncio import Semaphore
import concurrent.futures
from functools import lru_cache

# 配置日志
logger = logging.getLogger(__name__)

# 添加缓存装饰器
class CacheManager:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str):
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value):
        if len(self.cache) >= self.max_size:
            # 删除最旧的缓存项
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()

@dataclass
class MemoryNode:
    """记忆节点数据结构"""
    id: str
    type: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class MemoryRelation:
    """记忆关系数据结构"""
    from_node: str
    to_node: str
    relation_type: str
    strength: float
    metadata: Dict[str, Any]

class MemoryGraph:
    """基于Neo4j的图数据库记忆系统 - 优化版本"""

    def test_connection(self):
        """测试数据库连接"""
        with self.driver.session() as session:
            result = session.run("RETURN 1 AS test")
            if result.single()["test"] != 1:
                raise ConnectionError("Neo4j连接测试失败")
            else:
                logger.info("Neo4j连接测试成功")
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                deepseek_api_key: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        初始化记忆图系统
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.deepseek_api_key = deepseek_api_key
        
        # 添加缓存管理器
        self.knowledge_cache = CacheManager(max_size=500, ttl=1800)  # 知识查询缓存30分钟
        self.embedding_cache = CacheManager(max_size=1000, ttl=3600)  # 嵌入缓存1小时
        self.memory_cache = CacheManager(max_size=200, ttl=600)  # 记忆查询缓存10分钟
        
        # 并发控制
        self.api_semaphore = Semaphore(30)  # API调用并发控制
        self.db_semaphore = Semaphore(20)  # 数据库操作并发控制
        self.embedding_semaphore = Semaphore(10)  # 嵌入计算并发控制
        
        # 线程池用于CPU密集型任务
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=15, thread_name_prefix="memory")
        
        # 使用 atexit 确保在程序退出时关闭驱动
        import atexit
        atexit.register(self.close)
        
        try:
            # 延迟加载模型，只在第一次需要时加载
            self._embedding_model = None
            self.embedding_model_name = embedding_model
            self.test_connection()
            self._init_database()
        except Exception as e:
            logger.error(f"Neo4j连接或初始化失败: {str(e)}")
            raise

    @property
    def embedding_model(self):
        """延迟加载嵌入模型，避免启动时阻塞"""
        if self._embedding_model is None:
            logger.info(f"正在加载嵌入模型: {self.embedding_model_name}...")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("嵌入模型加载完成。")
        return self._embedding_model
    
    def _init_database(self):
        """初始化数据库约束和索引"""
        with self.driver.session() as session:
            # 创建唯一性约束
            constraints = [
                "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:USER) REQUIRE u.id IS UNIQUE",
                "CREATE CONSTRAINT conversation_id_unique IF NOT EXISTS FOR (c:CONVERSATION) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT topic_name_unique IF NOT EXISTS FOR (t:TOPIC) REQUIRE t.name IS UNIQUE",
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:ENTITY) REQUIRE e.name IS UNIQUE"
            ]
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"约束创建失败或已存在: {e}")
            
            # 创建索引
            indexes = [
                "CREATE INDEX conversation_timestamp IF NOT EXISTS FOR (c:CONVERSATION) ON (c.timestamp)",
                "CREATE INDEX conversation_session_id IF NOT EXISTS FOR (c:CONVERSATION) ON (c.session_id)",
                "CREATE INDEX topic_frequency IF NOT EXISTS FOR (t:TOPIC) ON (t.frequency)",
                "CREATE INDEX entity_importance IF NOT EXISTS FOR (e:ENTITY) ON (e.importance)"
            ]
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.debug(f"索引创建失败或已存在: {e}")
            logger.info("数据库约束和索引初始化完成")

    async def get_embedding(self, text: str) -> List[float]:
        """在线程中异步获取文本的向量嵌入，避免阻塞 - 添加缓存优化"""
        try:
            # 使用文本的哈希作为缓存键
            cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
            cached_embedding = self.embedding_cache.get(cache_key)
            if cached_embedding is not None:
                return cached_embedding
            
            # 使用信号量控制并发
            async with self.embedding_semaphore:
                # 如果缓存中没有，则计算嵌入
                embedding = await asyncio.to_thread(self.embedding_model.encode, text)
                embedding_list = embedding.tolist()
                
                # 存入缓存
                self.embedding_cache.set(cache_key, embedding_list)
                return embedding_list
        except Exception as e:
            logger.error(f"生成嵌入失败: {e}")
            return []

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except Exception:
            return 0.0

    async def _make_api_call(self, system_prompt: str, user_prompt: str, temperature: float = 0.5, max_tokens: int = 500) -> str:
        """通用的API调用函数 - 优化版本"""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.deepseek_api_key}", "Content-Type": "application/json"}
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            # 使用信号量控制并发和连接池
            async with self.api_semaphore:
                async with httpx.AsyncClient(
                    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                    timeout=httpx.Timeout(30.0, connect=5.0, read=25.0)
                ) as client:
                    response = await client.post(url, json=data, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    # 清理可能的markdown格式
                    return content.strip().replace("```json", "").replace("```", "").strip()
        except httpx.ReadTimeout:
            logger.error("API调用超时")
            return ""
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return ""

    async def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """分析文本情感"""
        prompt = f"""
        请分析以下文本的情感信息，包括情感类型和强度。文本：{text}
        请以JSON格式返回，格式如下：
        {{"sentiment_score": 0.5, "emotions": [{{"type": "happy", "intensity": 0.8}}]}}
        sentiment_score范围：-1到1，intensity范围：0到1。只返回JSON。
        """
        content = await self._make_api_call("你是一个专业的情感分析助手，只返回JSON格式的结果。", prompt, 0.2, 300)
        try:
            return json.loads(content) if content else {"sentiment_score": 0.0, "emotions": []}
        except json.JSONDecodeError:
            logger.warning(f"情感分析返回的不是有效JSON: {content}")
            return {"sentiment_score": 0.0, "emotions": []}

    async def extract_entities_with_ai(self, text: str) -> Dict[str, List[str]]:
        """使用AI提取实体信息"""
        prompt = f"""
        请从以下文本中提取关键实体，包括人名、地点、物品、时间、情感、话题。文本：{text}
        以JSON格式返回，格式如下：
        {{"persons": [], "locations": [], "objects": [], "times": [], "emotions": [], "topics": []}}
        只返回JSON。
        """
        content = await self._make_api_call("你是一个专业的实体提取助手，只返回JSON格式的结果。", prompt, 0.2, 500)
        try:
            return json.loads(content) if content else {}
        except json.JSONDecodeError:
            logger.warning(f"AI实体提取返回的不是有效JSON: {content}")
            return {}

    def create_or_get_user(self, user_id: str, user_info: Dict[str, Any] = None) -> bool:
        """创建或获取用户节点"""
        with self.driver.session() as session:
            try:
                result = session.run("MATCH (u:USER {id: $user_id}) RETURN u", user_id=user_id)
                if result.single(): return True
                user_data = {'id': user_id, 'created_at': datetime.now().isoformat()}
                if user_info: user_data.update(user_info)
                session.run("CREATE (u:USER $props)", props=user_data)
                logger.info(f"创建新用户: {user_id}")
                return True
            except Exception as e:
                logger.error(f"创建/获取用户失败: {e}")
                return False

    async def store_conversation(self, user_id: str, session_id: str, user_message: str, 
                                bot_response: str, persona_name: str = None, 
                                user_nickname: str = None) -> bool:
        """异步存储对话记录到图数据库"""
        try:
            if not user_id.isdigit(): user_id = ''.join(filter(str.isdigit, user_id))
            if not user_id: return False

            self.create_or_get_user(user_id)
            conversation_id = f"qq_{user_id}_{session_id}_{int(time.time() * 1000)}"

            # 并行处理所有阻塞和I/O密集型任务
            user_embedding, bot_embedding, entities, emotion_info = await asyncio.gather(
                self.get_embedding(user_message),
                self.get_embedding(bot_response),
                self.extract_entities_with_ai(user_message),
                self.analyze_emotion(user_message)
            )

            # 将数据库写入操作也放入线程中，以完全释放事件循环
            def _db_write():
                with self.driver.session() as session:
                    conversation_data = {
                        'id': conversation_id, 'user_id': user_id, 'user_nickname': user_nickname or '未知用户',
                        'session_id': session_id, 'user_message': user_message, 'bot_response': bot_response,
                        'persona_name': persona_name, 'timestamp': datetime.now().isoformat(),
                        'user_embedding': user_embedding, 'bot_embedding': bot_embedding,
                        'sentiment_score': emotion_info.get('sentiment_score', 0.0)
                    }
                    session.run("CREATE (c:CONVERSATION $props)", props=conversation_data)
                    session.run("""
                        MATCH (u:USER {id: $user_id}), (c:CONVERSATION {id: $conversation_id})
                        CREATE (u)-[:PARTICIPATED_IN]->(c)
                    """, user_id=user_id, conversation_id=conversation_id)

                    if persona_name:
                         session.run("""
                            MERGE (p:PERSONA {name: $persona_name}) SET p.usage_count = coalesce(p.usage_count, 0) + 1
                            WITH p MATCH (c:CONVERSATION {id: $cid}) CREATE (c)-[:USES_PERSONA]->(p)
                        """, persona_name=persona_name, cid=conversation_id)

                    if entities:
                        for entity_type, entity_list in entities.items():
                            for entity_name in entity_list:
                                if entity_name.strip():
                                    session.run("""
                                        MERGE (e:ENTITY {name: $name, type: $type})
                                        SET e.mention_count = coalesce(e.mention_count, 0) + 1
                                        WITH e MATCH (c:CONVERSATION {id: $cid}) CREATE (c)-[:MENTIONED]->(e)
                                    """, name=entity_name, type=entity_type, cid=conversation_id)
            
            # 使用信号量控制数据库并发
            async with self.db_semaphore:
                await asyncio.to_thread(_db_write)
            logger.info(f"存储对话记录: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"存储对话失败: {e}")
            return False

    async def retrieve_relevant_memories(self, user_id: str, current_message: str, 
                                        limit: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """异步检索相关记忆 - 优化版本"""
        try:
            # 1. 缓存检查
            cache_key = f"memory_{user_id}_{hashlib.md5(current_message.encode('utf-8')).hexdigest()}"
            cached_memories = self.memory_cache.get(cache_key)
            if cached_memories is not None:
                logger.info(f"从缓存获取相关记忆: {current_message[:30]}...")
                return cached_memories

            # 2. 获取当前消息的嵌入
            current_embedding = await self.get_embedding(current_message)
            if not current_embedding: return []

            def _optimized_memory_retrieval():
                with self.driver.session() as session:
                    # 优化查询：只查询最近30天的对话，并预过滤
                    result = session.run("""
                        MATCH (u:USER {id: $user_id})-[:PARTICIPATED_IN]->(c:CONVERSATION)
                        WHERE c.timestamp > datetime() - duration({days: 30})
                        AND c.user_embedding IS NOT NULL
                        AND size(c.user_embedding) > 0
                        RETURN c ORDER BY c.timestamp DESC LIMIT 100
                    """, user_id=user_id)
                    
                    memories = []
                    current_embedding_np = np.array(current_embedding)
                    
                    # 批量计算相似度
                    conversations = []
                    embeddings = []
                    
                    for record in result:
                        conv = dict(record['c'])
                        user_embedding = conv.get('user_embedding', [])
                        if len(user_embedding) == len(current_embedding):
                            conversations.append(conv)
                            embeddings.append(user_embedding)
                    
                    if not embeddings:
                        return []
                    
                    # 向量化计算相似度（更高效）
                    embeddings_np = np.array(embeddings)
                    similarities = np.dot(embeddings_np, current_embedding_np) / (
                        np.linalg.norm(embeddings_np, axis=1) * np.linalg.norm(current_embedding_np)
                    )
                    
                    # 过滤并排序
                    for i, sim in enumerate(similarities):
                        if sim >= similarity_threshold:
                            memories.append({
                                'conversation': conversations[i], 
                                'relevance_score': float(sim)
                            })
                    
                    memories.sort(key=lambda x: x['relevance_score'], reverse=True)
                    return memories[:limit]
            
            result = await asyncio.to_thread(_optimized_memory_retrieval)
            
            # 3. 缓存结果（时间较短，因为记忆更动态）
            self.memory_cache.set(cache_key, result)
            logger.info(f"检索到 {len(result)} 条相关记忆")
            return result
            
        except Exception as e:
            logger.error(f"检索记忆失败: {e}")
            return []

    async def get_conversation_context(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """异步获取对话上下文"""
        def _db_read():
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:CONVERSATION {session_id: $session_id})
                    RETURN c ORDER BY c.timestamp DESC LIMIT $limit
                """, session_id=session_id, limit=limit)
                # 使用列表推导式更简洁
                return [dict(record['c']) for record in result]
        try:
            context = await asyncio.to_thread(_db_read)
            return list(reversed(context))
        except Exception as e:
            logger.error(f"获取对话上下文失败: {e}")
            return []

    def format_memory_for_prompt(self, memories: List[Dict[str, Any]], context_type: str) -> str:
        """统一格式化记忆和上下文用于prompt"""
        if not memories: return ""
        
        formatted_parts = []
        for mem in memories:
            if context_type == "memory":
                conv = mem.get('conversation', {})
                score = mem.get('relevance_score', 0.0)
                user_display = conv.get('user_nickname') or conv.get('user_id', '用户')
                formatted_parts.append(f"[相关度:{score:.2f}] {user_display}: {conv.get('user_message', '')} | 你: {conv.get('bot_response', '')}")
            elif context_type == "context":
                user_display = mem.get('user_nickname') or mem.get('user_id', '用户')
                formatted_parts.append(f"{user_display}: {mem.get('user_message', '')}\n你: {mem.get('bot_response', '')}")
        
        title = "== 相关记忆 ==" if context_type == "memory" else "== 近期对话 =="
        return f"{title}\n" + "\n".join(formatted_parts)

    async def generate_contextual_response(self, user_id: str, session_id: str,
                                         current_message: str, persona_content: str,
                                         user_nickname: str = None, use_memory: bool = True,
                                         group_members: List[Dict[str, str]] = None) -> str:
        """基于记忆、事实核查、外部知识和工具生成带上下文的回复 - 增强版本"""
        try:
            # 1. 预处理：判断是否需要外部知识搜索
            needs_external_search = await self._should_search_external(current_message)
            
            # 2. 并行获取所有需要的数据（最大化并行度）
            tasks = []
            
            # 核心任务
            if use_memory:
                tasks.append(self.retrieve_relevant_memories(user_id, current_message))
                tasks.append(self.get_conversation_context(session_id, 5))
            else:
                tasks.extend([asyncio.sleep(0, result=[]), asyncio.sleep(0, result=[])])
            
            tasks.append(self.query_knowledge_graph(current_message))
            
            if group_members:
                tasks.append(self._fact_check_and_augment_prompt(current_message, group_members, user_id))
            else:
                tasks.append(asyncio.sleep(0, result=""))
            
            # 外部知识搜索（如果需要）
            if needs_external_search:
                # 提取查询关键词
                query_keywords = await self._extract_search_keywords(current_message)
                if query_keywords:
                    tasks.append(self.search_external_knowledge(query_keywords))
                else:
                    tasks.append(asyncio.sleep(0, result=""))
            else:
                tasks.append(asyncio.sleep(0, result=""))

            # 等待所有任务完成
            results = await asyncio.gather(*tasks)
            relevant_memories, recent_context, knowledge_results, fact_checking_results, external_knowledge = results

            # 3. 智能信息融合和优先级排序
            context_parts = []
            
            # 优先级1: 外部专业知识（如果可用）
            if external_knowledge:
                context_parts.append(external_knowledge)
            
            # 优先级2: 内部知识图谱
            if knowledge_results:
                context_parts.append(knowledge_results)
            
            # 优先级3: 事实核查结果
            if fact_checking_results:
                context_parts.append(fact_checking_results)
            
            # 优先级4: 近期对话上下文
            if recent_context:
                context_prompt = self.format_memory_for_prompt(recent_context, "context")
                if context_prompt:
                    context_parts.append(context_prompt)
            
            # 优先级5: 相关历史记忆
            if relevant_memories:
                memory_prompt = self.format_memory_for_prompt(relevant_memories, "memory")
                if memory_prompt:
                    context_parts.append(memory_prompt)

            # 4. 构建优化的系统prompt
            system_prompt = persona_content
            system_prompt += (
                "\n\n【对话指令】\n"
                "--- [绝对首要的规则] ---\n"
                "**回复长度**: 日常对话回复1-2句简短话语（5-15字）。当用户明确要求详细解释、代码、长回答时，可以完整回复3-6句或更多内容。对于编程问题、技术解答、代码示例等需要完整回答的内容，请提供完整详细的回复。\n"
                "--- [其他重要规则] ---\n"
                "- **信息优先级**: 如果提供了多种信息来源，请优先使用外部知识和知识库信息。\n"
                "- **语气**: 像真人一样聊天，语气自然。\n"
                "- **标点**: 可以使用各种标点符号（如 `，` `？` `…`）来表达情感，但请不要在单句末尾使用句号 `。`或者感叹号`！`。\n"
                "- **严禁事项**: 绝对禁止在回复中使用 `||` 或 `｜｜` 这类分隔符。\n"
                "- **事实**: 必须严格根据\"事实核查\"的结果进行回复，严禁伪造关于人物的记忆。\n"
            )
            
            # 5. 添加上下文信息（限制总长度）
            if context_parts:
                combined_context = "\n\n".join(context_parts)
                # 限制上下文长度以避免token过多
                if len(combined_context) > 3000:
                    combined_context = combined_context[:3000] + "..."
                system_prompt += "\n" + combined_context
            
            # 6. 调用API生成最终回复
            bot_response = await self._make_api_call(system_prompt, current_message, 0.7, 500)

            # 清理生成的回复
            if bot_response:
                cleaned_response = clean_response_text(bot_response)
                # 确保清理后还有内容
                if not cleaned_response or len(cleaned_response.strip()) == 0:
                    logger.warning(f"记忆系统回复清理后为空，使用简单回复")
                    cleaned_response = "嗯"
                bot_response = cleaned_response

            # 7. 异步存储本次对话，不阻塞回复发送
            if bot_response:
                asyncio.create_task(self.store_conversation(
                    user_id, session_id, current_message, bot_response,
                    persona_name=getattr(self, 'current_persona_name', None),
                    user_nickname=user_nickname
                ))
            
            return bot_response
            
        except Exception as e:
            logger.error(f"生成带上下文的回复失败: {e}")
            return "我好像有点短路了，稍等一下"

    async def query_knowledge_graph(self, message: str) -> str:
        """
        根据用户消息查询知识图谱，并将结果格式化用于prompt - 增强版本。
        """
        try:
            # 1. 缓存检查
            cache_key = hashlib.md5(message.encode('utf-8')).hexdigest()
            cached_result = self.knowledge_cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"从缓存获取知识图谱结果: {message[:50]}...")
                return cached_result
            
            # 2. 从消息中提取关键实体
            prompt = f'从以下问句中，提取出最核心的1到3个查询实体（例如人名、地名、概念），以JSON数组的格式返回。例如，对于"洛阳理工学院的办学理念是什么？"，应返回["洛阳理工学院", "办学理念"]。\n\n问句："{message}"'
            content = await self._make_api_call("你是一个实体提取助手，只返回JSON。", prompt, 0.0, 150)
            entities = json.loads(content) if content else []
            if not entities:
                return ""

            # 3. 在数据库中查询这些实体及其关系（优化版）
            def _enhanced_db_read():
                with self.driver.session() as session:
                    all_triples = set()
                    
                    for entity in entities[:3]:  # 最多查询前3个实体
                        # 查询多层关系（2层）并限制结果数量
                        cypher_query = """
                        MATCH (n) WHERE n.name CONTAINS $entity OR n.id CONTAINS $entity
                        OPTIONAL MATCH p=(n)-[r1]-(m)-[r2]-(o)
                        WHERE length(p) <= 2
                        RETURN p
                        UNION
                        MATCH (n) WHERE n.name CONTAINS $entity OR n.id CONTAINS $entity  
                        OPTIONAL MATCH p=(n)-[r]-(m)
                        RETURN p
                        LIMIT 15
                        """
                        records = session.run(cypher_query, entity=entity)
                        
                        for record in records:
                            path = record["p"]
                            if path is None: continue
                            
                            # 处理路径中的每个关系
                            nodes = list(path.nodes)
                            relationships = list(path.relationships)
                            
                            for i, rel in enumerate(relationships):
                                start_node = nodes[i]
                                end_node = nodes[i + 1]
                                
                                start_name = start_node.get('name', start_node.get('id', '未知节点'))
                                end_name = end_node.get('name', end_node.get('id', '未知节点'))
                                rel_type = rel.type
                                
                                all_triples.add(f"({start_name})-[{rel_type}]->({end_name})")
                    
                    return list(all_triples)[:10]  # 限制返回结果数量

            knowledge_triples = await asyncio.to_thread(_enhanced_db_read)
            
            # 4. 格式化结果
            if not knowledge_triples:
                result = ""
            else:
                result = "== 知识库参考 ==\n" + "\n".join(knowledge_triples)
                logger.info(f"从知识库检索到信息: \n{result}")
            
            # 5. 缓存结果
            self.knowledge_cache.set(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"知识库查询失败: {e}")
            return ""

    async def _fact_check_and_augment_prompt(self, message: str, group_members: List[Dict[str, str]], current_user_id: str) -> str:
        """
        事实核查：提取消息中的人名，检查他们是否存在于群聊或记忆中,
        并生成用于增强prompt的文本。
        """
        try:
            prompt = f'从以下句子中提取所有人名或昵称，以JSON数组格式返回。如果句子中没有人名，则返回空数组[]。\n\n句子："{message}"'
            content = await self._make_api_call("你是一个实体提取助手，只返回JSON。", prompt, 0.0, 100)
            mentioned_names = json.loads(content) if content else []
            if not mentioned_names: return ""

            facts = []
            for name in mentioned_names:
                fact_line = f'- 关于"{name}":'
                found_in_group = False
                if group_members:
                    for member in group_members:
                        if name in member.get("nickname", ""):
                            found_in_group = True
                            break
                fact_line += " 是群成员" if found_in_group else " 不是群成员"
                
                memories = await self.retrieve_memories_by_entity(current_user_id, name, entity_type="persons")
                fact_line += "；记忆中有相关记录。" if memories else "；记忆中无相关记录。"
                facts.append(fact_line)
            
            return "== 事实核查 ==\n" + "\n".join(facts) if facts else ""
        except Exception as e:
            logger.error(f"事实核查失败: {e}")
            return ""

    async def calculate_interjection_relevance(self, session_id: str, current_message: str, persona_content: str) -> float:
        """
        使用LLM计算新消息与对话上下文和人设的相关性，以决定是否插话。
        """
        try:
            # 获取最近的对话作为上下文
            recent_context = await self.get_conversation_context(session_id, 8)
            context_prompt = self.format_memory_for_prompt(recent_context, "context")

            prompt = f"""
你的任务是评估一条新消息是否值得一个AI助手在群聊中插话。你需要判断这条消息与AI的身份以及最近的聊天内容有多相关。

[AI人设]
{persona_content}

[最近的聊天记录]
{context_prompt if context_prompt else "（暂无）"}

[新消息]
"{current_message}"

请根据AI的人设和聊天记录，评估AI插话的合适度，并给出一个0.0到1.0的相关性分数。
- 0.0: 完全无关，AI绝对不应该插话。
- 0.5: 有点关系，可以考虑插话。
- 1.0: 关系非常紧密，AI应该立即插话回应。

请只返回一个JSON对象，格式如下：
{{"relevance_score": 0.85}}
"""
            system_prompt = "你是一个精确的相关性评估分析师，你的回答必须是一个JSON对象。"
            
            content = await self._make_api_call(system_prompt, prompt, temperature=0.0, max_tokens=100)
            
            if content:
                result = json.loads(content)
                relevance = float(result.get("relevance_score", 0.0))
                logger.info(f"计算插话相关性: 消息='{current_message}', 上下文相关度={relevance:.2f}")
                return relevance
            return 0.0
        except Exception as e:
            logger.error(f"计算插话相关性失败: {e}")
            return 0.0

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
        
        # 关闭线程池
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            logger.info("记忆系统线程池已关闭")

    # 其他函数如 get_memory_statistics, retrieve_memories_by_entity 等可以保持原样或按需异步化
    async def retrieve_memories_by_entity(self, user_id: str, entity_name: str, 
                                        entity_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """根据实体异步检索记忆"""
        def _db_read():
            with self.driver.session() as session:
                query = "MATCH (:USER {id: $user_id})-[:PARTICIPATED_IN]->(c)-[:MENTIONED]->(e:ENTITY {name: $entity_name})"
                if entity_type: query += " WHERE e.type = $entity_type"
                query += " RETURN c ORDER BY c.timestamp DESC LIMIT $limit"
                params = {'user_id': user_id, 'entity_name': entity_name, 'limit': limit}
                if entity_type: params['entity_type'] = entity_type
                
                result = session.run(query, params)
                return [dict(record['c']) for record in result]
        try:
            return await asyncio.to_thread(_db_read)
        except Exception as e:
            logger.error(f"根据实体检索记忆失败: {e}")
            return []

    async def store_knowledge_graph(self, graph_data: Dict[str, List[Dict[str, Any]]]) -> bool:
        """
        将从知识中提取的图数据异步存入Neo4j。
        """
        def _db_write():
            with self.driver.session() as session:
                # 存储节点
                for node in graph_data.get('nodes', []):
                    node_id = node.get('id')
                    node_type = node.get('type', 'Knowledge')
                    properties = node.get('properties', {})
                    if not node_id:
                        continue
                    
                    properties['id'] = node_id
                    properties['name'] = node_id # 使用name作为通用显示
                    
                    session.run(f"""
                        MERGE (n:{node_type} {{id: $id}})
                        SET n += $props, n.last_updated = datetime()
                    """, id=node_id, props=properties)
                
                # 存储关系
                for relation in graph_data.get('relations', []):
                    source_id = relation.get('source')
                    target_id = relation.get('target')
                    relation_type = relation.get('type')
                    properties = relation.get('properties', {})

                    if not all([source_id, target_id, relation_type]):
                        continue
                    
                    # 清理并验证关系类型，防止Cypher注入
                    safe_relation_type = re.sub(r'[^a-zA-Z0-9_]', '', relation_type.upper().replace(' ', '_'))
                    if not safe_relation_type:
                        logger.warning(f"跳过无效的关系类型: {relation_type}")
                        continue
                    
                    query = f"""
                        MATCH (source {{id: $source_id}}), (target {{id: $target_id}})
                        MERGE (source)-[r:`{safe_relation_type}`]->(target)
                        SET r += $props, r.last_updated = datetime()
                    """
                    session.run(query, source_id=source_id, target_id=target_id, props=properties)

        try:
            await asyncio.to_thread(_db_write)
            logger.info(f"成功存储知识图谱，包含 {len(graph_data.get('nodes', []))} 个节点和 {len(graph_data.get('relations', []))} 个关系")
            return True
        except Exception as e:
            logger.error(f"存储知识图谱失败: {e}")
            return False

    async def search_external_knowledge(self, query: str, source: str = "wikipedia") -> str:
        """
        搜索外部知识源以补充内部知识库
        """
        try:
            # 缓存检查
            cache_key = f"external_{source}_{hashlib.md5(query.encode('utf-8')).hexdigest()}"
            cached_result = self.knowledge_cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"从缓存获取外部知识: {query[:30]}...")
                return cached_result

            result = ""
            
            if source == "wikipedia":
                result = await self._search_wikipedia(query)
            elif source == "web":
                result = await self._search_web(query)
            
            # 缓存结果
            if result:
                self.knowledge_cache.set(cache_key, result)
                
            return result
            
        except Exception as e:
            logger.error(f"外部知识搜索失败: {e}")
            return ""

    async def _search_wikipedia(self, query: str) -> str:
        """
        搜索维基百科
        """
        try:
            async with httpx.AsyncClient() as client:
                # 先搜索页面
                search_url = "https://zh.wikipedia.org/w/api.php"
                search_params = {
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 3
                }
                
                search_response = await client.get(search_url, params=search_params, timeout=10.0)
                search_data = search_response.json()
                
                if not search_data.get("query", {}).get("search"):
                    return ""
                
                # 获取第一个结果的详细内容
                page_title = search_data["query"]["search"][0]["title"]
                content_params = {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "titles": page_title,
                    "exintro": True,
                    "explaintext": True,
                    "exsectionformat": "plain"
                }
                
                content_response = await client.get(search_url, params=content_params, timeout=10.0)
                content_data = content_response.json()
                
                pages = content_data.get("query", {}).get("pages", {})
                if pages:
                    page = next(iter(pages.values()))
                    extract = page.get("extract", "")
                    if extract:
                        # 限制长度并格式化
                        extract = extract[:500] + "..." if len(extract) > 500 else extract
                        return f"== 维基百科参考 ==\n标题: {page_title}\n内容: {extract}"
                
                return ""
                
        except Exception as e:
            logger.error(f"维基百科搜索失败: {e}")
            return ""

    async def _search_web(self, query: str) -> str:
        """
        使用搜索引擎（示例实现，需要API密钥）
        """
        # 这里可以集成Google、Bing等搜索API
        # 目前返回空字符串，用户可根据需要集成具体的搜索API
        logger.info(f"网络搜索功能待开发: {query}")
        return ""

    async def _should_search_external(self, message: str) -> bool:
        """
        判断是否需要搜索外部知识源
        """
        try:
            # 使用关键词和AI判断
            knowledge_keywords = ["什么是", "介绍", "历史", "定义", "原理", "如何", "为什么", "百科", "资料"]
            if any(keyword in message for keyword in knowledge_keywords):
                return True
            
            # 使用AI进行更精确的判断
            prompt = f'判断以下问题是否需要查找外部专业知识来回答（如百科知识、专业概念、历史事实等）。只返回true或false。\n\n问题："{message}"'
            content = await self._make_api_call("你是一个知识需求判断助手，只返回true或false。", prompt, 0.0, 50)
            return content.strip().lower() == "true"
            
        except Exception:
            return False

    async def _extract_search_keywords(self, message: str) -> str:
        """
        从消息中提取用于搜索的关键词
        """
        try:
            prompt = f'从以下问题中提取最适合用于搜索的1-2个关键词，返回格式为用空格分隔的关键词。\n\n问题："{message}"'
            content = await self._make_api_call("你是一个关键词提取助手。", prompt, 0.0, 100)
            return content.strip() if content else ""
        except Exception:
            return ""

    def get_memory_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户记忆统计信息"""
        with self.driver.session() as session:
            try:
                result = session.run("""
                    MATCH (u:USER {id: $user_id})
                    OPTIONAL MATCH (u)-[:PARTICIPATED_IN]->(c:CONVERSATION)
                    OPTIONAL MATCH (c)-[:USES_PERSONA]->(p:PERSONA)
                    OPTIONAL MATCH (c)-[:MENTIONED]->(e:ENTITY)
                    RETURN 
                        count(DISTINCT c) as total_conversations,
                        count(DISTINCT p) as personas_used,
                        count(DISTINCT e) as entities_mentioned,
                        min(c.timestamp) as first_conversation,
                        max(c.timestamp) as last_conversation
                """, user_id=user_id)
                
                record = result.single()
                if record:
                    return {
                        'total_conversations': record['total_conversations'],
                        'personas_used': record['personas_used'],
                        'entities_mentioned': record['entities_mentioned'],
                        'first_conversation': record['first_conversation'],
                        'last_conversation': record['last_conversation'],
                        'topics_discussed': 0  # 可以进一步实现
                    }
                return {}
            except Exception as e:
                logger.error(f"获取记忆统计失败: {e}")
                return {}

    def get_memory_health_report(self) -> Dict[str, Any]:
        """获取记忆系统健康报告"""
        with self.driver.session() as session:
            try:
                # 获取节点统计
                node_result = session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] as node_type, count(n) as count
                """)
                
                node_counts = {}
                for record in node_result:
                    node_counts[record['node_type']] = record['count']
                
                # 生成建议
                recommendations = []
                total_conversations = node_counts.get('CONVERSATION', 0)
                
                if total_conversations > 10000:
                    recommendations.append("建议清理旧对话记录")
                if total_conversations < 100:
                    recommendations.append("对话数据较少，建议增加互动")
                
                return {
                    'node_counts': node_counts,
                    'recommendations': recommendations,
                    'status': 'healthy'
                }
                
            except Exception as e:
                logger.error(f"获取健康报告失败: {e}")
                return {'error': str(e)}

    def cleanup_old_memories(self, days: int = 90) -> int:
        """清理旧记忆"""
        with self.driver.session() as session:
            try:
                result = session.run("""
                    MATCH (c:CONVERSATION)
                    WHERE c.timestamp < datetime() - duration({days: $days})
                    DETACH DELETE c
                    RETURN count(c) as deleted_count
                """, days=days)
                
                record = result.single()
                deleted_count = record['deleted_count'] if record else 0
                logger.info(f"已清理 {deleted_count} 条旧记忆")
                return deleted_count
                
            except Exception as e:
                logger.error(f"清理记忆失败: {e}")
                return 0


def clean_response_text(text: str) -> str:
    """清理回复文本，去除括号、描述性内容等不需要的元素"""
    if not text:
        return text
    
    # 去除各种括号及其内容
    text = re.sub(r'\([^)]*\)', '', text)  # 去除 (内容)
    text = re.sub(r'\[[^\]]*\]', '', text)  # 去除 [内容]
    text = re.sub(r'\{[^}]*\}', '', text)  # 去除 {内容}
    text = re.sub(r'（[^）]*）', '', text)  # 去除 （内容）
    text = re.sub(r'【[^】]*】', '', text)  # 去除 【内容】
    
    # 去除常见的描述性标记
    text = re.sub(r'\*[^*]*\*', '', text)  # 去除 *动作*
    text = re.sub(r'<[^>]*>', '', text)    # 去除 <标记>
    text = re.sub(r'「[^」]*」', '', text)  # 去除 「内容」
    text = re.sub(r'『[^』]*』', '', text)  # 去除 『内容』
    
    # 去除可能的角色标识或描述性前缀
    text = re.sub(r'^[^：:]*[：:]', '', text)  # 去除开头的 "角色名:" 格式
    text = re.sub(r'^\w+\s*:', '', text)       # 去除开头的 "Name:" 格式
    
    # 去除多余的空白字符和标点
    text = re.sub(r'\s+', ' ', text)  # 多个空格变成一个
    text = text.strip()  # 去除首尾空白
    
    # 去除开头和结尾的多余标点
    text = re.sub(r'^[。，！？\.,!?\s]+', '', text)
    text = re.sub(r'[。，\.,\s]+$', '', text)
    
    return text

