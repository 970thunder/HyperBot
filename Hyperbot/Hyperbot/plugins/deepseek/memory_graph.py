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

# 配置日志
logger = logging.getLogger(__name__)

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
    """基于Neo4j的图数据库记忆系统"""

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
        """在线程中异步获取文本的向量嵌入，避免阻塞"""
        try:
            embedding = await asyncio.to_thread(self.embedding_model.encode, text)
            return embedding.tolist()
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
        """通用的API调用函数"""
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
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers, timeout=45.0)
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
            
            await asyncio.to_thread(_db_write)
            logger.info(f"存储对话记录: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"存储对话失败: {e}")
            return False

    async def retrieve_relevant_memories(self, user_id: str, current_message: str, 
                                        limit: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """异步检索相关记忆"""
        try:
            current_embedding = await self.get_embedding(current_message)
            if not current_embedding: return []

            def _blocking_db_and_cpu_work():
                with self.driver.session() as session:
                    result = session.run("""
                        MATCH (u:USER {id: $user_id})-[:PARTICIPATED_IN]->(c:CONVERSATION)
                        WHERE c.timestamp > datetime() - duration({days: 30})
                        RETURN c ORDER BY c.timestamp DESC LIMIT 50
                    """, user_id=user_id)
                    
                    memories = []
                    for record in result:
                        conv = dict(record['c'])
                        sim = self.cosine_similarity(current_embedding, conv.get('user_embedding', []))
                        if sim >= similarity_threshold:
                            memories.append({'conversation': conv, 'relevance_score': sim})
                    
                    memories.sort(key=lambda x: x['relevance_score'], reverse=True)
                    return memories[:limit]
            
            return await asyncio.to_thread(_blocking_db_and_cpu_work)
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

    async def _fact_check_and_augment_prompt(self, message: str, group_members: List[Dict[str, str]], current_user_id: str) -> str:
        """事实核查，并生成用于增强prompt的文本"""
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

    async def generate_contextual_response(self, user_id: str, session_id: str,
                                         current_message: str, persona_content: str,
                                         user_nickname: str = None, use_memory: bool = True,
                                         group_members: List[Dict[str, str]] = None) -> str:
        """基于记忆、事实核查和工具生成带上下文的回复"""
        try:
            # 1. 并行获取所有需要的数据
            memory_task = self.retrieve_relevant_memories(user_id, current_message) if use_memory else asyncio.sleep(0, result=[])
            context_task = self.get_conversation_context(session_id, 5) if use_memory else asyncio.sleep(0, result=[])
            fact_check_task = self._fact_check_and_augment_prompt(current_message, group_members, user_id) if group_members else asyncio.sleep(0, result="")

            relevant_memories, recent_context, fact_checking_results = await asyncio.gather(
                memory_task, context_task, fact_check_task
            )

            # 2. 格式化记忆和上下文
            memory_prompt = self.format_memory_for_prompt(relevant_memories, "memory")
            context_prompt = self.format_memory_for_prompt(recent_context, "context")

            # 3. 构建最终的prompt
            system_prompt = persona_content
            system_prompt += (
                "\n\n【对话指令】\n"
                "--- [绝对首要的规则] ---\n"
                "**回复长度**: 你的核心任务是进行简短、自然的对话。在任何情况下，单次回复都不能超过3句话。默认只回复1-2句简短的话。只有当用户明确要求长篇幅（如“详细解释”、“写一段文字”）时，才能打破此限制。\n"
                "--- [其他重要规则] ---\n"
                "- **分段**: 如果需要进行长回复（已获得用户许可），请像真人一样聊天，自然地分段。在需要另起一段话说时，使用 `||` 作为分隔符。\n"
                "- **标点**: 严禁在回复中使用句号 `。`。\n"
                "- **事实**: 必须严格根据“事实核查”的结果进行回复，严禁伪造关于人物的记忆。\n"
            )
            
            # 组合所有信息
            full_context = "\n\n".join(filter(None, [fact_checking_results, context_prompt, memory_prompt]))
            if full_context:
                system_prompt += "\n" + full_context
            
            # 4. 调用API生成最终回复
            bot_response = await self._make_api_call(system_prompt, current_message, 0.8, 1000)

            # 5. 异步存储本次对话，不阻塞回复发送
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

