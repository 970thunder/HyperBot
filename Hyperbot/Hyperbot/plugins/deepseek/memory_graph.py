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
        
        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            deepseek_api_key: Deepseek API密钥
            embedding_model: 嵌入模型名称
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.deepseek_api_key = deepseek_api_key
        self.embedding_model = SentenceTransformer(embedding_model)
        
                # 然后进行连接测试
        try:
            self.test_connection()
        except Exception as e:
            logger.error(f"Neo4j连接失败: {str(e)}")
            raise

        # 初始化数据库约束和索引
        self._init_database()
        
        # 节点类型定义
        self.node_types = {
            'USER': '用户信息',
            'CONVERSATION': '对话记录',
            'TOPIC': '话题概念',
            'ENTITY': '实体信息',
            'EMOTION': '情感状态',
            'PREFERENCE': '偏好设置',
            'EVENT': '事件记录',
            'PERSONA': '人设信息',
            'CONTEXT': '上下文场景'
        }
        
        # 关系类型定义
        self.relation_types = {
            'PARTICIPATED_IN': '参与了',
            'MENTIONED': '提到了',
            'LIKES': '喜欢',
            'DISLIKES': '不喜欢',
            'KNOWS': '了解',
            'RELATED_TO': '相关联',
            'CAUSED_BY': '由...引起',
            'HAPPENED_AT': '发生在',
            'HAS_EMOTION': '有情感',
            'USES_PERSONA': '使用人设',
            'SIMILAR_TO': '相似于'
        }
    
    def _init_database(self):
        """初始化数据库约束和索引"""
        try:
        # 测试数据库连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                if result.single()["test"] != 1:
                    logger.error("Neo4j连接测试失败")
                    return False
                else:
                    logger.info("Neo4j连接测试成功")
                    
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
                "CREATE INDEX topic_frequency IF NOT EXISTS FOR (t:TOPIC) ON (t.frequency)",
                "CREATE INDEX entity_importance IF NOT EXISTS FOR (e:ENTITY) ON (e.importance)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.debug(f"索引创建失败或已存在: {e}")

            logger.info("数据库约束和索引初始化完成")
            return True
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}")
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量嵌入"""
        try:
            embedding = self.embedding_model.encode(text)
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
    
    async def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """分析文本情感"""
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            请分析以下文本的情感信息，包括情感类型和强度。
            
            文本：{text}
            
            请以JSON格式返回，格式如下：
            {{
                "sentiment_score": 0.5,
                "emotions": [
                    {{"type": "happy", "intensity": 0.8}},
                    {{"type": "excited", "intensity": 0.6}}
                ]
            }}
            
            sentiment_score范围：-1(非常负面)到1(非常正面)
            intensity范围：0到1
            只返回JSON，不要其他内容。
            """
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一个专业的情感分析助手，只返回JSON格式的结果。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 300
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers, timeout=30.0)
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 尝试解析JSON
                try:
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    
                    emotion_data = json.loads(content)
                    return emotion_data
                except json.JSONDecodeError:
                    logger.warning(f"情感分析返回的不是有效JSON: {content}")
                    return {"sentiment_score": 0.0, "emotions": []}
        
        except Exception as e:
            logger.error(f"情感分析失败: {e}")
            return {"sentiment_score": 0.0, "emotions": []}
    
    async def extract_preferences(self, text: str) -> List[Dict[str, Any]]:
        """提取用户偏好信息"""
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            请从以下文本中提取用户的偏好信息，包括喜欢、不喜欢、兴趣等。
            
            文本：{text}
            
            请以JSON格式返回，格式如下：
            {{
                "preferences": [
                    {{"type": "likes", "content": "音乐", "strength": 0.8}},
                    {{"type": "dislikes", "content": "噪音", "strength": 0.9}},
                    {{"type": "interests", "content": "编程", "strength": 0.7}}
                ]
            }}
            
            strength范围：0到1，表示偏好强度
            只返回JSON，不要其他内容。如果没有明显偏好，返回空数组。
            """
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一个专业的偏好提取助手，只返回JSON格式的结果。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 400
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers, timeout=30.0)
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 尝试解析JSON
                try:
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    
                    preference_data = json.loads(content)
                    return preference_data.get("preferences", [])
                except json.JSONDecodeError:
                    logger.warning(f"偏好提取返回的不是有效JSON: {content}")
                    return []
        
        except Exception as e:
            logger.error(f"偏好提取失败: {e}")
            return []

    async def extract_entities_with_ai(self, text: str) -> Dict[str, List[str]]:
        """使用AI提取实体信息"""
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            请从以下文本中提取关键实体信息，包括人名、地点、物品、时间、情感等。
            
            文本：{text}
            
            请以JSON格式返回，格式如下：
            {{
                "persons": ["人名1", "人名2"],
                "locations": ["地点1", "地点2"],
                "objects": ["物品1", "物品2"],
                "times": ["时间1", "时间2"],
                "emotions": ["情感1", "情感2"],
                "topics": ["话题1", "话题2"]
            }}
            
            只返回JSON，不要其他内容。
            """
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一个专业的实体提取助手，只返回JSON格式的结果。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers, timeout=30.0)
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 尝试解析JSON
                try:
                    # 清理可能的markdown格式
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    
                    entities = json.loads(content)
                    return entities
                except json.JSONDecodeError:
                    logger.warning(f"AI返回的不是有效JSON: {content}")
                    return {}
        
        except Exception as e:
            logger.error(f"AI实体提取失败: {e}")
            return {}
    
    def create_or_get_user(self, user_id: str, user_info: Dict[str, Any] = None) -> bool:
        """创建或获取用户节点"""
        with self.driver.session() as session:
            try:
                # 检查用户是否存在
                result = session.run(
                    "MATCH (u:USER {id: $user_id}) RETURN u",
                    user_id=user_id
                )
                
                if result.single():
                    return True
                
                # 创建新用户
                user_data = {
                    'id': user_id,
                    'created_at': datetime.now().isoformat(),
                    'total_conversations': 0,
                    'last_active': datetime.now().isoformat()
                }
                
                if user_info:
                    user_data.update(user_info)
                
                session.run(
                    "CREATE (u:USER $props)",
                    props=user_data
                )
                
                logger.info(f"创建新用户: {user_id}")
                return True
                
            except Exception as e:
                logger.error(f"创建/获取用户失败: {e}")
                return False
    
    async def store_conversation(self, user_id: str, session_id: str, user_message: str, 
                            bot_response: str, persona_name: str = None, 
                            context: Dict[str, Any] = None) -> bool:
        
        logger.info(f"开始存储对话 - 用户: {user_id}, 会话: {session_id}")

        """存储对话记录到图数据库"""
        try:
            # 确保用户ID是QQ号格式
            if not user_id.isdigit():
                logger.warning(f"非标准QQ号格式: {user_id}")
                # 尝试提取数字部分作为QQ号
                qq_id = ''.join(filter(str.isdigit, user_id))
                if qq_id:
                    user_id = qq_id
                else:
                    logger.error(f"无法提取有效QQ号: {user_id}")
                    return False
            # 确保用户存在
            self.create_or_get_user(user_id)
            
            # 生成对话ID - 使用QQ号和会话ID
            conversation_id = f"qq_{user_id}_{session_id}_{int(time.time() * 1000)}"
            
            # 获取文本嵌入
            user_embedding = self.get_embedding(user_message)
            bot_embedding = self.get_embedding(bot_response)
            
            # 提取实体信息
            entities = await self.extract_entities_with_ai(user_message)
            
            # 分析情感
            emotion_info = await self.analyze_emotion(user_message)
            
            with self.driver.session() as session:
                # 创建对话节点
                conversation_data = {
                    'id': conversation_id,
                    'user_id': user_id,
                    'session_id': session_id,
                    'user_message': user_message,
                    'bot_response': bot_response,
                    'persona_name': persona_name,
                    'timestamp': datetime.now().isoformat(),
                    'user_embedding': user_embedding,
                    'bot_embedding': bot_embedding,
                    'message_length': len(user_message),
                    'response_length': len(bot_response),
                    'sentiment_score': emotion_info.get('sentiment_score', 0.0)
                }
                
                if context:
                    conversation_data.update(context)
                
                session.run(
                    "CREATE (c:CONVERSATION $props)",
                    props=conversation_data
                )
                
                # 创建用户-对话关系
                session.run("""
                    MATCH (u:USER {id: $user_id}), (c:CONVERSATION {id: $conversation_id})
                    CREATE (u)-[:PARTICIPATED_IN {timestamp: datetime(), strength: 1.0}]->(c)
                """, user_id=user_id, conversation_id=conversation_id)
                
                # 更新用户统计
                session.run("""
                    MATCH (u:USER {id: $user_id})
                    SET u.total_conversations = u.total_conversations + 1,
                        u.last_active = datetime()
                """, user_id=user_id)
                
                # 处理人设关系
                if persona_name:
                    await self._create_persona_relationship(session, conversation_id, persona_name)
                
                # 处理实体关系
                await self._create_entity_relationships(session, conversation_id, entities)
                
                # 处理情感关系
                if emotion_info.get('emotions'):
                    await self._create_emotion_relationships(session, conversation_id, emotion_info['emotions'])
                
                # 处理偏好信息
                preferences = await self.extract_preferences(user_message)
                if preferences:
                    await self._create_preference_relationships(session, user_id, conversation_id, preferences)
                
                logger.info(f"存储对话记录: {conversation_id}")
                return True
                
        except Exception as e:
            logger.error(f"存储对话失败: {e}")
            return False
    
    async def _create_emotion_relationships(self, session, conversation_id: str, emotions: List[Dict[str, Any]]):
        """创建情感关系"""
        try:
            for emotion in emotions:
                emotion_type = emotion.get('type', '')
                intensity = emotion.get('intensity', 0.0)
                
                if not emotion_type:
                    continue
                
                # 创建或获取情感节点
                session.run("""
                    MERGE (e:EMOTION {type: $emotion_type})
                    ON CREATE SET e.created_at = datetime(), e.total_intensity = 0.0, e.occurrence_count = 0
                    SET e.total_intensity = e.total_intensity + $intensity,
                        e.occurrence_count = e.occurrence_count + 1,
                        e.average_intensity = e.total_intensity / e.occurrence_count,
                        e.last_occurred = datetime()
                """, emotion_type=emotion_type, intensity=intensity)
                
                # 创建对话-情感关系
                session.run("""
                    MATCH (c:CONVERSATION {id: $conversation_id}), (e:EMOTION {type: $emotion_type})
                    CREATE (c)-[:HAS_EMOTION {
                        timestamp: datetime(), 
                        intensity: $intensity,
                        strength: $intensity
                    }]->(e)
                """, conversation_id=conversation_id, emotion_type=emotion_type, intensity=intensity)
                
        except Exception as e:
            logger.error(f"创建情感关系失败: {e}")
    
    async def _create_preference_relationships(self, session, user_id: str, conversation_id: str, preferences: List[Dict[str, Any]]):
        """创建偏好关系"""
        try:
            for preference in preferences:
                pref_type = preference.get('type', '')
                content = preference.get('content', '')
                strength = preference.get('strength', 0.0)
                
                if not pref_type or not content:
                    continue
                
                # 创建或更新偏好节点
                session.run("""
                    MERGE (p:PREFERENCE {type: $pref_type, content: $content})
                    ON CREATE SET p.created_at = datetime(), p.total_strength = 0.0, p.mention_count = 0
                    SET p.total_strength = p.total_strength + $strength,
                        p.mention_count = p.mention_count + 1,
                        p.average_strength = p.total_strength / p.mention_count,
                        p.last_updated = datetime()
                """, pref_type=pref_type, content=content, strength=strength)
                
                # 创建用户-偏好关系
                session.run("""
                    MATCH (u:USER {id: $user_id}), (p:PREFERENCE {type: $pref_type, content: $content})
                    MERGE (u)-[r:HAS_PREFERENCE]->(p)
                    ON CREATE SET r.created_at = datetime(), r.strength = $strength
                    ON MATCH SET r.strength = (r.strength + $strength) / 2, r.last_updated = datetime()
                """, user_id=user_id, pref_type=pref_type, content=content, strength=strength)
                
                # 创建对话-偏好关系
                session.run("""
                    MATCH (c:CONVERSATION {id: $conversation_id}), (p:PREFERENCE {type: $pref_type, content: $content})
                    CREATE (c)-[:EXPRESSED_PREFERENCE {
                        timestamp: datetime(),
                        strength: $strength
                    }]->(p)
                """, conversation_id=conversation_id, pref_type=pref_type, content=content, strength=strength)
                
        except Exception as e:
            logger.error(f"创建偏好关系失败: {e}")

    async def _create_persona_relationship(self, session, conversation_id: str, persona_name: str):
        """创建人设关系"""
        try:
            # 创建或获取人设节点
            session.run("""
                MERGE (p:PERSONA {name: $persona_name})
                ON CREATE SET p.created_at = datetime(), p.usage_count = 0
                SET p.usage_count = p.usage_count + 1, p.last_used = datetime()
            """, persona_name=persona_name)
            
            # 创建对话-人设关系
            session.run("""
                MATCH (c:CONVERSATION {id: $conversation_id}), (p:PERSONA {name: $persona_name})
                CREATE (c)-[:USES_PERSONA {timestamp: datetime()}]->(p)
            """, conversation_id=conversation_id, persona_name=persona_name)
            
        except Exception as e:
            logger.error(f"创建人设关系失败: {e}")
    
    async def _create_entity_relationships(self, session, conversation_id: str, entities: Dict[str, List[str]]):
        """创建实体关系"""
        try:
            for entity_type, entity_list in entities.items():
                for entity_name in entity_list:
                    if not entity_name.strip():
                        continue
                    
                    # 计算实体重要性（基于类型和频率）
                    importance_multiplier = {
                        'persons': 1.2,
                        'locations': 1.0,
                        'objects': 0.8,
                        'times': 0.9,
                        'emotions': 1.1,
                        'topics': 1.3
                    }.get(entity_type, 1.0)
                    
                    # 创建或获取实体节点
                    session.run("""
                        MERGE (e:ENTITY {name: $entity_name, type: $entity_type})
                        ON CREATE SET e.created_at = datetime(), e.mention_count = 0, e.importance = $base_importance
                        SET e.mention_count = e.mention_count + 1, 
                            e.last_mentioned = datetime(),
                            e.importance = e.importance * 0.95 + $importance_boost
                    """, entity_name=entity_name, entity_type=entity_type, 
                         base_importance=importance_multiplier, 
                         importance_boost=importance_multiplier * 0.1)
                    
                    # 创建对话-实体关系
                    session.run("""
                        MATCH (c:CONVERSATION {id: $conversation_id}), (e:ENTITY {name: $entity_name})
                        CREATE (c)-[:MENTIONED {
                            timestamp: datetime(), 
                            context: $entity_type,
                            strength: 1.0
                        }]->(e)
                    """, conversation_id=conversation_id, entity_name=entity_name, entity_type=entity_type)
                    
                    # 创建话题节点（特殊处理）
                    if entity_type == 'topics':
                        session.run("""
                            MERGE (t:TOPIC {name: $entity_name})
                            ON CREATE SET t.created_at = datetime(), t.frequency = 0, t.importance = 1.0
                            SET t.frequency = t.frequency + 1, 
                                t.last_discussed = datetime(),
                                t.importance = t.importance * 0.98 + 0.05
                        """, entity_name=entity_name)
                        
                        # 创建对话-话题关系
                        session.run("""
                            MATCH (c:CONVERSATION {id: $conversation_id}), (t:TOPIC {name: $entity_name})
                            CREATE (c)-[:RELATED_TO {
                                timestamp: datetime(),
                                strength: 1.0
                            }]->(t)
                        """, conversation_id=conversation_id, entity_name=entity_name)
                        
                        # 创建实体-话题关系
                        session.run("""
                            MATCH (e:ENTITY {name: $entity_name, type: 'topics'}), (t:TOPIC {name: $entity_name})
                            MERGE (e)-[:REPRESENTS]->(t)
                        """, entity_name=entity_name)
            
            # 创建实体间的关联关系
            await self._create_entity_associations(session, conversation_id, entities)
            
        except Exception as e:
            logger.error(f"创建实体关系失败: {e}")
    
    async def _create_entity_associations(self, session, conversation_id: str, entities: Dict[str, List[str]]):
        """创建实体间的关联关系"""
        try:
            # 获取所有实体
            all_entities = []
            for entity_type, entity_list in entities.items():
                for entity_name in entity_list:
                    if entity_name.strip():
                        all_entities.append((entity_name, entity_type))
            
            # 创建实体间的共现关系
            for i, (entity1, type1) in enumerate(all_entities):
                for entity2, type2 in all_entities[i+1:]:
                    # 计算关联强度（基于实体类型的相关性）
                    association_strength = self._calculate_association_strength(type1, type2)
                    
                    if association_strength > 0.3:  # 只创建强关联
                        session.run("""
                            MATCH (e1:ENTITY {name: $entity1}), (e2:ENTITY {name: $entity2})
                            MERGE (e1)-[r:ASSOCIATED_WITH]-(e2)
                            ON CREATE SET r.strength = $strength, r.created_at = datetime(), r.co_occurrence = 1
                            ON MATCH SET r.strength = (r.strength + $strength) / 2, 
                                        r.co_occurrence = r.co_occurrence + 1,
                                        r.last_updated = datetime()
                        """, entity1=entity1, entity2=entity2, strength=association_strength)
            
        except Exception as e:
            logger.error(f"创建实体关联失败: {e}")
    
    def _calculate_association_strength(self, type1: str, type2: str) -> float:
        """计算实体类型间的关联强度"""
        # 定义实体类型间的关联矩阵
        association_matrix = {
            ('persons', 'emotions'): 0.9,
            ('persons', 'topics'): 0.8,
            ('persons', 'locations'): 0.7,
            ('topics', 'emotions'): 0.8,
            ('topics', 'objects'): 0.6,
            ('locations', 'times'): 0.7,
            ('emotions', 'times'): 0.6,
            ('objects', 'topics'): 0.5,
        }
        
        # 查找关联强度（双向）
        key1 = (type1, type2)
        key2 = (type2, type1)
        
        return association_matrix.get(key1, association_matrix.get(key2, 0.4))
    
    async def store_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """存储用户档案信息"""
        try:
            with self.driver.session() as session:
                # 更新用户节点
                session.run("""
                    MATCH (u:USER {id: $user_id})
                    SET u += $profile_data, u.profile_updated = datetime()
                """, user_id=user_id, profile_data=profile_data)
                
                # 如果包含偏好信息，创建偏好节点
                if 'preferences' in profile_data:
                    for pref in profile_data['preferences']:
                        await self._create_preference_relationships(
                            session, user_id, None, [pref]
                        )
                
                logger.info(f"更新用户档案: {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"存储用户档案失败: {e}")
            return False
    
    async def store_event(self, user_id: str, event_data: Dict[str, Any]) -> bool:
        """存储事件记录"""
        try:
            event_id = f"event_{user_id}_{int(time.time() * 1000)}"
            
            with self.driver.session() as session:
                # 创建事件节点
                event_props = {
                    'id': event_id,
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat(),
                    'importance': event_data.get('importance', 1.0)
                }
                event_props.update(event_data)
                
                session.run(
                    "CREATE (e:EVENT $props)",
                    props=event_props
                )
                
                # 创建用户-事件关系
                session.run("""
                    MATCH (u:USER {id: $user_id}), (e:EVENT {id: $event_id})
                    CREATE (u)-[:EXPERIENCED {timestamp: datetime()}]->(e)
                """, user_id=user_id, event_id=event_id)
                
                logger.info(f"存储事件记录: {event_id}")
                return True
                
        except Exception as e:
            logger.error(f"存储事件失败: {e}")
            return False
    
    def retrieve_relevant_memories(self, user_id: str, current_message: str, 
                                 limit: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """检索相关记忆"""
        try:
            # 获取当前消息的嵌入
            current_embedding = self.get_embedding(current_message)
            
            with self.driver.session() as session:
                # 查询用户的历史对话
                result = session.run("""
                    MATCH (u:USER {id: $user_id})-[:PARTICIPATED_IN]->(c:CONVERSATION)
                    OPTIONAL MATCH (c)-[:RELATED_TO]->(t:TOPIC)
                    OPTIONAL MATCH (c)-[:MENTIONED]->(e:ENTITY)
                    OPTIONAL MATCH (c)-[:USES_PERSONA]->(p:PERSONA)
                    OPTIONAL MATCH (c)-[:HAS_EMOTION]->(em:EMOTION)
                    WHERE c.timestamp > datetime() - duration({days: 30})
                    RETURN c, 
                           collect(DISTINCT t.name) as topics,
                           collect(DISTINCT e.name) as entities,
                           collect(DISTINCT p.name) as personas,
                           collect(DISTINCT em.type) as emotions
                    ORDER BY c.timestamp DESC
                    LIMIT 50
                """, user_id=user_id)
                
                memories = []
                for record in result:
                    conversation = dict(record['c'])
                    
                    # 计算语义相似度
                    semantic_similarity = 0.0
                    if conversation.get('user_embedding'):
                        semantic_similarity = self.cosine_similarity(
                            current_embedding, 
                            conversation['user_embedding']
                        )
                    
                    # 计算综合相关性分数
                    relevance_score = self._calculate_relevance_score(
                        current_message, conversation, record, semantic_similarity
                    )
                    
                    if relevance_score >= similarity_threshold:
                        memories.append({
                            'conversation': conversation,
                            'topics': record['topics'],
                            'entities': record['entities'],
                            'personas': record['personas'],
                            'emotions': record['emotions'],
                            'semantic_similarity': semantic_similarity,
                            'relevance_score': relevance_score
                        })
                
                # 按相关性分数排序
                memories.sort(key=lambda x: x['relevance_score'], reverse=True)
                
                return memories[:limit]
                
        except Exception as e:
            logger.error(f"检索记忆失败: {e}")
            return []
    
    def _calculate_relevance_score(self, current_message: str, conversation: Dict, 
                                 record: Dict, semantic_similarity: float) -> float:
        """计算综合相关性分数"""
        try:
            # 基础语义相似度权重
            score = semantic_similarity * 0.4
            
            # 时间衰减因子
            timestamp_str = conversation.get('timestamp', '')
            if timestamp_str:
                try:
                    conv_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    time_diff = datetime.now() - conv_time.replace(tzinfo=None)
                    time_decay = max(0.1, 1.0 - (time_diff.days / 30.0))  # 30天内线性衰减
                    score += time_decay * 0.2
                except:
                    score += 0.1  # 默认时间权重
            
            # 实体匹配加分
            current_entities = set(re.findall(r'\b\w+\b', current_message.lower()))
            conversation_entities = set([e.lower() for e in record.get('entities', [])])
            entity_overlap = len(current_entities & conversation_entities)
            if entity_overlap > 0:
                score += min(entity_overlap * 0.1, 0.2)
            
            # 话题匹配加分
            conversation_topics = set([t.lower() for t in record.get('topics', [])])
            topic_overlap = len(current_entities & conversation_topics)
            if topic_overlap > 0:
                score += min(topic_overlap * 0.15, 0.3)
            
            # 情感一致性加分
            current_sentiment = self._quick_sentiment_analysis(current_message)
            conversation_sentiment = conversation.get('sentiment_score', 0.0)
            if abs(current_sentiment - conversation_sentiment) < 0.3:
                score += 0.1
            
            # 消息长度相似性
            current_length = len(current_message)
            conv_length = conversation.get('message_length', 0)
            if conv_length > 0:
                length_similarity = 1.0 - abs(current_length - conv_length) / max(current_length, conv_length)
                score += length_similarity * 0.05
            
            return min(score, 1.0)  # 限制最大分数为1.0
            
        except Exception as e:
            logger.error(f"计算相关性分数失败: {e}")
            return semantic_similarity * 0.5  # 降级到基础语义相似度
    
    def _quick_sentiment_analysis(self, text: str) -> float:
        """快速情感分析（简化版）"""
        positive_words = ['好', '喜欢', '开心', '高兴', '棒', '赞', '爱', '美好', '快乐', '满意']
        negative_words = ['不好', '讨厌', '难过', '生气', '糟糕', '差', '恨', '痛苦', '失望', '烦']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def retrieve_memories_by_entity(self, user_id: str, entity_name: str, 
                                  entity_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """根据实体检索记忆"""
        try:
            with self.driver.session() as session:
                query = """
                    MATCH (u:USER {id: $user_id})-[:PARTICIPATED_IN]->(c:CONVERSATION)-[:MENTIONED]->(e:ENTITY {name: $entity_name})
                """
                params = {'user_id': user_id, 'entity_name': entity_name}
                
                if entity_type:
                    query += " WHERE e.type = $entity_type"
                    params['entity_type'] = entity_type
                
                query += """
                    OPTIONAL MATCH (c)-[:RELATED_TO]->(t:TOPIC)
                    OPTIONAL MATCH (c)-[:USES_PERSONA]->(p:PERSONA)
                    RETURN c, e, 
                           collect(DISTINCT t.name) as topics,
                           collect(DISTINCT p.name) as personas
                    ORDER BY c.timestamp DESC
                    LIMIT $limit
                """
                params['limit'] = limit
                
                result = session.run(query, params)
                
                memories = []
                for record in result:
                    conversation = dict(record['c'])
                    entity = dict(record['e'])
                    
                    memories.append({
                        'conversation': conversation,
                        'entity': entity,
                        'topics': record['topics'],
                        'personas': record['personas']
                    })
                
                return memories
                
        except Exception as e:
            logger.error(f"根据实体检索记忆失败: {e}")
            return []
    
    def retrieve_memories_by_topic(self, user_id: str, topic_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """根据话题检索记忆"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:USER {id: $user_id})-[:PARTICIPATED_IN]->(c:CONVERSATION)-[:RELATED_TO]->(t:TOPIC {name: $topic_name})
                    OPTIONAL MATCH (c)-[:MENTIONED]->(e:ENTITY)
                    OPTIONAL MATCH (c)-[:USES_PERSONA]->(p:PERSONA)
                    OPTIONAL MATCH (c)-[:HAS_EMOTION]->(em:EMOTION)
                    RETURN c, t,
                           collect(DISTINCT e.name) as entities,
                           collect(DISTINCT p.name) as personas,
                           collect(DISTINCT em.type) as emotions
                    ORDER BY c.timestamp DESC
                    LIMIT $limit
                """, user_id=user_id, topic_name=topic_name, limit=limit)
                
                memories = []
                for record in result:
                    conversation = dict(record['c'])
                    topic = dict(record['t'])
                    
                    memories.append({
                        'conversation': conversation,
                        'topic': topic,
                        'entities': record['entities'],
                        'personas': record['personas'],
                        'emotions': record['emotions']
                    })
                
                return memories
                
        except Exception as e:
            logger.error(f"根据话题检索记忆失败: {e}")
            return []
    
    def retrieve_memories_by_emotion(self, user_id: str, emotion_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """根据情感检索记忆"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:USER {id: $user_id})-[:PARTICIPATED_IN]->(c:CONVERSATION)-[:HAS_EMOTION]->(em:EMOTION {type: $emotion_type})
                    OPTIONAL MATCH (c)-[:RELATED_TO]->(t:TOPIC)
                    OPTIONAL MATCH (c)-[:MENTIONED]->(e:ENTITY)
                    OPTIONAL MATCH (c)-[:USES_PERSONA]->(p:PERSONA)
                    RETURN c, em,
                           collect(DISTINCT t.name) as topics,
                           collect(DISTINCT e.name) as entities,
                           collect(DISTINCT p.name) as personas
                    ORDER BY c.timestamp DESC
                    LIMIT $limit
                """, user_id=user_id, emotion_type=emotion_type, limit=limit)
                
                memories = []
                for record in result:
                    conversation = dict(record['c'])
                    emotion = dict(record['em'])
                    
                    memories.append({
                        'conversation': conversation,
                        'emotion': emotion,
                        'topics': record['topics'],
                        'entities': record['entities'],
                        'personas': record['personas']
                    })
                
                return memories
                
        except Exception as e:
            logger.error(f"根据情感检索记忆失败: {e}")
            return []
    
    def retrieve_memories_by_timerange(self, user_id: str, start_time: datetime, 
                                     end_time: datetime, limit: int = 20) -> List[Dict[str, Any]]:
        """根据时间范围检索记忆"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:USER {id: $user_id})-[:PARTICIPATED_IN]->(c:CONVERSATION)
                    WHERE datetime($start_time) <= datetime(c.timestamp) <= datetime($end_time)
                    OPTIONAL MATCH (c)-[:RELATED_TO]->(t:TOPIC)
                    OPTIONAL MATCH (c)-[:MENTIONED]->(e:ENTITY)
                    OPTIONAL MATCH (c)-[:USES_PERSONA]->(p:PERSONA)
                    OPTIONAL MATCH (c)-[:HAS_EMOTION]->(em:EMOTION)
                    RETURN c,
                           collect(DISTINCT t.name) as topics,
                           collect(DISTINCT e.name) as entities,
                           collect(DISTINCT p.name) as personas,
                           collect(DISTINCT em.type) as emotions
                    ORDER BY c.timestamp DESC
                    LIMIT $limit
                """, user_id=user_id, 
                     start_time=start_time.isoformat(), 
                     end_time=end_time.isoformat(), 
                     limit=limit)
                
                memories = []
                for record in result:
                    conversation = dict(record['c'])
                    
                    memories.append({
                        'conversation': conversation,
                        'topics': record['topics'],
                        'entities': record['entities'],
                        'personas': record['personas'],
                        'emotions': record['emotions']
                    })
                
                return memories
                
        except Exception as e:
            logger.error(f"根据时间范围检索记忆失败: {e}")
            return []
    
    def search_memories_by_keywords(self, user_id: str, keywords: List[str], 
                                  search_type: str = 'any', limit: int = 15) -> List[Dict[str, Any]]:
        """根据关键词搜索记忆"""
        try:
            with self.driver.session() as session:
                # 构建搜索条件
                if search_type == 'all':
                    # 所有关键词都必须匹配
                    conditions = []
                    for i, keyword in enumerate(keywords):
                        conditions.append(f"(c.user_message CONTAINS $keyword{i} OR c.bot_response CONTAINS $keyword{i})")
                    where_clause = " AND ".join(conditions)
                else:
                    # 任意关键词匹配
                    conditions = []
                    for i, keyword in enumerate(keywords):
                        conditions.append(f"(c.user_message CONTAINS $keyword{i} OR c.bot_response CONTAINS $keyword{i})")
                    where_clause = " OR ".join(conditions)
                
                query = f"""
                    MATCH (u:USER {{id: $user_id}})-[:PARTICIPATED_IN]->(c:CONVERSATION)
                    WHERE {where_clause}
                    OPTIONAL MATCH (c)-[:RELATED_TO]->(t:TOPIC)
                    OPTIONAL MATCH (c)-[:MENTIONED]->(e:ENTITY)
                    OPTIONAL MATCH (c)-[:USES_PERSONA]->(p:PERSONA)
                    RETURN c,
                           collect(DISTINCT t.name) as topics,
                           collect(DISTINCT e.name) as entities,
                           collect(DISTINCT p.name) as personas
                    ORDER BY c.timestamp DESC
                    LIMIT $limit
                """
                
                params = {'user_id': user_id, 'limit': limit}
                for i, keyword in enumerate(keywords):
                    params[f'keyword{i}'] = keyword
                
                result = session.run(query, params)
                
                memories = []
                for record in result:
                    conversation = dict(record['c'])
                    
                    # 计算关键词匹配分数
                    match_score = self._calculate_keyword_match_score(
                        conversation, keywords
                    )
                    
                    memories.append({
                        'conversation': conversation,
                        'topics': record['topics'],
                        'entities': record['entities'],
                        'personas': record['personas'],
                        'match_score': match_score
                    })
                
                # 按匹配分数排序
                memories.sort(key=lambda x: x['match_score'], reverse=True)
                
                return memories
                
        except Exception as e:
            logger.error(f"根据关键词搜索记忆失败: {e}")
            return []
    
    def _calculate_keyword_match_score(self, conversation: Dict, keywords: List[str]) -> float:
        """计算关键词匹配分数"""
        try:
            user_message = conversation.get('user_message', '').lower()
            bot_response = conversation.get('bot_response', '').lower()
            full_text = user_message + ' ' + bot_response
            
            total_matches = 0
            for keyword in keywords:
                keyword_lower = keyword.lower()
                matches = full_text.count(keyword_lower)
                total_matches += matches
            
            # 归一化分数
            text_length = len(full_text.split())
            if text_length == 0:
                return 0.0
            
            return min(total_matches / text_length, 1.0)
            
        except Exception as e:
            logger.error(f"计算关键词匹配分数失败: {e}")
            return 0.0
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """获取用户偏好"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:USER {id: $user_id})-[:PARTICIPATED_IN]->(c:CONVERSATION)
                    OPTIONAL MATCH (c)-[:USES_PERSONA]->(p:PERSONA)
                    OPTIONAL MATCH (c)-[:RELATED_TO]->(t:TOPIC)
                    OPTIONAL MATCH (c)-[:MENTIONED]->(e:ENTITY {type: 'emotions'})
                    RETURN 
                        collect(DISTINCT p.name) as used_personas,
                        collect(DISTINCT t.name) as discussed_topics,
                        collect(DISTINCT e.name) as expressed_emotions,
                        count(c) as total_conversations,
                        u.created_at as user_created_at
                """, user_id=user_id)
                
                record = result.single()
                if record:
                    return {
                        'used_personas': record['used_personas'],
                        'discussed_topics': record['discussed_topics'],
                        'expressed_emotions': record['expressed_emotions'],
                        'total_conversations': record['total_conversations'],
                        'user_created_at': record['user_created_at']
                    }
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"获取用户偏好失败: {e}")
            return {}
    
    def get_conversation_context(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取对话上下文"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:CONVERSATION {session_id: $session_id})
                    OPTIONAL MATCH (c)-[:USES_PERSONA]->(p:PERSONA)
                    RETURN c, p.name as persona_name
                    ORDER BY c.timestamp DESC
                    LIMIT $limit
                """, session_id=session_id, limit=limit)
                
                context = []
                for record in result:
                    conversation = dict(record['c'])
                    context.append({
                        'user_message': conversation.get('user_message', ''),
                        'bot_response': conversation.get('bot_response', ''),
                        'persona_name': record['persona_name'],
                        'timestamp': conversation.get('timestamp', '')
                    })
                
                return list(reversed(context))  # 按时间正序返回
                
        except Exception as e:
            logger.error(f"获取对话上下文失败: {e}")
            return []
    
    def format_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """格式化记忆上下文"""
        if not memories:
            return ""
        
        context_parts = []
        for memory in memories:
            conv = memory['conversation']
            similarity = memory['similarity']
            
            context_part = f"[相似度: {similarity:.2f}] "
            context_part += f"用户说: {conv.get('user_message', '')[:100]}..."
            context_part += f" | 回复: {conv.get('bot_response', '')[:100]}..."
            
            if memory['topics']:
                context_part += f" | 话题: {', '.join(memory['topics'][:3])}"
            
            if memory['entities']:
                context_part += f" | 实体: {', '.join(memory['entities'][:3])}"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    async def generate_contextual_response(self, user_id: str, session_id: str, 
                                         current_message: str, persona_content: str,
                                         use_memory: bool = True) -> str:
        system_prompt += (
            "\n\n【重要指令】根据问题复杂度调整回答长度："
            "- 简单问题：1-8字简短回答（如'嗯'、'好的'、'明白了'）"
            "- 中等问题：9-20字自然回答"
            "- 复杂解释：20-50字详细说明"
            "像人类对话一样自然变换回答长度！"
        )
        
        """基于记忆生成带上下文的回复"""
        try:
            memory_context = ""
            conversation_context = ""
            
            if use_memory:
                # 获取相关记忆
                relevant_memories = self.retrieve_relevant_memories(user_id, current_message)
                if relevant_memories:
                    memory_context = self.format_memory_context(relevant_memories)
                
                # 获取近期对话上下文
                recent_context = self.get_conversation_context(session_id, 5)
                if recent_context:
                    context_parts = []
                    for ctx in recent_context:
                        context_parts.append(f"用户: {ctx['user_message']}")
                        context_parts.append(f"回复: {ctx['bot_response']}")
                    conversation_context = "\n".join(context_parts)
            
            # 构建包含记忆的prompt
            system_prompt = persona_content
            
            if memory_context or conversation_context:
                system_prompt += "\n\n=== 记忆和上下文信息 ==="
                
                if conversation_context:
                    system_prompt += f"\n\n近期对话:\n{conversation_context}"
                
                if memory_context:
                    system_prompt += f"\n\n相关记忆:\n{memory_context}"
                
                system_prompt += "\n\n请基于以上记忆和上下文信息，以一个有记忆、有感情的朋友身份回复用户。体现出你记得之前的对话和用户的偏好。"
            
            system_prompt += "\n\n重要提醒：在回复时请不要使用句号（。），可以使用其他标点符号如问号、感叹号等。"
            
            # 调用API生成回复
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_message}
                ],
                "temperature": 0.8,
                "max_tokens": 1000
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers, timeout=30.0)
                response.raise_for_status()
                
                result = response.json()
                bot_response = result["choices"][0]["message"]["content"]
                
                # 存储这次对话
                await self.store_conversation(
                    user_id, session_id, current_message, bot_response,
                    persona_name=getattr(self, 'current_persona_name', None)
                )
                
                return bot_response
                
        except Exception as e:
            logger.error(f"生成带上下文的回复失败: {e}")
            return "抱歉，我现在有点记忆混乱，请稍后再试"
    
    def get_memory_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取记忆统计信息"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (u:USER {id: $user_id})-[:PARTICIPATED_IN]->(c:CONVERSATION)
                    OPTIONAL MATCH (c)-[:USES_PERSONA]->(p:PERSONA)
                    OPTIONAL MATCH (c)-[:RELATED_TO]->(t:TOPIC)
                    OPTIONAL MATCH (c)-[:MENTIONED]->(e:ENTITY)
                    RETURN 
                        count(DISTINCT c) as total_conversations,
                        count(DISTINCT p) as personas_used,
                        count(DISTINCT t) as topics_discussed,
                        count(DISTINCT e) as entities_mentioned,
                        min(c.timestamp) as first_conversation,
                        max(c.timestamp) as last_conversation
                """, user_id=user_id)
                
                record = result.single()
                if record:
                    return dict(record)
                else:
                    return {}
                    
        except Exception as e:
            logger.error(f"获取记忆统计失败: {e}")
            return {}
    
    def update_memory_importance(self, user_id: str, conversation_id: str, 
                               importance_boost: float = 0.1) -> bool:
        """更新记忆重要性"""
        try:
            with self.driver.session() as session:
                # 更新对话重要性
                session.run("""
                    MATCH (c:CONVERSATION {id: $conversation_id})
                    SET c.importance = COALESCE(c.importance, 1.0) + $boost,
                        c.last_accessed = datetime()
                """, conversation_id=conversation_id, boost=importance_boost)
                
                # 更新相关实体重要性
                session.run("""
                    MATCH (c:CONVERSATION {id: $conversation_id})-[:MENTIONED]->(e:ENTITY)
                    SET e.importance = e.importance + $boost * 0.5
                """, conversation_id=conversation_id, boost=importance_boost)
                
                # 更新相关话题重要性
                session.run("""
                    MATCH (c:CONVERSATION {id: $conversation_id})-[:RELATED_TO]->(t:TOPIC)
                    SET t.importance = t.importance + $boost * 0.3
                """, conversation_id=conversation_id, boost=importance_boost)
                
                logger.info(f"更新记忆重要性: {conversation_id}")
                return True
                
        except Exception as e:
            logger.error(f"更新记忆重要性失败: {e}")
            return False
    
    def merge_similar_entities(self, similarity_threshold: float = 0.8) -> int:
        """合并相似实体"""
        try:
            merged_count = 0
            
            with self.driver.session() as session:
                # 查找相似实体
                result = session.run("""
                    MATCH (e1:ENTITY), (e2:ENTITY)
                    WHERE e1.type = e2.type AND e1.name <> e2.name AND id(e1) < id(e2)
                    RETURN e1, e2
                """)
                
                for record in result:
                    entity1 = dict(record['e1'])
                    entity2 = dict(record['e2'])
                    
                    # 计算名称相似度
                    similarity = self._calculate_string_similarity(
                        entity1['name'], entity2['name']
                    )
                    
                    if similarity >= similarity_threshold:
                        # 合并实体
                        if self._merge_entities(session, entity1, entity2):
                            merged_count += 1
                
                logger.info(f"合并了 {merged_count} 个相似实体")
                return merged_count
                
        except Exception as e:
            logger.error(f"合并相似实体失败: {e}")
            return 0
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度（简化版编辑距离）"""
        try:
            if str1 == str2:
                return 1.0
            
            # 简化的相似度计算
            str1_lower = str1.lower()
            str2_lower = str2.lower()
            
            # 检查包含关系
            if str1_lower in str2_lower or str2_lower in str1_lower:
                return 0.9
            
            # 计算公共字符比例
            common_chars = set(str1_lower) & set(str2_lower)
            total_chars = set(str1_lower) | set(str2_lower)
            
            if len(total_chars) == 0:
                return 0.0
            
            return len(common_chars) / len(total_chars)
            
        except Exception as e:
            logger.error(f"计算字符串相似度失败: {e}")
            return 0.0
    
    def _merge_entities(self, session, entity1: Dict, entity2: Dict) -> bool:
        """合并两个实体"""
        try:
            # 选择保留的实体（通常是重要性更高的）
            keep_entity = entity1 if entity1.get('importance', 0) >= entity2.get('importance', 0) else entity2
            remove_entity = entity2 if keep_entity == entity1 else entity1
            
            # 转移所有关系到保留的实体
            session.run("""
                MATCH (remove:ENTITY {name: $remove_name})-[r]->(target)
                MATCH (keep:ENTITY {name: $keep_name})
                CREATE (keep)-[new_r:MENTIONED]->(target)
                SET new_r = r
                DELETE r
            """, remove_name=remove_entity['name'], keep_name=keep_entity['name'])
            
            session.run("""
                MATCH (source)-[r]->(remove:ENTITY {name: $remove_name})
                MATCH (keep:ENTITY {name: $keep_name})
                CREATE (source)-[new_r:MENTIONED]->(keep)
                SET new_r = r
                DELETE r
            """, remove_name=remove_entity['name'], keep_name=keep_entity['name'])
            
            # 更新保留实体的统计信息
            session.run("""
                MATCH (keep:ENTITY {name: $keep_name}), (remove:ENTITY {name: $remove_name})
                SET keep.mention_count = keep.mention_count + remove.mention_count,
                    keep.importance = (keep.importance + remove.importance) / 2
                DELETE remove
            """, keep_name=keep_entity['name'], remove_name=remove_entity['name'])
            
            logger.info(f"合并实体: {remove_entity['name']} -> {keep_entity['name']}")
            return True
            
        except Exception as e:
            logger.error(f"合并实体失败: {e}")
            return False
    
    def update_entity_relationships(self, user_id: str) -> bool:
        """更新实体关系强度"""
        try:
            with self.driver.session() as session:
                # 更新实体共现关系强度
                session.run("""
                    MATCH (u:USER {id: $user_id})-[:PARTICIPATED_IN]->(c:CONVERSATION)
                    MATCH (c)-[:MENTIONED]->(e1:ENTITY), (c)-[:MENTIONED]->(e2:ENTITY)
                    WHERE e1 <> e2
                    MERGE (e1)-[r:ASSOCIATED_WITH]-(e2)
                    ON CREATE SET r.strength = 0.1, r.co_occurrence = 1, r.created_at = datetime()
                    ON MATCH SET r.strength = r.strength + 0.05, 
                                r.co_occurrence = r.co_occurrence + 1,
                                r.last_updated = datetime()
                """, user_id=user_id)
                
                # 衰减长时间未更新的关系
                session.run("""
                    MATCH ()-[r:ASSOCIATED_WITH]-()
                    WHERE r.last_updated < datetime() - duration({days: 30})
                    SET r.strength = r.strength * 0.9
                """)
                
                logger.info(f"更新用户 {user_id} 的实体关系")
                return True
                
        except Exception as e:
            logger.error(f"更新实体关系失败: {e}")
            return False
    
    def cleanup_old_memories(self, days_threshold: int = 90) -> int:
        """清理旧记忆"""
        try:
            with self.driver.session() as session:
                # 删除旧的低重要性对话记录
                result = session.run("""
                    MATCH (c:CONVERSATION)
                    WHERE c.timestamp < datetime() - duration({days: $days})
                    AND COALESCE(c.importance, 1.0) < 2.0
                    AND NOT exists((c)-[:IMPORTANT]-())
                    DETACH DELETE c
                    RETURN count(c) as deleted_count
                """, days=days_threshold)
                
                deleted_conversations = result.single()['deleted_count']
                
                # 清理孤立的实体节点
                result = session.run("""
                    MATCH (e:ENTITY)
                    WHERE NOT exists((e)<-[:MENTIONED]-())
                    DELETE e
                    RETURN count(e) as deleted_count
                """)
                
                deleted_entities = result.single()['deleted_count']
                
                # 清理孤立的话题节点
                result = session.run("""
                    MATCH (t:TOPIC)
                    WHERE NOT exists((t)<-[:RELATED_TO]-())
                    DELETE t
                    RETURN count(t) as deleted_count
                """)
                
                deleted_topics = result.single()['deleted_count']
                
                total_deleted = deleted_conversations + deleted_entities + deleted_topics
                logger.info(f"清理了 {total_deleted} 个旧记忆节点")
                
                return total_deleted
                
        except Exception as e:
            logger.error(f"清理旧记忆失败: {e}")
            return 0
    
    def optimize_memory_graph(self) -> bool:
        """优化记忆图结构"""
        try:
            with self.driver.session() as session:
                # 重新计算实体重要性
                session.run("""
                    MATCH (e:ENTITY)<-[:MENTIONED]-(c:CONVERSATION)
                    WITH e, count(c) as mention_count, 
                         avg(COALESCE(c.importance, 1.0)) as avg_importance
                    SET e.mention_count = mention_count,
                        e.importance = mention_count * 0.1 + avg_importance * 0.9
                """)
                
                # 重新计算话题重要性
                session.run("""
                    MATCH (t:TOPIC)<-[:RELATED_TO]-(c:CONVERSATION)
                    WITH t, count(c) as frequency,
                         avg(COALESCE(c.importance, 1.0)) as avg_importance
                    SET t.frequency = frequency,
                        t.importance = frequency * 0.2 + avg_importance * 0.8
                """)
                
                # 删除弱关系
                session.run("""
                    MATCH ()-[r:ASSOCIATED_WITH]-()
                    WHERE r.strength < 0.1
                    DELETE r
                """)
                
                # 更新用户统计
                session.run("""
                    MATCH (u:USER)-[:PARTICIPATED_IN]->(c:CONVERSATION)
                    WITH u, count(c) as total_conversations,
                         max(c.timestamp) as last_active
                    SET u.total_conversations = total_conversations,
                        u.last_active = last_active
                """)
                
                logger.info("记忆图优化完成")
                return True
                
        except Exception as e:
            logger.error(f"优化记忆图失败: {e}")
            return False
    
    def backup_user_memories(self, user_id: str, backup_path: str) -> bool:
        """备份用户记忆"""
        try:
            with self.driver.session() as session:
                # 导出用户的所有记忆数据
                result = session.run("""
                    MATCH (u:USER {id: $user_id})-[:PARTICIPATED_IN]->(c:CONVERSATION)
                    OPTIONAL MATCH (c)-[:RELATED_TO]->(t:TOPIC)
                    OPTIONAL MATCH (c)-[:MENTIONED]->(e:ENTITY)
                    OPTIONAL MATCH (c)-[:USES_PERSONA]->(p:PERSONA)
                    OPTIONAL MATCH (c)-[:HAS_EMOTION]->(em:EMOTION)
                    OPTIONAL MATCH (u)-[:HAS_PREFERENCE]->(pref:PREFERENCE)
                    RETURN u, c, 
                           collect(DISTINCT t) as topics,
                           collect(DISTINCT e) as entities,
                           collect(DISTINCT p) as personas,
                           collect(DISTINCT em) as emotions,
                           collect(DISTINCT pref) as preferences
                    ORDER BY c.timestamp
                """, user_id=user_id)
                
                backup_data = {
                    'user_id': user_id,
                    'backup_timestamp': datetime.now().isoformat(),
                    'conversations': [],
                    'topics': [],
                    'entities': [],
                    'personas': [],
                    'emotions': [],
                    'preferences': []
                }
                
                for record in result:
                    user_data = dict(record['u'])
                    conversation = dict(record['c'])
                    
                    backup_data['conversations'].append(conversation)
                    backup_data['topics'].extend([dict(t) for t in record['topics']])
                    backup_data['entities'].extend([dict(e) for e in record['entities']])
                    backup_data['personas'].extend([dict(p) for p in record['personas']])
                    backup_data['emotions'].extend([dict(em) for em in record['emotions']])
                    backup_data['preferences'].extend([dict(pref) for pref in record['preferences']])
                
                # 去重
                backup_data['topics'] = list({t['name']: t for t in backup_data['topics']}.values())
                backup_data['entities'] = list({e['name']: e for e in backup_data['entities']}.values())
                backup_data['personas'] = list({p['name']: p for p in backup_data['personas']}.values())
                backup_data['emotions'] = list({em['type']: em for em in backup_data['emotions']}.values())
                backup_data['preferences'] = list({pref['content']: pref for pref in backup_data['preferences']}.values())
                
                # 保存到文件
                import json
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"用户 {user_id} 的记忆已备份到 {backup_path}")
                return True
                
        except Exception as e:
            logger.error(f"备份用户记忆失败: {e}")
            return False
    
    def restore_user_memories(self, backup_path: str) -> bool:
        """恢复用户记忆"""
        try:
            import json
            
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            user_id = backup_data['user_id']
            
            with self.driver.session() as session:
                # 恢复用户节点
                session.run("""
                    MERGE (u:USER {id: $user_id})
                    SET u.restored_at = datetime()
                """, user_id=user_id)
                
                # 恢复对话记录
                for conversation in backup_data['conversations']:
                    session.run("""
                        CREATE (c:CONVERSATION $props)
                    """, props=conversation)
                    
                    # 重新创建用户-对话关系
                    session.run("""
                        MATCH (u:USER {id: $user_id}), (c:CONVERSATION {id: $conversation_id})
                        CREATE (u)-[:PARTICIPATED_IN]->(c)
                    """, user_id=user_id, conversation_id=conversation['id'])
                
                # 恢复其他节点和关系...
                # (这里可以根据需要扩展完整的恢复逻辑)
                
                logger.info(f"用户 {user_id} 的记忆已从 {backup_path} 恢复")
                return True
                
        except Exception as e:
            logger.error(f"恢复用户记忆失败: {e}")
            return False
    
    def get_memory_health_report(self) -> Dict[str, Any]:
        """获取记忆系统健康报告"""
        try:
            with self.driver.session() as session:
                # 统计各类节点数量
                result = session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] as node_type, count(n) as count
                """)
                
                node_counts = {}
                for record in result:
                    node_counts[record['node_type']] = record['count']
                
                # 统计关系数量
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as relation_type, count(r) as count
                """)
                
                relation_counts = {}
                for record in result:
                    relation_counts[record['relation_type']] = record['count']
                
                # 统计数据库大小和性能指标
                result = session.run("""
                    CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Store file sizes")
                    YIELD attributes
                    RETURN attributes
                """)
                
                db_size_info = {}
                try:
                    for record in result:
                        db_size_info = record['attributes']
                        break
                except:
                    db_size_info = {"error": "无法获取数据库大小信息"}
                
                # 检查数据质量
                result = session.run("""
                    MATCH (c:CONVERSATION)
                    WHERE c.user_embedding IS NULL OR c.timestamp IS NULL
                    RETURN count(c) as incomplete_conversations
                """)
                
                incomplete_conversations = result.single()['incomplete_conversations']
                
                # 检查孤立节点
                result = session.run("""
                    MATCH (n)
                    WHERE NOT (n)--()
                    RETURN labels(n)[0] as node_type, count(n) as orphan_count
                """)
                
                orphan_nodes = {}
                for record in result:
                    orphan_nodes[record['node_type']] = record['orphan_count']
                
                health_report = {
                    'timestamp': datetime.now().isoformat(),
                    'node_counts': node_counts,
                    'relation_counts': relation_counts,
                    'database_size': db_size_info,
                    'data_quality': {
                        'incomplete_conversations': incomplete_conversations,
                        'orphan_nodes': orphan_nodes
                    },
                    'recommendations': self._generate_health_recommendations(
                        node_counts, incomplete_conversations, orphan_nodes
                    )
                }
                
                return health_report
                
        except Exception as e:
            logger.error(f"获取记忆健康报告失败: {e}")
            return {'error': str(e)}
    
    def _generate_health_recommendations(self, node_counts: Dict, 
                                       incomplete_conversations: int, 
                                       orphan_nodes: Dict) -> List[str]:
        """生成健康建议"""
        recommendations = []
        
        # 检查数据完整性
        if incomplete_conversations > 0:
            recommendations.append(f"发现 {incomplete_conversations} 个不完整的对话记录，建议修复")
        
        # 检查孤立节点
        total_orphans = sum(orphan_nodes.values())
        if total_orphans > 0:
            recommendations.append(f"发现 {total_orphans} 个孤立节点，建议清理")
        
        # 检查数据规模
        total_conversations = node_counts.get('CONVERSATION', 0)
        if total_conversations > 10000:
            recommendations.append("对话记录较多，建议定期清理旧记录")
        
        # 检查实体数量
        total_entities = node_counts.get('ENTITY', 0)
        if total_entities > 5000:
            recommendations.append("实体数量较多，建议合并相似实体")
        
        if not recommendations:
            recommendations.append("记忆系统运行良好")
        
        return recommendations

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
    
    def __del__(self):
        """析构函数"""
        self.close()

