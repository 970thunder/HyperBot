import asyncio
import json
import os
import re
import random
import httpx
import time
from pathlib import Path
from typing import Dict, List, Optional
from nonebot import on_message, get_driver, logger
from nonebot.adapters.onebot.v11 import Bot, MessageEvent
from nonebot.plugin import PluginMetadata

# 导入记忆系统
from .memory_graph import MemoryGraph
from .memory_config import memory_config

__plugin_meta__ = PluginMetadata(
    name="增强版Deepseek人设聊天",
    description="支持多人设和长期记忆的智能聊天机器人",
    usage="发送消息聊天，使用特定关键词切换人设，支持长期记忆功能",
)

# 获取配置
driver = get_driver()
config = driver.config
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# 群聊回复配置
GROUP_REPLY_PROBABILITY = getattr(config, 'group_reply_probability', 0.6)
TRIGGER_KEYWORDS = getattr(config, 'trigger_keywords', ['希儿'])

# 人设文件夹路径
PERSONA_DIR = Path(__file__).parent / "personas"
PERSONA_DIR.mkdir(exist_ok=True)

# 人设配置文件路径
PERSONA_CONFIG_FILE = PERSONA_DIR / "personas.json"

# 用户当前人设存储 {user_id: persona_name}
current_personas: Dict[str, str] = {}

# 消息缓存和定时器 - 用于消息整合
message_cache: Dict[str, List[Dict]] = {} # 修改为包含event的缓存
message_timers: Dict[str, asyncio.Task] = {}
MESSAGE_MERGE_TIMEOUT = 10

# 记忆系统实例
memory_system: Optional[MemoryGraph] = None

class PersonaManager:
    def __init__(self):
        self.personas: Dict[str, Dict] = {}
        self.load_personas()
    
    def load_personas(self):
        """加载人设配置和内容"""
        try:
            if not PERSONA_CONFIG_FILE.exists():
                logger.error(f"人设配置文件不存在: {PERSONA_CONFIG_FILE}")
                return
            
            with open(PERSONA_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            for persona_name, persona_config in config_data.items():
                persona_file = PERSONA_DIR / persona_config['file']
                if persona_file.exists():
                    with open(persona_file, 'r', encoding='utf-8') as f:
                        persona_content = f.read().strip()
                    
                    keywords = persona_config.get('keywords', [])
                    if isinstance(keywords, str):
                        keywords = [keywords]
                    
                    self.personas[persona_name] = {
                        'name': persona_name,
                        'content': persona_content,
                        'keywords': keywords,
                        'is_default': persona_config.get('is_default', False),
                        'description': persona_config.get('description', '')
                    }
                else:
                    logger.warning(f"人设文件不存在: {persona_file}")
            
            logger.info(f"加载了 {len(self.personas)} 个人设")
        except Exception as e:
            logger.error(f"加载人设失败: {e}")
    
    def get_persona(self, name: str) -> Optional[Dict]:
        """获取指定人设"""
        return self.personas.get(name)
    
    def get_persona_by_keyword(self, keyword: str) -> Optional[Dict]:
        """通过关键词获取人设"""
        for persona in self.personas.values():
            keywords = persona.get('keywords', [])
            if keyword in keywords or any(keyword in k for k in keywords):
                return persona
        return None
    
    def list_personas(self) -> List[Dict]:
        """获取所有人设信息"""
        return [
            {
                'name': persona['name'],
                'description': persona['description'],
                'keywords': persona['keywords']
            }
            for persona in self.personas.values()
        ]
    
    def get_default_persona(self) -> Optional[Dict]:
        """获取默认人设"""
        for persona in self.personas.values():
            if persona.get('is_default', False):
                return persona
        return None
    
    def reload_personas(self):
        """重新加载人设"""
        self.personas.clear()
        self.load_personas()

# 初始化人设管理器
persona_manager = PersonaManager()

def clean_text(text: str) -> str:
    """清理文本，去掉句号和其他不必要的标点符号"""
    text = text.replace('。', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_text_by_sentences(text: str, min_sentences: int = 1, max_sentences: int = 3) -> List[str]:
    # 特殊处理短回复（不超过10字）
    if len(text) <= 10:
        return [text]
    
    # 保留原始短句（如"嗯"、"好的"）
    if re.match(r'^[嗯啊哦咦诶]{1,3}[！？。~]*$', text):
        return [text]
    
    """将文本按句子分割成多个部分，并清理句号"""
    text = clean_text(text)
    
    sentences = re.split(r'([！？!?]+)', text)
    
    result_sentences = []
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            sentence = sentences[i].strip()
            if sentence:
                if i + 1 < len(sentences) and sentences[i + 1].strip():
                    sentence += sentences[i + 1]
                result_sentences.append(sentence)
    
    if not result_sentences:
        return [text] if text else []
    
    paragraphs = []
    current_paragraph = []
    
    for sentence in result_sentences:
        if sentence.strip():
            current_paragraph.append(sentence)
        
        if len(current_paragraph) >= min_sentences:
            if len(current_paragraph) >= max_sentences or random.random() < 0.6:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
    
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    return paragraphs if paragraphs else [text]

def split_message_for_sending(text: str) -> List[str]:
    """
    将一段完整的回复，基于标点符号精确地分割成多条独立的句子。
    """
    if not text:
        return []

    # 清理文本，为分割做准备
    text = clean_text(text)
    
    # 使用正则表达式分割句子。这个表达式会匹配各种结束标点，
    # 并保留这些标点在句子的末尾。
    sentences = re.split(r'(?<=[，。？！…~,!?])\s*', text)
    
    # 清理分割后的结果，去除空字符串
    # 过滤掉只包含空格或标点的无意义短句
    final_sentences = [s.strip() for s in sentences if s and s.strip() and not re.match(r'^[，。？！…~,!?\s]+$', s)]

    return final_sentences if final_sentences else [text]

async def send_messages_with_typing(bot: Bot, event: MessageEvent, messages: List[str]):
    """模拟真人打字效果发送消息"""
    for i, message in enumerate(messages):
        if not message.strip(): continue
        # 根据消息长度估算打字时间
        typing_time = max(0.5, min(len(message) * 0.08, 2.5))
        await asyncio.sleep(typing_time)
        await bot.send(event, message)
        # 在多条消息之间加入随机延迟
        if i < len(messages) - 1:
            await asyncio.sleep(random.uniform(0.8, 1.8))

async def process_merged_messages(bot: Bot, session_id: str):
    """处理整合后的消息，现在不直接依赖event，更通用"""
    try:
        cached_data = message_cache.get(session_id)
        if not cached_data: return

        # 使用最后一条消息的event进行回复和身份识别
        last_message_event = cached_data[-1]['event']
        user_id = str(last_message_event.user_id)
        user_nickname = last_message_event.sender.card or last_message_event.sender.nickname
        
        # 整合所有消息文本
        merged_message = ' '.join([msg['text'] for msg in cached_data])
        message_cache.pop(session_id, None)

        logger.info(f"处理整合消息 - 会话 {session_id}: {merged_message}")

        group_members = None
        if last_message_event.message_type == 'group' and hasattr(last_message_event, 'group_id'):
            try:
                group_id = int(last_message_event.group_id)
                member_list = await bot.get_group_member_list(group_id=group_id, no_cache=True)
                group_members = [{"user_id": str(m["user_id"]), "nickname": m.get("card") or m.get("nickname")} for m in member_list]
            except Exception as e:
                logger.error(f"获取群成员列表失败: {e}")

        current_persona = persona_manager.get_persona(current_personas.get(session_id)) or persona_manager.get_default_persona()
        if not current_persona:
            await bot.send(last_message_event, "请先设置人设或创建默认人设")
            return

        response = ""
        if memory_system and memory_config.ENABLE_MEMORY_SYSTEM:
            response = await memory_system.generate_contextual_response(
                user_id=user_id,
                session_id=session_id,
                current_message=merged_message,
                persona_content=current_persona['content'],
                user_nickname=user_nickname,
                group_members=group_members
            )
        else:
            response = await call_deepseek_api(merged_message, current_persona)
        
        if response:
            # 最终过滤器：在分割前强行移除所有 || 符号
            cleaned_text = re.sub(r'\s*[||]{2}\s*', ' ', response).strip()
            message_parts = split_message_for_sending(cleaned_text)
            await send_messages_with_typing(bot, last_message_event, message_parts)
            logger.info(f"已用 {current_persona['name']} 人设回复整合消息 - 会话 {session_id}")
        
    except Exception as e:
        logger.error(f"处理整合消息时出错: {e}")
    finally:
        message_timers.pop(session_id, None)

async def schedule_message_processing(bot: Bot, event: MessageEvent, session_id: str):
    """安排消息处理定时器"""
    await asyncio.sleep(MESSAGE_MERGE_TIMEOUT)
    
    if session_id in message_timers:
        await process_merged_messages(bot, session_id) # 修改为直接调用 process_merged_messages

async def call_deepseek_api(message: str, persona: Dict) -> str:
    """调用Deepseek API with persona"""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = persona['content'] + "\n\n重要提醒：在回复时请不要使用句号（。），可以使用其他标点符号如问号、感叹号等。"
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": message
            }
        ],
        "temperature": 0.8,
        "max_tokens": 1000
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, headers=headers, timeout=30.0)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]

def is_command_message(message_text: str) -> bool:
    """判断消息是否为命令消息（不应该被整合）"""
    commands = [
        '人设列表', '查看人设', '人设', 
        '重新加载人设', '刷新人设', '重载人设',
        '记忆统计', '记忆报告', '清理记忆', '记忆健康'
    ]
    
    if message_text in commands:
        return True
    
    if persona_manager.get_persona_by_keyword(message_text):
        return True
    
    return False

async def should_reply_in_group(session_id: str, event: MessageEvent, text: str) -> bool:
    """更智能地判断在群聊中是否应该回复"""
    # 1. 私聊必定回复
    if event.message_type == "private":
        return True
    
    # 2. 被@时必定回复
    if hasattr(event, 'to_me') and event.to_me:
        logger.info("群聊消息：被@，触发回复")
        return True

    # 3. 消息以触发词开头时回复 (例如 "希儿，你好")
    for keyword in TRIGGER_KEYWORDS:
        if re.match(rf'^\s*{re.escape(keyword)}[\s,，!！?？]*', text, re.IGNORECASE):
            logger.info(f"群聊消息：以触发词'{keyword}'开头，触发回复")
            return True
            
    # 4. 基于相关性判断是否插话
    if memory_system and memory_config.ENABLE_MEMORY_SYSTEM:
        # 获取当前人设用于评估相关性
        current_persona = persona_manager.get_persona(current_personas.get(session_id)) or persona_manager.get_default_persona()
        if not current_persona:
            return False # 没有人设无法判断相关性

        relevance = await memory_system.calculate_interjection_relevance(
            session_id=session_id,
            current_message=text,
            persona_content=current_persona['content']
        )
        if relevance > 0.5:
            logger.info(f"群聊消息相关度 ({relevance:.2f}) > 0.5，触发插话")
            return True

    return False

# 创建消息处理器
chat_handler = on_message(priority=99, block=False) # 改为非阻塞，允许插话

@chat_handler.handle()
async def handle_message(bot: Bot, event: MessageEvent):
    user_id = str(event.user_id)
    message_text = event.get_plaintext().strip()
    
    if not message_text:
        return
    
    # 获取会话标识
    if event.message_type == "private":
        session_id = f"private_{user_id}"
        logger.info(f"收到私聊用户 {user_id} 的消息: {message_text}")
    else:
        group_id = str(event.group_id) if hasattr(event, 'group_id') else "unknown"
        session_id = f"group_{group_id}_{user_id}"
        logger.info(f"收到群聊 {group_id} 用户 {user_id} 的消息: {message_text}")
    
    # 检查是否是命令消息
    if is_command_message(message_text):
        if session_id in message_timers:
            message_timers[session_id].cancel()
            if session_id in message_cache and message_cache[session_id]:
                await process_merged_messages(bot, session_id) # 修改为直接调用 process_merged_messages
        
        await handle_command_message(bot, event, message_text, session_id)
        return
    
    # 群聊回复概率判断
    if not await should_reply_in_group(session_id, event, message_text):
        return
    
    # 对于普通消息，进行整合处理
    await handle_regular_message(bot, event, message_text, session_id)

async def handle_command_message(bot: Bot, event: MessageEvent, message_text: str, session_id: str):
    """处理命令消息"""
    user_id = str(event.user_id)
    
    # 检查是否是人设切换命令
    persona = persona_manager.get_persona_by_keyword(message_text)
    if persona:
        current_personas[session_id] = persona['name']
        await bot.send(event, f"已切换到 {persona['name']} 人设~")
        logger.info(f"会话 {session_id} 切换到人设: {persona['name']}")
        return
    
    # 检查是否是查看人设列表命令
    if message_text in ['人设列表', '查看人设', '人设']:
        persona_list = persona_manager.list_personas()
        if persona_list:
            reply = "🎭 可用的人设：\n\n"
            for persona in persona_list:
                keywords_str = " | ".join(persona['keywords'])
                reply += f"• {persona['name']}\n"
                reply += f"  描述: {persona['description']}\n"
                reply += f"  触发词: {keywords_str}\n\n"
            await bot.send(event, reply.strip())
        else:
            await bot.send(event, "暂无可用人设")
        return
    
    # 检查是否是重新加载人设命令
    if message_text in ['重新加载人设', '刷新人设', '重载人设']:
        persona_manager.reload_personas()
        await bot.send(event, f"人设已重新加载！当前有 {len(persona_manager.personas)} 个人设")
        return
    
    # 记忆系统相关命令
    if memory_system and memory_config.ENABLE_MEMORY_SYSTEM:
        if message_text in ['记忆统计', '记忆报告']:
            try:
                stats = memory_system.get_memory_statistics(user_id)
                if stats:
                    reply = f"📊 您的记忆统计：\n\n"
                    reply += f"总对话数: {stats.get('total_conversations', 0)}\n"
                    reply += f"使用人设数: {stats.get('personas_used', 0)}\n"
                    reply += f"讨论话题数: {stats.get('topics_discussed', 0)}\n"
                    reply += f"提及实体数: {stats.get('entities_mentioned', 0)}\n"
                    
                    first_conv = stats.get('first_conversation')
                    last_conv = stats.get('last_conversation')
                    if first_conv:
                        reply += f"首次对话: {first_conv}\n"
                    if last_conv:
                        reply += f"最近对话: {last_conv}\n"
                    
                    await bot.send(event, reply)
                else:
                    await bot.send(event, "暂无记忆数据")
            except Exception as e:
                logger.error(f"获取记忆统计失败: {e}")
                await bot.send(event, "获取记忆统计失败")
            return
        
        if message_text in ['记忆健康', '系统健康']:
            try:
                health_report = memory_system.get_memory_health_report()
                if 'error' not in health_report:
                    reply = "🏥 记忆系统健康报告：\n\n"
                    
                    node_counts = health_report.get('node_counts', {})
                    reply += "节点统计:\n"
                    for node_type, count in node_counts.items():
                        reply += f"  {node_type}: {count}\n"
                    
                    recommendations = health_report.get('recommendations', [])
                    reply += f"\n建议:\n"
                    for rec in recommendations:
                        reply += f"  • {rec}\n"
                    
                    await bot.send(event, reply)
                else:
                    await bot.send(event, f"获取健康报告失败: {health_report['error']}")
            except Exception as e:
                logger.error(f"获取健康报告失败: {e}")
                await bot.send(event, "获取健康报告失败")
            return
        
        if message_text in ['清理记忆']:
            try:
                deleted_count = memory_system.cleanup_old_memories()
                await bot.send(event, f"已清理 {deleted_count} 条旧记忆")
            except Exception as e:
                logger.error(f"清理记忆失败: {e}")
                await bot.send(event, "清理记忆失败")
            return

async def handle_regular_message(bot: Bot, event: MessageEvent, message_text: str, session_id: str):
    """处理普通消息（需要整合的消息）"""
    if session_id in message_timers:
        message_timers[session_id].cancel()
    
    if session_id not in message_cache:
        message_cache[session_id] = []
    message_cache[session_id].append({'text': message_text, 'event': event}) # 存储event
    
    timer_task = asyncio.create_task(schedule_message_processing(bot, event, session_id))
    message_timers[session_id] = timer_task
    
    logger.info(f"消息已添加到缓存 - 会话 {session_id}: {message_text}")

async def init_memory_system():
    """初始化记忆系统"""
    global memory_system
    
    if not memory_config.ENABLE_MEMORY_SYSTEM:
        logger.info("记忆系统已禁用")
        return
    
    if not memory_config.validate():
        logger.error("记忆系统配置验证失败")
        return
    
    try:
        memory_system = MemoryGraph(
            neo4j_uri=memory_config.NEO4J_URI,
            neo4j_user=memory_config.NEO4J_USER,
            neo4j_password=memory_config.NEO4J_PASSWORD,
            deepseek_api_key=DEEPSEEK_API_KEY,  # 确保使用全局的DEEPSEEK_API_KEY
            embedding_model=memory_config.EMBEDDING_MODEL
        )
        
        logger.info("记忆系统初始化成功")
        
        # 测试连接
        test_user_id = "test_connection"
        memory_system.create_or_get_user(test_user_id)
        logger.info("记忆系统连接测试成功")
        
    except Exception as e:
        logger.error(f"记忆系统初始化失败: {e}")
        memory_system = None

# 启动时检查
@driver.on_startup
async def startup_check():
    """启动时检查"""
    # 检查人设文件
    if not persona_manager.personas:
        logger.warning("未找到任何人设文件，请在 personas 文件夹中创建人设文件")
    else:
        logger.info(f"已加载 {len(persona_manager.personas)} 个人设")
        for persona in persona_manager.list_personas():
            logger.info(f"- {persona['name']}: {', '.join(persona['keywords'])}")
    
    # 显示群聊回复配置
    logger.info(f"群聊回复概率: {GROUP_REPLY_PROBABILITY*100:.1f}%")
    logger.info(f"触发词: {', '.join(TRIGGER_KEYWORDS)}")
    
    # 初始化记忆系统
    await init_memory_system()

# 关闭时清理
@driver.on_shutdown
async def cleanup():
    """关闭时清理所有资源"""
    # 清理定时器
    for timer in message_timers.values():
        if not timer.done():
            timer.cancel()
    message_timers.clear()
    message_cache.clear()
    logger.info("已清理所有消息定时器和缓存")
    
    # 关闭记忆系统
    if memory_system:
        memory_system.close()
        logger.info("记忆系统已关闭")

