import asyncio
import json
import time
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from nonebot import logger
import os
from asyncio import Semaphore
import concurrent.futures
from functools import lru_cache

# 导入优化组件
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'deepseek'))
from api_pool import get_siliconflow_manager, cleanup_global_api_pool
from cache_manager import get_global_cache, AnalysisCache

# 添加dotenv支持
try:
    from dotenv import load_dotenv
    load_dotenv()  # 显式加载.env文件
except ImportError:
    pass

# 导入配置
from .config import HeartflowConfig, get_heartflow_config

@dataclass
class GroupState:
    """群聊状态"""
    group_id: str
    energy: float = 1.0  # 精力值 0.1-1.0
    last_reply_time: float = 0  # 上次回复时间戳
    last_recovery_time: float = 0  # 上次精力恢复时间戳
    total_messages_today: int = 0  # 今日总消息数
    bot_replies_today: int = 0  # 今日机器人回复数
    last_reset_day: str = ""  # 上次重置日期
    recent_messages: List[Dict] = None  # 最近的消息历史
    
    def __post_init__(self):
        if self.recent_messages is None:
            self.recent_messages = []

class HeartflowEngine:
    """心流引擎 - 智能回复决策系统"""
    
    def __init__(self, config: HeartflowConfig = None):
        """初始化心流引擎"""
        # 使用延迟初始化的配置
        self.config = config or get_heartflow_config()
        self.group_states: Dict[str, GroupState] = {}
        
        # 多种方式尝试获取API密钥
        self.api_key = self._get_api_key()
        
        # 并发控制
        self.api_semaphore = Semaphore(100)  # API调用并发控制
        self.cache_semaphore = Semaphore(5)  # 缓存操作并发控制
        
        # 缓存
        self.decision_cache = {}  # 决策缓存
        self.analysis_cache = {}  # AI分析缓存
        
        # 全局缓存管理器 - 延迟初始化
        self.global_cache = None
        self.analysis_cache_manager = None
        
        # 线程池用于CPU密集型任务
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix="heartflow")
        
        if not self.api_key:
            logger.warning("未设置SILICONFLOW_API_KEY环境变量，将无法使用心流机制")
        else:
            logger.info(f"心流机制已启用，API密钥: {self.api_key[:8]}...")
    
    def _get_api_key(self) -> str:
        """多种方式获取API密钥"""
        # 方式1: 直接从环境变量获取
        api_key = os.getenv("SILICONFLOW_API_KEY", "")
        if api_key:
            return api_key
        
        # 方式2: 尝试从NoneBot配置获取
        try:
            from nonebot import get_driver
            driver = get_driver()
            config = driver.config
            api_key = getattr(config, 'siliconflow_api_key', "")
            if api_key:
                return api_key
        except:
            pass
        
        # 方式3: 再次尝试加载.env文件
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)  # 强制重新加载
            api_key = os.getenv("SILICONFLOW_API_KEY", "")
            if api_key:
                return api_key
        except:
            pass
        
        return ""
    
    def get_group_state(self, group_id: str) -> GroupState:
        """获取群聊状态，如果不存在则创建"""
        if group_id not in self.group_states:
            self.group_states[group_id] = GroupState(group_id=group_id)
        
        state = self.group_states[group_id]
        self._update_daily_reset(state)
        self._recover_energy(state)
        return state
    
    def _update_daily_reset(self, state: GroupState):
        """检查并执行每日重置"""
        today = datetime.now().strftime("%Y-%m-%d")
        if state.last_reset_day != today:
            # 每日重置
            state.total_messages_today = 0
            state.bot_replies_today = 0
            state.last_reset_day = today
            # 每日精力恢复奖励
            state.energy = min(1.0, state.energy + self.config.daily_energy_bonus)
            logger.info(f"群 {state.group_id} 执行每日重置，精力恢复到 {state.energy:.2f}")
    
    def _recover_energy(self, state: GroupState):
        """精力自然恢复"""
        current_time = time.time()
        if state.last_recovery_time == 0:
            state.last_recovery_time = current_time
            return
        
        # 计算恢复时间（秒）
        time_diff = current_time - state.last_recovery_time
        if time_diff > 60:  # 每分钟恢复一次
            recovery_amount = self.config.energy_recovery_rate * (time_diff / 60)
            state.energy = min(1.0, state.energy + recovery_amount)
            state.last_recovery_time = current_time
    
    def consume_energy(self, group_id: str):
        """消耗精力（回复后调用）"""
        state = self.get_group_state(group_id)
        state.energy = max(0.1, state.energy - self.config.energy_decay_rate)
        state.last_reply_time = time.time()
        state.bot_replies_today += 1
        logger.info(f"群 {group_id} 精力消耗，当前精力: {state.energy:.2f}")
    
    def add_message(self, group_id: str, user_id: str, message: str, nickname: str = ""):
        """添加消息到上下文"""
        state = self.get_group_state(group_id)
        state.total_messages_today += 1

        message_data = {
            "user_id": user_id,
            "message": message,
            "nickname": nickname,
            "timestamp": time.time(),
            "is_bot_reply": False
        }

        state.recent_messages.append(message_data)
        # 保持最近N条消息
        if len(state.recent_messages) > self.config.context_messages_count:
            state.recent_messages.pop(0)
    
    async def _add_message_async(self, group_id: str, user_id: str, message: str, nickname: str = ""):
        """异步添加消息到上下文"""
        # 在线程池中执行，避免阻塞
        await asyncio.to_thread(self.add_message, group_id, user_id, message, nickname)
    
    async def should_reply(self, group_id: str, user_id: str, message: str, 
                          nickname: str = "", persona_name: str = "") -> Tuple[bool, Dict]:
        """
        核心决策函数：判断是否应该回复 - 优化版本
        返回: (是否回复, 决策详情)
        """
        if not self.api_key:
            # 如果没有API key，使用简单的回退策略
            return await self._fallback_decision(group_id, message)
        
        state = self.get_group_state(group_id)
        current_time = time.time()
        
        # 1. 基础限制检查
        basic_checks = self._check_basic_limits(state, current_time)
        if not basic_checks["can_reply"]:
            return False, {
                "decision": "rejected",
                "reason": basic_checks["reason"],
                "energy": state.energy,
                "details": basic_checks
            }
        
        # 2. 并行处理：添加消息到上下文和检查缓存
        cache_key = f"{group_id}_{hash(message)}_{persona_name}"
        
        # 并行任务
        tasks = []
        tasks.append(self._add_message_async(group_id, user_id, message, nickname))
        tasks.append(self._get_cached_analysis(cache_key, message, persona_name, state))
        
        # 等待任务完成
        _, cached_analysis = await asyncio.gather(*tasks)
        
        # 3. 使用缓存或AI分析
        try:
            if cached_analysis:
                analysis = cached_analysis
                logger.info(f"使用缓存的AI分析结果")
            else:
                analysis = await self._analyze_message_with_ai_optimized(state, message, persona_name)
                # 缓存结果
                asyncio.create_task(self._cache_analysis(cache_key, analysis))
            
            # 4. 综合评分决策
            decision_result = self._make_final_decision(state, analysis)
            
            return decision_result["should_reply"], {
                "decision": "approved" if decision_result["should_reply"] else "rejected",
                "analysis": analysis,
                "final_score": decision_result["final_score"],
                "threshold": self.config.reply_threshold,
                "energy": state.energy,
                "details": decision_result
            }
            
        except Exception as e:
            logger.error(f"心流AI分析失败: {e}")
            # 降级到简单策略
            return await self._fallback_decision(group_id, message)
    
    def _check_basic_limits(self, state: GroupState, current_time: float) -> Dict:
        """检查基础限制条件"""
        # 检查精力值
        if state.energy < self.config.min_energy_threshold:
            return {
                "can_reply": False,
                "reason": "精力不足",
                "energy_too_low": True
            }

        # 检查回复间隔 - 为连贯对话提供更宽松的限制
        time_since_last_reply = current_time - state.last_reply_time

        # 如果是连贯对话（最近有用户消息），允许更短的回复间隔
        is_coherent_conversation = False
        if state.recent_messages:
            last_user_message_time = max(msg.get('timestamp', 0) for msg in state.recent_messages
                                       if not msg.get('is_bot_reply', False))
            time_since_last_user_message = current_time - last_user_message_time
            # 如果最近30秒内有用户消息，认为是连贯对话
            is_coherent_conversation = time_since_last_user_message < 30

        # 连贯对话的最小回复间隔为10秒，普通对话为30秒
        min_interval = 10 if is_coherent_conversation else self.config.min_reply_interval

        # 已移除回复间隔检查 - 跳过此条件

        # 检查每小时回复限制
        hour_ago = current_time - 3600
        recent_replies = sum(1 for msg in state.recent_messages
                           if msg.get("is_bot_reply") and msg["timestamp"] > hour_ago)

        if recent_replies >= self.config.max_replies_per_hour:
            return {
                "can_reply": False,
                "reason": "达到每小时回复上限",
                "hourly_limit_reached": True
            }

        return {"can_reply": True, "reason": "通过基础检查", "is_coherent_conversation": is_coherent_conversation}

    def _analyze_conversation_coherence(self, recent_messages: List[Dict], new_message: str) -> float:
        """分析对话连贯性，返回0-1的连贯性分数"""
        if len(recent_messages) < 2:
            return 0.5  # 没有足够上下文，默认中等连贯性

        coherence_score = 0.5  # 基础分数

        # 检查最后一条消息是否来自同一用户
        last_message = recent_messages[-1]
        if len(recent_messages) > 1:
            second_last = recent_messages[-2]

            # 如果最后两条消息来自同一用户，可能是连续发言
            if last_message.get('user_id') == second_last.get('user_id'):
                coherence_score += 0.2

            # 检查时间间隔 - 短时间内连续发言
            time_diff = last_message.get('timestamp', 0) - second_last.get('timestamp', 0)
            if time_diff < 60:  # 1分钟内
                coherence_score += 0.1

        # 检查新消息是否包含对话延续的线索
        new_message_lower = new_message.lower()

        # 对话延续关键词
        continuation_keywords = [
            "然后", "接着", "还有", "另外", "而且", "但是", "不过",
            "所以", "因此", "于是", "接下来", "继续", "接着说",
            "嗯", "哦", "啊", "好吧", "好的", "行", "可以"
        ]

        # 问题关键词
        question_keywords = ["？", "?", "什么", "怎么", "为什么", "如何", "哪里", "谁", "什么时候"]

        # 提及机器人关键词
        bot_mention_keywords = ["机器人", "bot", "@", "你", "帮忙", "帮助"]

        # 分析新消息特征
        if any(keyword in new_message_lower for keyword in continuation_keywords):
            coherence_score += 0.2

        if any(keyword in new_message_lower for keyword in question_keywords):
            coherence_score += 0.3

        if any(keyword in new_message_lower for keyword in bot_mention_keywords):
            coherence_score += 0.4

        # 限制在0-1范围内
        return max(0.0, min(1.0, coherence_score))
    
    async def _analyze_message_with_ai_optimized(self, state: GroupState, message: str, persona_name: str) -> Dict:
        """使用AI分析消息 - 优化上下文连贯性"""

        # 构建上下文 - 使用更多消息和对话连贯性分析
        context_messages = []
        for msg in state.recent_messages[-5:]:  # 增加到最近5条消息作为上下文
            sender = msg.get('nickname', msg['user_id'])
            context_messages.append(f"{sender}: {msg['message']}")

        context_str = "\n".join(context_messages) if context_messages else "无上下文"

        # 分析对话连贯性
        conversation_coherence = self._analyze_conversation_coherence(state.recent_messages, message)
        
        # 构建prompt - 优化为更稳定的JSON输出，增强上下文连贯性判断
        analysis_prompt = f"""你是一个聊天群的智能回复决策助手。请分析以下消息是否值得机器人回复。

当前群聊上下文（最近对话）：
{context_str}

新消息: {message}
机器人当前人设: {persona_name or "默认"}
当前精力值: {state.energy:.2f}/1.0
今日已回复: {state.bot_replies_today}次

**特别注意对话连贯性**：
- 如果新消息明显是对话的延续，请提高回复意愿
- 如果消息直接提及机器人或包含问题，请提高内容相关度
- 如果对话正在进行中，请考虑社交适宜性

请从以下4个维度评分(0-10分)，并给出简短理由：
1. 内容相关度：消息是否有价值、有趣、适合回复，是否与上下文连贯
2. 回复意愿：基于当前精力状态和对话连贯性的回复意愿
3. 社交适宜性：回复是否符合群聊氛围和当前对话节奏
4. 时机恰当性：考虑频率控制、时间间隔和对话时机

**重要**: 请直接返回JSON格式，不要使用markdown包装：
{{
  "content_relevance": 整数(0-10),
  "reply_willingness": 整数(0-10),
  "social_appropriateness": 整数(0-10),
  "timing_appropriateness": 整数(0-10),
  "reasoning": "简短的分析理由",
  "key_factors": ["关键因素1", "关键因素2"],
  "conversation_coherence": {conversation_coherence}
}}"""

        try:
            url = "https://api.siliconflow.cn/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "Qwen/QwQ-32B",
                "max_tokens": self.config.max_analysis_tokens,
                "temperature": self.config.analysis_temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ]
            }
            
            # 使用信号量控制并发和连接池
            async with self.api_semaphore:
                async with httpx.AsyncClient(
                    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                    timeout=httpx.Timeout(self.config.api_timeout, connect=5.0, read=20.0)
                ) as client:
                    response = await client.post(
                        url, 
                        json=payload, 
                        headers=headers
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    ai_response = result["choices"][0]["message"]["content"]
                    
                    # 尝试解析JSON响应 - 使用与测试脚本相同的成功逻辑
                    try:
                        # 首先尝试直接解析JSON
                        analysis = json.loads(ai_response)
                        logger.debug(f"直接解析JSON成功")
                    except json.JSONDecodeError:
                        # 如果直接解析失败，尝试提取markdown中的JSON
                        try:
                            import re
                            # 查找```json...```或```...```包装的JSON
                            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', ai_response, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1).strip()
                                analysis = json.loads(json_str)
                                logger.debug(f"从markdown中成功解析JSON")
                            else:
                                # 尝试查找第一个完整的JSON对象
                                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                                if json_match:
                                    json_str = json_match.group(0)
                                    analysis = json.loads(json_str)
                                    logger.debug(f"成功提取JSON对象")
                                else:
                                    raise ValueError("无法找到有效的JSON内容")
                        except (json.JSONDecodeError, ValueError) as e2:
                            logger.warning(f"AI响应JSON解析完全失败: {e2}")
                            logger.warning(f"原始响应: {ai_response}")
                            # 尝试更宽松的JSON提取
                            try:
                                # 移除所有markdown标记
                                cleaned_response = re.sub(r'```.*?```', '', ai_response, flags=re.DOTALL)
                                cleaned_response = re.sub(r'`.*?`', '', cleaned_response)
                                cleaned_response = cleaned_response.strip()
                                
                                # 查找JSON对象
                                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response)
                                if json_match:
                                    json_str = json_match.group(0)
                                    analysis = json.loads(json_str)
                                    logger.info(f"使用宽松模式成功解析JSON")
                                else:
                                    raise ValueError("无法找到有效的JSON内容")
                            except (json.JSONDecodeError, ValueError) as e3:
                                logger.error(f"宽松模式JSON解析也失败: {e3}")
                                return self._generate_fallback_analysis(message)
                    
                    # 验证必要字段
                    required_fields = ["content_relevance", "reply_willingness", 
                                     "social_appropriateness", "timing_appropriateness"]
                    
                    # 检查字段存在性和类型
                    missing_fields = []
                    invalid_fields = []
                    
                    for field in required_fields:
                        if field not in analysis:
                            missing_fields.append(field)
                        elif not isinstance(analysis[field], (int, float)):
                            invalid_fields.append(f"{field}={analysis[field]} (type: {type(analysis[field]).__name__})")
                    
                    if missing_fields or invalid_fields:
                        logger.warning(f"AI响应字段验证失败:")
                        if missing_fields:
                            logger.warning(f"  缺少字段: {missing_fields}")
                        if invalid_fields:
                            logger.warning(f"  无效字段: {invalid_fields}")
                        logger.warning(f"  原始分析结果: {analysis}")
                        return self._generate_fallback_analysis(message)
                    
                    # 验证字段值范围（0-10）
                    out_of_range_fields = []
                    for field in required_fields:
                        value = analysis[field]
                        if not (0 <= value <= 10):
                            out_of_range_fields.append(f"{field}={value}")
                    
                    if out_of_range_fields:
                        logger.warning(f"AI响应字段值超出范围(0-10): {out_of_range_fields}")
                        # 不使用回退策略，而是修正数值
                        for field in required_fields:
                            analysis[field] = max(0, min(10, analysis[field]))
                        logger.info(f"已修正字段值范围")
                    
                    logger.debug(f"AI分析成功解析: {analysis}")
                    return analysis
                
        except Exception as e:
            logger.error(f"调用硅基流动API失败: {type(e).__name__}: {e}")
            logger.error(f"API URL: {url}")
            logger.error(f"API Key: {self.api_key[:10]}..." if self.api_key else "API Key: None")
            return self._generate_fallback_analysis(message)
    
    def _generate_fallback_analysis(self, message: str) -> Dict:
        """生成回退分析结果"""
        # 简单的规则基础分析
        content_score = 5
        if any(keyword in message.lower() for keyword in ["?", "？", "怎么", "为什么", "帮助"]):
            content_score = 7
        if len(message) < 5:
            content_score = 3
        
        return {
            "content_relevance": content_score,
            "reply_willingness": 6,
            "social_appropriateness": 6,
            "timing_appropriateness": 5,
            "reasoning": "使用回退分析策略",
            "key_factors": ["基础规则判断"]
        }
    
    def _make_final_decision(self, state: GroupState, analysis: Dict) -> Dict:
        """做出最终决策 - 优化考虑对话连贯性"""
        # 使用配置中的权重
        weights = self.config.get_weights()

        final_score = 0
        for dimension, weight in weights.items():
            score = analysis.get(dimension, 5)
            final_score += (score / 10) * weight

        # 精力状态影响
        energy_multiplier = 0.5 + (state.energy * 0.5)  # 0.5-1.0
        final_score *= energy_multiplier

        # 对话连贯性加成
        conversation_coherence = analysis.get('conversation_coherence', 0.5)
        coherence_bonus = conversation_coherence * 0.2  # 最高20%加成
        final_score *= (1.0 + coherence_bonus)

        should_reply = final_score > self.config.reply_threshold

        return {
            "should_reply": should_reply,
            "final_score": final_score,
            "energy_multiplier": energy_multiplier,
            "conversation_coherence": conversation_coherence,
            "coherence_bonus": coherence_bonus,
            "threshold_met": final_score > self.config.reply_threshold
        }
    
    async def _fallback_decision(self, group_id: str, message: str) -> Tuple[bool, Dict]:
        """简单的回退决策策略"""
        state = self.get_group_state(group_id)
        
        # 基于精力和消息特征的简单判断
        base_probability = state.energy * 0.4  # 基础概率与精力相关
        
        # 消息特征加分
        if any(trigger in message for trigger in ["?", "？", "帮助", "求助"]):
            base_probability += 0.3
        if len(message) > 20:
            base_probability += 0.1
        
        should_reply = base_probability > 0.5
        
        return should_reply, {
            "decision": "approved" if should_reply else "rejected", 
            "method": "fallback",
            "probability": base_probability,
            "energy": state.energy
        }
    
    def get_stats(self, group_id: str) -> Dict:
        """获取群聊统计信息"""
        if group_id not in self.group_states:
            return {"error": "群聊状态不存在"}
        
        state = self.group_states[group_id]
        current_time = time.time()
        
        return {
            "group_id": group_id,
            "energy": round(state.energy, 2),
            "total_messages_today": state.total_messages_today,
            "bot_replies_today": state.bot_replies_today,
            "reply_rate": round(state.bot_replies_today / max(1, state.total_messages_today), 2),
            "last_reply_ago": round((current_time - state.last_reply_time) / 60, 1) if state.last_reply_time > 0 else "从未回复",
            "recent_messages_count": len(state.recent_messages),
            "config": asdict(self.config)
        }
    
    async def _get_cached_analysis(self, cache_key: str, message: str, persona_name: str, state: GroupState) -> Optional[Dict]:
        """获取缓存的AI分析结果"""
        # 初始化缓存管理器（如果需要）
        if self.global_cache is None:
            self.global_cache = get_global_cache()
            self.analysis_cache_manager = AnalysisCache(self.global_cache)
        
        # 尝试从全局缓存获取
        if self.analysis_cache_manager:
            try:
                cached_analysis = await self.analysis_cache_manager.get_analysis(
                    message, "", persona_name  # 这里可以传入context
                )
                if cached_analysis:
                    return cached_analysis
            except Exception as e:
                logger.error(f"获取全局缓存分析失败: {e}")
        
        # 回退到本地缓存
        try:
            async with self.cache_semaphore:
                if cache_key in self.analysis_cache:
                    cached_data = self.analysis_cache[cache_key]
                    # 检查缓存是否过期（5分钟）
                    if time.time() - cached_data['timestamp'] < 300:
                        return cached_data['analysis']
                    else:
                        # 清理过期缓存
                        del self.analysis_cache[cache_key]
        except Exception as e:
            logger.error(f"获取本地缓存分析失败: {e}")
        return None
    
    async def _cache_analysis(self, cache_key: str, analysis: Dict):
        """缓存AI分析结果"""
        # 缓存到全局缓存
        if self.analysis_cache_manager:
            try:
                # 这里需要传入正确的参数，暂时跳过
                pass
            except Exception as e:
                logger.error(f"缓存到全局缓存失败: {e}")
        
        # 缓存到本地缓存
        try:
            async with self.cache_semaphore:
                self.analysis_cache[cache_key] = {
                    'analysis': analysis,
                    'timestamp': time.time()
                }
                # 限制缓存大小
                if len(self.analysis_cache) > 100:
                    # 删除最旧的缓存
                    oldest_key = min(self.analysis_cache.keys(), 
                                   key=lambda k: self.analysis_cache[k]['timestamp'])
                    del self.analysis_cache[oldest_key]
        except Exception as e:
            logger.error(f"缓存分析失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        self.analysis_cache.clear()
        self.decision_cache.clear()

# 全局心流引擎实例 - 延迟初始化
heartflow_engine = HeartflowEngine()

# 清理函数
def cleanup_heartflow():
    """清理心流引擎资源"""
    if 'heartflow_engine' in globals():
        heartflow_engine.cleanup()
        logger.info("心流引擎资源已清理") 