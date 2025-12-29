"""
群聊上下文管理器
参考DeepSeek多轮对话实现，为每个群聊维护独立的对话上下文
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class GroupContext:
    """群聊上下文数据结构"""
    group_id: str
    messages: List[Dict[str, Any]] = None  # 消息历史
    last_activity: float = 0  # 最后活动时间戳
    session_id: str = ""  # 会话ID
    context_summary: str = ""  # 上下文摘要
    topic: str = ""  # 当前话题
    participants: List[str] = None  # 参与者列表

    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.participants is None:
            self.participants = []

class GroupContextManager:
    """群聊上下文管理器 - 参考DeepSeek多轮对话实现"""

    def __init__(self, max_context_messages: int = 20, context_timeout: int = 3600):
        """
        初始化群聊上下文管理器

        Args:
            max_context_messages: 最大上下文消息数
            context_timeout: 上下文超时时间（秒）
        """
        self.max_context_messages = max_context_messages
        self.context_timeout = context_timeout
        self.group_contexts: Dict[str, GroupContext] = {}

        # 缓存管理
        self.context_cache: Dict[str, Dict] = {}
        self.summary_cache: Dict[str, str] = {}

        # 统计信息
        self.stats = {
            'total_groups': 0,
            'total_messages': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def get_group_context(self, group_id: str) -> GroupContext:
        """获取群聊上下文，如果不存在则创建"""
        if group_id not in self.group_contexts:
            self.group_contexts[group_id] = GroupContext(
                group_id=group_id,
                session_id=f"group_{group_id}_{int(time.time())}"
            )
            self.stats['total_groups'] += 1

        context = self.group_contexts[group_id]
        self._cleanup_old_messages(context)
        return context

    def add_message(self, group_id: str, user_id: str, message: str,
                   nickname: str = "", is_bot_reply: bool = False) -> None:
        """添加消息到群聊上下文"""
        context = self.get_group_context(group_id)

        message_data = {
            "user_id": user_id,
            "nickname": nickname,
            "message": message,
            "timestamp": time.time(),
            "is_bot_reply": is_bot_reply
        }

        # 添加消息
        context.messages.append(message_data)

        # 更新最后活动时间
        context.last_activity = time.time()

        # 添加参与者
        if user_id not in context.participants:
            context.participants.append(user_id)

        # 限制消息数量
        if len(context.messages) > self.max_context_messages:
            context.messages.pop(0)

        self.stats['total_messages'] += 1

        # 清除相关缓存
        cache_key = f"summary_{group_id}"
        if cache_key in self.summary_cache:
            del self.summary_cache[cache_key]

    def get_recent_messages(self, group_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的群聊消息"""
        context = self.get_group_context(group_id)
        return context.messages[-limit:]

    async def generate_context_summary(self, group_id: str) -> str:
        """生成群聊上下文摘要"""
        cache_key = f"summary_{group_id}"

        # 检查缓存
        if cache_key in self.summary_cache:
            self.stats['cache_hits'] += 1
            return self.summary_cache[cache_key]

        self.stats['cache_misses'] += 1

        context = self.get_group_context(group_id)
        if not context.messages:
            return "暂无对话历史"

        # 提取最近消息作为上下文
        recent_messages = context.messages[-8:]  # 最近8条消息

        # 构建摘要
        summary_parts = []
        current_topic = ""

        for msg in recent_messages:
            sender = msg.get('nickname', msg['user_id'])
            content = msg['message']

            # 简单的对话连贯性分析
            if len(summary_parts) > 0:
                last_sender = summary_parts[-1].split(':')[0]
                if sender != last_sender:
                    summary_parts.append(f"{sender}: {content}")
            else:
                summary_parts.append(f"{sender}: {content}")

        summary = "\n".join(summary_parts)

        # 缓存摘要
        self.summary_cache[cache_key] = summary
        return summary

    def get_conversation_flow(self, group_id: str) -> Dict[str, Any]:
        """获取对话流程分析"""
        context = self.get_group_context(group_id)

        if len(context.messages) < 2:
            return {
                "active": False,
                "turn_count": 0,
                "participant_count": len(context.participants),
                "coherence_score": 0.0
            }

        # 分析对话轮次
        turns = 0
        last_sender = None

        for msg in context.messages:
            current_sender = msg['user_id']
            if current_sender != last_sender:
                turns += 1
                last_sender = current_sender

        # 计算连贯性分数
        coherence_score = self._calculate_coherence_score(context.messages)

        return {
            "active": turns >= 3,  # 至少有3轮对话
            "turn_count": turns,
            "participant_count": len(context.participants),
            "coherence_score": coherence_score,
            "last_activity_ago": time.time() - context.last_activity
        }

    def _calculate_coherence_score(self, messages: List[Dict]) -> float:
        """计算对话连贯性分数"""
        if len(messages) < 2:
            return 0.0

        score = 0.5  # 基础分数

        # 检查对话轮次
        turn_changes = 0
        last_sender = None

        for msg in messages:
            if msg['user_id'] != last_sender:
                turn_changes += 1
                last_sender = msg['user_id']

        # 对话轮次越多，连贯性越高
        if turn_changes > 1:
            score += min(0.3, (turn_changes - 1) * 0.1)

        # 检查时间间隔
        time_intervals = []
        for i in range(1, len(messages)):
            time_diff = messages[i]['timestamp'] - messages[i-1]['timestamp']
            time_intervals.append(time_diff)

        if time_intervals:
            avg_interval = sum(time_intervals) / len(time_intervals)
            # 平均间隔越短，连贯性越高
            if avg_interval < 60:  # 1分钟内
                score += 0.2

        return min(1.0, score)

    def _cleanup_old_messages(self, context: GroupContext) -> None:
        """清理过期消息"""
        current_time = time.time()

        # 清理超时的上下文
        if current_time - context.last_activity > self.context_timeout:
            context.messages.clear()
            context.participants.clear()
            context.topic = ""
            context.context_summary = ""
            logger.info(f"群 {context.group_id} 上下文已超时清理")

        # 清理过期的缓存
        current_time = time.time()
        expired_keys = []

        for key in list(self.summary_cache.keys()):
            # 摘要缓存30分钟过期
            if current_time - float(key.split('_')[-1]) > 1800:
                expired_keys.append(key)

        for key in expired_keys:
            del self.summary_cache[key]

    def get_context_for_prompt(self, group_id: str, include_summary: bool = True) -> str:
        """获取用于prompt的上下文格式"""
        context = self.get_group_context(group_id)

        if not context.messages:
            return ""

        # 构建上下文字符串
        context_parts = []

        if include_summary:
            # 异步获取摘要
            try:
                summary = asyncio.run(self.generate_context_summary(group_id))
                if summary and summary != "暂无对话历史":
                    context_parts.append(f"【群聊上下文摘要】\n{summary}")
            except Exception as e:
                logger.warning(f"获取群聊摘要失败: {e}")

        # 添加最近消息
        recent_messages = context.messages[-5:]  # 最近5条消息
        if recent_messages:
            message_lines = []
            for msg in recent_messages:
                sender = msg.get('nickname', msg['user_id'])
                message_lines.append(f"{sender}: {msg['message']}")

            context_parts.append(f"【最近对话】\n" + "\n".join(message_lines))

        # 添加对话流程分析
        flow = self.get_conversation_flow(group_id)
        if flow['active']:
            context_parts.append(f"【对话状态】正在进行中，参与者: {flow['participant_count']}人")

        return "\n\n".join(context_parts) if context_parts else ""

    def clear_group_context(self, group_id: str) -> bool:
        """清空群聊上下文"""
        if group_id in self.group_contexts:
            del self.group_contexts[group_id]

            # 清理相关缓存
            cache_key = f"summary_{group_id}"
            if cache_key in self.summary_cache:
                del self.summary_cache[cache_key]

            logger.info(f"群 {group_id} 上下文已清空")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        active_groups = sum(1 for ctx in self.group_contexts.values()
                          if time.time() - ctx.last_activity < self.context_timeout)

        return {
            **self.stats,
            'active_groups': active_groups,
            'inactive_groups': len(self.group_contexts) - active_groups,
            'cache_size': len(self.summary_cache)
        }

    def cleanup_all_contexts(self) -> int:
        """清理所有过期的群聊上下文"""
        current_time = time.time()
        expired_groups = []

        for group_id, context in self.group_contexts.items():
            if current_time - context.last_activity > self.context_timeout:
                expired_groups.append(group_id)

        for group_id in expired_groups:
            self.clear_group_context(group_id)

        return len(expired_groups)

# 全局上下文管理器实例
global_group_context_manager = GroupContextManager()