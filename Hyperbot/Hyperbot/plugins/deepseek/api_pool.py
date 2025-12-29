"""
API并发池管理器
支持无限制并发调用，优化API响应性能
"""

import asyncio
import time
import httpx
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from nonebot import logger
import json
from functools import wraps


@dataclass
class APICall:
    """API调用请求"""
    url: str
    headers: Dict[str, str]
    data: Dict[str, Any]
    timeout: float = 30.0
    retry_count: int = 3
    call_id: str = ""


class APIPool:
    """API并发池管理器"""
    
    def __init__(self, max_concurrent: int = 100, max_connections: int = 200):
        """
        初始化API池
        
        Args:
            max_concurrent: 最大并发调用数
            max_connections: 最大连接数
        """
        self.max_concurrent = max_concurrent
        self.max_connections = max_connections
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client_pool: Optional[httpx.AsyncClient] = None
        self.active_calls: Dict[str, asyncio.Task] = {}
        self.call_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'average_response_time': 0.0
        }
        self.response_times: List[float] = []
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._init_client()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def _init_client(self):
        """初始化HTTP客户端"""
        if not self.client_pool:
            limits = httpx.Limits(
                max_keepalive_connections=self.max_connections,
                max_connections=self.max_connections
            )
            timeout = httpx.Timeout(
                connect=3.0,    # 连接超时从5秒减少到3秒
                read=15.0,     # 读取超时从30秒减少到15秒
                write=5.0,     # 写入超时从10秒减少到5秒
                pool=30.0      # 连接池超时从60秒减少到30秒
            )
            self.client_pool = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                http2=True  # 启用HTTP/2支持
            )
    
    async def call_api(self, api_call: APICall) -> Dict[str, Any]:
        """
        执行API调用
        
        Args:
            api_call: API调用请求
            
        Returns:
            API响应结果
        """
        call_id = api_call.call_id or f"call_{int(time.time() * 1000)}_{id(api_call)}"
        
        async with self.semaphore:
            return await self._execute_call(api_call, call_id)
    
    async def _execute_call(self, api_call: APICall, call_id: str) -> Dict[str, Any]:
        """执行单个API调用"""
        start_time = time.time()
        self.call_stats['total_calls'] += 1
        
        try:
            await self._init_client()
            
            # 执行HTTP请求
            response = await self.client_pool.post(
                api_call.url,
                json=api_call.data,
                headers=api_call.headers
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            # 记录成功统计
            response_time = time.time() - start_time
            self._record_response_time(response_time)
            self.call_stats['successful_calls'] += 1
            
            logger.debug(f"API调用成功: {call_id}, 耗时: {response_time:.3f}s")
            
            return {
                'success': True,
                'data': result,
                'response_time': response_time,
                'call_id': call_id
            }
            
        except Exception as e:
            # 记录失败统计
            response_time = time.time() - start_time
            self.call_stats['failed_calls'] += 1
            
            logger.error(f"API调用失败: {call_id}, 错误: {e}, 耗时: {response_time:.3f}s")
            
            return {
                'success': False,
                'error': str(e),
                'response_time': response_time,
                'call_id': call_id
            }
    
    async def call_multiple_apis(self, api_calls: List[APICall]) -> List[Dict[str, Any]]:
        """
        并发执行多个API调用
        
        Args:
            api_calls: API调用列表
            
        Returns:
            API响应结果列表
        """
        tasks = []
        for i, api_call in enumerate(api_calls):
            api_call.call_id = api_call.call_id or f"batch_call_{i}"
            task = asyncio.create_task(self.call_api(api_call))
            tasks.append(task)
            self.active_calls[api_call.call_id] = task
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({
                        'success': False,
                        'error': str(result),
                        'response_time': 0.0,
                        'call_id': 'unknown'
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        finally:
            # 清理活跃调用记录
            for api_call in api_calls:
                self.active_calls.pop(api_call.call_id, None)
    
    def _record_response_time(self, response_time: float):
        """记录响应时间"""
        self.response_times.append(response_time)
        
        # 保持最近1000次调用的记录
        if len(self.response_times) > 1000:
            self.response_times.pop(0)
        
        # 更新平均响应时间
        self.call_stats['average_response_time'] = sum(self.response_times) / len(self.response_times)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取API池统计信息"""
        success_rate = 0.0
        if self.call_stats['total_calls'] > 0:
            success_rate = self.call_stats['successful_calls'] / self.call_stats['total_calls']
        
        return {
            'total_calls': self.call_stats['total_calls'],
            'successful_calls': self.call_stats['successful_calls'],
            'failed_calls': self.call_stats['failed_calls'],
            'success_rate': success_rate,
            'average_response_time': self.call_stats['average_response_time'],
            'active_calls': len(self.active_calls),
            'max_concurrent': self.max_concurrent,
            'max_connections': self.max_connections
        }
    
    async def close(self):
        """关闭API池"""
        # 取消所有活跃调用
        for call_id, task in self.active_calls.items():
            if not task.done():
                task.cancel()
                logger.info(f"取消活跃API调用: {call_id}")
        
        self.active_calls.clear()
        
        # 关闭HTTP客户端
        if self.client_pool:
            await self.client_pool.aclose()
            self.client_pool = None
        
        logger.info("API池已关闭")


class DeepSeekAPIManager:
    """DeepSeek API管理器"""
    
    def __init__(self, api_key: str, api_pool: Optional[APIPool] = None):
        self.api_key = api_key
        self.api_pool = api_pool or APIPool(max_concurrent=50, max_connections=100)
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        
    async def generate_response(self, messages: List[Dict[str, str]],
                              model: str = "deepseek-reasoner",
                              temperature: float = 0.7,
                              max_tokens: int = 1500) -> str:
        """生成回复"""
        api_call = APICall(
            url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            data={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        
        result = await self.api_pool.call_api(api_call)
        
        if result['success']:
            return result['data']['choices'][0]['message']['content']
        else:
            logger.error(f"DeepSeek API调用失败: {result['error']}")
            return ""
    
    async def batch_generate_responses(self, message_batches: List[List[Dict[str, str]]],
                                     model: str = "deepseek-reasoner",
                                     temperature: float = 0.7,
                                     max_tokens: int = 1500) -> List[str]:
        """批量生成回复"""
        api_calls = []
        for messages in message_batches:
            api_call = APICall(
                url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                data={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            api_calls.append(api_call)
        
        results = await self.api_pool.call_multiple_apis(api_calls)
        
        responses = []
        for result in results:
            if result['success']:
                response_text = result['data']['choices'][0]['message']['content']
                responses.append(response_text)
            else:
                logger.error(f"批量API调用失败: {result['error']}")
                responses.append("")
        
        return responses


class SiliconFlowAPIManager:
    """SiliconFlow API管理器"""
    
    def __init__(self, api_key: str, api_pool: Optional[APIPool] = None):
        self.api_key = api_key
        self.api_pool = api_pool or APIPool(max_concurrent=30, max_connections=60)
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        
    async def analyze_message(self, message: str, context: str = "", 
                            persona_name: str = "", energy: float = 1.0,
                            replies_today: int = 0) -> Dict[str, Any]:
        """分析消息"""
        prompt = f"""你是一个聊天群的智能回复决策助手。请分析以下消息是否值得机器人回复。

当前群聊上下文：
{context}

新消息: {message}
机器人当前人设: {persona_name or "默认"}
当前精力值: {energy:.2f}/1.0
今日已回复: {replies_today}次

请从以下4个维度评分(0-10分)，并给出简短理由：
1. 内容相关度：消息是否有价值、有趣、适合回复
2. 回复意愿：基于当前精力状态的回复意愿
3. 社交适宜性：回复是否符合群聊氛围
4. 时机恰当性：考虑频率控制和时间间隔

**重要**: 请直接返回JSON格式，不要使用markdown包装：
{{
  "content_relevance": 整数(0-10),
  "reply_willingness": 整数(0-10),
  "social_appropriateness": 整数(0-10),
  "timing_appropriateness": 整数(0-10),
  "reasoning": "简短的分析理由",
  "key_factors": ["关键因素1", "关键因素2"]
}}"""
        
        api_call = APICall(
            url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            data={
                "model": "Qwen/QwQ-32B",
                "max_tokens": 300,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )
        
        result = await self.api_pool.call_api(api_call)
        
        if result['success']:
            try:
                ai_response = result['data']['choices'][0]['message']['content']
                # 解析JSON响应
                analysis = json.loads(ai_response)
                return analysis
            except json.JSONDecodeError as e:
                logger.error(f"SiliconFlow API响应解析失败: {e}")
                return self._generate_fallback_analysis(message)
        else:
            logger.error(f"SiliconFlow API调用失败: {result['error']}")
            return self._generate_fallback_analysis(message)
    
    def _generate_fallback_analysis(self, message: str) -> Dict[str, Any]:
        """生成回退分析结果"""
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


# 全局API池实例
_global_api_pool: Optional[APIPool] = None
_deepseek_manager: Optional[DeepSeekAPIManager] = None
_siliconflow_manager: Optional[SiliconFlowAPIManager] = None


async def get_global_api_pool() -> APIPool:
    """获取全局API池"""
    global _global_api_pool
    if _global_api_pool is None:
        _global_api_pool = APIPool(max_concurrent=100, max_connections=200)
        await _global_api_pool._init_client()
    return _global_api_pool


async def get_deepseek_manager(api_key: str) -> DeepSeekAPIManager:
    """获取DeepSeek API管理器"""
    global _deepseek_manager
    if _deepseek_manager is None:
        api_pool = await get_global_api_pool()
        _deepseek_manager = DeepSeekAPIManager(api_key, api_pool)
    return _deepseek_manager


async def get_siliconflow_manager(api_key: str) -> SiliconFlowAPIManager:
    """获取SiliconFlow API管理器"""
    global _siliconflow_manager
    if _siliconflow_manager is None:
        api_pool = await get_global_api_pool()
        _siliconflow_manager = SiliconFlowAPIManager(api_key, api_pool)
    return _siliconflow_manager


async def cleanup_global_api_pool():
    """清理全局API池"""
    global _global_api_pool, _deepseek_manager, _siliconflow_manager
    
    if _global_api_pool:
        await _global_api_pool.close()
        _global_api_pool = None
    
    _deepseek_manager = None
    _siliconflow_manager = None
    
    logger.info("全局API池已清理")


# 装饰器：自动管理API调用
def api_call_monitor(func: Callable):
    """API调用监控装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"API调用 {func.__name__} 成功，耗时: {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"API调用 {func.__name__} 失败，耗时: {duration:.3f}s，错误: {e}")
            raise
    return wrapper
