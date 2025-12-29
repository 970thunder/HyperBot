"""
多层缓存管理器
提供内存缓存、Redis缓存和持久化缓存支持
"""

import asyncio
import time
import json
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from nonebot import logger
import threading
from collections import OrderedDict
import weakref


@dataclass
class CacheConfig:
    """缓存配置"""
    max_size: int = 1000
    ttl: int = 3600  # 生存时间（秒）
    cleanup_interval: int = 300  # 清理间隔（秒）
    persistent_cache: bool = True
    cache_dir: str = "./cache"
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379"


class MemoryCache:
    """内存缓存实现"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.access_counts = {}
        self.lock = threading.RLock()
        self._cleanup_task = None
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                # 检查是否过期
                if time.time() - self.timestamps[key] < self.ttl:
                    # 更新访问时间（LRU）
                    self.cache.move_to_end(key)
                    self.access_counts[key] = self.access_counts.get(key, 0) + 1
                    return self.cache[key]
                else:
                    # 过期，删除
                    self._remove_key(key)
        return None
    
    def set(self, key: str, value: Any):
        """设置缓存值"""
        with self.lock:
            # 如果达到最大大小，删除最少使用的项
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            # 启动清理任务
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        with self.lock:
            return self._remove_key(key)
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_counts.clear()
    
    def _remove_key(self, key: str) -> bool:
        """移除缓存键"""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
            del self.access_counts[key]
            return True
        return False
    
    def _evict_lru(self):
        """驱逐最少使用的项"""
        if self.cache:
            # 找到最少使用的项
            lru_key = min(self.access_counts.keys(), 
                         key=lambda k: self.access_counts[k])
            self._remove_key(lru_key)
    
    async def _periodic_cleanup(self):
        """定期清理过期项"""
        try:
            while True:
                await asyncio.sleep(300)  # 5分钟清理一次
                with self.lock:
                    current_time = time.time()
                    expired_keys = [
                        key for key, timestamp in self.timestamps.items()
                        if current_time - timestamp >= self.ttl
                    ]
                    for key in expired_keys:
                        self._remove_key(key)
                    
                    if expired_keys:
                        logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")
        except asyncio.CancelledError:
            pass
        finally:
            self._cleanup_task = None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'ttl': self.ttl,
                'hit_rate': 0.0,  # 需要外部统计
                'access_counts': dict(self.access_counts)
            }


class PersistentCache:
    """持久化缓存实现"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.meta_file = self.cache_dir / "cache_meta.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """加载缓存元数据"""
        if self.meta_file.exists():
            try:
                with open(self.meta_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载缓存元数据失败: {e}")
        return {}
    
    def _save_metadata(self):
        """保存缓存元数据"""
        try:
            with open(self.meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存缓存元数据失败: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key not in self.metadata:
            return None
        
        meta = self.metadata[key]
        cache_file = self.cache_dir / f"{key}.cache"
        
        # 检查是否过期
        if time.time() - meta['timestamp'] >= meta['ttl']:
            self.delete(key)
            return None
        
        # 检查文件是否存在
        if not cache_file.exists():
            self.delete(key)
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"读取缓存文件失败: {e}")
            self.delete(key)
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存值"""
        cache_file = self.cache_dir / f"{key}.cache"
        
        try:
            # 保存数据
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            # 更新元数据
            self.metadata[key] = {
                'timestamp': time.time(),
                'ttl': ttl,
                'size': cache_file.stat().st_size
            }
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"保存缓存文件失败: {e}")
    
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        if key not in self.metadata:
            return False
        
        try:
            # 删除文件
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                cache_file.unlink()
            
            # 删除元数据
            del self.metadata[key]
            self._save_metadata()
            
            return True
        except Exception as e:
            logger.error(f"删除缓存文件失败: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        
        for key, meta in self.metadata.items():
            if current_time - meta['timestamp'] >= meta['ttl']:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.delete(key)
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_size = sum(meta.get('size', 0) for meta in self.metadata.values())
        return {
            'count': len(self.metadata),
            'total_size': total_size,
            'cache_dir': str(self.cache_dir)
        }


class MultiLayerCache:
    """多层缓存管理器"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = MemoryCache(config.max_size, config.ttl)
        self.persistent_cache = PersistentCache(config.cache_dir) if config.persistent_cache else None
        self.stats = {
            'memory_hits': 0,
            'persistent_hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        self._cleanup_task = None
        
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """生成缓存键"""
        # 将所有参数序列化并生成哈希
        key_data = {
            'prefix': prefix,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    async def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        # 确保清理任务已启动
        self.ensure_cleanup_task_started()
        
        self.stats['total_requests'] += 1
        
        # 第一层：内存缓存
        value = self.memory_cache.get(key)
        if value is not None:
            self.stats['memory_hits'] += 1
            return value
        
        # 第二层：持久化缓存
        if self.persistent_cache:
            value = self.persistent_cache.get(key)
            if value is not None:
                self.stats['persistent_hits'] += 1
                # 回填到内存缓存
                self.memory_cache.set(key, value)
                return value
        
        self.stats['misses'] += 1
        return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        if ttl is None:
            ttl = self.config.ttl
        
        # 设置到内存缓存
        self.memory_cache.set(key, value)
        
        # 设置到持久化缓存
        if self.persistent_cache:
            self.persistent_cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        memory_deleted = self.memory_cache.delete(key)
        persistent_deleted = False
        
        if self.persistent_cache:
            persistent_deleted = self.persistent_cache.delete(key)
        
        return memory_deleted or persistent_deleted
    
    async def clear(self):
        """清空所有缓存"""
        self.memory_cache.clear()
        if self.persistent_cache:
            # 清理持久化缓存目录
            for cache_file in self.persistent_cache.cache_dir.glob("*.cache"):
                cache_file.unlink()
            self.persistent_cache.metadata.clear()
            self.persistent_cache._save_metadata()
    
    async def cleanup_expired(self):
        """清理过期缓存"""
        if self.persistent_cache:
            expired_count = self.persistent_cache.cleanup_expired()
            if expired_count > 0:
                logger.info(f"清理了 {expired_count} 个过期持久化缓存项")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.stats['total_requests']
        hit_rate = 0.0
        if total_requests > 0:
            total_hits = self.stats['memory_hits'] + self.stats['persistent_hits']
            hit_rate = total_hits / total_requests
        
        stats = {
            'total_requests': total_requests,
            'memory_hits': self.stats['memory_hits'],
            'persistent_hits': self.stats['persistent_hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'memory_cache': self.memory_cache.get_stats()
        }
        
        if self.persistent_cache:
            stats['persistent_cache'] = self.persistent_cache.get_stats()
        
        return stats
    
    async def start_cleanup_task(self):
        """启动清理任务"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    def ensure_cleanup_task_started(self):
        """确保清理任务已启动（在事件循环可用时调用）"""
        try:
            loop = asyncio.get_running_loop()
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        except RuntimeError:
            pass
    
    async def stop_cleanup_task(self):
        """停止清理任务"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    async def _periodic_cleanup(self):
        """定期清理任务"""
        try:
            while True:
                await asyncio.sleep(self.config.cleanup_interval)
                await self.cleanup_expired()
        except asyncio.CancelledError:
            pass


# 缓存装饰器
def cached(prefix: str, ttl: int = 3600, cache_instance: Optional[MultiLayerCache] = None):
    """缓存装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if cache_instance is None:
                return await func(*args, **kwargs)
            
            # 生成缓存键
            cache_key = cache_instance._generate_key(prefix, *args, **kwargs)
            
            # 尝试从缓存获取
            cached_result = await cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            await cache_instance.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


# 全局缓存实例
_global_cache: Optional[MultiLayerCache] = None


def get_global_cache() -> MultiLayerCache:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        config = CacheConfig(
            max_size=2000,
            ttl=1800,  # 30分钟
            persistent_cache=True,
            cache_dir="./cache"
        )
        _global_cache = MultiLayerCache(config)
        # 延迟启动清理任务，等到事件循环可用时
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(_global_cache.start_cleanup_task())
        except RuntimeError:
            # 没有运行中的事件循环，将在第一次使用时启动
            pass
    return _global_cache


async def cleanup_global_cache():
    """清理全局缓存"""
    global _global_cache
    if _global_cache:
        await _global_cache.stop_cleanup_task()
        await _global_cache.clear()
        _global_cache = None
        logger.info("全局缓存已清理")


# 专用缓存管理器
class EmbeddingCache:
    """嵌入向量缓存"""
    
    def __init__(self, cache: MultiLayerCache):
        self.cache = cache
        self.prefix = "embedding"
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """获取嵌入向量"""
        key = self.cache._generate_key(self.prefix, text)
        return await self.cache.get(key)
    
    async def set_embedding(self, text: str, embedding: List[float]):
        """设置嵌入向量"""
        key = self.cache._generate_key(self.prefix, text)
        await self.cache.set(key, embedding, ttl=7200)  # 2小时


class AnalysisCache:
    """AI分析结果缓存"""
    
    def __init__(self, cache: MultiLayerCache):
        self.cache = cache
        self.prefix = "analysis"
    
    async def get_analysis(self, message: str, context: str, persona: str) -> Optional[Dict[str, Any]]:
        """获取分析结果"""
        key = self.cache._generate_key(self.prefix, message, context, persona)
        return await self.cache.get(key)
    
    async def set_analysis(self, message: str, context: str, persona: str, analysis: Dict[str, Any]):
        """设置分析结果"""
        key = self.cache._generate_key(self.prefix, message, context, persona)
        await self.cache.set(key, analysis, ttl=900)  # 15分钟


class ResponseCache:
    """API响应缓存"""
    
    def __init__(self, cache: MultiLayerCache):
        self.cache = cache
        self.prefix = "response"
    
    async def get_response(self, messages: List[Dict[str, str]], model: str) -> Optional[str]:
        """获取API响应"""
        key = self.cache._generate_key(self.prefix, messages, model)
        return await self.cache.get(key)
    
    async def set_response(self, messages: List[Dict[str, str]], model: str, response: str):
        """设置API响应"""
        key = self.cache._generate_key(self.prefix, messages, model)
        await self.cache.set(key, response, ttl=600)  # 10分钟
