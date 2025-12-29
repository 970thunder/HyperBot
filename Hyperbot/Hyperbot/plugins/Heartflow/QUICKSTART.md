# 🚀 心流机制快速上手指南

## 1️⃣ 安装配置

### 步骤1: 获取API密钥
1. 注册[硅基流动](https://siliconflow.cn/)账号
2. 获取API密钥
3. 设置环境变量：
```bash
export SILICONFLOW_API_KEY="your_api_key_here"
```

### 步骤2: 基础配置
```bash
# 可选：调整回复阈值（默认0.6）
export HEARTFLOW_REPLY_THRESHOLD=0.6

# 可选：设置每小时最大回复数（默认10）
export HEARTFLOW_MAX_REPLIES_PER_HOUR=10
```

## 2️⃣ 启动测试

### 启动机器人
```bash
cd Hyperbot
python -m nonebot2 run
```

### 测试命令
在群聊中发送以下命令测试：

1. **查看心流状态**：
   ```
   心流状态
   ```
   应该看到类似输出：
   ```
   💫 群聊心流状态报告:
   🔋 当前精力: 1.0/1.0
   📊 今日消息: 0 条
   💬 今日回复: 0 次
   ```

2. **测试智能回复**：
   发送以下类型的消息观察回复行为：
   - ✅ `这个问题怎么解决？` (问题类，容易触发)
   - ✅ `有人知道这个吗？` (求助类，容易触发)
   - ❓ `哈哈哈` (简单回应，不易触发)
   - ❓ `。。。` (无意义消息，不易触发)

## 3️⃣ 配置调优

### 场景1: 机器人太活跃
如果机器人回复过于频繁：

```bash
# 提高回复阈值
export HEARTFLOW_REPLY_THRESHOLD=0.7

# 增加回复间隔
export HEARTFLOW_MIN_REPLY_INTERVAL=60

# 减少每小时回复上限
export HEARTFLOW_MAX_REPLIES_PER_HOUR=5
```

### 场景2: 机器人太沉默
如果机器人很少回复：

```bash
# 降低回复阈值
export HEARTFLOW_REPLY_THRESHOLD=0.4

# 缩短回复间隔
export HEARTFLOW_MIN_REPLY_INTERVAL=20

# 增加每小时回复上限
export HEARTFLOW_MAX_REPLIES_PER_HOUR=15
```

### 场景3: 不同群聊环境
```python
# 在代码中针对不同群聊使用不同配置
from Hyperbot.plugins.Heartflow.config import (
    get_active_config,      # 活跃群聊
    get_conservative_config, # 保守群聊
    get_balanced_config     # 平衡配置
)
```

## 4️⃣ 监控运行

### 实时监控
定期在群聊中发送 `心流状态` 查看：
- 🔋 **精力水平**：应该在 0.3-1.0 之间
- 📈 **回复率**：建议控制在 5-15% 之间
- ⏰ **回复间隔**：观察是否合理

### 日志监控
查看机器人日志中的心流相关信息：
```
[INFO] 心流决策：群聊 123456 触发回复 - 通过综合分析
[INFO] 群 123456 精力消耗，当前精力: 0.75
```

## 5️⃣ 常见问题

### ❌ 机器人完全不回复
**检查清单**：
1. API密钥是否正确设置
2. 网络连接是否正常
3. 发送 `心流状态` 查看精力是否过低
4. 尝试降低回复阈值：`export HEARTFLOW_REPLY_THRESHOLD=0.3`

### ❌ 日志显示"心流AI分析失败"
**解决方案**：
1. 检查API密钥有效性
2. 确认网络可访问 `api.siliconflow.cn`
3. 增加超时时间：`export HEARTFLOW_API_TIMEOUT=20`

### ❌ 精力值一直很低
**原因**：回复过于频繁导致精力耗尽
**解决**：
1. 等待精力自然恢复（每分钟少量恢复）
2. 等到第二天精力重置
3. 调整参数减少精力消耗：`export HEARTFLOW_ENERGY_DECAY_RATE=0.1`

## 6️⃣ 进阶使用

### 自定义配置
```python
from Hyperbot.plugins.Heartflow.config import HeartflowConfig

# 创建专门的配置
my_config = HeartflowConfig(
    reply_threshold=0.55,           # 中等回复阈值
    energy_decay_rate=0.12,         # 较少精力消耗
    max_replies_per_hour=8,         # 适中的回复频率
    min_reply_interval=45,          # 适中的间隔
    content_relevance_weight=0.4,   # 更重视内容相关性
    social_appropriateness_weight=0.3  # 重视社交适宜性
)

# 使用自定义配置
from Hyperbot.plugins.Heartflow import HeartflowEngine
engine = HeartflowEngine(my_config)
```

### 性能优化
1. **减少API调用**：适当提高 `HEARTFLOW_MIN_REPLY_INTERVAL`
2. **降低延迟**：减少 `HEARTFLOW_API_TIMEOUT`（但要确保网络稳定）
3. **节省精力**：降低 `HEARTFLOW_ENERGY_DECAY_RATE`

## 7️⃣ 最佳实践

### 推荐配置（大多数场景）
```bash
export SILICONFLOW_API_KEY="your_api_key"
export HEARTFLOW_REPLY_THRESHOLD=0.6
export HEARTFLOW_MAX_REPLIES_PER_HOUR=10
export HEARTFLOW_MIN_REPLY_INTERVAL=30
export HEARTFLOW_ENERGY_DECAY_RATE=0.15
export HEARTFLOW_ENERGY_RECOVERY_RATE=0.02
```

### 运行建议
1. **初期测试**：使用较低阈值（0.4-0.5）观察效果
2. **稳定运行**：根据群聊活跃度调整到合适阈值（0.5-0.7）
3. **定期检查**：每周查看心流状态，调整参数
4. **群聊区分**：不同性质的群聊使用不同配置

### 监控指标
- **回复率**: 5-15% 为佳
- **精力水平**: 保持在 0.3 以上
- **用户反馈**: 观察群友对机器人回复的反应

---

🎉 **恭喜！** 你已经掌握了心流机制的基础使用方法。根据实际情况调整配置，享受智能的群聊机器人体验吧！

如有问题，请查看完整的 [README.md](./README.md) 文档或提交 Issue。 