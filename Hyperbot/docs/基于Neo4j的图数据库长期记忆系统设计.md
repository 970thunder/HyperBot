# 基于Neo4j的图数据库长期记忆系统设计

## 1. 记忆节点设计

### 1.1 核心节点类型

#### USER（用户节点）
- **属性**：
  - `id`: 用户唯一标识符
  - `created_at`: 创建时间
  - `total_conversations`: 总对话数
  - `last_active`: 最后活跃时间
  - `nickname`: 用户昵称（可选）
  - `preferences`: 用户偏好设置（JSON格式）

#### CONVERSATION（对话节点）
- **属性**：
  - `id`: 对话唯一标识符
  - `user_id`: 用户ID
  - `session_id`: 会话ID
  - `user_message`: 用户消息内容
  - `bot_response`: 机器人回复内容
  - `persona_name`: 使用的人设名称
  - `timestamp`: 对话时间戳
  - `user_embedding`: 用户消息的向量嵌入
  - `bot_embedding`: 机器人回复的向量嵌入
  - `message_length`: 消息长度
  - `response_length`: 回复长度
  - `sentiment_score`: 情感分数（可选）

#### TOPIC（话题节点）
- **属性**：
  - `name`: 话题名称
  - `created_at`: 创建时间
  - `frequency`: 讨论频次
  - `last_discussed`: 最后讨论时间
  - `importance`: 重要性评分
  - `category`: 话题分类

#### ENTITY（实体节点）
- **属性**：
  - `name`: 实体名称
  - `type`: 实体类型（persons, locations, objects, times, emotions等）
  - `created_at`: 创建时间
  - `mention_count`: 提及次数
  - `last_mentioned`: 最后提及时间
  - `importance`: 重要性评分
  - `description`: 实体描述

#### PERSONA（人设节点）
- **属性**：
  - `name`: 人设名称
  - `created_at`: 创建时间
  - `usage_count`: 使用次数
  - `last_used`: 最后使用时间
  - `description`: 人设描述
  - `keywords`: 触发关键词

#### EMOTION（情感节点）
- **属性**：
  - `type`: 情感类型（happy, sad, angry, excited等）
  - `intensity`: 情感强度（0-1）
  - `created_at`: 创建时间
  - `context`: 情感上下文

#### PREFERENCE（偏好节点）
- **属性**：
  - `type`: 偏好类型（likes, dislikes, interests等）
  - `content`: 偏好内容
  - `strength`: 偏好强度
  - `created_at`: 创建时间
  - `last_updated`: 最后更新时间

#### EVENT（事件节点）
- **属性**：
  - `id`: 事件唯一标识符
  - `type`: 事件类型
  - `description`: 事件描述
  - `timestamp`: 事件时间
  - `importance`: 重要性评分
  - `participants`: 参与者列表

#### CONTEXT（上下文节点）
- **属性**：
  - `id`: 上下文标识符
  - `type`: 上下文类型（location, time, situation等）
  - `description`: 上下文描述
  - `created_at`: 创建时间

### 1.2 关系类型设计

#### 用户相关关系
- `PARTICIPATED_IN`: 用户参与对话
- `HAS_PREFERENCE`: 用户有偏好
- `EXPERIENCED_EMOTION`: 用户体验情感
- `KNOWS_ABOUT`: 用户了解某个实体

#### 对话相关关系
- `MENTIONED`: 对话中提到实体
- `RELATED_TO`: 对话关联话题
- `USES_PERSONA`: 对话使用人设
- `HAPPENED_IN`: 对话发生在某个上下文
- `TRIGGERED_EMOTION`: 对话触发情感

#### 实体相关关系
- `SIMILAR_TO`: 实体相似性
- `PART_OF`: 实体包含关系
- `CAUSED_BY`: 因果关系
- `ASSOCIATED_WITH`: 关联关系

#### 时间相关关系
- `BEFORE`: 时间先后关系
- `DURING`: 时间包含关系
- `AFTER`: 时间后续关系

### 1.3 关系属性设计

每个关系都包含以下基础属性：
- `timestamp`: 关系创建时间
- `strength`: 关系强度（0-1）
- `confidence`: 置信度（0-1）
- `context`: 关系上下文信息
- `metadata`: 额外元数据（JSON格式）

## 2. 图数据库模式设计

### 2.1 索引策略

#### 唯一性约束
```cypher
CREATE CONSTRAINT user_id_unique FOR (u:USER) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT conversation_id_unique FOR (c:CONVERSATION) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT topic_name_unique FOR (t:TOPIC) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT entity_name_unique FOR (e:ENTITY) REQUIRE e.name IS UNIQUE;
```

#### 性能索引
```cypher
CREATE INDEX conversation_timestamp FOR (c:CONVERSATION) ON (c.timestamp);
CREATE INDEX topic_frequency FOR (t:TOPIC) ON (t.frequency);
CREATE INDEX entity_importance FOR (e:ENTITY) ON (e.importance);
CREATE INDEX user_last_active FOR (u:USER) ON (u.last_active);
```

### 2.2 数据分层策略

#### 热数据层（最近30天）
- 频繁访问的对话记录
- 活跃的话题和实体
- 当前使用的人设

#### 温数据层（30-90天）
- 中等频率访问的历史记录
- 重要的长期记忆
- 用户偏好历史

#### 冷数据层（90天以上）
- 归档的历史对话
- 低频实体和话题
- 统计汇总数据

### 2.3 记忆重要性评分

#### 评分因子
1. **时间衰减因子**：越近的记忆重要性越高
2. **频率因子**：被提及越多的内容重要性越高
3. **情感因子**：情感强烈的对话重要性越高
4. **关联度因子**：与当前话题相关度越高重要性越高
5. **用户反馈因子**：用户明确表示重要的内容

#### 计算公式
```
importance_score = (
    time_factor * 0.3 +
    frequency_factor * 0.25 +
    emotion_factor * 0.2 +
    relevance_factor * 0.15 +
    user_feedback_factor * 0.1
)
```

## 3. 记忆检索策略

### 3.1 多维度检索

#### 语义相似度检索
- 使用向量嵌入计算余弦相似度
- 阈值设置：0.7以上为高相关，0.5-0.7为中等相关

#### 实体匹配检索
- 提取当前消息中的实体
- 查找历史对话中包含相同实体的记录

#### 话题关联检索
- 识别当前对话的话题
- 检索相关话题的历史对话

#### 时间窗口检索
- 优先检索最近的对话记录
- 根据重要性扩展时间窗口

### 3.2 检索优化

#### 缓存策略
- 缓存用户的常用记忆
- 缓存热门话题和实体

#### 分页检索
- 限制单次检索的记录数量
- 按重要性排序返回结果

#### 并行检索
- 同时进行多种检索策略
- 合并和排序检索结果

## 4. 记忆更新和维护

### 4.1 实时更新

#### 对话后处理
- 提取新的实体和话题
- 更新实体的提及次数
- 计算新的关系强度

#### 增量学习
- 根据新对话调整实体重要性
- 更新用户偏好模型
- 优化检索权重

### 4.2 定期维护

#### 记忆整理
- 合并相似的实体和话题
- 清理低质量的记忆节点
- 更新重要性评分

#### 数据清理
- 删除过期的临时数据
- 归档长期不用的记忆
- 压缩历史统计数据

#### 性能优化
- 重建索引
- 优化查询路径
- 调整缓存策略

### 4.3 记忆遗忘机制

#### 自然遗忘
- 基于时间的重要性衰减
- 低频记忆的自动清理
- 冗余信息的合并

#### 主动遗忘
- 用户请求删除特定记忆
- 隐私敏感信息的清理
- 错误记忆的纠正

## 5. 隐私和安全考虑

### 5.1 数据加密
- 敏感信息的加密存储
- 传输过程的加密保护
- 访问权限的严格控制

### 5.2 数据脱敏
- 个人身份信息的匿名化
- 敏感内容的模糊处理
- 可配置的隐私级别

### 5.3 数据治理
- 数据保留期限的设置
- 用户数据的导出和删除
- 合规性检查和审计

这个设计提供了一个完整的图数据库记忆系统架构，支持复杂的记忆存储、检索和维护功能，同时考虑了性能优化和隐私保护。

