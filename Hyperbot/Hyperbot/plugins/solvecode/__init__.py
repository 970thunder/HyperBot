import asyncio
import os
import httpx
import json
import re
from typing import Optional
from nonebot import on_message, get_driver, logger
from nonebot.adapters.onebot.v11 import Bot, MessageEvent
from nonebot.plugin import PluginMetadata

__plugin_meta__ = PluginMetadata(
    name="代码解决助手",
    description="专业的代码问题分析和解决方案",
    usage="发送 /代码 + 问题描述，获取专业的代码解决方案",
)

# 获取配置
driver = get_driver()
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V3.2-Exp"

# 创建消息处理器 - 使用高优先级和block=True拦截代码相关消息
code_handler = on_message(priority=20, block=True)


@code_handler.handle()
async def handle_code_message(bot: Bot, event: MessageEvent):
    """处理 /代码 开头的消息"""
    user_id = str(event.user_id)
    message_text = event.get_plaintext().strip()
    
    # 检查是否是 /代码 开头的消息
    if not message_text.startswith("/代码"):
        return
    
    # 提取问题描述
    problem = message_text.replace("/代码", "", 1).strip()
    if not problem:
        await bot.send(event, "请提供代码问题描述，格式：/代码 + 问题描述")
        return
    
    logger.info(f"[solvecode] 收到代码问题 - 用户 {user_id}: {problem[:100]}...")
    
    # 调用API生成解决方案
    try:
        response = await generate_code_solution(problem)
        if response:
            # 分割并发送多个消息
            await send_split_messages(bot, event, response)
            logger.info(f"[solvecode] 已成功回复代码问题 - 用户 {user_id}")
        else:
            await bot.send(event, "抱歉，生成解决方案时出现问题，请稍后再试。")
            logger.error(f"[solvecode] API调用失败 - 用户 {user_id}")
    except Exception as e:
        logger.error(f"[solvecode] 处理代码问题时出错: {e}")
        await bot.send(event, "处理代码问题时出现错误，请稍后再试。")


async def generate_code_solution(problem: str) -> Optional[str]:
    """
    调用硅基流动API生成代码解决方案
    
    Args:
        problem: 代码问题描述
        
    Returns:
        格式化的问题分析和代码解决方案
    """
    if not SILICONFLOW_API_KEY:
        logger.error("[solvecode] 未配置SILICONFLOW_API_KEY")
        return None
    
    # 构建提示词
    system_prompt = """你是一位专业的代码问题解决专家。请为用户的问题提供完整的解决方案。

格式要求：
1. 首先提供"问题分析："部分，清晰说明问题的核心思路和解决方向
2. 然后提供"代码及解决方案："部分，包含完整可运行的代码和详细解释

重要要求：
- 代码必须完整、可运行
- 需要详细解释关键代码的逻辑
- 对于算法题，需要说明时间复杂度和空间复杂度
- 对于编程语言语法题，需要讲解相关概念
- 代码要注重代码风格和最佳实践"""
    
    user_prompt = f"请解决以下代码问题：\n\n{problem}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                SILICONFLOW_API_URL,
                headers={
                    "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 10000  # 确保代码和解释完整
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 提取回复内容
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                
                logger.info(f"[solvecode] API调用成功，生成了 {len(content)} 字符的回复")
                return content
            else:
                logger.error(f"[solvecode] API响应格式异常: {result}")
                return None
                
    except httpx.TimeoutException:
        logger.error("[solvecode] API调用超时")
        return None
    except Exception as e:
        logger.error(f"[solvecode] API调用失败: {e}")
        return None


async def send_split_messages(bot: Bot, event: MessageEvent, content: str):
    """
    将内容分割并分成多次发送
    
    Args:
        bot: Bot 实例
        event: 消息事件
        content: 待发送的内容
    """
    # 先提取所有代码块（在清理 markdown 之前）
    code_blocks = []
    code_pattern = r'```(\w+)?\n?([\s\S]*?)```'
    code_matches = re.findall(code_pattern, content)
    
    # 清理 markdown 格式
    content = clean_markdown(content)
    
    # 分离问题分析部分
    analysis_part = ""
    
    # 尝试找到"问题分析"部分
    if "问题分析" in content:
        parts = content.split("问题分析")
        if len(parts) > 1:
            remaining = parts[1]
            # 尝试找到"代码"的开始
            if "代码" in remaining:
                # 分离分析和代码
                analysis_part = "问题分析" + remaining.split("代码")[0].strip()
            else:
                analysis_part = "问题分析" + remaining
        else:
            analysis_part = content
    else:
        # 如果没有明确的问题分析部分，取前 500 字符作为分析
        analysis_part = content[:500] if len(content) > 500 else content
    
    # 发送问题分析部分
    if analysis_part and analysis_part.strip():
        analysis_part = clean_text(analysis_part)
        if analysis_part.strip() and len(analysis_part) > 20:  # 确保有实际内容
            await bot.send(event, analysis_part)
            await asyncio.sleep(1)  # 短暂延迟，避免消息过快
    
    # 发送代码部分
    if code_matches:
        for i, (lang, code) in enumerate(code_matches):
            if code.strip():
                # 检测编程语言
                language = lang if lang else detect_language(code)
                
                code_text = f"【{language}代码方案】\n\n{code.strip()}"
                
                # 如果代码太长，分割成多段
                if len(code_text) > 2000:
                    await send_long_code(bot, event, code_text)
                else:
                    await bot.send(event, code_text)
                
                if i < len(code_matches) - 1:
                    await asyncio.sleep(1)
    elif "代码" in content or "解法" in content:
        # 如果没有找到代码块，尝试提取文本中的解决方案
        if "代码" in content:
            code_section = content.split("代码")[-1][:3000]
            if code_section.strip():
                await bot.send(event, f"【解决方案】\n\n{code_section}")


def detect_language(code: str) -> str:
    """
    检测编程语言
    
    Args:
        code: 代码文本
        
    Returns:
        语言名称
    """
    if 'int main' in code or '#include' in code:
        return "C/C++"
    elif 'def ' in code or 'import ' in code:
        return "Python"
    elif 'class ' in code and 'public static' in code:
        return "Java"
    elif 'function ' in code or '=>' in code:
        return "JavaScript"
    elif '#include <iostream>' in code:
        return "C++"
    else:
        return "代码"


async def send_long_code(bot: Bot, event: MessageEvent, code_text: str):
    """
    发送超长的代码，自动分割成多个消息
    
    Args:
        bot: Bot 实例
        event: 消息事件
        code_text: 代码文本
    """
    # 按行分割，尽量保持在合理的长度
    lines = code_text.split('\n')
    current_message = ""
    
    for line in lines:
        # 如果加上这一行会超过2000字符，先发送当前消息
        if len(current_message) + len(line) + 1 > 2000 and current_message:
            await bot.send(event, current_message.strip())
            await asyncio.sleep(0.5)
            current_message = line + '\n'
        else:
            current_message += line + '\n'
    
    # 发送最后一部分
    if current_message.strip():
        await bot.send(event, current_message.strip())


def clean_markdown(text: str) -> str:
    """
    清理 markdown 格式符号
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 移除代码块标记
    text = text.replace('```python', '').replace('```cpp', '').replace('```c', '')
    text = text.replace('```java', '').replace('```javascript', '').replace('```js', '')
    text = text.replace('```py', '').replace('```', '')
    
    # 移除粗体和斜体
    text = text.replace('**', '').replace('*', '')
    
    # 移除标题标记
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # 移除链接
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # 移除图片
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
    
    # 清理多余的空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def clean_text(text: str) -> str:
    """
    清理文本，移除多余的空格和换行
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 移除多个连续空格
    text = re.sub(r' +', ' ', text)
    # 移除多个连续换行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


# 启动检查
@driver.on_startup
async def startup_check():
    """启动时检查配置"""
    if not SILICONFLOW_API_KEY:
        logger.warning("[solvecode] 未设置SILICONFLOW_API_KEY环境变量，代码解决功能将无法使用")
    else:
        logger.info(f"[solvecode] 代码解决助手已启用，API密钥: {SILICONFLOW_API_KEY[:8]}...")
        logger.info(f"[solvecode] 使用模型: {MODEL}")
