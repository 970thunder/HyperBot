import asyncio
from nonebot import on_command, logger
from nonebot.params import CommandArg
from nonebot.adapters.onebot.v11 import Bot, MessageEvent, Message

# 从 .knowledge 模块导入知识处理器
from .knowledge_processor import knowledge_processor

# 从 .. 目录导入 deepseek 插件模块，而不是直接导入变量
from .. import deepseek

__plugin_meta__ = {
    "name": "知识库收录",
    "description": "通过指令将结构化知识存入Neo4j知识图谱",
    "usage": "使用 /收录知识 [要录入的文本] 来添加新知识",
}

# 创建命令处理器
knowledge_handler = on_command("收录知识", aliases={"knowledge", "录入知识"}, priority=10, block=True)

@knowledge_handler.handle()
async def handle_knowledge_command(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    """
    处理 "/收录知识" 命令。
    """
    knowledge_text = args.extract_plain_text().strip()

    if not knowledge_text:
        await knowledge_handler.finish("请输入需要收录的知识内容。用法：/收录知识 [内容]")

    if not knowledge_processor:
        await knowledge_handler.finish("错误：知识处理模块未正确初始化，请检查API Key配置。")

    # 在运行时动态访问 memory_system 实例
    if not deepseek.memory_system:
        await knowledge_handler.finish("错误：记忆系统未初始化，无法存储知识。请检查deepseek插件的Neo4j和API Key配置。")

    await bot.send(event, "正在分析知识并提取实体关系，请稍候...")
    
    try:
        # 1. 使用KnowledgeProcessor从文本中提取图数据
        graph_data = await knowledge_processor.extract_graph_from_text(knowledge_text)

        if not graph_data or not graph_data.get("nodes"):
            await knowledge_handler.finish("未能从文本中提取有效的知识节点，请检查您的输入。")

        # 2. 使用MemoryGraph将图数据存入Neo4j
        success = await deepseek.memory_system.store_knowledge_graph(graph_data)

        if success:
            nodes_count = len(graph_data.get('nodes', []))
            relations_count = len(graph_data.get('relations', []))
            reply = f"知识收录成功！\n- 新增/更新节点: {nodes_count}个\n- 新增/更新关系: {relations_count}个"
            await bot.send(event, reply)
        else:
            await bot.send(event, "知识存储失败，请查看后台日志了解详情。")

    except Exception as e:
        logger.error(f"处理知识收录命令时出错: {e}")
        await bot.send(event, f"处理失败，发生内部错误: {str(e)}")

# 启动时检查依赖
@knowledge_handler.handle()
async def _startup_check():
    if not knowledge_processor:
        logger.warning("Knowledge plugin: KnowledgeProcessor 未初始化, 可能是缺少 DEEPSEEK_API_KEY。")
    # 同样在启动检查时也使用动态访问
    if not deepseek.memory_system:
        logger.warning("Knowledge plugin: MemoryGraph (memory_system) 未初始化。") 