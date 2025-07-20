import asyncio
import json
import httpx
from typing import Dict, List, Any
from nonebot import get_driver, logger

# 从全局配置中获取API Key
DEEPSEEK_API_KEY = getattr(get_driver().config, "deepseek_api_key", None)

class KnowledgeProcessor:
    """
    负责处理知识文本，调用大模型提取实体和关系。
    """

    def __init__(self, deepseek_api_key: str):
        if not deepseek_api_key:
            raise ValueError("DeepSeek API Key未配置")
        self.deepseek_api_key = deepseek_api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"

    async def extract_graph_from_text(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        使用DeepSeek从文本中提取实体和关系图谱。

        Args:
            text: 待处理的知识文本。

        Returns:
            一个包含 "nodes" 和 "relations" 的字典。
            例如:
            {
                "nodes": [
                    {"id": "洛阳理工学院", "type": "学校", "properties": {"校区数量": 3}},
                    {"id": "开元校区", "type": "校区"},
                    ...
                ],
                "relations": [
                    {"source": "洛阳理工学院", "target": "开元校区", "type": "拥有"},
                    ...
                ]
            }
        """
        prompt = f"""
        请根据以下原则，从提供的文本中细致地提取知识图谱的节点和关系，并以JSON格式返回。

        **提取原则:**
        1.  **全面识别节点**: 不仅要识别物理实体（如学校、校区、专业），也要识别抽象概念（如校训、办学理念、宗旨、地址等）。
        2.  **属性与关系**:
            - 如果一个信息是另一个实体的主要属性（例如，地址是校区的一个属性），请将其作为`properties`添加到节点中。
            - 如果一个信息本身是一个重要的概念（例如，校训、理念），请将其创建为一个独立的节点，并与主实体建立关系。
        3.  **节点定义**: 每个节点必须有 `id` (唯一标识，通常是名称) 和 `type` (类型)。
        4.  **关系定义**: 每个关系必须有 `source` (源节点id), `target` (目标节点id), 和 `type` (关系类型，如“拥有”、“位于”、“校训是”）。

        **文本:**
        "{text}"

        **JSON输出格式示例 (以洛阳理工学院为例):**
        {{
          "nodes": [
            {{
              "id": "洛阳理工学院",
              "type": "高等院校",
              "properties": {{
                "性质": "河南省属全日制普通本科",
                "根本任务": "落实立德树人"
              }}
            }},
            {{"id": "让政府放心，让社会满意，让学生受益", "type": "办学理念"}},
            {{"id": "致知、致善、致能、致新", "type": "校训"}},
            {{
              "id": "王城校区",
              "type": "校区",
              "properties": {{"地址": "洛阳市洛龙区王城大道90号"}}
            }},
            {{
              "id": "开元校区",
              "type": "校区",
              "properties": {{"地址": "洛阳市洛龙区学子街8号"}}
            }},
            {{
              "id": "九都校区",
              "type": "校区",
              "properties": {{"地址": "洛阳市涧西区九都西路71号"}}
            }},
            {{"id": "计算机专业", "type": "专业"}}
          ],
          "relations": [
            {{"source": "洛阳理工学院", "target": "让政府放心，让社会满意，让学生受益", "type": "秉承"}},
            {{"source": "洛阳理工学院", "target": "致知、致善、致能、致新", "type": "校训是"}},
            {{"source": "洛阳理工学院", "target": "王城校区", "type": "拥有"}},
            {{"source": "洛阳理工学院", "target": "开元校区", "type": "拥有"}},
            {{"source": "洛阳理工学院", "target": "九都校区", "type": "拥有"}},
            {{"source": "计算机专业", "target": "开元校区", "type": "位于"}}
          ]
        }}
        
        请严格按照此格式只返回JSON对象，不要包含任何其他说明或markdown标记。
        """

        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个极其细致的知识图谱构建专家。你的任务是深入分析用户提供的文本，捕捉所有显式和隐式的实体、概念及其关系，并以指定的JSON格式返回。请特别注意提取如校训、理念、地址等细节信息。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 2048
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.api_url, json=data, headers=headers, timeout=60.0)
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 清理和解析JSON
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                
                graph_data = json.loads(content)
                logger.info(f"成功从文本中提取知识图谱: {graph_data}")
                return graph_data

        except httpx.HTTPStatusError as e:
            logger.error(f"调用DeepSeek API时发生HTTP错误: {e.response.status_code} - {e.response.text}")
            raise
        except json.JSONDecodeError:
            logger.error(f"无法解析DeepSeek API返回的JSON: {content}")
            raise
        except Exception as e:
            logger.error(f"提取知识图谱时发生未知错误: {e}")
            raise

# 全局实例
try:
    knowledge_processor = KnowledgeProcessor(deepseek_api_key=DEEPSEEK_API_KEY)
except ValueError as e:
    logger.warning(f"无法初始化KnowledgeProcessor: {e}")
    knowledge_processor = None 