#!/usr/bin/env python3
"""
硅基流动API连通性测试脚本
用于验证API密钥、网络连接和模型响应
"""

import os
import json
import time
import asyncio
import httpx
from pathlib import Path

# 尝试加载.env文件
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ 已加载.env文件")
except ImportError:
    print("⚠️  未安装python-dotenv，直接使用环境变量")

class SiliconFlowTester:
    def __init__(self):
        self.api_key = os.getenv("SILICONFLOW_API_KEY", "")
        self.base_url = "https://api.siliconflow.cn/v1"
        self.timeout = 30
        
    def check_api_key(self):
        """检查API密钥"""
        print("\n🔑 检查API密钥...")
        if not self.api_key:
            print("❌ 未找到SILICONFLOW_API_KEY环境变量")
            print("请在.env文件中设置: SILICONFLOW_API_KEY=your_api_key")
            return False
        
        print(f"✅ API密钥已设置: {self.api_key[:8]}...{self.api_key[-4:]}")
        return True
    
    async def test_network_connection(self):
        """测试网络连接"""
        print("\n🌐 测试网络连接...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    print("✅ 网络连接正常")
                    models = response.json()
                    print(f"📋 可用模型数量: {len(models.get('data', []))}")
                    return True
                elif response.status_code == 401:
                    print("❌ API密钥无效 (401 Unauthorized)")
                    return False
                else:
                    print(f"⚠️  连接异常，状态码: {response.status_code}")
                    print(f"响应: {response.text}")
                    return False
                    
        except httpx.TimeoutException:
            print("❌ 连接超时，请检查网络")
            return False
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    async def test_chat_completion(self):
        """测试聊天完成API"""
        print("\n💬 测试聊天完成API...")
        
        test_messages = [
            {
                "role": "user",
                "content": "请简单回复：你好"
            }
        ]
        
        payload = {
            "model": "Qwen/QwQ-32B",
            "messages": test_messages,
            "max_tokens": 50,
            "temperature": 0.3
        }
        
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=self.timeout
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    print("✅ 聊天完成API测试成功")
                    print(f"⏱️  响应时间: {elapsed:.2f}秒")
                    print(f"🤖 模型回复: {content}")
                    print(f"📊 使用tokens: {result.get('usage', {})}")
                    return True
                    
                else:
                    print(f"❌ API调用失败，状态码: {response.status_code}")
                    print(f"错误信息: {response.text}")
                    return False
                    
        except httpx.TimeoutException:
            print(f"❌ API调用超时 (>{self.timeout}秒)")
            return False
        except Exception as e:
            print(f"❌ API调用异常: {e}")
            return False
    
    async def test_heartflow_scenario(self):
        """测试心流机制场景"""
        print("\n🧠 测试心流判断场景...")
        
        test_scenarios = [
            {
                "name": "问题类消息",
                "message": "这个问题怎么解决？",
                "expected": "高分"
            },
            {
                "name": "无意义消息", 
                "message": "...",
                "expected": "低分"
            },
            {
                "name": "求助类消息",
                "message": "有人知道这个吗？",
                "expected": "高分"
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n测试场景: {scenario['name']}")
            print(f"消息内容: {scenario['message']}")
            
            prompt = f"""你是一个聊天群的智能回复决策助手。请分析以下消息是否值得机器人回复。

新消息: {scenario['message']}
机器人当前人设: 默认
当前精力值: 0.8/1.0
今日已回复: 5次

请从以下4个维度评分(0-10分)：
1. 内容相关度：消息是否有价值、有趣、适合回复
2. 回复意愿：基于当前精力状态的回复意愿  
3. 社交适宜性：回复是否符合群聊氛围
4. 时机恰当性：考虑频率控制和时间间隔

请严格按照以下JSON格式回复：
{{
  "content_relevance": 数字,
  "reply_willingness": 数字,
  "social_appropriateness": 数字,
  "timing_appropriateness": 数字,
  "reasoning": "简短的分析理由"
}}"""

            payload = {
                "model": "Qwen/QwQ-32B",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.3
            }
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        ai_response = result["choices"][0]["message"]["content"]
                        
                        try:
                            # 尝试解析JSON - 使用与心流引擎相同的逻辑
                            try:
                                # 首先尝试直接解析JSON
                                analysis = json.loads(ai_response)
                            except json.JSONDecodeError:
                                # 如果直接解析失败，尝试提取markdown中的JSON
                                import re
                                # 查找```json...```或```...```包装的JSON
                                json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', ai_response, re.DOTALL)
                                if json_match:
                                    json_str = json_match.group(1).strip()
                                    analysis = json.loads(json_str)
                                    print(f"✅ 从markdown中成功解析JSON")
                                else:
                                    # 尝试查找第一个完整的JSON对象
                                    json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                                    if json_match:
                                        json_str = json_match.group(0)
                                        analysis = json.loads(json_str)
                                        print(f"✅ 成功提取JSON对象")
                                    else:
                                        raise ValueError("无法找到有效的JSON内容")
                            
                            print(f"✅ 分析成功:")
                            print(f"   内容相关度: {analysis.get('content_relevance', 'N/A')}")
                            print(f"   回复意愿: {analysis.get('reply_willingness', 'N/A')}")
                            print(f"   社交适宜性: {analysis.get('social_appropriateness', 'N/A')}")
                            print(f"   时机恰当性: {analysis.get('timing_appropriateness', 'N/A')}")
                            print(f"   分析理由: {analysis.get('reasoning', 'N/A')}")
                            
                            # 计算总分
                            total_score = sum([
                                analysis.get('content_relevance', 0) * 0.3,
                                analysis.get('reply_willingness', 0) * 0.25,
                                analysis.get('social_appropriateness', 0) * 0.25,
                                analysis.get('timing_appropriateness', 0) * 0.2
                            ]) / 10
                            
                            print(f"   综合得分: {total_score:.2f}/1.0")
                            
                        except (json.JSONDecodeError, ValueError) as e:
                            print(f"⚠️  JSON解析失败: {e}")
                            print(f"原始响应: {ai_response}")
                            
                    else:
                        print(f"❌ 场景测试失败: {response.status_code}")
                        
            except Exception as e:
                print(f"❌ 场景测试异常: {e}")
                
            # 添加延迟避免频率限制
            await asyncio.sleep(1)
    
    def print_env_info(self):
        """打印环境信息"""
        print("\n📋 环境信息:")
        print(f"   Python: {os.sys.version}")
        print(f"   工作目录: {Path.cwd()}")
        print(f"   .env文件: {'存在' if Path('.env').exists() else '不存在'}")
        
        # 检查相关环境变量
        env_vars = [
            "SILICONFLOW_API_KEY",
            "HEARTFLOW_REPLY_THRESHOLD", 
            "HEARTFLOW_MAX_REPLIES_PER_HOUR"
        ]
        
        print("   环境变量:")
        for var in env_vars:
            value = os.getenv(var)
            if var == "SILICONFLOW_API_KEY" and value:
                value = f"{value[:8]}...{value[-4:]}"
            print(f"     {var}: {value or '未设置'}")
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 硅基流动API连通性测试开始")
        print("=" * 50)
        
        self.print_env_info()
        
        # 检查API密钥
        if not self.check_api_key():
            return False
        
        # 测试网络连接
        if not await self.test_network_connection():
            return False
        
        # 测试基础聊天
        if not await self.test_chat_completion():
            return False
        
        # 测试心流场景
        await self.test_heartflow_scenario()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试完成！硅基流动API连接正常")
        return True

async def main():
    """主函数"""
    tester = SiliconFlowTester()
    
    try:
        success = await tester.run_all_tests()
        if success:
            print("\n✅ 测试结果: 连接正常，可以使用心流机制")
        else:
            print("\n❌ 测试结果: 连接异常，请检查配置")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现异常: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 