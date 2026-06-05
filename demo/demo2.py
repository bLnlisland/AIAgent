import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent

# 1. 读取 .env 文件
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")

if not api_key:
    raise ValueError("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件")

# 2. 创建模型对象
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url=base_url,
    temperature=0.5,
)

def multiply(a:int, b:int)->int:
    "计算两个数的乘积"
    return a * b

agent = create_agent(
    model=llm,
    tools=[multiply],
    system_prompt="""
你是一名擅长辅导大学生的计算机课程助教。
回答要清晰、准确、分点说明。

如果是计算问题，可以调用工具；
如果不是，就直接解释知识点。
"""
)
# 5. 循环交互
print("=== AI 学习助手已启动 ===")

while True:
    topic = input("\n请输入知识点（输入 q 退出）：").strip()

    if topic.lower() == "q":
        print("已退出。")
        break

    if not topic:
        print("请输入一个知识点，不要留空。")
        continue

    try:
        response = agent.invoke({
            "messages": [
                {"role": "user", "content": topic}
            ]
        })
        print("\n===== AI讲解结果 =====")
        print(response["messages"][-1].content)
        print("======================")
    except Exception as e:
        print(f"\n调用失败：{e}")