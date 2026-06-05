import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 读取环境变量
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")

if not api_key:
    raise ValueError("❌ 没读取到 API KEY，请检查 .env 文件")

# 初始化模型
#temperature 越低，回答越稳、越保守、越像标准答案
#temperature 越高，回答越活、越随机、越有创造性
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url=base_url,
    temperature=0.7,
)

# 调用模型
response = llm.invoke("你好，用一句话告诉我你是谁")

print("✅ 模型回复：")
print(response.content)