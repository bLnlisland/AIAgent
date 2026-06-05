# demo4_langgraph_agent_toolnode_fixed.py
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import datetime

# 读取 .env
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")


#定义 State
class ChatState(TypedDict):
    message: str
    response: str
    history: List[dict]
    weather_info: str   # 存放天气信息
    time_info: str      # 存放时间信息


# 定义工具函数
def get_weather() -> str:
    """返回天气"""
    return f"今天武汉天气晴，28°C"

def get_time() -> str:
    """返回当前时间"""
    return f"当前时间: {datetime.datetime.now().strftime('%H:%M:%S')}"


#节点函数
def greet_node(state: ChatState) -> dict:
    """可选的问候节点，不做实质修改，仅打印日志"""
    print(f"[greet_node] 收到消息: {state['message']}")
    # 返回空字典表示不更新状态
    return {}

def weather_node(state: ChatState) -> dict:
    """调用天气工具，将结果存入 state['weather_info']"""
    result = get_weather()
    print(f"[weather_node] 获取天气: {result}")
    return {"weather_info": result}

def time_node(state: ChatState) -> dict:
    """调用时间工具，将结果存入 state['time_info']"""
    result = get_time()
    print(f"[time_node] 获取时间: {result}")
    return {"time_info": result}

def llm_node(state: ChatState) -> dict:
    """
    调用 DeepSeek LLM，将工具的输出作为上下文注入系统消息
    """
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url=base_url
    )

    messages = []

    # 历史对话
    messages.extend(state["history"])

    # 组装工具信息
    tool_context = ""
    if state.get("weather_info"):
        tool_context += f"天气信息: {state['weather_info']}\n"
    if state.get("time_info"):
        tool_context += f"时间信息: {state['time_info']}\n"

    if tool_context:
        # 将工具信息作为 system 消息插入
        messages.append({
            "role": "system",
            "content": f"以下是已经获取到的实时信息，请基于这些信息回答用户的问题：\n{tool_context}"
        })

    # 当前用户消息
    messages.append({
        "role": "user",
        "content": state["message"]
    })

    response = llm.invoke(messages)
    print(f"[llm_node] DeepSeek 输出: {response.content}")
    return {"response": response.content}


# 构建图
builder = StateGraph(ChatState)

builder.add_node("greet", greet_node)
builder.add_node("weather", weather_node)
builder.add_node("time", time_node)
builder.add_node("llm", llm_node)

# 定义工作流: START -> greet -> 同时运行 weather 和 time -> llm -> END
builder.add_edge(START, "greet")
builder.add_edge("greet", "weather")
builder.add_edge("greet", "time")
# 等待两个工具都完成后，再进入 llm（这里使用普通边即可，因为它们并行但不阻塞，但实际 Graph 会等待所有前驱节点完成后才执行下一个）
# 为了让 weather 和 time 都完成后才进入 llm，需要添加边：
builder.add_edge("weather", "llm")
builder.add_edge("time", "llm")
builder.add_edge("llm", END)

graph = builder.compile()

# REPL 循环
history = []

print("=== LangGraph DeepSeek Agent Demo (Tool Nodes) ===")
print("输入消息，AI会调用天气和时间工具（每次都会调用），然后回答。输入 exit 退出。")

while True:
    user_input = input("\n你: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("程序结束")
        break

    state = {
        "message": user_input,
        "response": "",
        "history": history,
        "weather_info": "",
        "time_info": ""
    }

    result = graph.invoke(state)

    print("\nAI:")
    print(result["response"])

    # 更新历史
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": result["response"]})