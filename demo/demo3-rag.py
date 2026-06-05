import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# =========================
# 1. 聊天模型：继续用 DeepSeek
# =========================
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

# =========================
# 2. Embedding 模型：本地版
# =========================
# 这里用 HuggingFace 的本地 embedding，不需要 OPENAI_API_KEY，也不用充值。
# 第一次运行时会下载模型，可能稍微慢一点，之后会快很多。
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# 如果以后你想改成 OpenAI embedding，就把上面这一段替换成下面这段：
#
# from langchain_openai import OpenAIEmbeddings
#
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small"
# )
#
# 同时你需要在 .env 里加入：
# OPENAI_API_KEY=你的key
# -------------------------

# =========================
# 3. 准备知识库文本
# =========================
raw_text = """
数据库课程复习资料

第一章：关系模型
关系模型用二维表表示数据，核心概念包括属性、元组、码、域。

第二章：SQL
SQL常见操作包括 SELECT、INSERT、UPDATE、DELETE。
多表查询要重点掌握 JOIN。

第三章：范式
考试重点是 1NF、2NF、3NF，要求会判断函数依赖、部分依赖、传递依赖。

第四章：事务
事务四大特性是 ACID：原子性、一致性、隔离性、持久性。

考试安排
期末考试时间：2026年6月20日
闭卷，考试时长 120 分钟。
"""

docs = [Document(page_content=raw_text)]

# =========================
# 4. 切块
# =========================
# chunk_size：每块大概多长
# chunk_overlap：相邻块重复一点，避免信息被切断
splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=30,
)

split_docs = splitter.split_documents(docs)

# =========================
# 5. 建立向量库
# =========================
# InMemoryVectorStore：内存版向量库，适合学习和小项目
# 它会把 split_docs 用 embeddings 转成向量，并存到内存里
vectorstore = InMemoryVectorStore.from_documents(
    documents=split_docs,
    embedding=embeddings,
)

# =========================
# 6. 变成 retriever
# =========================
# retriever 的作用：给它一个问题，它去向量库里找最相关的文本块
retriever = vectorstore.as_retriever()

print("=== RAG 学习助手已启动 ===")

while True:
    question = input("\n请输入问题（输入 q 退出）：").strip()

    if question.lower() == "q":
        print("已退出。")
        break

    if not question:
        print("请输入内容，不要留空。")
        continue

    try:
        # =========================
        # 7. 检索相关资料
        # =========================
        retrieved_docs = retriever.invoke(question)

        # 把检索到的多个文档块拼成一个 context
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # =========================
        # 8. 让模型基于资料回答
        # =========================
        prompt = f"""
你是一名擅长辅导大学生的计算机课程助教。
请严格根据“资料”回答问题。
如果资料里没有答案，请明确回答：资料中没有提到。
回答要清晰、准确、分点说明。

资料：
{context}

问题：
{question}
"""

        response = llm.invoke(prompt)

        print("\n===== 检索到的资料 =====")
        print(context)

        print("\n===== AI回答 =====")
        print(response.content)
        print("===================")

    except Exception as e:
        print(f"\n调用失败：{e}")