大家可以先熟悉一下MAS的基础开发：

1. 自己本地通过lm studio部署一个本地大模型（例如qwen3.6-27B，4bit）

2. 本地搭建longchain/longgraph/longsmith的agent基础环境

3. 构建本地sqllite来实现memory，存储对话的上下文，并进行上下文管理

4. 构建本地Chroma向量数据库来实现本地的rag知识库，将业务文档固化在其中，并进行知识库初始化和预处理

5. 提供本地执行的函数作为tool

6. 最终实现：

Round 1：测试持久化记忆

你：你好，我是研发总监，我叫张伟。

AI：（不触发检索工具）正常回答，打招呼并记住你的身份。

Round 2：测试 RAG 检索 (不干扰记忆)

你：帮我查一下我们最新文档里，关于“XXX项目”的预算/指标是多少？

(此时你会在终端看到：🔍 [动作] AI 正在翻阅本地知识库...)

AI：精准输出文档里的数值。

Round 3：测试“断电恢复”的终极震撼

动作：按 q 退出程序，彻底关闭终端（甚至你可以重启 Mac）。

动作：再次打开终端，重新运行 python agent_rag_memory_demo.py。

你：你还记得我是谁吗？我刚才问了你哪个项目？

AI：准确回答出你是研发总监张伟，以及刚才探讨的项目。



答案代码如下，大家如果对Agent开发不熟悉，先跑通流程，看懂下面的代码：

import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# ==========================================
# 1. 基础资源连接 (完全离线模式)
# ==========================================
print("🔄 正在初始化系统组件，请稍候...")

# 1.1 连接本地大语言模型 (基于 LM Studio)
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="local-model",
    temperature=0.3 # 平衡严谨与聊天流畅度
)

# 1.2 连接本地知识库 (基于 M 芯片加速的 Embedding)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5", 
    model_kwargs={'device': 'mps'}
)
vector_store = Chroma(
    persist_directory="./local_chroma_db", 
    embedding_function=embeddings
)

# ==========================================
# 2. 核心功能：定义 RAG 检索工具
# ==========================================
@tool
def retrieve_local_knowledge(query: str) -> str:
    """
    当你需要回答关于具体文档、内部指南、数据指标或专业技术问题时，必须调用此工具。
    它会从本地的私有知识库中检索出最相关的片段。
    """
    print(f"\n🔍 [动作] AI 正在翻阅本地知识库，搜索关键词: {query} ...")
    docs = vector_store.similarity_search(query, k=3)
    if not docs:
        return "本地知识库中未找到相关内容，请基于你的常识如实回答。"
    
    # 将找到的知识切片合并返回给 AI 作为参考资料
    return "\n\n".join([f"【参考片段】: {d.page_content}" for d in docs])

# 将工具绑定到 LLM 上，使其具备调用能力
tools = [retrieve_local_knowledge]
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. 构建 LangGraph 状态机工作流
# ==========================================
# 状态定义：不断累加消息历史
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 思考节点：注入系统人设，并调用 LLM
def assistant_node(state: AgentState) -> dict:
    # 动态插入 System Prompt 规范 AI 行为
    sys_msg = SystemMessage(content="""
    你是一个安全、专业的高级本地 AI 助手。
    1. 你拥有查阅本地知识库的工具，不要臆造专业数据，查不到就说不知道。
    2. 你像人类一样有记忆，请记住用户的名字和上下文。
    3. 用中文友好作答。
    """)
    # 将系统消息和历史消息一起喂给大模型
    messages_to_prompt = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages_to_prompt)
    return {"messages": [response]}

# 构建流转图
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools)) # 官方预制工具节点，负责执行 Python 函数

# 定义路由边 (核心大脑)
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition) # 自动判断：调用工具 -> tools，聊天 -> END
builder.add_edge("tools", "assistant") # 工具执行完，带着结果回到 assistant 进行最后总结

# ==========================================
# 4. 挂载持久化存储并运行
# ==========================================
# 使用 with 语句确保 SQLite 数据库在使用后安全断开连接
with SqliteSaver.from_conn_string("memory_with_rag.db") as memory:
    
    # 编译图：将记忆引擎注入 Agent
    app = builder.compile(checkpointer=memory)

    if __name__ == "__main__":
        # 定义当前用户的唯一会话 ID
        # 只要这个 ID 不变，记忆就永远存在；换个 ID，就是全新的开始。
        config = {"configurable": {"thread_id": "boss_local_session_001"}}
        
        print("\n" + "="*50)
        print("🚀 【终极本地私有化 Agent】已启动！")
        print("💡 数据安全说明：")
        print("  - 聊天记忆存储于: memory_with_rag.db")
        print("  - 业务文档存储于: ./docs (已向量化至 local_chroma_db)")
        print("  - 所有计算在 Mac 本地完成，完全断网可用。")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("👤 你的输入 (输入 'q' 退出): ")
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("👋 正在安全关闭数据库连接并退出系统。再见！")
                    break
                if not user_input.strip():
                    continue

                # 将用户输入传入状态机
                print("🤖 思考中...", end="\r")
                events = app.stream(
                    {"messages": [HumanMessage(content=user_input)]}, 
                    config, 
                    stream_mode="values"
                )
                
                # 实时捕获图流转的最终输出
                for event in events:
                    last_msg = event["messages"][-1]
                    # 只打印 AI 的最终文本回复，跳过中间工具调用的占位符
                    if last_msg.type == "ai" and last_msg.content:
                        print(f"🤖 助手: {last_msg.content}\n")
                        
            except KeyboardInterrupt:
                print("\n👋 被强制中断，退出系统。")
                break