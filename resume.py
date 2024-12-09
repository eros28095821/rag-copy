from operator import itemgetter
import ollama
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory

from chainlit.types import ThreadDict
import chainlit as cl

load_dotenv()  # 加载 .env 配置

# 设置 Runnable
def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    # 设置 Ollama 模型
    model = "kenneth85/llama-3-taiwan:8b-instruct"  # 选择 Ollama 模型
    client = ollama  # 使用 Ollama 客户端

    # 创建一个简单的 prompt 模板
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful chatbot"),
            ("human", "{question}"),
        ]
    )

    # 获取历史记录并将其转换为字符串
    history = memory.load_memory_variables({"question": ""})  # 必须传递 `inputs` 参数，这里传递空字典或问题
    history_str = "\n".join([msg["content"] for msg in history.get("history", [])])

    # 格式化 prompt，并传递历史记录
    prompt_with_history = prompt.format(question="{question}", history=history_str)

    # 设置可执行流程
    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt_with_history  # 使用已经格式化的 prompt
        | client.chat  # 使用 Ollama 客户端的 chat 方法
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)

# 用户身份验证
@cl.password_auth_callback
def auth():
    return cl.User(identifier="test")

# 当用户开始聊天时
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))  # 创建一个新的聊天内存
    setup_runnable()  # 设置可执行的聊天流程

# 当用户恢复聊天时
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)  # 创建一个新的聊天内存
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)  # 设置新的聊天内存
    setup_runnable()  # 设置可执行的聊天流程

# 当用户发送消息时
@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    runnable = cl.user_session.get("runnable")  # type: Runnable

    res = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)  # 实时返回流式数据

    await res.send()  # 发送回应

    memory.chat_memory.add_user_message(message.content)  # 将用户消息添加到内存
    memory.chat_memory.add_ai_message(res.content)  # 将模型回复添加到内存
