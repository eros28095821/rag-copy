from operator import itemgetter
import os
import ollama
from dotenv import load_dotenv
import json
from chainlit.types import ThreadDict
import chainlit as cl

load_dotenv()

@cl.password_auth_callback
def auth():
    return cl.User(identifier="test")


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_history", [])


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    cl.user_session.set("chat_history", [])

    # user_session = thread["metadata"]
    
    for message in thread["steps"]:
        if message["type"] == "user_message":
            cl.user_session.get("chat_history").append({"role": "user", "content": message["output"]})
        elif message["type"] == "assistant_message":
            cl.user_session.get("chat_history").append({"role": "assistant", "content": message["output"]})


@cl.on_message
async def on_message(message: cl.Message):
    # Note: by default, the list of messages is saved and the entire user session is saved in the thread metadata
    chat_history = cl.user_session.get("chat_history")

    model = "kenneth85/llama-3-taiwan:8b-instruct"  # Ollama 模型名稱
    client = ollama  # 使用 Ollama 客戶端

    chat_history.append({"role": "user", "content": message.content})

    # 使用 Ollama 模型進行聊天
    chat_response = client.chat(
        model=model,
        messages=chat_history
    )

    # 取得回應內容
    response_content = chat_response["message"]["content"]

    # 更新聊天歷史
    chat_history.append({"role": "assistant", "content": response_content})

    # 發送回應
    await cl.Message(content=response_content).send()
