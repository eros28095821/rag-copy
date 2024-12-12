import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from operator import itemgetter
from langchain.schema import HumanMessage, AIMessage

# 全域變數儲存歷史對話
global_history = []

# 初始化模型
def setup_runnable():
    memory = cl.user_session.get("memory")  # 獲取目前的記憶體對象
    model = Ollama(model="kenneth85/llama-3-taiwan:8b-instruct")  # 替換為 Ollama 模型
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一個樂於助人的聊天機器人。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
    )
    cl.user_session.set("runnable", runnable)

# 啟動時初始化
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    setup_runnable()

    # 提供操作選單
    await cl.Message(
        content=(
            "歡迎使用聊天機器人！\n"
            "請輸入以下選項進行操作：\n"
            "new_chat - 開啟新對話\n"
            "view_history - 查看歷史對話"
        )
    ).send()

# 處理訊息
@cl.on_message
async def on_message(message: cl.Message):
    global global_history

    memory = cl.user_session.get("memory")
    runnable = cl.user_session.get("runnable")

    # 根據用戶輸入進行操作
    if message.content.strip() == "new_chat":
        # 保存目前對話到歷史
        if memory.chat_memory.messages:
            global_history.append(memory.chat_memory.messages)

        # 清除上下文，開啟新對話
        cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
        setup_runnable()

        await cl.Message(content="新對話已啟動！").send()

    elif message.content.strip() == "view_history":
        # 顯示所有歷史對話
         # **打印原始 global_history 到終端**
        print("\n=== Global History (Raw Structure) ===")
        print(global_history)
        print("=== End of Global History ===\n")

        # 顯示所有歷史對話
        if not global_history:
            await cl.Message(content="目前沒有歷史對話記錄。").send()
        else:
            for i, history in enumerate(global_history, start=1):
                history_text = []
                for msg in history:
                    # 判斷訊息角色
                    if isinstance(msg, HumanMessage):
                        role = "用戶"
                    elif isinstance(msg, AIMessage):
                        role = "AI"
                    else:
                        role = "未知角色"  # 預防其他情況
                    history_text.append(f"**{role}：** {msg.content}")
                await cl.Message(content=f"### 歷史對話 {i}\n" + "\n".join(history_text)).send()

    else:
        # 處理用戶輸入作為對話
        res = cl.Message(content="")
        async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await res.stream_token(chunk)
        await res.send()

        # 儲存對話記錄
         # 儲存對話記錄
        memory.chat_memory.add_user_message(HumanMessage(content=message.content))
        memory.chat_memory.add_ai_message(AIMessage(content=res.content))