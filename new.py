import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="歡迎使用聊天機器人！請選擇操作：",
        buttons=[
            cl.Button(name="new_chat", label="開啟新對話"),
            cl.Button(name="view_history", label="查看歷史對話"),
        ],
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    if message.content == "new_chat":
        await cl.Message(content="新對話已啟動！").send()
    elif message.content == "view_history":
        await cl.Message(content="歷史對話記錄顯示如下：...").send()
    else:
        await cl.Message(content="無效操作，請使用按鈕選擇功能！").send()
