from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chainlit as cl

# 初始化 Ollama LLM 和其他組件
llm = Ollama(model="kenneth85/llama-3-taiwan:8b-instruct")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 加載並分割文檔
loader = TextLoader("/home/chen/rag-copy/起訴狀.txt")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20, separators=[" ", ",", "\n"])
splited_docs = text_splitter.split_documents(loader.load())

# 創建向量數據庫
vector_db = Chroma.from_documents(
    documents=splited_docs,
    embedding=embeddings,
    persist_directory="db",
    collection_name="about",
)

# 創建檢索器
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 用來保存對話歷史
conversation_history = []

# 上下文摘要函數
def summarize_context(context):
    """
    簡單模擬一個上下文摘要功能。可以替換為自動化摘要模型或自定義邏輯。
    """
    MAX_HISTORY = 10
    if len(context) > MAX_HISTORY * 2:
        # 模擬摘要 (可以替換為更智能的摘要邏輯)
        summary = "之前的對話摘要：用戶與助手就一般性問題進行了互動。"
        return [summary] + context[-MAX_HISTORY * 2:]
    return context

@cl.on_chat_start
async def on_chat_start():
    # 創建 system 提示模板
    system_prompt = "你是台灣民法法學專家，擁有深入的法律知識，並能夠準確解釋和分析台灣民法相關的問題，提供精確且法律依據充分的答案。"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{question}")]
    )
    # 設置用戶會話
    runnable = prompt_template | llm | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    global conversation_history

    # 如果用戶請求清除記憶
    if message.content.strip().lower() in ["清除記憶", "reset"]:
        conversation_history = []
        await cl.Message(content="記憶已清除！").send()
        return

    try:
        # 將當前用戶消息添加到歷史中
        conversation_history.append(f"用戶: {message.content}")

        # 簡化對話歷史以節省 Token
        summarized_history = summarize_context(conversation_history)
        full_context = "\n".join(summarized_history)

        # 使用檢索器檢索相關文檔
        retriever_result = retriever.get_relevant_documents(full_context)
        context = "\n".join([doc.page_content for doc in retriever_result])

        # 從上下文生成最終的提示語
        prompt = f"""你是一個中華民國的民法專長的律師，你要根據中華民國民法正確引用民法法條，並且要正確引用以下的中華民國民法內容。
######
第 184 條
因故意或過失，不法侵害他人之權利者，負損害賠償責任。故意以背於善良風俗之方法，加損害於他人者亦同。
違反保護他人之法律，致生損害於他人者，負賠償責任。但能證明其行為無過失者，不在此限。
第 185 條
數人共同不法侵害他人之權利者，連帶負損害賠償責任。不能知其中孰為加害人者亦同。
造意人及幫助人，視為共同行為人。
第 191 條
土地上之建築物或其他工作物所致他人權利之損害，由工作物之所有人負賠償責任。但其對於設置或保管並無欠缺，或損害非因設置或保管有欠缺，或於防止損害之發生，已盡相當之注意者，不在此限。
前項損害之發生，如別有應負責任之人時，賠償損害之所有人，對於該應負責者，有求償權。
第 191-1 條
商品製造人因其商品之通常使用或消費所致他人之損害，負賠償責任。但其對於商品之生產、製造或加工、設計並無欠缺或其損害非因該項欠缺所致或於防止損害之發生，已盡相當之注意者，不在此限。
前項所稱商品製造人，謂商品之生產、製造、加工業者。其在商品上附加標章或其他文字、符號，足以表彰係其自己所生產、製造、加工者，視為商品製造人。
商品之生產、製造或加工、設計，與其說明書或廣告內容不符者，視為有欠缺。
商品輸入業者，應與商品製造人負同一之責任。
第 191-2 條
汽車、機車或其他非依軌道行駛之動力車輛，在使用中加損害於他人者，駕駛人應賠償因此所生之損害。但於防止損害之發生，已盡相當之注意者，不在此限。
第 191-3 條
經營一定事業或從事其他工作或活動之人，其工作或活動之性質或其使用之工具或方法有生損害於他人之危險者，對他人之損害應負賠償責任。但損害非由於其工作或活動或其使用之工具或方法所致，或於防止損害之發生已盡相當之注意者，不在此限。
第 193 條
不法侵害他人之身體或健康者，對於被害人因此喪失或減少勞動能力或增加生活上之需要時，應負損害賠償責任。
前項損害賠償，法院得因當事人之聲請，定為支付定期金。但須命加害人提出擔保。
第 195 條
不法侵害他人之身體、健康、名譽、自由、信用、隱私、貞操，或不法侵害其他人格法益而情節重大者，被害人雖非財產上之損害，亦得請求賠償相當之金額。其名譽被侵害者，並得請求回復名譽之適當處分。
前項請求權，不得讓與或繼承。但以金額賠償之請求權已依契約承諾，或已起訴者，不在此限。
前二項規定，於不法侵害他人基於父、母、子、女或配偶關係之身分法益而情節重大者，準用之。
######

你主要是要撰寫交通事故的民事起訴狀，並且只針對慰撫金部分的案件作撰寫。慰撫金基本上只會用到以上民法的內容，你閱讀完需要改寫的內容之後思考一下再使用正確的民法法條。
以下是一個正確的民事交通事故起訴狀的範本，請你根據使用者提供的內容，然後生成符合範本的內容，以下是我要你生成的訴訟狀的法律結構：

 """

        model = Ollama(model="kenneth85/llama-3-taiwan:8b-instruct")
        prompt_template = ChatPromptTemplate.from_messages([("human", prompt)])
        runnable = prompt_template | model | StrOutputParser()

        msg = cl.Message(content="")

        # 使用流模式發送請求
        async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)

        # 將模型回應發送給用戶
        await msg.send()

        # 將模型回應添加到歷史中
        conversation_history.append(f"助手: {msg.content}")

    except Exception as e:
        # 捕獲錯誤並通知用戶
        await cl.Message(content=f"發生錯誤：{str(e)}").send()
