# data_utils.py
import json
from langchain.schema import Document  # 修正導入路徑
from langchain.vectorstores import FAISS  # 修正導入路徑

def save_documents(documents, file_path):
    """
    將分割後的 Document 物件保存為 JSON 文件。
    """
    # 使用 `doc.dict()` 代替 `doc.to_dict()`
    docs_dict = [doc.dict() for doc in documents]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(docs_dict, f, ensure_ascii=False, indent=4)
    print(f"Documents saved to {file_path}")

def load_documents(file_path):
    """
    從 JSON 文件載入 Document 物件。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        docs_dict = json.load(f)
    documents = [Document(**doc) for doc in docs_dict]
    print(f"Documents loaded from {file_path}")
    return documents

def save_faiss(vectordb, directory):
    """
    將 FAISS 向量資料庫保存到指定目錄。
    """
    vectordb.save_local(directory)
    print(f"FAISS index saved to '{directory}' directory")

def load_faiss(directory, embeddings):
    """
    從指定目錄載入 FAISS 向量資料庫。
    """
    vectordb = FAISS.load_local(directory, embeddings)
    print(f"FAISS index loaded from '{directory}' directory")
    return vectordb
