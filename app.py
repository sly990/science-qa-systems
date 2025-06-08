import os
import streamlit as st
import torch
import pickle
import glob
from typing import List
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import numpy as np
import faiss
import traceback
from langchain.docstore import InMemoryDocstore
from langchain_community.chat_models import ChatTongyi
from pathlib import Path

#当时是用于解决stramlit的某个冲突问题
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# ======== 提示词统一管理（用SystemMessage和HumanMessage） ========
PROMPTS = {
    "is_scientific_query": [
        SystemMessage(content="请判断以下问题是否属于“科普相关内容”（如科学、技术、演讲等），请务必只回答“是”或“否”："),
        HumanMessage(content="{question}")
    ],
    "rag_answer": [
        SystemMessage(content="你是理性且友好的科学问答助手。请根据以下科普内容，做个全面的总结和思考，回答用户问题。如果找不到答案，请回复“抱歉哦，我的知识库里没有找到相关的内容！”。"),
        HumanMessage(content="上下文：\n{context}\n\n问题：\n{question}")
    ],
    "simple_answer": [
        SystemMessage(content="你是一个活泼可爱又理性的科普问答小助手，请简洁、友好、礼貌地回答以下问题："),
        HumanMessage(content="{question}")
    ],
    #添加一个模板，用于处理多轮问答
    "multi_turn": [
        SystemMessage(content="你是理性且友好的科学问答助手。请根据对话历史和以下科普内容回答问题：\n{context}"),
        HumanMessage(content="当前对话历史：\n{history}\n\n问题：\n{question}")
    ],
}


# # 配置路径和环境变量
# DATA_DIR = "/rag_science_speech/data_pdf/merged_pdfs"
# VECTOR_STORE_PATH = "/rag_science_speech/vector_store.pkl"
# ES_INDEX_NAME = "rag_docs"
# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


# #配置路径和环境变量，路径需要修改
# DATA_DIR = "/root/autodl-tmp/jinxiangshao_projects1/rag_science_speech/data_pdf/merged_pdfs"
# VECTOR_STORE_PATH = "/root/autodl-tmp/jinxiangshao_projects1/rag_science_speech/vector_store.pkl"
# ES_INDEX_NAME = "rag_docs"
# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


# # ==== 动态获取项目根目录（关键修改）====
# BASE_DIR = Path(__file__).parent  # 获取当前文件（app.py）所在的目录，即项目根目录

# # ==== 修改 DATA_DIR 和 VECTOR_STORE_PATH 为相对路径 ====
# # 数据目录：相对于项目根目录的 "data_pdf/merged_pdfs"
# DATA_DIR = BASE_DIR / "data_pdf/merged_pdfs"
# # 向量存储路径：相对于项目根目录的 "vector_store/vector_store.pkl"
# VECTOR_STORE_PATH = BASE_DIR / "vector_store/vector_store.pkl"

# # 将 Path 对象转换为字符串（供后续代码使用）
# DATA_DIR = str(DATA_DIR)
# VECTOR_STORE_PATH = str(VECTOR_STORE_PATH)

# 直接从当前工作目录出发找子目录/文件
DATA_DIR = "data_pdf/merged_pdfs"  
VECTOR_STORE_PATH = "vector_store.pkl"  

ES_INDEX_NAME = "rag_docs"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"



#获取向量模型
@st.cache_resource
def get_embeddings():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_kwargs = {'device': device}
        embedder = HuggingFaceEmbeddings(
            model_name="lier007/xiaobu-embedding-v2",
            model_kwargs=model_kwargs,
            encode_kwargs={'normalize_embeddings': True}
        )
        # 测试嵌入功能是否正常
        _ = embedder.embed_query("测试")
        return embedder
    except Exception:
        return None

#设置成自己的密钥
os.environ["TONGYI_API_KEY"] = "sk-506a5c243ac3445caea6be389b0025fb"  



# 辅助函数：格式化消息为通义所需格式，与之前有不同
def format_messages(template_messages, **kwargs):
    messages = []
    for msg in template_messages:
        content = msg.content.format(**kwargs)
        if isinstance(msg, SystemMessage):
            messages.append({"role": "system", "content": content})
        else:
            messages.append({"role": "user", "content": content})
    return messages




# 获取通义千问 LLM 客户端
def get_llm():
    api_key = os.getenv("TONGYI_API_KEY")
    if not api_key:
        st.error("通义API密钥未配置！")
        return None
    try:
        return ChatTongyi(
            model_name="qwen-plus",
            dashscope_api_key=api_key,
            temperature=0.7
        )
    except Exception as e:
        st.error(f"通义模型加载失败: {str(e)}")
        return None



def load_documents() -> List[Document]:
    docs = []
    for file_path in glob.glob(f"{DATA_DIR}/*.pdf"):
        try:
            loader = PDFPlumberLoader(file_path)
            pages = loader.load()
            full_text = "\n".join([p.page_content for p in pages])
            file_name = os.path.basename(file_path)
            docs.append(Document(page_content=full_text, metadata={"source": file_name}))
        except Exception:
            continue
    return docs





# 定义一个简单向量数据库替代faiss（如果faiss失效的话）
# 可以不用这部分

class InMemorySimpleVectorStore:
    def __init__(self, embedding_function, texts, metadatas=None):
        self.embedding_function = embedding_function
        self.texts = texts
        self.metadatas = metadatas or [{} for _ in texts]
        self.doc_embeddings = []
        for text in texts:
            try:
                embedding = embedding_function.embed_query(text)
                self.doc_embeddings.append(embedding)
            except Exception:
                self.doc_embeddings.append(np.zeros(768, dtype=np.float32))

    def similarity_search(self, query, k=4):
        query_embedding = self.embedding_function.embed_query(query)
        query_vec = np.array(query_embedding, dtype=np.float32)
        similarities = []
        for doc_embedding in self.doc_embeddings:
            doc_vec = np.array(doc_embedding, dtype=np.float32)
            dot_product = np.dot(query_vec, doc_vec)
            query_norm = np.linalg.norm(query_vec)
            doc_norm = np.linalg.norm(doc_vec)
            similarity = dot_product / (query_norm * doc_norm) if query_norm > 0 and doc_norm > 0 else 0
            similarities.append(similarity)
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [Document(page_content=self.texts[i], metadata=self.metadatas[i]) for i in top_indices]




#加载向量数据库的函数
def load_or_create_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            with open(VECTOR_STORE_PATH, "rb") as f:
                vector_store = pickle.load(f)
            return vector_store
        except Exception:
            return None
    return None


#构建向量数据库和索引
@st.cache_resource
def build_vector_store_and_index():
    vector_store = load_or_create_vector_store()
    if vector_store:
        try:
            es = Elasticsearch("http://localhost:9200")
            es.info()
            return vector_store, es
        except Exception:
            return vector_store, None

    docs = load_documents()
    embedder = get_embeddings()
    if not embedder:
        return None, None

    docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
    if not docs:
        return None, None

    try:
        try:
            vector_store = FAISS.from_documents(docs, embedder)
        except Exception:
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            vector_store = InMemorySimpleVectorStore(embedder, texts, metadatas)

        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        with open(VECTOR_STORE_PATH, "wb") as f:
            pickle.dump(vector_store, f)

        try:
            es = Elasticsearch("http://localhost:9200")
            es.info()
            if not es.indices.exists(index=ES_INDEX_NAME):
                actions = [{
                    "_index": ES_INDEX_NAME,
                    "_id": i,
                    "_source": {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown")
                    }
                } for i, doc in enumerate(docs)]
                bulk(es, actions)
            return vector_store, es
        except Exception:
            return vector_store, None

    except Exception:
        return None, None



#关键字匹配        
def keyword_search(es, query, top_k=5):
    if not es:
        return []
    try:
        res = es.search(
            index=ES_INDEX_NAME,
            query={"match": {"content": query}},
            size=top_k
        )
        return [Document(page_content=hit["_source"]["content"], metadata={"source": hit["_source"]["source"]}) 
                for hit in res["hits"]["hits"]]
    except Exception:
        return []


# 判断问题是否为科普问题 返回True或False
def is_scientific_query(question: str) -> bool:
    llm = get_llm()
    if not llm:
        return True
    try:
        messages = format_messages(PROMPTS["is_scientific_query"], question=question)
        #这里和原先不一样是因为如果直接使用原来的代码，会额外输出一些其他内容，这里是为了调整输出格式，后面类似的部分也要修改
        response = llm.invoke(messages)
        # 提取纯文本内容
        if hasattr(response, 'content'):
            result = response.content
        else:
            result = str(response)
        result = result.strip().lower()
        return result.startswith("是") or "是" in result or result.startswith("yes")
    except Exception:
        return True


#修改回答生成部分，根据是否是多轮对话动态构建提示，把main函数里面的回答生成部分封装成一个函数，直接调用
def generate_response(user_input, is_sci_question, rag_docs=None):
    llm = get_llm()
    if not llm:
        return "系统错误：无法加载语言模型。"
    
    # 构建对话历史字符串
    history_str = "\n".join(
        [f"{msg['role']}: {msg['content']}" 
         for msg in st.session_state.chat_history]
    )
    
    if is_sci_question and rag_docs:
        # 构建RAG上下文
        context = "\n".join(
            [f"文档({i+1}): {doc.page_content}" 
             for i, doc in enumerate(rag_docs)]
        )
        
        # 使用多轮对话模板
        messages = format_messages(
            PROMPTS["multi_turn"],
            context=context,
            history=history_str,
            question=user_input
        )
    else:
        # 普通对话使用简单模板
        messages = format_messages(
            PROMPTS["simple_answer"],
            question=user_input
        )
        # 添加历史上下文
        messages.insert(0, {"role": "system", "content": f"对话历史：\n{history_str}"})
    
    try:
        response = llm.invoke(messages)
        return response.content.strip() if hasattr(response, 'content') else str(response).strip()
    except Exception as e:
        return f"生成回答时出错：{str(e)}"


def main():

    st.set_page_config(page_title="科普知识问答助手", layout="wide")
    st.title("🌟 科普知识问答助手")

    vector_store, es = build_vector_store_and_index()
    if vector_store is None:
        st.error("向量数据库初始化失败，系统可能无法正常回答科普相关问题。")
    if es is None:
        st.warning("Elasticsearch连接失败，将仅使用向量搜索。")

    #修改初始化
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        {"role": "system", "content": "你是理性且友好的科学问答助手"}
                 ]

    # 显示历史消息（跳过系统消息）
    for msg in st.session_state.chat_history:
        if msg["role"] == "system":  # 跳过系统消息
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("请输入您的问题~")

    if user_input:
        # 添加用户消息到历史
        user_message = {"role": "user", "content": user_input}
        st.session_state.chat_history.append(user_message)
        
        with st.chat_message("User"):
            st.markdown(user_input)

        with st.spinner("思考中..."):
            try:
                is_sci_question = is_scientific_query(user_input)
                rag_docs = []
                if is_sci_question and vector_store:
                    

                    try:
                        vector_results = vector_store.similarity_search(user_input, k=5)
                        print("vector_results:",vector_results)
                        rag_docs.extend(vector_results)
                        
                    except Exception:
                        pass

                    if es:
                        try:
                            keyword_results = keyword_search(es, user_input, top_k=5)
                            print("keyword_results:", keyword_results)
                            rag_docs.extend(keyword_results)
                        except Exception:
                            pass
                     # 去重和排序
                    if rag_docs:
                        seen_content = set()
                        unique_docs = []
                        for doc in rag_docs:
                            if doc.page_content not in seen_content:
                                seen_content.add(doc.page_content)
                                unique_docs.append(doc)
                        rag_docs = sorted(unique_docs, key=lambda x: len(x.page_content))[:5]
                answer = generate_response(
                                        user_input, 
                                         is_sci_question, 
                                     rag_docs if rag_docs else None  # 如果rag_docs不为空则传入，否则传入None
                                            )
                    # 添加助手回复到历史
                assistant_message = {"role": "assistant", "content": answer}
                st.session_state.chat_history.append(assistant_message)
                    # 显示助手回复
                with st.chat_message("assistant"):  # 角色名称小写
                    st.markdown(answer)

            except Exception as e:
                # 错误处理
                error_msg = f"系统错误：{str(e)}"
                st.error(error_msg)
                st.error(traceback.format_exc())
                
                # 添加错误消息到历史
                assistant_message = {"role": "assistant", "content": error_msg}
                st.session_state.chat_history.append(assistant_message)
                
                # 显示错误消息
                with st.chat_message("assistant"):
                    st.markdown(error_msg)

if __name__ == "__main__":
    main()









