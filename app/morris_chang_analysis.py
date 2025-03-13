import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import re
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import SupabaseVectorStore
from supabase import create_client

# 載入 .env 檔案中的環境變數
load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
ORGANIZATION_ID = os.environ['ORGANIZATION_ID']

def load_multiple_pdfs(pdf_paths: List[str]) -> list:
    all_docs = []

    for path in pdf_paths:
        loader = PyMuPDFLoader(path)
        # 添加自定义元数据
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = path  # 记录文件来源
        all_docs.extend(docs)

    print(f"共加载 {len(pdf_paths)} 个PDF，总页数 {len(all_docs)}")
    return all_docs

def clean_text(text: str) -> str:
    """清除危险字符但保留格式符号"""
    # 保留的格式字符：\n (换行) \t (制表符)
    # 匹配范围：空字符(\x00) + 控制字符(除\t外的\x01-\x1F) + DEL(\x7F)
    return re.sub(r'[\x00\x01-\x08\x0B-\x1F\x7F]', '', text)

def clean_document(doc: Document) -> Document:
    """清洗文档内容及元数据"""
    # 清洗正文内容
    cleaned_content = clean_text(doc.page_content)

    # 清洗元数据
    cleaned_metadata = {}
    for key, value in doc.metadata.items():
        if isinstance(value, str):
            cleaned_metadata[key] = clean_text(value)
        elif isinstance(value, (list, dict)):
            # 处理嵌套结构
            cleaned_metadata[key] = json.loads(
                clean_text(json.dumps(value)))
        else:
            cleaned_metadata[key] = value

    return Document(
        page_content=cleaned_content,
        metadata=cleaned_metadata
    )


def split_data():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    # 适合中文的推荐值
        chunk_overlap=100,  # 建议为 chunk_size 的 20%
        separators=[
            "\n\n",        # 空行（段落分隔）
            "\n",           # 换行
            "(?<!\d)\.(?!\d)", # 中文句号（排除小数点）
            "？", "！",     # 中文问号/感叹号
            ";", "…",       # 分号/省略号
            ",", "、",      # 中文逗号
            " "
        ]
    )
    splits = text_splitter.split_documents(combined_docs)


    # 验证清洗结果
    cleaned_docs = [clean_document(doc) for doc in combined_docs]
    sample_doc = cleaned_docs[5]
    print("清洗后内容示例：", sample_doc.page_content[:100])
    print("清洗后元数据：", sample_doc.metadata)

    splits = text_splitter.split_documents(cleaned_docs)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

def response_morris_data(query:str) -> str:
    # 初始化客戶端
    supabase = create_client(
        SUPABASE_URL,
        SUPABASE_KEY 
    )

    vectorstore = SupabaseVectorStore(
        client=supabase,
        embedding=OpenAIEmbeddings(),
        table_name="morris_autobio",
        query_name="match_morris_autobio"
    )

    llm = ChatOpenAI(temperature=0, model_name='gpt-4o')
    retriever = vectorstore.as_retriever()
    # Step 6: 建立 RetrievalQA Chain，使用 retriever 來檢索相關資料
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # Step 8: 進行檢索並生成回答 (qa_chain)
    response = qa_chain.run(query)

    return response