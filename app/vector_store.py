import os
from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase import create_client, Client

# 載入 .env 檔案中的環境變數
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings()

def get_retriever():
    vectorstore = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents", # 資料表名稱
    query_name="match_documents", # 查詢函式名稱
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return retriever



