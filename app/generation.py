from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from vector_store import get_retriever

# Constants
TEMPERATURE = 0.5
LLM_MODEL = 'gpt-4o-mini'

def generate_answer(query):
    # 初始化 LLM 來生成多查詢
    llm = ChatOpenAI(temperature=TEMPERATURE, model_name=LLM_MODEL)
    retriever = get_retriever()

    # Convert the dist data output into string
    str_parser = StrOutputParser()

    # 使用 MultiQueryRetriever 增加查詢變體來增強檢索效果
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm = llm,
    )

    template="""
    您是一位專業的財經新聞分析師。請根據提供的財經影片片段，以專業、客觀、精準的角度回答使用者的問題。
    請務必只依據影片片段中的資訊進行分析，並避免自行腦補或使用外部資訊。
    您的回答應著重於事實分析和邏輯推演，希望見解能以分條列點的方式來呈現。
    請在答案的最後務必放上影片連結，格式為 video_name 加上超連結 video_url，方便使用者追溯資訊來源。

    --- 影片片段 ---
    {context}
    ---

    使用者問題：{question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": multi_query_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 修改為串流輸出
    for chunk in chain.stream(query):
        yield chunk