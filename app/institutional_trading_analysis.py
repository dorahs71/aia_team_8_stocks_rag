#langchain 相關套件
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from sqlalchemy.exc import OperationalError
from langchain_core.runnables import RunnablePassthrough
from typing import Iterator

import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

db = SQLDatabase.from_uri("sqlite:///data/aia_sql_db.db")
db.get_usable_table_names()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def get_db_schema(_):
    return db.run("select * from table_comments")

def run_query(query):
    try:
        return db.run(query)
    except (OperationalError, Exception) as e:
        return str(e)


gen_sql_prompt = ChatPromptTemplate.from_messages([
    ('system', '根據下列的DB的Schema進行參考，陣列的內容分別是資料表名稱、英文欄位、中文說明、備註, \
      請根據user\'s 的問題: {db_schema}，並參考各資料表中說明或備註，寫出不要有任何換行符號的SQL來回覆答案'),
    ('user', '請根據下列問題產生一段SQL的查詢語法: "{input}". \
     查詢的格式應如下所示，無需任何附加說明: SQL> <sql_query>'),
])


class SqlQueryParser(StrOutputParser):
    def parse(self, s):
        r = s.split('SQL> ')
        print("r", r)
        if len(r) > 0:
            return r[1]
        return s

def response_data(query: str) -> Iterator[str]:

    gen_query_chain = (
        RunnablePassthrough.assign(db_schema=get_db_schema)
        | gen_sql_prompt
        | llm
        | SqlQueryParser()
    )


    gen_answer_prompt = ChatPromptTemplate.from_template("""
    根據提供的問題、SQL 查詢和查詢結果，編寫自然語言回應。
    不應包含任何額外的解釋。

    問題: {input}
    SQL查詢語法: {query}
    查詢結果: {result}

    """)


    chain = (
        RunnablePassthrough.assign(query=gen_query_chain).assign(
            result=lambda x: run_query(x["query"]),
        )
        | gen_answer_prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({
    'input': query,})

    for chunk in response:
        yield chunk