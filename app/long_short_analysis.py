import pandas as pd
import requests as req
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from typing import Iterator

# 載入環境變數
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

### 資料處理

# 資料截取Class
class StockScraper:

    def __init__(self):
        """
        初始化 StockScraper 類別
        """
        self.data=self.get_stock_number()

    def get_information(self,Stock_code:str) -> dict:
        """
        取得所有資訊
        輸入為股票代號（Str Type）
        :return: 字典形式的資料 ["相關新聞","即時交易資訊","法人歷史資訊","即時法人資訊"]
        """

        info = dict()
        # 取得雅戶股市網址
        result_main = req.get(f"https://tw.stock.yahoo.com/quote/{Stock＿code}.TW")
        soup_main = BeautifulSoup(result_main.text, "html.parser")
        # 取得法人資訊網址
        result_tech = req.get(f"https://tw.stock.yahoo.com/quote/{Stock＿code}.TW/institutional-trading")
        soup_tech = BeautifulSoup(result_tech.text, "html.parser")

        #取得各項資訊
        info["相關新聞"] = self.get_news(soup_main)
        info["即時交易資訊"] = self.get_quote(soup_main)
        info["法人歷史資訊"] = self.get_ttmii_history_data(soup_tech)
        info["即時法人資訊"] = self.get_ttmii_data(soup_tech)

        return str(info)

    def get_news(self,soup) -> list:
        """
        取得相關新聞標題列表
        :return: 包含新聞標題的列表
        """
        stock_news = []
        for html_row in soup.find_all("li", class_="js-stream-content Pos(r)"):
            title = html_row.find('a')
            if title:
                stock_news.append(title.text.strip())
        return stock_news

    def get_quote(self,soup) -> str:
        """
        取得即時股市交易資訊，返回格式化的字串
        :return: 即時股市交易資訊字串
        """
        stock_quote = ""
        ul_tag = soup.find("ul", class_="D(f) Fld(c) Flw(w) H(192px) Mx(-16px)")
        if not ul_tag:
            return stock_quote

        for li in ul_tag.find_all("li"):
            span_tags = li.find_all("span")
            if len(span_tags) >= 2:
                name = span_tags[0].get_text(strip=True)
                value = ''.join(span_tags[1].stripped_strings)
                stock_quote += f"{name}: {value}\n"
        return stock_quote

    def get_ttmii_history_data(self,soup) -> list:
        """
        取得法人買賣歷史數據，返回字典列表格式的數據
        :return: 法人買賣歷史數據（字典列表）
        """
        section = soup.find('section', id="qsp-trading-by-day")
        if not section:
            return []

        soup_info_history = section.find('div', class_="Pos(r) Ov(h)")
        if not soup_info_history:
            return []

        columns = ["日期", "外資(張)", "投信(張)", "自營商(張)", "合計(張)", "外資籌碼", "漲跌幅(%)", "成交量"]
        table_data = []

        rows = soup_info_history.find_all("li", class_="List(n)")
        for row in rows:
            row_data = []
            date_div = row.find("div", class_="W(96px) Ta(start)")
            if date_div:
                row_data.append(date_div.get_text(strip=True))

            cells = row.find_all("span", class_="Jc(fe)")
            for cell in cells:
                row_data.append(cell.get_text(strip=True))

            trend_span = row.find("span", class_="Fw(600) Jc(fe) D(f) Ai(c)")
            if trend_span:
                trend_text = ''.join(trend_span.stripped_strings)
                row_data.append(trend_text)

            if len(row_data) == len(columns):
                table_data.append(row_data)

        df = pd.DataFrame(table_data, columns=columns)
        return df.to_dict(orient="records")

    def get_ttmii_data(self,soup) -> list:
        """
        取得法人買賣紀錄數據，返回字典列表格式的數據
        :return: 法人買賣紀錄數據（字典列表）
        """
        soup_info = soup.find('div', id="main-3-QuoteChipTrade-Proxy")
        if not soup_info:
            return []

        soup_info_history = soup_info.find('div', class_="Pos(r) Ov(h)")
        if not soup_info_history:
            return []

        table_data = []
        rows = soup_info_history.find_all("li", class_="List(n)")
        for row in rows:
            cols = row.find_all("span")
            row_data = [col.get_text(strip=True) for col in cols]
            table_data.append(row_data)

        columns = ["法人類型", "買進", "賣出", "買賣超", "趨勢"]

        df = pd.DataFrame(table_data, columns=columns)
        return df.to_dict(orient="records")

    # 抓取台灣上市公司股票代號
    def get_stock_number(self):
        # 發送 HTTP 請求並取得網頁內容
        url = 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=2'
        response = req.get(url)
        # html_content = response.content.decode('big5')  # 該網站使用 Big5 編碼

        # 解析 HTML 並提取表格資料
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'h4'})  # 根據實際情況調整選擇器

        # 遍歷表格行並提取資料資料
        data = []
        rows = table.find_all('tr')[1:]  # 跳過表頭
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 5:
                if len(cols[0].get_text().split("\u3000")[0])==4:
                    [k,v] = cols[0].get_text().split("\u3000")
                    record = {v:k}
                    data.append(record)
        return str(data)


class LLM_Model():
    def __init__(self,OPENAI_API_KEY):

        self.OPENAI_API_KEY = OPENAI_API_KEY

        self.Name_Analyze_LLM = self.build_Analyze__Name_LLM()

        self.LLM = self.build_LLM_model()
        # 建立 StockScraper
        self.stock_clawer = StockScraper()

    # 創建股票分析模型
    def build_LLM_model(self):
        # 創建一個 OpenAI LLM 實例
        llm = ChatOpenAI(openai_api_key=self.OPENAI_API_KEY,temperature=0.5)

        # 設置prompt模板
        # 系統template
        system_message = SystemMessagePromptTemplate.from_template(
            """你是一個專業的股市分析師，你會根據各種股市消息結合來判斷市場走勢，像是即時新聞、法人買賣超、技術分析、交易資訊，
                請解釋你因為哪些原因和新聞而判斷多空的可能，而且最後總會在「利空」和「利多」兩個選擇中給出一個明確的股市方向。
                其中利空為股市下跌，利多為股市上漲。使用者不會透漏訊息給你，所以說明時要假裝那些即時資料都是自己上網找的。"""
        )

        # 使用者template
        human_message = HumanMessagePromptTemplate.from_template(
            "即時參考資料: {info_data} \n 問題: {question}"
        )

        # 串起template
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        # 建立 LLMChain
        chain = LLMChain(llm=llm, prompt=chat_prompt)

        return chain

    # 創建使用者提問分析模型
    def build_Analyze__Name_LLM(self):

        llm_analyze_name = ChatOpenAI(
            openai_api_key = self.OPENAI_API_KEY,
            temperature=0,
            # model="GPT-4o",
        )

        # 設置prompt模板
        # 設抓取股票名稱 系統template
        system_message_analyze_name  = SystemMessagePromptTemplate.from_template(
            """你是一個專業的股票名稱分析師，你可以從任意話中提取股票名稱，回覆只能固定格式，
               股票代號_股票名稱,...,股票代號_股票名稱 ，
               譬如： 0000_AAA,0000_BBB ，若找不到則回傳 0_0。"""
        )

        # 使用者template
        human_message_analyze_name  = HumanMessagePromptTemplate.from_template(
            "問題: {question} \n 參考股票代號和名稱: {name_reference}"
        )

        # 串起
        chat_prompt_analyze_name = ChatPromptTemplate.from_messages([system_message_analyze_name, human_message_analyze_name])


        # 建立 LLMChain
        chain_analyze_name = LLMChain(llm=llm_analyze_name, prompt=chat_prompt_analyze_name)

        return chain_analyze_name

    def infer_LLM(self, question: str) -> Iterator[str]:
        """
        進行推論並以串流方式返回結果
        Args:
            question: 使用者的問題
        Returns:
            Iterator[str]: 串流形式的回答
        """

        try:
            # 取得輸入主要股票代號
            stock_list = self.Name_Analyze_LLM.run({
                "name_reference": self.stock_clawer.data,
                "question": question
            }).split(",")
            
            # 找不到股票代號的情況
            if stock_list == ["0_0"]:
                yield "找不到股票代號"
                return

            # # 逐一取得股票資訊並直接使用串流輸出
            for stock in stock_list:
                stock_code, stock_name = stock.split("_")
                # 取得股票資訊
                info_data = self.stock_clawer.get_information(Stock_code=stock_code)

                # 使用串流方式生成回答
                for chunk in self.LLM.stream({
                    "info_data": info_data,
                    "question": "請你分析這些資訊，並逐條列點輸出重點資訊，減少冗餘資訊。"
                }):
                    yield chunk['text']

        except Exception as e:
            yield( f"發生錯誤: {str(e)}")

def analyze_stock(query: str) -> Iterator[str]:
    """
    分析股票走勢並以串流方式返回結果
    Args:
        query: 使用者的查詢
    Returns:
        Iterator[str]: 生成回答的串流
    """
    model = LLM_Model(OPENAI_API_KEY = OPENAI_API_KEY)
    for chunk in model.infer_LLM(query):
        yield chunk

