import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import tejapi
import json
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64

# 載入環境變數
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEJ_API_KEY = os.getenv("TEJ_API_KEY")

def set_basic_data():
    # TEJ API 設定
    tejapi.ApiConfig.api_key = TEJ_API_KEY

    # 取得 TEJ 資料
    data_chinese = tejapi.get('TRAIL/AIND', chinese_column_name=True)
    data_chinese_copy = data_chinese.copy()

    # 處理日期格式
    for col in data_chinese_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(data_chinese_copy[col]):
            try:
                data_chinese_copy[col] = data_chinese_copy[col].dt.strftime('%Y-%m-%d')
            except AttributeError:
                print(f"Column '{col}' is not a datetime object.")

    # 欄位名稱調整
    for col in data_chinese_copy.columns:
        # Check if the column name appears more than once
        if data_chinese_copy.columns.tolist().count(col) > 1:
            # Get the indices of the duplicate columns
            duplicate_indices = [i for i, c in enumerate(data_chinese_copy.columns) if c == col]
            # Rename the duplicate columns, but keep the first occurrence as is
            for i, index in enumerate(duplicate_indices):
                if i == 0:
                    new_col_name = f"{col}"
                else:
                    new_col_name = f"{col}_{i}"
                    data_chinese_copy.columns.values[index] = new_col_name

    # 選擇需要的欄位
    data_used = data_chinese_copy[['證期會代碼', '公司中文全稱', '公司中文簡稱', 'TSE新產業_代碼', 'TSE新產業_名稱', '董事長', '實收資本額(元)','危機事件大類別說明']].copy()



    # 轉換成 JSON
    data_used.to_json('data_used.json', orient='records')

# 類股API
sector_mapping = {
    "水泥工業": "1",
    "食品工業": "2",
    "塑膠工業": "3",
    "紡織纖維": "4",
    "電機機械": "6",
    "電器電纜": "7",
    "化學工業": "37",
    "生技醫療": "38",
    "玻璃陶瓷": "9",
    "造紙工業": "10",
    "鋼鐵工業": "11",
    "橡膠工業": "12",
    "汽車工業": "13",
    "半導體": "40",
    "電腦及週邊": "41",
    "光電業": "42",
    "通信網路業": "43",
    "電子零組件": "44",
    "電子通路業": "45",
    "資訊服務業": "46",
    "其他電子業": "47",
    "建材營造": "19",
    "航運業": "20",
    "觀光餐旅": "21",
    "金融業": "22",
    "貿易百貨": "24",
    "油電燃氣業": "39",
    "存託憑證": "25",
    "ETF": "26",
    "受益證券": "29",
    "ETN": "48",
    "創新板": "49",
    "其他": "30",
    "運動休閒": "95",
    "居家生活": "96",
    "數位雲端": "94",
    "綠能環保": "93",
}

def get_sector_id(sector_name):
    """根據類股名稱獲取對應的ID"""
    return sector_mapping.get(sector_name, None)

def fetch_stock_data(sector_id, exchange="TAI"):
    """爬取指定類股和交易所的股票資料"""
    url = f"https://tw.stock.yahoo.com/_td-stock/api/resource/StockServices.getClassQuotes;exchange={exchange};offset=0;sectorId={sector_id}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if 'list' in data and isinstance(data['list'], list):
            stock_list = data['list']
        else:
            print("Unexpected data format from API.")
            return []

        stocks_data = []
        for item in stock_list:
            stock_info = {
                "股票代碼": item.get("symbol", ""),
                "股票名稱": item.get("symbolName", ""),
                "產業類別": item.get("sectorName", ""),
                "當前價格": item.get("price", {}).get("fmt", "") if isinstance(item.get("price"), dict) else "",
                "漲跌": item.get("change", {}).get("fmt", "") if isinstance(item.get("change"), dict) else "",
                "漲跌幅": item.get("changePercent", ""),
                "成交量": item.get("volume", ""),
            }
            stocks_data.append(stock_info)
        return stocks_data
    else:
        print(f"請求失敗，狀態碼: {response.status_code}")
        return []

# 設計 Prompt Template

# 設定 OpenAI 提示模板
stock_analysis_template = """
請分析以下股票資訊並提供專業的見解：

股票基本資訊：
{stock_info}

同產業股票資訊：
{sector_info}

請提供以下分析：
1. 該股票在同產業中的市場地位
2. 主要競爭對手分析
3. 當前股價表現評估
4. 產業趨勢分析
5. 投資建議

請以專業且易懂的方式呈現分析結果。
"""

# 創建 LLM Chain
llm = ChatOpenAI(temperature=0.7)
stock_analysis_prompt = PromptTemplate(
    input_variables=["stock_info", "sector_info"],
    template=stock_analysis_template
)
stock_analysis_chain = LLMChain(
    llm=llm,
    prompt=stock_analysis_prompt
)

def analyze_stock_data(stock_info, sector_stocks):
    """使用 OpenAI 分析股票資料"""
    stock_info_str = json.dumps(stock_info, ensure_ascii=False, indent=2)
    sector_info_str = json.dumps(sector_stocks, ensure_ascii=False, indent=2)

    analysis = stock_analysis_chain.run(
        stock_info=stock_info_str,
        sector_info=sector_info_str
    )
    return analysis

# 設定基本字型
plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用系統內建的字型

# 接著是原本的視覺化函數
def create_stock_visualization(stocks):
    """Create an alternative visualization for stock comparison"""
    # Prepare data
    data = stocks[:10]  # Get top 10 stocks

    # Extract and process data
    codes = [stock['股票代碼'] for stock in data]
    prices = [float(stock['當前價格'].replace(',', '')) for stock in data]
    changes = [float(stock['漲跌幅'].replace('%', '')) for stock in data]

    # Create figure with specific size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Color palette
    colors = ['#FF4B4B' if c > 0 else '#00A99D' for c in changes]

    # Plot 1: Horizontal bar chart for prices
    sns.barplot(y=codes, x=prices, ax=ax1, palette=colors, orient='h')
    ax1.set_title('Stock Prices', pad=15)
    ax1.set_xlabel('Price (TWD)')
    ax1.set_ylabel('Stock Code')

    # Add price annotations
    for i, p in enumerate(prices):
        ax1.text(p, i, f' {p:,.2f}', va='center')

    # Plot 2: Heatmap-style visualization for price changes
    change_data = np.array(changes).reshape(-1, 1)
    sns.heatmap(change_data,
                yticklabels=codes,
                xticklabels=['Change %'],
                ax=ax2,
                cmap='RdYlGn',
                center=0,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Price Change (%)'})
    ax2.set_title('Price Changes (%)', pad=15)

    # Adjust layout
    plt.tight_layout()

    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    return base64.b64encode(image_png).decode()

def process_stock_query(query):
    """處理股票查詢並提供 AI 分析"""
    # 讀取股票資料
    with open('data_used.json', 'r') as f:
        stock_data = json.load(f)

    # 建立查詢索引
    stock_index = {}
    for stock in stock_data:
        stock_index[stock['證期會代碼']] = stock
        stock_index[stock['公司中文簡稱']] = stock

    # 查找股票
    stock_info = stock_index.get(query)
    if stock_info:
        sector_name = stock_info['TSE新產業_名稱']
        sector_id = get_sector_id(sector_name)

        # 基本資訊
        basic_info = f"""
        找到股票資訊：
        代碼：{stock_info['證期會代碼']}
        公司名稱：{stock_info['公司中文全稱']}
        產業類別：{sector_name}
        董事長：{stock_info['董事長']}
        """

        # 獲取同產業股票資料
        same_sector_stocks = []
        if sector_id:
            same_sector_stocks = fetch_stock_data(sector_id)

        # 使用 OpenAI 進行分析
        analysis = analyze_stock_data(stock_info, same_sector_stocks)

        # 組合完整回應
        response = f"{basic_info}\n\nAI 分析報告：\n{analysis}"

        if same_sector_stocks:
                # 生成視覺化圖表
                chart_base64 = create_stock_visualization(same_sector_stocks)

                # 添加文字說明
                response += "\n\nSector Stock Information:\n"
                for stock in same_sector_stocks[:10]:
                    response += f"{stock['股票代碼']} {stock['股票名稱']} - Price: {stock['當前價格']} Change: {stock['漲跌']} ({stock['漲跌幅']})\n"

                # 添加圖表
                response += f"\n\n<img src='data:image/png;base64,{chart_base64}' alt='Stock Analysis Visualization'>"

        return response
    else:
        return "找不到相關股票資訊"