import gradio as gr
from agent_orchestrator import AgentOrchestrator

# 自定義超連結 CSS 樣式
custom_css = """
    .message-wrap a {
        color: #2563eb !important;  /* 藍色連結 */
        text-decoration: underline !important;
    }
    .message-wrap a:hover {
        color: #1d4ed8 !important;  /* 滑鼠懸停時的深藍色 */
    }
"""

# 初始化代理編排器
orchestrator = AgentOrchestrator()

def respond(message, history):
    """
    處理聊天訊息並返回回應
    """
    response = orchestrator.process_query(message)
    if response.type == "generator":
        full_response = ""
        for chunk in response.content:
            full_response += chunk
            yield full_response
    elif response.type == "string":
        yield response.content

# 創建 Gradio 聊天界面
def create_app():
    chat_interface = gr.ChatInterface(
        respond,
        title="智能財經分析助手",
        description="請輸入您想詢問的問題：\n1. 股票名稱或代號（例如：2330、台積電）\n2. 股票走勢分析（例如：請分析台積電的多空走勢）\n3. 專家觀點分析（例如：專家對台積電赴美設廠的看法）\n4. 法人買賣情況（例如：0050的法人買賣狀況）\n5. 張忠謀相關事蹟或看法（例如：張忠謀對於台灣經濟發展的看法）",
        theme="soft",
        examples=[
            "台積電",
            "請分析台積電的多空走勢",
            "專家對台積電赴美設廠有什麼看法",
            "0050最高融資餘額時，當時的法人買賣狀況",
            "張忠謀對於台灣經濟發展的看法"
        ],
        css=custom_css
    )
    
    return chat_interface

# 啟動應用
if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)