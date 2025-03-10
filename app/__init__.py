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
    full_response = ""
    for chunk in orchestrator.process_query(message):
        full_response += chunk
        yield full_response

# 創建 Gradio 聊天界面
def create_app():
    chat_interface = gr.ChatInterface(
        respond,
        title="智能財經分析助手",
        description="請輸入您想詢問的問題：\n1. 股票走勢分析（例如：請分析台積電的多空走勢）\n2. 專家觀點分析（例如：專家對近期市場有什麼看法）",
        theme="soft",
        examples=[
            "請分析台積電的多空走勢",
            "請問台積電的類股資訊？",
            "專家對台積電赴美設廠有什麼看法？"
        ],
        css=custom_css
    )
    
    return chat_interface

# 啟動應用
if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)