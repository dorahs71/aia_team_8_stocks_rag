import gradio as gr
from generation import generate_answer

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

def respond(message, history):
    """
    處理聊天訊息並返回回應
    """
    response = ""
    for chunk in generate_answer(message):
        response += chunk
        yield response

# 創建 Gradio 聊天界面
def create_app():
    chat_interface = gr.ChatInterface(
        respond,
        title="股票趨勢分析AI",
        description="請輸入您想詢問的股票趨勢、融資融券的相關問題，我將以專業的方式回答。",
        theme="soft",
        examples=["台積電最近的交易資訊如何?", 
                "請問台積電的類股資訊？",
                "台積電赴美設廠對台股的影響？"],
        css=custom_css
    )
    
    return chat_interface

# 啟動應用
if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)