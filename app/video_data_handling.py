import yt_dlp
import os
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase import create_client, Client


# Constants
VIDEO_SUBSCRIBE_URL = 'https://www.youtube.com/watch?v=nFjZaBLi9TE'
VIDEO_AUDIO_URL = 'https://www.youtube.com/watch?v=_dILH2h6Dkk'
SUBTITLE_VTT_OUTPUT_DIR = './subtitles'
AUDIO_FILE_OUTPUT_DIR = './audio'
TRANSCRIBE_MODEL_SIZE = 'small'
TXT_OUTPUT_DIR = './data'

def download_youtube_subtitle(url, path):
    ydl_opts = {
        'writesubtitles': True,          # 寫入字幕
        'subtitleslangs': ['zh-TW'],    # 偏好的字幕語言
        'subtitlesformat': 'vtt',       # 字幕格式指定為 vtt
        'outtmpl': f'{path}/%(title)s', # 輸出檔名格式 (使用 f-string)
        'skip_download': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 先提取資訊，不下載
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict['title'].replace("?", "？").replace(":", "：").replace("!", "！").replace("/", "⧸")
            subtitle_path = f"{path}/{video_title}.zh-TW.vtt"  # 生成完整的檔案路徑

            #檢查檔案是否已存在
            if os.path.exists(subtitle_path):
                print(f"字幕檔案已存在：{subtitle_path}，跳過下載。")
                
            else:
                ydl.download([url])  # 執行下載
                print(f"字幕檔案已成功下載至：{subtitle_path}")
            
            return url, video_title, subtitle_path
        
    except Exception as e: # 捕捉更廣泛的異常，例如 yt_dlp 自己的 exceptions
        print(f"字幕下載失敗 (使用 yt_dlp 函式庫)，錯誤訊息：{e}")
    except FileNotFoundError: # 仍然保留 FileNotFoundError 處理，以防 yt_dlp 函式庫本身未安裝
        print("錯誤：yt-dlp 函式庫未找到，請確認您已安裝 yt-dlp (例如使用 pip install yt-dlp)。")

def download_youtube_audio(url, path):
    try:
        ydl_opts = {
            'extractaudio': True,  # 只下載音訊
            'audioformat': 'mp3',   # 音訊格式為 mp3
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': f'{path}/%(title)s',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            audio_title = info_dict['title'].replace("?", "？").replace(":", "：").replace("!", "！").replace("/", "⧸")
            audio_filepath = os.path.join(path, f"{audio_title}.mp3") # 假設輸出為 mp3 格式

            if os.path.exists(audio_filepath):
                print(f"音訊檔案已存在：{audio_filepath}，跳過下載。")
            
            else:
                info_dict = ydl.extract_info(url, download=True)
                print(f"成功下載 YouTube 音訊至: {audio_filepath}")
            
            return url, audio_title, audio_filepath

    except Exception as e:
        print(f"下載 YouTube 音訊時發生錯誤: {e}")
        return None
    
def transcribe_audio_to_text(audio_filepath, model_size): #  預設模型大小為 small
    try:
        # 加載 Whisper 模型 (根據 model_size 選擇模型)
        model = WhisperModel(model_size, device="cpu", compute_type="int8") #  device 可選 "cpu" 或 "cuda" (如果您的電腦有 NVIDIA GPU)

        # 進行音訊轉錄
        segments, info = model.transcribe(audio_filepath, beam_size=5, language="zh") #  beam_size 參數可以調整轉錄品質和速度

        print(f"偵測到的語言:{info.language}，開始轉錄文字") #  印出偵測到的語言和機率

        transcribed_text = []
        for i, segment in enumerate(segments, 1):# 迭代轉錄片段，將文字片段拼接起來
            if i <= 5:  # 僅打印前 5 個片段
                print(f"Segment {i}: {segment.text}")
            transcribed_text.append(segment.text)  # 將每個 segment 加入列表

        return transcribed_text

    except Exception as e:
        print(f"使用 faster-whisper 轉錄音訊時發生錯誤: {e}")
        return None

    
def clean_vtt_content(vtt_content):
    cleaned_subtitle_lines = []  # 存放清理後的字幕行
    lines = vtt_content.splitlines()  # 按行分割 VTT 內容

    for line in lines:
        line = line.strip()  # 去除每行首尾空白
        if not line or line.startswith(('WEBVTT', 'Kind:', 'Language:')) or "-->" in line:
            continue  # 忽略空白行、標頭行和時間戳記行
        cleaned_subtitle_lines.append(line)  # 保留字幕文字行
    
    print("已完成字幕清理")

    return cleaned_subtitle_lines

def save_subtitles_to_txt_and_return_output_path(text_lines, dir_path, video_title):
    try:
        # 使用 'w' 模式開啟檔案，以寫入文字 (如果檔案已存在會被覆蓋)
        output_path = os.path.join(dir_path, f"{video_title}.txt")
        with open(output_path, 'w', encoding='utf-8') as txt_file:
            for line in text_lines:
                txt_file.write(line + '\n')

        print(f"文字檔案已成功儲存至: {output_path}")

        return output_path

    except Exception as e:
        print(f"儲存文字至 TXT 檔案時發生錯誤: {e}")

def process_subtitles_and_return_txt_path():
    # Get subtitles from youtube
    download_result = download_youtube_subtitle(VIDEO_SUBSCRIBE_URL, SUBTITLE_VTT_OUTPUT_DIR)

    video_url, video_title, subtitle_path = download_result

    # Read VTT content
    with open(subtitle_path, 'r', encoding='utf-8') as file:
        vtt_content = file.read()

    # Clean subtitle lines
    cleaned_subtitles = clean_vtt_content(vtt_content)

    # Save cleaned subtitles to TXT and return output path
    txt_path = save_subtitles_to_txt_and_return_output_path(cleaned_subtitles, TXT_OUTPUT_DIR, video_title)

    return txt_path, video_url

def process_audio_and_return_txt_path():

    download_result = download_youtube_audio(VIDEO_AUDIO_URL, AUDIO_FILE_OUTPUT_DIR)

    video_url, audio_title, audio_filepath  = download_result

    transcribed_text = transcribe_audio_to_text(audio_filepath, TRANSCRIBE_MODEL_SIZE)

    txt_path = save_subtitles_to_txt_and_return_output_path(transcribed_text, TXT_OUTPUT_DIR,audio_title)

    return txt_path, video_url

def subtitle_split(txt_path, video_url):
    # Load sample data
    with open(txt_path) as f:
        state_of_the_union = f.read()

    file_name_parts = os.path.basename(txt_path).split('│') # 假設影片名稱在 '│' 之前
    # 取得影片名稱，如果分割失敗則使用完整檔名
    video_name = file_name_parts[0].strip() if file_name_parts else os.path.basename(txt_path) 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=200, 
        add_start_index=True,
        is_separator_regex=False,
    )

    combined_metadata = {'video_name': video_name, 'video_url': video_url}

    documents = text_splitter.create_documents([state_of_the_union], [combined_metadata])

    return documents

# 載入 .env 檔案中的環境變數
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings()


embeddings = OpenAIEmbeddings()

def save_to_vector_store(splits): 
    try:
        SupabaseVectorStore.from_documents(
            splits,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
        )
        print("成功將文件向量存入 Supabase")
    except Exception as e:
        print(f"將文件向量存入 Supabase 時發生錯誤: {str(e)}")

def get_retriever():
    vectorstore = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents", # 資料表名稱
    query_name="match_documents", # 查詢函式名稱
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return retriever



