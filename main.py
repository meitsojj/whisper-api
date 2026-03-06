from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import whisper
import tempfile
import os
import yt_dlp

app = FastAPI()
model = None

class YouTubeURL(BaseModel):
    url: str

def get_model():
    global model
    if model is None:
        model = whisper.load_model("base")
    return model

@app.get("/")
async def root():
    """根路由 - 健康檢查"""
    return {"status": "ok", "message": "Whisper API is running"}

@app.get("/health")
async def health():
    """健康檢查端點"""
    return {"status": "healthy"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """轉錄上傳的音頻文件"""
    model = get_model()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = model.transcribe(tmp_path, language=None)
        return {"text": result["text"], "language": result["language"]}
    finally:
        os.unlink(tmp_path)

@app.post("/transcribe-youtube")
async def transcribe_youtube(data: YouTubeURL):
    """從 YouTube URL 下載音頻並轉錄"""
    model = get_model()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # 使用 yt-dlp 下載音頻
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(tmpdir, '%(title)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(data.url, download=True)
                audio_file = os.path.join(tmpdir, f"{info['title']}.mp3")
            
            # 轉錄音頻
            result = model.transcribe(audio_file, language=None)
            return {
                "title": info['title'],
                "text": result["text"],
                "language": result["language"]
            }
        except Exception as e:
            return {"error": str(e)}
