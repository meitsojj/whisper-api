from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import tempfile
import os
import yt_dlp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 中間件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

model = None
executor = ThreadPoolExecutor(max_workers=2)

def get_model():
    global model
    if model is None:
        logger.info("Loading Whisper model...")
        model = whisper.load_model("base")
        logger.info("Model loaded successfully!")
    return model

class YouTubeURL(BaseModel):
    url: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        model = get_model()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            content = await file.read()
            logger.info(f"File size: {len(content)} bytes")
            tmp.write(content)
            tmp_path = tmp.name
        try:
            logger.info(f"Transcribing: {tmp_path}")
            result = model.transcribe(tmp_path, language=None)
            logger.info(f"Transcription successful: {result['text'][:50]}...")
            return {"text": result["text"], "language": result["language"]}
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        logger.error(f"Error in transcribe: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

def download_youtube_audio(url: str, tmpdir: str) -> str:
    """使用 yt-dlp Python 庫下載 YouTube 音頻"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(tmpdir, '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            # 重試機制
            'socket_timeout': 30,
            'retries': 3,
            'fragment_retries': 3,
            # 使用 node 運行時
            'js_runtimes': ['node'],
            # 禁用 age-gate 檢查
            'skip_unavailable_fragments': True,
            # 添加 User-Agent
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }
        
        logger.info(f"Downloading from: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_file = ydl.prepare_filename(info)
            logger.info(f"Downloaded: {audio_file}")
            return audio_file
    except Exception as e:
        logger.error(f"Download error: {str(e)}", exc_info=True)
        raise

@app.post("/transcribe-youtube")
async def transcribe_youtube(data: YouTubeURL):
    try:
        logger.info(f"Processing YouTube URL: {data.url}")
        model = get_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 在線程池中運行下載，避免阻塞
            loop = asyncio.get_event_loop()
            audio_file = await loop.run_in_executor(
                executor,
                download_youtube_audio,
                data.url,
                tmpdir
            )
            
            logger.info(f"Transcribing: {audio_file}")
            result = model.transcribe(audio_file, language=None)
            logger.info(f"Transcription successful: {result['text'][:50]}...")
            return {"text": result["text"], "language": result["language"]}
            
    except Exception as e:
        logger.error(f"Error in transcribe_youtube: {str(e)}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
