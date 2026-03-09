from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os
import tempfile
import json
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper API")

# CORS 設置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局變數
model = None
executor = ThreadPoolExecutor(max_workers=2)

@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Loading Whisper model...")
    try:
        model = whisper.load_model("base")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    轉譯上傳的音頻文件
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    temp_file = None
    try:
        # 保存上傳的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name
        
        logger.info(f"Processing file: {file.filename}, size: {len(content)} bytes")
        
        # 在線程池中運行轉譯（避免阻塞事件循環）
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: model.transcribe(temp_file, language="zh")
        )
        
        # 生成 SRT 字幕
        srt_content = generate_srt(result["segments"])
        
        return {
            "text": result["text"],
            "segments": result["segments"],
            "srt": srt_content,
            "language": result.get("language", "unknown")
        }
    
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 清理臨時文件
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

@app.post("/transcribe-srt")
async def transcribe_srt(file: UploadFile = File(...)):
    """
    轉譯並返回 SRT 文件下載
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name
        
        logger.info(f"Processing file for SRT: {file.filename}")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: model.transcribe(temp_file, language="zh")
        )
        
        srt_content = generate_srt(result["segments"])
        
        # 返回 SRT 文件下載
        srt_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".srt", encoding='utf-8')
        srt_file.write(srt_content)
        srt_file.close()
        
        return FileResponse(
            srt_file.name,
            media_type="text/plain",
            filename=f"{Path(file.filename).stem}.srt"
        )
    
    except Exception as e:
        logger.error(f"SRT generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

def generate_srt(segments):
    """
    從 Whisper 的 segments 生成 SRT 字幕
    """
    srt_lines = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")
    
    return "\n".join(srt_lines)

def format_timestamp(seconds):
    """
    將秒數轉換為 SRT 時間戳格式 (HH:MM:SS,mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# 靜態文件
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        timeout_keep_alive=3600,
        timeout_graceful_shutdown=3600
    )
