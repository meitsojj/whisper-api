from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import tempfile
import os
import yt_dlp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import json
from datetime import timedelta
import io

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper API", version="1.0.0")

# CORS 中間件配置 - 允許所有來源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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

class YouTubeRequest(BaseModel):
    url: str
    language: str = "zh"

class TranscribeResponse(BaseModel):
    text: str
    segments: list
    language: str

def seconds_to_srt_time(seconds):
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((td.total_seconds() % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def generate_srt(segments):
    """Generate SRT subtitle content from segments"""
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        start_time = seconds_to_srt_time(segment['start'])
        end_time = seconds_to_srt_time(segment['end'])
        text = segment['text'].strip()
        srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt_content

def download_youtube_audio(url: str) -> str:
    """Download audio from YouTube and return file path"""
    try:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "%(title)s.%(ext)s")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading audio from {url}")
            info = ydl.extract_info(url, download=True)
            audio_file = ydl.prepare_filename(info)
            # Convert to mp3 if needed
            mp3_file = audio_file.rsplit('.', 1)[0] + '.mp3'
            if os.path.exists(mp3_file):
                return mp3_file
            return audio_file
    except Exception as e:
        logger.error(f"Error downloading YouTube audio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download YouTube audio: {str(e)}")

def transcribe_audio(audio_path: str, language: str = None):
    """Transcribe audio file using Whisper"""
    try:
        model = get_model()
        logger.info(f"Transcribing audio: {audio_path}")
        
        options = {}
        if language:
            options['language'] = language
        
        result = model.transcribe(audio_path, **options)
        return result
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Whisper API"}

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_file(file: UploadFile = File(...), language: str = None):
    """
    Transcribe uploaded audio file
    
    Parameters:
    - file: Audio file (mp3, wav, m4a, etc.)
    - language: Language code (optional, e.g., 'zh', 'en')
    
    Returns:
    - text: Full transcription text
    - segments: List of segments with timestamps
    - language: Detected language
    """
    temp_file = None
    try:
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        # Transcribe
        result = await asyncio.get_event_loop().run_in_executor(
            executor, 
            transcribe_audio, 
            temp_file.name,
            language
        )
        
        return TranscribeResponse(
            text=result['text'],
            segments=result['segments'],
            language=result['language']
        )
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@app.post("/transcribe-youtube")
async def transcribe_youtube(request: YouTubeRequest):
    """
    Transcribe YouTube video
    
    Parameters:
    - url: YouTube URL
    - language: Language code (optional)
    
    Returns:
    - text: Full transcription text
    - segments: List of segments with timestamps
    - language: Detected language
    """
    audio_file = None
    try:
        # Download audio
        audio_file = await asyncio.get_event_loop().run_in_executor(
            executor,
            download_youtube_audio,
            request.url
        )
        
        # Transcribe
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            transcribe_audio,
            audio_file,
            request.language
        )
        
        return TranscribeResponse(
            text=result['text'],
            segments=result['segments'],
            language=result['language']
        )
    except Exception as e:
        logger.error(f"Error in transcribe-youtube endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if audio_file and os.path.exists(audio_file):
            os.unlink(audio_file)

@app.post("/transcribe/srt")
async def transcribe_file_srt(file: UploadFile = File(...), language: str = None):
    """
    Transcribe uploaded audio file and return SRT subtitle file
    
    Parameters:
    - file: Audio file (mp3, wav, m4a, etc.)
    - language: Language code (optional)
    
    Returns:
    - SRT subtitle file for download
    """
    temp_file = None
    try:
        # Save uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        
        # Transcribe
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            transcribe_audio,
            temp_file.name,
            language
        )
        
        # Generate SRT
        srt_content = generate_srt(result['segments'])
        
        # Return as file download
        return StreamingResponse(
            iter([srt_content]),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=subtitles.srt"}
        )
    except Exception as e:
        logger.error(f"Error in transcribe-srt endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

@app.post("/transcribe-youtube/srt")
async def transcribe_youtube_srt(request: YouTubeRequest):
    """
    Transcribe YouTube video and return SRT subtitle file
    
    Parameters:
    - url: YouTube URL
    - language: Language code (optional)
    
    Returns:
    - SRT subtitle file for download
    """
    audio_file = None
    try:
        # Download audio
        audio_file = await asyncio.get_event_loop().run_in_executor(
            executor,
            download_youtube_audio,
            request.url
        )
        
        # Transcribe
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            transcribe_audio,
            audio_file,
            request.language
        )
        
        # Generate SRT
        srt_content = generate_srt(result['segments'])
        
        # Return as file download
        return StreamingResponse(
            iter([srt_content]),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=subtitles.srt"}
        )
    except Exception as e:
        logger.error(f"Error in transcribe-youtube-srt endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if audio_file and os.path.exists(audio_file):
            os.unlink(audio_file)

@app.get("/")
async def root():
    """API documentation"""
    return {
        "service": "Whisper API",
        "version": "1.0.0",
        "endpoints": {
            "POST /transcribe": "Transcribe uploaded audio file",
            "POST /transcribe-youtube": "Transcribe YouTube video",
            "POST /transcribe/srt": "Transcribe audio and get SRT file",
            "POST /transcribe-youtube/srt": "Transcribe YouTube and get SRT file",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
