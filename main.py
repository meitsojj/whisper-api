from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper, tempfile, os, subprocess, base64

app = FastAPI()

# CORS 中间件配置 - 必须在最前面
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

model = None

def get_model():
    global model
    if model is None:
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Model loaded successfully!")
    return model

def setup_cookies():
    """从环境变量解码并设置 cookies 文件"""
    cookies_b64 = os.getenv("YT_DLP_COOKIES_B64")
    if not cookies_b64:
        print("No YT_DLP_COOKIES_B64 environment variable found")
        return None
    
    try:
        # 解码 Base64
        cookies_content = base64.b64decode(cookies_b64).decode('utf-8')
        
        # 保存到临时文件
        cookies_path = "/tmp/cookies.txt"
        with open(cookies_path, 'w') as f:
            f.write(cookies_content)
        
        print(f"Cookies file created at {cookies_path}")
        return cookies_path
    except Exception as e:
        print(f"Error setting up cookies: {str(e)}")
        return None

class YouTubeURL(BaseModel):
    url: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")
        model = get_model()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            content = await file.read()
            print(f"File size: {len(content)} bytes")
            tmp.write(content)
            tmp_path = tmp.name
        try:
            print(f"Transcribing: {tmp_path}")
            result = model.transcribe(tmp_path, language=None)
            print(f"Transcription successful: {result['text'][:50]}...")
            return {"text": result["text"], "language": result["language"]}
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"Error in transcribe: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/transcribe-youtube")
async def transcribe_youtube(data: YouTubeURL):
    try:
        print(f"Processing YouTube URL: {data.url}")
        model = get_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_template = os.path.join(tmpdir, "%(title)s.%(ext)s")
            
            # 构建 yt-dlp 命令
            cmd = [
                "yt-dlp",
                "-f", "bestaudio/best",
                "-x",
                "--audio-format", "mp3",
                "-o", output_template,
                data.url
            ]
            
            # 如果设置了 cookies 文件，添加到命令中
            cookies_path = setup_cookies()
            if cookies_path and os.path.exists(cookies_path):
                print(f"Using cookies file: {cookies_path}")
                cmd.insert(1, "--cookies")
                cmd.insert(2, cookies_path)
            else:
                print("No cookies file found, attempting download without authentication")
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                error_msg = result.stderr
                print(f"yt-dlp error: {error_msg}")
                return JSONResponse({"error": f"Download failed: {error_msg}"}, status_code=500)
            
            # 查找下载的文件
            files = os.listdir(tmpdir)
            if not files:
                return JSONResponse({"error": "No audio file downloaded"}, status_code=500)
            
            audio_file = os.path.join(tmpdir, files[0])
            print(f"Transcribing downloaded audio: {audio_file}")
            result = model.transcribe(audio_file, language=None)
            print(f"Transcription successful: {result['text'][:50]}...")
            return {"text": result["text"], "language": result["language"]}
            
    except subprocess.TimeoutExpired:
        print("Download timeout")
        return JSONResponse({"error": "Download timeout (5 minutes exceeded)"}, status_code=500)
    except Exception as e:
        print(f"Error in transcribe_youtube: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)
