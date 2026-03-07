from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper, tempfile, os, subprocess

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
            print(f"Downloading audio...")
            subprocess.run([
                "yt-dlp", "-x", "--audio-format", "mp3",
                "--js-runtimes", "node",
                "-o", output_template, data.url
            ], check=True, capture_output=True)
            
            files = os.listdir(tmpdir)
            if not files:
                return JSONResponse({"error": "Failed to download audio"}, status_code=500)
            
            audio_file = os.path.join(tmpdir, files[0])
            print(f"Transcribing: {audio_file}")
            result = model.transcribe(audio_file, language=None)
            print(f"Transcription successful: {result['text'][:50]}...")
            return {"text": result["text"], "language": result["language"]}
    except subprocess.CalledProcessError as e:
        error_msg = f"Download failed: {e.stderr.decode()}"
        print(f"Error: {error_msg}")
        return JSONResponse({"error": error_msg}, status_code=500)
    except Exception as e:
        print(f"Error in transcribe_youtube: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)
