from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper, tempfile, os, subprocess

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

def get_model():
    global model
    if model is None:
        model = whisper.load_model("base")
    return model

class YouTubeURL(BaseModel):
    url: str

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
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
    try:
        model = get_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_template = os.path.join(tmpdir, "%(title)s.%(ext)s")
            subprocess.run([
                "yt-dlp", "-x", "--audio-format", "mp3",
                "-o", output_template, data.url
            ], check=True, capture_output=True)
            
            files = os.listdir(tmpdir)
            if not files:
                return {"error": "Failed to download audio"}
            
            audio_file = os.path.join(tmpdir, files[0])
            result = model.transcribe(audio_file, language=None)
            return {"text": result["text"], "language": result["language"]}
    except subprocess.CalledProcessError as e:
        return {"error": f"Download failed: {e.stderr.decode()}"}
    except Exception as e:
        return {"error": str(e)}
