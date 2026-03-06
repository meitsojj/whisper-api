from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper, tempfile, os, subprocess

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    result = model.transcribe(tmp_path, language=None)
    os.unlink(tmp_path)
    return {"text": result["text"], "language": result["language"]}

@app.post("/transcribe-youtube")
async def transcribe_youtube(data: dict):
    url = data.get("url")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir="/tmp") as tmp:
        tmp_path = tmp.name
    subprocess.run([
        "yt-dlp", "-x", "--audio-format", "mp3",
        "-o", tmp_path, url
    ], check=True)
    result = model.transcribe(tmp_path, language=None)
    os.unlink(tmp_path)
    return {"text": result["text"], "language": result["language"]}
