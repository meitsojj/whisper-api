from fastapi import FastAPI, UploadFile, File
import whisper, tempfile, os

app = FastAPI()
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    result = model.transcribe(tmp_path, language=None)
    os.unlink(tmp_path)
    return {"text": result["text"], "language": result["language"]}
```

**`requirements.txt`**
```
fastapi
uvicorn
openai-whisper
python-multipart
