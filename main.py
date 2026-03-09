from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import whisper
import os
import tempfile
import json
from datetime import datetime
import uvicorn

app = FastAPI()

# Load Whisper model
model = whisper.load_model("base")

# HTML Web UI
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Whisper API - Audio Transcription</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px; }
        .container { background: white; border-radius: 12px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); max-width: 600px; width: 100%; padding: 40px; }
        h1 { color: #333; margin-bottom: 10px; font-size: 28px; }
        .subtitle { color: #666; margin-bottom: 30px; font-size: 14px; }
        .upload-area { border: 2px dashed #667eea; border-radius: 8px; padding: 40px; text-align: center; cursor: pointer; transition: all 0.3s; margin-bottom: 20px; }
        .upload-area:hover { border-color: #764ba2; background: #f8f9ff; }
        .upload-area.dragover { border-color: #764ba2; background: #f0f2ff; }
        .upload-area input { display: none; }
        .upload-icon { font-size: 48px; margin-bottom: 10px; }
        .upload-text { color: #333; font-weight: 500; margin-bottom: 5px; }
        .upload-hint { color: #999; font-size: 12px; }
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 30px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 600; transition: transform 0.2s; width: 100%; margin-top: 10px; }
        button:hover { transform: translateY(-2px); }
        button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .result { margin-top: 30px; padding: 20px; background: #f8f9ff; border-radius: 8px; display: none; }
        .result.show { display: block; }
        .result-title { color: #333; font-weight: 600; margin-bottom: 15px; }
        .transcript-text { background: white; padding: 15px; border-radius: 6px; color: #333; line-height: 1.6; margin-bottom: 15px; max-height: 300px; overflow-y: auto; }
        .segments { background: white; padding: 15px; border-radius: 6px; max-height: 300px; overflow-y: auto; }
        .segment { padding: 10px; border-bottom: 1px solid #eee; font-size: 13px; }
        .segment:last-child { border-bottom: none; }
        .segment-time { color: #667eea; font-weight: 600; }
        .segment-text { color: #333; margin-top: 5px; }
        .download-btn { background: #10b981; margin-top: 10px; }
        .download-btn:hover { background: #059669; }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { color: #dc2626; padding: 15px; background: #fee2e2; border-radius: 6px; margin-top: 15px; display: none; }
        .error.show { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Whisper Transcription</h1>
        <p class="subtitle">Upload audio files to transcribe with OpenAI's Whisper</p>
        
        <div class="upload-area" id="uploadArea">
            <input type="file" id="fileInput" accept="audio/*" />
            <div class="upload-icon">📁</div>
            <div class="upload-text">Click to upload or drag and drop</div>
            <div class="upload-hint">MP3, WAV, M4A, FLAC, OGG (Max 25MB)</div>
        </div>
        
        <button id="transcribeBtn" disabled>Transcribe Audio</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 10px; color: #666;">Processing audio...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="result" id="result">
            <div class="result-title">Transcription Result</div>
            <div class="transcript-text" id="transcript"></div>
            <button class="download-btn" id="downloadBtn">⬇️ Download SRT</button>
            <div style="margin-top: 15px;">
                <div class="result-title" style="margin-bottom: 10px;">Segments with Timestamps</div>
                <div class="segments" id="segments"></div>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const transcribeBtn = document.getElementById('transcribeBtn');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        const error = document.getElementById('error');
        const transcript = document.getElementById('transcript');
        const segments = document.getElementById('segments');
        const downloadBtn = document.getElementById('downloadBtn');
        
        let selectedFile = null;
        let transcriptionData = null;

        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect();
            }
        });
        
        fileInput.addEventListener('change', handleFileSelect);
        
        function handleFileSelect() {
            selectedFile = fileInput.files[0];
            if (selectedFile) {
                transcribeBtn.disabled = false;
                uploadArea.querySelector('.upload-text').textContent = selectedFile.name;
            }
        }
        
        transcribeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            loading.style.display = 'block';
            error.classList.remove('show');
            result.classList.remove('show');
            transcribeBtn.disabled = true;
            
            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Transcription failed');
                }
                
                transcriptionData = await response.json();
                displayResult(transcriptionData);
            } catch (err) {
                showError(err.message);
            } finally {
                loading.style.display = 'none';
                transcribeBtn.disabled = false;
            }
        });
        
        function displayResult(data) {
            transcript.textContent = data.text;
            
            segments.innerHTML = '';
            data.segments.forEach(seg => {
                const div = document.createElement('div');
                div.className = 'segment';
                div.innerHTML = `
                    <div class="segment-time">${formatTime(seg.start)} → ${formatTime(seg.end)}</div>
                    <div class="segment-text">${seg.text}</div>
                `;
                segments.appendChild(div);
            });
            
            result.classList.add('show');
        }
        
        function formatTime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
        }
        
        function showError(message) {
            error.textContent = message;
            error.classList.add('show');
        }
        
        downloadBtn.addEventListener('click', () => {
            if (!transcriptionData) return;
            
            let srtContent = '';
            transcriptionData.segments.forEach((seg, idx) => {
                const start = formatSRTTime(seg.start);
                const end = formatSRTTime(seg.end);
                srtContent += `${idx + 1}\n${start} --> ${end}\n${seg.text}\n\n`;
            });
            
            const blob = new Blob([srtContent], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `transcription_${new Date().getTime()}.srt`;
            a.click();
            URL.revokeObjectURL(url);
        });
        
        function formatSRTTime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            const ms = Math.floor((seconds % 1) * 1000);
            return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the Web UI"""
    return HTML_CONTENT

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio file"""
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Transcribe with Whisper
            result = model.transcribe(tmp_path)
            
            # Format response with segments
            return {
                "text": result["text"],
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip()
                    }
                    for seg in result["segments"]
                ]
            }
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
