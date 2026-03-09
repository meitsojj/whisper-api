from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse
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
    minutes, secs = divmod(remainder, 60)
    milliseconds = int((td.total_seconds() % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def generate_srt(segments):
    """Generate SRT subtitle content from segments"""
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        start_time = seconds_to_srt_time(segment['start'])
        end_time = seconds_to_srt_time(segment['end'])
        text = segment['text'].strip()
        srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt_content

@app.get("/", response_class=HTMLResponse)
async def root():
    """Web UI for Whisper API"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Whisper API - 語音轉文字</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 600px;
                width: 100%;
                padding: 40px;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 28px;
            }
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 14px;
            }
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 30px;
                border-bottom: 2px solid #eee;
            }
            .tab-btn {
                padding: 12px 20px;
                border: none;
                background: none;
                cursor: pointer;
                font-size: 14px;
                color: #666;
                border-bottom: 3px solid transparent;
                transition: all 0.3s;
            }
            .tab-btn.active {
                color: #667eea;
                border-bottom-color: #667eea;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 500;
                font-size: 14px;
            }
            input[type="text"],
            input[type="file"],
            select {
                width: 100%;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            input[type="text"]:focus,
            input[type="file"]:focus,
            select:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            .file-input-wrapper {
                position: relative;
                overflow: hidden;
            }
            .file-input-wrapper input[type="file"] {
                position: absolute;
                left: -9999px;
            }
            .file-input-label {
                display: block;
                padding: 12px;
                background: #f5f5f5;
                border: 2px dashed #ddd;
                border-radius: 6px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
            }
            .file-input-label:hover {
                border-color: #667eea;
                background: #f9f7ff;
            }
            .file-name {
                margin-top: 8px;
                color: #667eea;
                font-size: 12px;
            }
            .button-group {
                display: flex;
                gap: 10px;
                margin-top: 30px;
            }
            button {
                flex: 1;
                padding: 12px 20px;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            .btn-primary:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .btn-secondary {
                background: #f5f5f5;
                color: #333;
            }
            .btn-secondary:hover {
                background: #eee;
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                background: #f9f7ff;
                border-radius: 6px;
                border-left: 4px solid #667eea;
                display: none;
            }
            .result.show {
                display: block;
            }
            .result-title {
                color: #667eea;
                font-weight: 600;
                margin-bottom: 12px;
            }
            .result-text {
                color: #333;
                line-height: 1.6;
                font-size: 14px;
                word-break: break-word;
                max-height: 300px;
                overflow-y: auto;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .loading.show {
                display: block;
            }
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error {
                color: #e74c3c;
                margin-top: 10px;
                padding: 10px;
                background: #fadbd8;
                border-radius: 4px;
                display: none;
            }
            .error.show {
                display: block;
            }
            .info {
                background: #e8f4f8;
                border-left: 4px solid #3498db;
                padding: 12px;
                border-radius: 4px;
                font-size: 13px;
                color: #2c3e50;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Whisper API</h1>
            <p class="subtitle">AI 語音轉文字服務</p>
            
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('upload')">上傳音檔</button>
                <button class="tab-btn" onclick="switchTab('youtube')">YouTube 影片</button>
            </div>
            
            <!-- Upload Tab -->
            <div id="upload" class="tab-content active">
                <div class="info">上傳音檔進行語音轉文字，支援 MP3、WAV、M4A 等格式</div>
                <div class="form-group">
                    <label>選擇音檔</label>
                    <div class="file-input-wrapper">
                        <input type="file" id="audioFile" accept="audio/*" onchange="updateFileName()">
                        <label for="audioFile" class="file-input-label">
                            📁 點擊選擇或拖放音檔
                        </label>
                        <div class="file-name" id="fileName"></div>
                    </div>
                </div>
                <div class="form-group">
                    <label>語言</label>
                    <select id="language">
                        <option value="zh">中文</option>
                        <option value="en">English</option>
                        <option value="ja">日本語</option>
                        <option value="ko">한국어</option>
                    </select>
                </div>
                <div class="button-group">
                    <button class="btn-primary" onclick="transcribeAudio()">轉錄文字</button>
                    <button class="btn-secondary" onclick="downloadSRT('audio')">下載字幕</button>
                </div>
            </div>
            
            <!-- YouTube Tab -->
            <div id="youtube" class="tab-content">
                <div class="info">輸入 YouTube 影片連結進行語音轉文字</div>
                <div class="form-group">
                    <label>YouTube 連結</label>
                    <input type="text" id="youtubeUrl" placeholder="https://www.youtube.com/watch?v=...">
                </div>
                <div class="form-group">
                    <label>語言</label>
                    <select id="youtubeLanguage">
                        <option value="zh">中文</option>
                        <option value="en">English</option>
                        <option value="ja">日本語</option>
                        <option value="ko">한국어</option>
                    </select>
                </div>
                <div class="button-group">
                    <button class="btn-primary" onclick="transcribeYouTube()">轉錄文字</button>
                    <button class="btn-secondary" onclick="downloadSRT('youtube')">下載字幕</button>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 10px; color: #666;">處理中，請稍候...</p>
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="result" id="result">
                <div class="result-title">轉錄結果</div>
                <div class="result-text" id="resultText"></div>
            </div>
        </div>
        
        <script>
            let lastResult = null;
            
            function switchTab(tab) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
                document.getElementById(tab).classList.add('active');
                event.target.classList.add('active');
            }
            
            function updateFileName() {
                const file = document.getElementById('audioFile').files[0];
                if (file) {
                    document.getElementById('fileName').textContent = '✓ ' + file.name;
                }
            }
            
            function showLoading(show) {
                document.getElementById('loading').classList[show ? 'add' : 'remove']('show');
            }
            
            function showError(message) {
                const errorEl = document.getElementById('error');
                errorEl.textContent = message;
                errorEl.classList.add('show');
                setTimeout(() => errorEl.classList.remove('show'), 5000);
            }
            
            function showResult(text) {
                document.getElementById('resultText').textContent = text;
                document.getElementById('result').classList.add('show');
            }
            
            async function transcribeAudio() {
                const file = document.getElementById('audioFile').files[0];
                if (!file) {
                    showError('請選擇音檔');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('language', document.getElementById('language').value);
                
                showLoading(true);
                document.getElementById('error').classList.remove('show');
                
                try {
                    const response = await fetch('/transcribe', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) throw new Error('轉錄失敗');
                    
                    lastResult = await response.json();
                    showResult(lastResult.text);
                } catch (error) {
                    showError('錯誤: ' + error.message);
                } finally {
                    showLoading(false);
                }
            }
            
            async function transcribeYouTube() {
                const url = document.getElementById('youtubeUrl').value;
                if (!url) {
                    showError('請輸入 YouTube 連結');
                    return;
                }
                
                showLoading(true);
                document.getElementById('error').classList.remove('show');
                
                try {
                    const response = await fetch('/transcribe-youtube', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            url: url,
                            language: document.getElementById('youtubeLanguage').value
                        })
                    });
                    
                    if (!response.ok) throw new Error('轉錄失敗');
                    
                    lastResult = await response.json();
                    showResult(lastResult.text);
                } catch (error) {
                    showError('錯誤: ' + error.message);
                } finally {
                    showLoading(false);
                }
            }
            
            async function downloadSRT(type) {
                if (!lastResult) {
                    showError('請先進行轉錄');
                    return;
                }
                
                try {
                    const endpoint = type === 'audio' ? '/transcribe/srt' : '/transcribe-youtube/srt';
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(lastResult)
                    });
                    
                    if (!response.ok) throw new Error('下載失敗');
                    
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'subtitles.srt';
                    a.click();
                } catch (error) {
                    showError('錯誤: ' + error.message);
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = "zh"):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        model = get_model()
        result = model.transcribe(tmp_path, language=language)
        
        return TranscribeResponse(
            text=result["text"],
            segments=result["segments"],
            language=result["language"]
        )
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/transcribe-youtube")
async def transcribe_youtube(request: YouTubeRequest):
    audio_file = None
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': tempfile.gettempdir() + '/%(id)s',
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=True)
            audio_file = tempfile.gettempdir() + '/' + info['id'] + '.wav'
        
        model = get_model()
        result = model.transcribe(audio_file, language=request.language)
        
        return TranscribeResponse(
            text=result["text"],
            segments=result["segments"],
            language=result["language"]
        )
    except Exception as e:
        logger.error(f"Error in transcribe-youtube endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if audio_file and os.path.exists(audio_file):
            os.unlink(audio_file)

@app.post("/transcribe/srt")
async def transcribe_srt(response: TranscribeResponse):
    try:
        srt_content = generate_srt(response.segments)
        return StreamingResponse(
            iter([srt_content]),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=subtitles.srt"}
        )
    except Exception as e:
        logger.error(f"Error in transcribe-srt endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe-youtube/srt")
async def transcribe_youtube_srt(response: TranscribeResponse):
    try:
        srt_content = generate_srt(response.segments)
        return StreamingResponse(
            iter([srt_content]),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=subtitles.srt"}
        )
    except Exception as e:
        logger.error(f"Error in transcribe-youtube-srt endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
