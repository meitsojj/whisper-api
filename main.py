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

class
