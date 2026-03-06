FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg git

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

**問題 2：requirements.txt 內容不對（圖4）**

裡面放的是資料夾結構說明，不是套件清單。同樣點進去編輯，**全部清掉**換成：
```
fastapi
uvicorn
openai-whisper
python-multipart
