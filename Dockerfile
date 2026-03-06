FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg git

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

**第三步：回到 Zeabur 填入**
```
Git URL: https://github.com/你的帳號/whisper-api.git
分支: main
認證方式: 匿名（Public repo 選這個就好）
