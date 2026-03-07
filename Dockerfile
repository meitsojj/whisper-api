FROM python:3.11-slim

LABEL "language"="python"
LABEL "framework"="fastapi"

WORKDIR /app

# Install system dependencies including ffmpeg, git, and nodejs for yt-dlp
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
