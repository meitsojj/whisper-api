FROM python:3.11-slim

LABEL "language"="python"
LABEL "framework"="fastapi"

WORKDIR /app

# Install system dependencies including ffmpeg, git, and deno for yt-dlp
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && curl -fsSL https://deno.land/x/install/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add deno to PATH
ENV PATH="/root/.deno/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
