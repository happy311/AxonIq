FROM python:3.12-slim

RUN apt-get update && apt-get install -y gcc g++ git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace persistent storage — mounted at runtime via Space Settings → Storage
RUN mkdir -p /data/db /data/chroma /data/backups

ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=1
ENV NEUROCHECK_DB=/data/db/neurocheck.db
ENV CHROMA_DB_PATH=/data/chroma
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "/app", "--timeout-keep-alive", "1800"]
