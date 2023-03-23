FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi uvicorn \
    chromadb \
    sentence-transformers \
    transformers peft \
    torch \
    ollama \
    python-dotenv

COPY api/ ./api/
COPY src/ ./src/
COPY data/chroma_db ./data/chroma_db
COPY data/raft_model ./data/raft_model

EXPOSE 8000
CMD ["uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
