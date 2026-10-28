FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY eval ./eval

# Deterministic offline defaults. Point OLLAMA_BASE_URL at a reachable host to use the LLM.
ENV EMBEDDING_BACKEND=lsa \
    LLM_PROVIDER=ollama \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
