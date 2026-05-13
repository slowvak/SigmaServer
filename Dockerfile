FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    nibabel \
    numpy \
    httpx \
    python-multipart

COPY server.py models_registry.json ./
RUN mkdir -p models

EXPOSE 8050

CMD ["python3", "server.py", "--host", "0.0.0.0", "--port", "8050"]
