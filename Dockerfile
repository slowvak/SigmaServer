FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install deps from lockfile (torch runs on CPU if no GPU detected)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY server.py models_registry.json ./
RUN mkdir -p models

EXPOSE 8050

CMD [".venv/bin/python", "server.py", "--host", "0.0.0.0", "--port", "8050"]
