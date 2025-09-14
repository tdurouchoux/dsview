FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /bin

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

ENV CONF_DIR="./config"
ENV PROMPT_DIR="./prompts"

WORKDIR /app
RUN uv sync --frozen --no-cache

RUN uv run dsview/obsidian/setup.py

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "dsview/api.py", "--host",  "0.0.0.0"]
