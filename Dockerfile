FROM python:3.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

ENV CONF_DIR="./config"
ENV PROMPT_DIR="./prompts"

WORKDIR /app
RUN uv sync --frozen --no-cache
RUN uv run python dsview/obsidian/setup.py

EXPOSE 8000
EXPOSE 8501

CMD ["/bin/sh", "-c", "/app/.venv/bin/fastapi run dsview/api.py & /app/.venv/bin/streamlit run dsview/labelling/interface.py"]
