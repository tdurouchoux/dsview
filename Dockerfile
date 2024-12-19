FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

ENV CONF_DIR="./config"
ENV PROMPT_DIR="./prompts"

WORKDIR /app
RUN uv sync --frozen --no-cache

EXPOSE 8000
EXPOSE 8501

CMD ["/bin/sh", "-c", "/app/docker_startup.sh"]
