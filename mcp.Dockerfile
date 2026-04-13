FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /bin

COPY . /app

ENV CONF_DIR="./config"
ENV PROMPT_DIR="./prompts"

WORKDIR /app
RUN uv sync --frozen --no-cache

EXPOSE 8000

CMD ["uv", "run", "python", "dsview/mcp/server.py"]
