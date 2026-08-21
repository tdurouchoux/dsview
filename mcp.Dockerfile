FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /bin

COPY . /app

ENV CONF_DIR="./config"
ENV PROMPT_DIR="./prompts"

WORKDIR /app
RUN uv sync --no-default-groups --group mcp --frozen --no-cache

EXPOSE 8000

CMD ["uv", "run", "dsview/mcp/server.py"]
