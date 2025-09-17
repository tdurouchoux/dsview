FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /bin

COPY . /app

ENV CONF_DIR="./config"
ENV PROMPT_DIR="./prompts"

WORKDIR /app
RUN uv sync --group labels --frozen --no-cache

EXPOSE 8080

CMD ["uv", "run", "streamlit", "run", "dsview/interface/labelling_interface.py", "--server.address","0.0.0.0"]
