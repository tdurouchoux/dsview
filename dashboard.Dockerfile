FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /bin

COPY . /app

ENV CONF_DIR="./config"

WORKDIR /app
RUN uv sync --no-default-groups --group dashboard --frozen --no-cache

EXPOSE 2718

CMD ["uv", "run", "dsview/interface/run_dashboard.py"]
