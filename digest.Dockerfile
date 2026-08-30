FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /bin

COPY . /app

ENV CONF_DIR="./config"
ENV PROMPT_DIR="./prompts"

WORKDIR /app
# `dsview digest` runs through dsview/cli.py, which unconditionally imports the
# `evaluate` subcommand (dsview/evaluation/cli.py imports mlflow at module level).
# The evaluation group is pulled in to make that import succeed, not because the
# digest command itself uses mlflow.
RUN uv sync --no-default-groups --group evaluation --frozen --no-cache

CMD ["uv", "run", "dsview", "digest"]
