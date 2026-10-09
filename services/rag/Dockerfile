# Cookmate RAG service image.
#
# Two stages. The builder has uv and installs dependencies into a virtualenv.
# The runner gets only that virtualenv and the source, so build tools never
# ship, and the image is smaller and has less to patch.

# ---- builder ----
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

# Compile .pyc files at build time so the container does not do it on first
# request, and copy rather than hard-link because the venv moves stages.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, source second. This layer only rebuilds when the lock
# file changes, not on every code edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY README.md ./
RUN uv sync --locked --no-dev

# ---- runner ----
FROM python:3.12-slim AS runner

# A non-root user. If the process is ever compromised it cannot write to the
# image or install anything.
RUN useradd --create-home --uid 10001 rag

WORKDIR /app
COPY --from=builder --chown=rag:rag /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER rag
EXPOSE 8000

# Python is already in the image, so the health check needs no curl.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"]

# Exec form, so uvicorn is PID 1 and receives SIGTERM directly and shuts down
# cleanly. Shell form would put /bin/sh in between and swallow the signal.
# --host 0.0.0.0: a server bound to 127.0.0.1 inside a container is unreachable
# from outside it.
CMD ["uvicorn", "--factory", "rag.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
