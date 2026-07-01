# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.5

# Stage 1: compile the Rust extension into a Linux wheel
FROM python:${PYTHON_VERSION}-slim AS rust-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

RUN pip install maturin

WORKDIR /build
COPY rust_knn/ .
RUN maturin build --release

# Stage 2: runtime image
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Install the compiled Rust extension wheel
COPY --from=rust-builder /build/target/wheels/ /tmp/wheels/
RUN pip install /tmp/wheels/rust_knn*.whl

USER appuser

COPY . ./stage1/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

CMD ["uvicorn", "stage1.adapters.http.serve:app", "--host", "0.0.0.0", "--port", "8000"]
