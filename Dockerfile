# Multi-stage Dockerfile for the InvoiceOps Agent service (Cloud Run).
# Section 8: FastAPI + uvicorn, non-root user, Python 3.12-slim.

# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# --- Stage 2: Runtime ---
FROM python:3.12-slim AS runtime

RUN groupadd --gid 1001 invoiceops \
    && useradd --uid 1001 --gid invoiceops --shell /bin/false invoiceops

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ src/
COPY configs/ configs/
COPY data/generated/ data/generated/
COPY data/splits/ data/splits/

# Ensure tmp directory exists for uploaded PDFs
RUN mkdir -p /app/tmp/uploads && chown -R invoiceops:invoiceops /app

USER invoiceops

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "-m", "invoiceops.server"]
