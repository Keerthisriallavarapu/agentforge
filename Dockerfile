FROM python:3.12-slim

WORKDIR /app

# System deps for sentence-transformers (numpy, scipy wheels available; no need for build-essential here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# Copy source
COPY agentforge/ ./agentforge/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "agentforge.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
