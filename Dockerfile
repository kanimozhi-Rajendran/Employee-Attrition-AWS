# Production Multi-Platform Dockerfile for Employee Attrition API + Web Dashboard
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase and artifacts
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY static/ ./static/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Start FastAPI ASGI server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
