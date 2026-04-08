# Email Triage OpenEnv — Dockerfile
# Compatible with HuggingFace Spaces (port 7860)
# Resources: 2 vCPU, 8 GB RAM

FROM python:3.12-slim-bookworm

# Labels for HuggingFace Spaces
LABEL maintainer="hackathon-team"
LABEL description="Email Triage OpenEnv — real-world email triage for RL agents"

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py .
COPY tasks_data.py .
COPY environment.py .
COPY server.py .
COPY client.py .
COPY inference.py .
COPY openenv.yaml .

# Change ownership
RUN chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:7860/health'); assert r.status_code == 200"

# Start the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
