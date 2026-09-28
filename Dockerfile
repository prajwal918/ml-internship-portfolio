FROM python:3.14-slim

# Security Metadata
LABEL maintainer="ml-internship-portfolio"
LABEL version="2.0.0"
LABEL description="ML Internship Portfolio - Hardened Production ML Pipeline"
LABEL security.hardened="true"

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src"

# Create a dedicated unprivileged non-root user
RUN addgroup --system appgroup && adduser --system --group appuser

WORKDIR /app

# Install dependencies as a separate cached layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and assign unprivileged ownership
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

EXPOSE 8501

# Native Python healthcheck avoiding external curl dependency on slim image
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
CMD ["--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=true"]
