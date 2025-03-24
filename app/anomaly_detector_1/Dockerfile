FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3

# Expose port
EXPOSE 5000

# Set entrypoint
ENTRYPOINT ["python", "app.py"]

# Default command (if not overridden in docker-compose)
CMD ["--api", "--host", "0.0.0.0", "--port", "5000", "--model-dir", "/app/models"]