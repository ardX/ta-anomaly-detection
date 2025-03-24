FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

# Environment variables to disable GPU and reduce TensorFlow warnings
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV TF_ENABLE_ONEDNN_OPTS="0"

# Expose port
EXPOSE 5000

# Run the API server
CMD ["python", "app.py", "--api", "--port", "5000"]
