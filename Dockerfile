# Dockerfile for Ray Train & Serve deployment
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# For GPU support, uncomment the following lines and rebuild:
# RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY src/ ./src/

# Train model if artifacts don't exist (for fresh deployments)
# In production, you'd download from cloud storage instead
# Create artifacts directory and train model
RUN mkdir -p artifacts && \
    if [ ! -f "artifacts/model.pt" ]; then \
        python src/train_ray.py && \
        ls -la artifacts/ && \
        test -f artifacts/model.pt && echo "Model file verified" || echo "Warning: Model file not found"; \
    else \
        echo "Model already exists, skipping training"; \
    fi

# Expose Ray Serve port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/predict || exit 1

# Start Ray Serve
CMD ["python", "src/serve_ray.py"]

