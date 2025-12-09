#!/bin/bash
# Deployment script for Ray Train & Serve

set -e

echo "🚀 Ray Train & Serve Deployment Script"
echo "========================================"

# Step 1: Train the model
echo ""
echo "Step 1: Training model..."
python src/train_ray.py

if [ ! -f "artifacts/model.pt" ]; then
    echo "❌ Error: Model training failed or model not found"
    exit 1
fi

echo "✅ Model trained successfully!"

# Step 2: Build Docker image (optional)
if [ "$1" == "docker" ]; then
    echo ""
    echo "Step 2: Building Docker image..."
    docker build -t ray-gpu-train-serve:latest .
    echo "✅ Docker image built!"
    echo ""
    echo "To run with Docker:"
    echo "  docker run -p 8000:8000 ray-gpu-train-serve:latest"
fi

echo ""
echo "✅ Deployment ready!"
echo ""
echo "To start serving locally:"
echo "  python src/serve_ray.py"
echo ""
echo "Or with Docker Compose:"
echo "  docker-compose up"

