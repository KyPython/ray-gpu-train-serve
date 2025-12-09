# Ray GPU Train & Serve

A minimal but complete Ray project demonstrating PyTorch model training with Ray Train and serving with Ray Serve.

## Overview

This project includes:
- **Model Definition**: Simple MLP model and toy dataset (`src/model_def.py`)
- **Training**: Ray Train script for distributed training (`src/train_ray.py`)
- **Serving**: Ray Serve deployment with HTTP endpoint (`src/serve_ray.py`)
- **Utilities**: Model save/load and device detection (`src/utils.py`)

## Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-capable GPU for GPU training/serving

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**For GPU support** (if you have CUDA installed):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Training

Train the model using Ray Train:

```bash
python src/train_ray.py
```

This will:
- Initialize Ray
- Train the model for 5 epochs
- Save the model weights to `artifacts/model.pt`
- Log training metrics to stdout

**Expected output:**
```
INFO: Starting Ray Train...
INFO: Epoch 1/5 - Train Loss: 0.xxxx, Val Loss: 0.xxxx
...
INFO: Training completed successfully!
INFO: Model checkpoint verified at artifacts/model.pt
```

### Serving

Start the Ray Serve deployment:

```bash
python src/serve_ray.py
```

This will:
- Load the trained model from `artifacts/model.pt`
- Start Ray Serve on `http://localhost:8000`
- Deploy the model as an HTTP endpoint

**Expected output:**
```
INFO: Starting Ray Serve...
INFO: Model loaded and ready on device: cuda (or cpu)
INFO: ============================================================
INFO: Ray Serve is running!
INFO: ============================================================
INFO: Model endpoint: http://localhost:8000/predict
...
```

### Making Predictions

Once the server is running, make a prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'
```

**Expected response:**
```json
{
  "prediction": 5.1234,
  "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}
```

## Monitoring

### Ray Dashboard

Ray automatically starts a dashboard for monitoring. Access it at:

**http://localhost:8265**

The dashboard provides:
- Cluster resource usage
- Job status and metrics
- Ray Serve deployment status
- Real-time logs

### Logs

- **Training logs**: Printed to stdout during training (loss per epoch)
- **Serving logs**: Printed to stdout for each prediction request
- **Ray logs**: Available in `~/ray_results/` directory

## Project Structure

```
ray-gpu-train-serve/
├── src/
│   ├── model_def.py      # PyTorch model and dataset definitions
│   ├── train_ray.py      # Ray Train training script
│   ├── serve_ray.py      # Ray Serve deployment
│   └── utils.py          # Utility functions (save/load, device detection)
├── artifacts/            # Generated during training (model.pt)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Production Considerations

This is a minimal demo. For production use, consider:

### Training (`train_ray.py`)
- **Data Loading**: Replace `ToyDataset` with actual data loaders from S3/HDFS/etc.
- **Multi-node Training**: Set `num_workers > 1` in `ScalingConfig` and configure Ray cluster
- **Checkpointing**: Save checkpoints to cloud storage (S3, GCS) instead of local disk
- **Experiment Tracking**: Integrate MLflow, Weights & Biases, or TensorBoard
- **Hyperparameter Tuning**: Use Ray Tune for automated hyperparameter search

### Serving (`serve_ray.py`)
- **Model Versioning**: Load models from a model registry (MLflow, S3 with versioning)
- **Request Batching**: Enable batching for better throughput
- **Authentication**: Add API keys or OAuth for production endpoints
- **Health Checks**: Implement `/health` endpoint for monitoring
- **Scaling**: Configure `num_replicas` and autoscaling based on traffic
- **Deployment**: Deploy to Kubernetes with Ray Operator or use Ray Serve CLI

### Infrastructure
- **Ray Cluster**: Deploy multi-node Ray cluster for distributed training/serving
- **GPU Management**: Configure GPU allocation and scheduling policies
- **Monitoring**: Set up Prometheus/Grafana for production monitoring
- **CI/CD**: Automate training and deployment pipelines

## Troubleshooting

### Model not found error
If you see `FileNotFoundError` when running `serve_ray.py`, make sure you've run training first:
```bash
python src/train_ray.py
```

### CUDA/GPU issues
- Verify CUDA is installed: `nvidia-smi`
- Install PyTorch with CUDA support (see Setup section)
- Ray will automatically use GPU if available

### Port already in use
If port 8000 is already in use, modify the port in `serve_ray.py`:
```python
serve.run(create_app(), host="0.0.0.0", port=8001)  # Change port
```

## License

MIT

