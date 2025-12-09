# Quick Deployment Guide

## 🚀 Deploy to Public Cloud

### Railway (Easiest - Recommended)

1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Select `KyPython/ray-gpu-train-serve`
4. Railway will auto-detect the configuration
5. The service will:
   - Train the model automatically on first deploy
   - Start Ray Serve on port 8000
   - Be available at `https://your-app.railway.app`

**Your endpoint will be**: `https://your-app.railway.app/predict`

### Render

1. Go to [render.com](https://render.com) and sign up
2. Create a new "Web Service"
3. Connect your GitHub account and select `ray-gpu-train-serve`
4. Render will use `render.yaml` automatically
5. Deploy!

**Your endpoint will be**: `https://your-app.onrender.com/predict`

### Fly.io

1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Sign up: `fly auth signup`
3. Create app: `fly launch`
4. Deploy: `fly deploy`

### Docker + Any Cloud Provider

Build and push to a container registry, then deploy:

```bash
# Build
docker build -t ray-gpu-train-serve:latest .

# Tag for your registry (example: Docker Hub)
docker tag ray-gpu-train-serve:latest yourusername/ray-gpu-train-serve:latest

# Push
docker push yourusername/ray-gpu-train-serve:latest
```

Then deploy to:
- **AWS**: ECS, EKS, or EC2
- **GCP**: Cloud Run, GKE, or Compute Engine
- **Azure**: Container Instances, AKS, or App Service

## 📝 Testing Your Deployment

Once deployed, test with:

```bash
curl -X POST https://your-deployed-url/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'
```

Expected response:
```json
{
  "prediction": 5.1234,
  "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}
```

## 🔧 Environment Variables

Set these in your deployment platform:

- `PORT`: Server port (default: 8000, auto-detected by most platforms)
- `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE`: Set to `1` for slower storage

## 📊 Monitoring

- **Ray Dashboard**: Available at `http://your-url:8265` (if ports are exposed)
- **Application Logs**: Check your platform's logging dashboard
- **Health Check**: The `/predict` endpoint serves as a health check

## 🐛 Troubleshooting

### Model Training Fails
- Check build logs for Python/PyTorch installation issues
- Ensure sufficient memory (at least 2GB recommended)
- Training takes ~1-2 minutes on CPU

### Service Won't Start
- Verify port 8000 is exposed
- Check that all dependencies installed correctly
- Review application logs

### Predictions Fail
- Ensure model was trained successfully (check logs)
- Verify request format matches expected JSON schema
- Check that features array has exactly 10 values

