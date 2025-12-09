"""
Ray Serve deployment for serving the trained PyTorch model.

This script:
1. Loads the trained model from artifacts/model.pt
2. Deploys it as a Ray Serve HTTP endpoint
3. Exposes a /predict endpoint that accepts JSON input

To run: python src/serve_ray.py

Then call: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'
"""

import os
import logging
import torch
from ray import serve
from ray.serve import Application
from model_def import SimpleMLP
from utils import load_model, get_device_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.5 if torch.cuda.is_available() else 0}
)
class ModelDeployment:
    """
    Ray Serve deployment for the PyTorch model.
    
    In production, this would:
    - Load model from cloud storage (S3, GCS, etc.)
    - Use model versioning
    - Implement request batching for better throughput
    - Add authentication/authorization
    - Include health checks and metrics
    """
    
    def __init__(self):
        """Initialize the deployment by loading the trained model."""
        # Get device info
        self.device = get_device_info()
        
        # Model configuration (should match training config)
        input_dim = 10
        hidden_dim = 64
        output_dim = 1
        
        # Initialize model architecture
        self.model = SimpleMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # Load trained weights
        model_path = "artifacts/model.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                "Please run training first: python src/train_ray.py"
            )
        
        self.model = load_model(self.model, model_path, device=self.device)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded and ready on device: {self.device}")
    
    def predict(self, features: list) -> dict:
        """
        Make a prediction given input features.
        
        Args:
            features: List of feature values (should match input_dim)
        
        Returns:
            dict: Prediction result with 'prediction' key
        """
        try:
            # Convert to tensor
            input_tensor = torch.tensor(features, dtype=torch.float32)
            
            # Ensure correct shape: (batch_size, input_dim)
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)
            
            # Move to device
            input_tensor = input_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(input_tensor)
                prediction_value = prediction.cpu().item()
            
            logger.info(f"Prediction request: features={features}, prediction={prediction_value:.4f}")
            
            return {
                "prediction": float(prediction_value),
                "features": features
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "error": str(e),
                "prediction": None
            }


@serve.deployment(
    num_replicas=1
)
class PredictDeployment:
    """
    HTTP deployment for the prediction endpoint.
    """
    
    def __init__(self, model_deployment):
        self.model = model_deployment
    
    async def __call__(self, request: dict) -> dict:
        """
        HTTP POST endpoint for predictions.
        
        Expected JSON body:
        {
            "features": [0.1, 0.2, 0.3, ...]  # List of 10 feature values
        }
        
        Returns:
            JSON response with prediction
        """
        features = request.get("features", [])
        
        if not features:
            return {
                "error": "Missing 'features' in request body",
                "prediction": None
            }
        
        if len(features) != 10:
            return {
                "error": f"Expected 10 features, got {len(features)}",
                "prediction": None
            }
        
        return self.model.predict(features)


def create_app() -> Application:
    """
    Create the Ray Serve application.
    
    Returns:
        Application: Configured Ray Serve application
    """
    model_deployment = ModelDeployment.bind()
    return PredictDeployment.bind(model_deployment)


def main():
    """
    Main serving entrypoint.
    """
    logger.info("Starting Ray Serve...")
    
    # Check if model exists, train if not
    model_path = "artifacts/model.pt"
    if not os.path.exists(model_path):
        logger.warning(
            f"Model not found at {model_path}. Attempting to train..."
        )
        try:
            # Try to train the model
            import sys
            import subprocess
            script_dir = os.path.dirname(os.path.abspath(__file__))
            train_script = os.path.join(script_dir, "train_ray.py")
            result = subprocess.run(
                [sys.executable, train_script],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            if result.returncode == 0 and os.path.exists(model_path):
                logger.info("Model trained successfully!")
            else:
                logger.error(f"Training failed: {result.stderr}")
                logger.error(
                    "Please run training manually: python src/train_ray.py"
                )
                return
        except Exception as e:
            logger.error(
                f"Failed to train model: {e}. "
                "Please run training manually: python src/train_ray.py"
            )
            return
    
    # Start Ray Serve
    # In production, this would be deployed to a Ray cluster
    # and managed via Ray Serve CLI or Kubernetes
    port = int(os.environ.get("PORT", 8000))
    
    logger.info("=" * 60)
    logger.info("Starting Ray Serve...")
    logger.info(f"Binding to 0.0.0.0:{port}")
    logger.info("=" * 60)
    
    # Use route_prefix in serve.run (new API)
    serve.run(
        create_app(),
        host="0.0.0.0",
        port=port,
        route_prefix="/predict"
    )
    
    logger.info("=" * 60)
    logger.info("Ray Serve is running!")
    logger.info("=" * 60)
    logger.info(f"Model endpoint: http://0.0.0.0:{port}/predict")
    logger.info("")
    logger.info("Example curl request:")
    logger.info(
        'curl -X POST http://localhost:8000/predict \\'
    )
    logger.info(
        '  -H "Content-Type: application/json" \\'
    )
    logger.info(
        '  -d \'{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}\''
    )
    logger.info("")
    logger.info("Ray Dashboard: http://localhost:8265")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

