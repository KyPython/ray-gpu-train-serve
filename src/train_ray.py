"""
Ray Train entrypoint for training the PyTorch model.

This script:
1. Initializes Ray
2. Defines a training function
3. Runs training with Ray Train
4. Saves the model to artifacts/model.pt

To run: python src/train_ray.py
"""

import os
import logging
import torch
import torch.nn as nn
from ray import train
from ray.train import ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer
from model_def import SimpleMLP, get_data_loaders
from utils import save_model, get_device_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_func(config: dict):
    """
    Training function executed by each Ray Train worker.
    
    In production, this would:
    - Load data from distributed storage (S3, HDFS, etc.)
    - Use distributed data loading
    - Sync gradients across workers for multi-node training
    - Log metrics to MLflow/Weights & Biases
    
    Args:
        config: Training configuration dictionary
    """
    # Get device info
    device = get_device_info()
    logger.info(f"Training on device: {device}")
    
    # Get data loaders
    batch_size = config.get("batch_size", 32)
    num_samples = config.get("num_samples", 1000)
    input_dim = config.get("input_dim", 10)
    
    train_loader, val_loader = get_data_loaders(
        batch_size=batch_size,
        num_samples=num_samples,
        input_dim=input_dim
    )
    
    # Initialize model
    model = SimpleMLP(input_dim=input_dim, hidden_dim=64, output_dim=1)
    model = train.torch.prepare_model(model)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    
    # Training loop
    num_epochs = config.get("num_epochs", 5)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(X)
            loss = criterion(predictions, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                predictions = model(X)
                loss = criterion(predictions, y)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # Log metrics
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )
        
        # Report metrics to Ray Train (for distributed training tracking)
        train.report({
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "epoch": epoch + 1
        })
    
    # Save model checkpoint
    # In production, this would save to cloud storage (S3, GCS, etc.)
    # Use absolute path to ensure it's saved in the right location
    artifacts_dir = os.path.abspath("artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "model.pt")
    
    # Get the underlying PyTorch model from Ray's wrapper
    # For single worker, model is not wrapped in DDP, so we can access it directly
    # For multi-worker, we'd need to unwrap from DDP: model.module if hasattr(model, 'module') else model
    unwrapped_model = model.module if hasattr(model, 'module') else model
    torch.save(unwrapped_model.state_dict(), model_path)
    
    logger.info(f"Training completed. Model saved to {model_path}")
    logger.info(f"Model file exists: {os.path.exists(model_path)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Artifacts directory: {artifacts_dir}")
    
    # Create a checkpoint for Ray Train
    checkpoint = Checkpoint.from_directory(artifacts_dir)
    train.report({}, checkpoint=checkpoint)


def main():
    """
    Main training entrypoint.
    """
    # Suppress deprecation warnings for cleaner output
    import warnings
    import os
    os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"
    
    # Training configuration
    # In production, this would come from a config file (YAML, JSON, etc.)
    train_config = {
        "batch_size": 32,
        "num_samples": 1000,
        "input_dim": 10,
        "lr": 0.001,
        "num_epochs": 5
    }
    
    # Scaling configuration
    # For multi-node: use num_workers > 1 and configure Ray cluster
    # For GPU: set use_gpu=True (Ray will auto-detect available GPUs)
    scaling_config = ScalingConfig(
        num_workers=1,  # Single node for this demo
        use_gpu=torch.cuda.is_available()  # Use GPU if available
    )
    
    logger.info("Starting Ray Train...")
    logger.info(f"Scaling config: {scaling_config}")
    
    # Initialize trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config,
        scaling_config=scaling_config
    )
    
    # Run training
    result = trainer.fit()
    
    logger.info("Training completed successfully!")
    logger.info(f"Final metrics: {result.metrics}")
    
    # Ensure model is saved locally
    # The checkpoint is already saved in train_func, but we verify it exists
    if os.path.exists("artifacts/model.pt"):
        logger.info("Model checkpoint verified at artifacts/model.pt")
    else:
        logger.warning("Model checkpoint not found at artifacts/model.pt")


if __name__ == "__main__":
    main()

