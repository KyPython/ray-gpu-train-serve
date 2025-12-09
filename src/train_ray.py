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
    
    # Initialize model (reduced hidden_dim for memory efficiency)
    model = SimpleMLP(input_dim=input_dim, hidden_dim=32, output_dim=1)
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
    # Optimized for low memory usage (Render free tier: 512MB)
    train_config = {
        "batch_size": 16,  # Reduced from 32
        "num_samples": 500,  # Reduced from 1000
        "input_dim": 10,
        "lr": 0.001,
        "num_epochs": 3  # Reduced from 5
    }
    
    # Scaling configuration
    # For multi-node: use num_workers > 1 and configure Ray cluster
    # For GPU: set use_gpu=True (Ray will auto-detect available GPUs)
    # Optimized for low memory usage
    scaling_config = ScalingConfig(
        num_workers=1,  # Single node for this demo
        use_gpu=False,  # Disable GPU for Render free tier
        resources_per_worker={"CPU": 1, "memory": 400 * 1024 * 1024}  # Limit memory to 400MB
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
    
    # Extract model from checkpoint if it exists
    # The model is saved in train_func, but we need to ensure it's accessible
    checkpoint = result.checkpoint
    if checkpoint:
        logger.info(f"Checkpoint available at: {checkpoint.path}")
        # Try to load model from checkpoint
        try:
            import shutil
            checkpoint_path = checkpoint.path
            # Check if model.pt exists in checkpoint directory
            checkpoint_model_path = os.path.join(checkpoint_path, "model.pt")
            if os.path.exists(checkpoint_model_path):
                # Copy to artifacts directory
                os.makedirs("artifacts", exist_ok=True)
                shutil.copy2(checkpoint_model_path, "artifacts/model.pt")
                logger.info("Model copied from checkpoint to artifacts/model.pt")
        except Exception as e:
            logger.warning(f"Could not copy model from checkpoint: {e}")
    
    # Ensure model is saved locally
    # The checkpoint is already saved in train_func, but we verify it exists
    model_path = "artifacts/model.pt"
    abs_model_path = os.path.abspath(model_path)
    
    # Check both relative and absolute paths
    found_path = None
    if os.path.exists(model_path):
        found_path = model_path
    elif os.path.exists(abs_model_path):
        found_path = abs_model_path
    
    if found_path:
        logger.info(f"Model checkpoint verified at {found_path}")
        logger.info(f"File size: {os.path.getsize(found_path)} bytes")
    else:
        logger.warning(f"Model checkpoint not found at {model_path} or {abs_model_path}")
        # List artifacts directory
        if os.path.exists("artifacts"):
            logger.info(f"Artifacts directory contents: {os.listdir('artifacts')}")
        else:
            logger.warning("Artifacts directory does not exist!")


if __name__ == "__main__":
    main()

