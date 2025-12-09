"""
Simple PyTorch model and dataset definitions for Ray Train/Serve demo.

This module defines:
- A small MLP model for regression/classification
- A toy dataset generator for demonstration
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron for demonstration.
    
    In production, this would be replaced with your actual model architecture.
    Optimized for low memory usage.
    """
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, output_dim: int = 1):
        super(SimpleMLP, self).__init__()
        # Reduced hidden_dim from 64 to 32 to save memory
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ToyDataset(Dataset):
    """
    A toy dataset that generates random data for demonstration.
    
    In production, this would load from your actual data source
    (e.g., files, databases, cloud storage).
    """
    def __init__(self, num_samples: int = 1000, input_dim: int = 10, seed: int = 42):
        """
        Args:
            num_samples: Number of samples to generate
            input_dim: Dimension of input features
            seed: Random seed for reproducibility
        """
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.input_dim = input_dim
        
        # Generate random features
        self.X = torch.randn(num_samples, input_dim)
        
        # Generate targets (simple linear relationship + noise)
        # In production, this would be your actual labels
        self.y = (self.X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(num_samples, 1))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_data_loaders(batch_size: int = 16, num_samples: int = 500, input_dim: int = 10):
    """
    Create train and validation data loaders.
    
    In production, this would load from your actual data source
    and split into train/val/test sets appropriately.
    
    Args:
        batch_size: Batch size for training
        num_samples: Total number of samples to generate
        input_dim: Dimension of input features
    
    Returns:
        train_loader, val_loader: DataLoader instances
    """
    # Split dataset: 80% train, 20% val
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    full_dataset = ToyDataset(num_samples=num_samples, input_dim=input_dim)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

