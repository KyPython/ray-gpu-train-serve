#!/usr/bin/env python3
"""
Initialize model by training if it doesn't exist.
This script can be run in deployment environments to ensure the model exists.
"""

import os
import sys

def main():
    model_path = "artifacts/model.pt"
    
    if os.path.exists(model_path):
        print(f"✅ Model already exists at {model_path}")
        return 0
    
    print("📦 Model not found. Training model...")
    print("=" * 60)
    
    # Import and run training
    try:
        from train_ray import main as train_main
        train_main()
        
        if os.path.exists(model_path):
            print(f"✅ Model trained successfully at {model_path}")
            return 0
        else:
            print(f"❌ Training completed but model not found at {model_path}")
            return 1
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

