import os
import torch
import numpy as np
import hashlib
import pandas as pd
import json
import pickle

class ModelRegistry:
    """
    Manages caching and retrieval of trained JEPA models based on dataset fingerprints.
    """
    def __init__(self, base_dir='checkpoints/registry'):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def compute_dataset_hash(self, data):
        """
        Generates a unique hash for a dataset.
        
        Args:
            data: Can be a pandas DataFrame, a numpy array, a torch Tensor, 
                  or a file path (str).
        """
        if isinstance(data, str) and os.path.exists(data):
            # It's a file path
            hasher = hashlib.sha256()
            with open(data, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
            
        elif isinstance(data, pd.DataFrame):
            # Hash pandas dataframe
            # Use a robust method: convert to json/csv string and hash, or hash_pandas_object
            # Combining shape + columns + sample of values for speed, or full content?
            # Full content is safer.
            try:
                content = pd.util.hash_pandas_object(data, index=True).values
                return hashlib.sha256(content).hexdigest()
            except:
                # Fallback
                return hashlib.sha256(pickle.dumps(data)).hexdigest()
                
        elif isinstance(data, (torch.Tensor, np.ndarray)):
             # Convert to bytes
             if isinstance(data, torch.Tensor):
                 data = data.cpu().numpy()
             return hashlib.sha256(data.tobytes()).hexdigest()
             
        elif isinstance(data, dict):
            # Dictionary (e.g. JSON input)
            return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            
        else:
            raise ValueError(f"Unsupported data type for hashing: {type(data)}")

    def get_model_path(self, data_hash):
        return os.path.join(self.base_dir, f"{data_hash}.pt")

    def save_model(self, model, data_hash, metadata=None):
        """
        Saves a trained model and optional metadata.
        """
        path = self.get_model_path(data_hash)
        save_dict = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata or {}
        }
        torch.save(save_dict, path)
        print(f"[Registry] Model saved to {path}")

    def load_model(self, data_hash, model_class_instance=None):
        """
        Loads a model if it exists.
        
        Args:
            data_hash: The hash of the dataset.
            model_class_instance: Optional initialized model to load weights into. 
                                  If None, returns the state dict.
        
        Returns:
            Loaded model (if instance provided) or state_dict, or None if not found.
        """
        path = self.get_model_path(data_hash)
        if not os.path.exists(path):
            return None
            
        print(f"[Registry] Loading cached model from {path}")
        checkpoint = torch.load(path, map_location='cpu')
        
        if model_class_instance:
            model_class_instance.load_state_dict(checkpoint['model_state_dict'])
            return model_class_instance
        
        return checkpoint['model_state_dict']

    def has_model(self, data_hash):
        return os.path.exists(self.get_model_path(data_hash))
