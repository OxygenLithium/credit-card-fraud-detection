# Handles saving and loading data
import json
import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Creates directory
def create_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

# Saves a trained model 
def save_model(model, filepath):
    create_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

# Loads a trained model off the disk
def load_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model

def save_json(data, filepath):
    create_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Data saved to {filepath}")

def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data
