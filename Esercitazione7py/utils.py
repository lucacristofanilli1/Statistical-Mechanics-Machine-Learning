import os
import json
import numpy as np
from numba import njit

def check_numba_compatibility():
    """
    Checks if Numba is working correctly.
    """
    try:
        @njit
        def test_numba(x):
            return x * 2
        
        assert test_numba(5) == 10
        print("Numba is working correctly.")
        return True
    except Exception as e:
        print(f"Numba check failed: {e}")
        return False

def ensure_directory(path):
    """
    Ensures that the directory exists.
    """
    if path and not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def save_results(filepath, data):
    """
    Saves the results dictionary to a JSON file.
    Appends or updates existing data if the file exists.
    """
    ensure_directory(os.path.dirname(filepath))
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {filepath}. Overwriting.")
            existing_data = {}
    else:
        existing_data = {}
    
    # Update existing data with new data
    # This is a shallow merge. For deep merge, more logic is needed.
    # Assuming top-level keys are experiment names.
    existing_data.update(data)
    
    with open(filepath, 'w') as f:
        json.dump(existing_data, f, indent=4)
    print(f"Results saved to {filepath}")

def load_results(filepath):
    """
    Loads results from a JSON file.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding {filepath}")
        return {}

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
