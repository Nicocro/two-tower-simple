import json
from pathlib import Path
from typing import List, Dict

def load_toy_dataset(filepath: str) -> list[dict]:
    """Load toy QA dataset from JSON file.
    
    Args:
        filepath: Path to the JSON file relative to the data/raw directory
    """
    data_dir = Path(__file__).parent.parent / 'raw'
    data_path = data_dir / filepath
    
    with open(data_path, 'r') as f:
        data = json.load(f)
        
    return data['examples']