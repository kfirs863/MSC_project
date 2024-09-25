# utils.py

import yaml

def load_config(config_path='config.yaml'):
    """Load the configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

import json

def load_feature_descriptions(file_path='feature_descriptions.json'):
    with open(file_path, 'r') as f:
        return json.load(f)
