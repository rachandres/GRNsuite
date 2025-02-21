import yaml
from datetime import datetime

def load_parameters(param_file='parameters.yaml'):
    """Load parameters from YAML file and add current date"""
    with open(param_file, 'r') as file:
        params = yaml.safe_load(file)
    
    # Set analysis date to today
    params['analysis_date'] = datetime.now().strftime('%Y-%m-%d')
    
    return params

def parse_filename_metadata(filename, params):
    """
    Parse metadata from filename based on configuration.
    
    Parameters:
        filename (str): Name of the file without extension
        params (dict): Parameters dictionary from parameters.yaml
    
    Returns:
        dict: Metadata dictionary with fields as keys
    """
    # Get parsing configuration
    separator = params['filename_parsing']['separator']
    fields = params['filename_parsing']['fields']
    
    # Split filename
    parts = filename.split(separator)
    
    # Create metadata dictionary
    metadata = {}
    for field, value in zip(fields, parts):
        metadata[field] = value
    
    return metadata
