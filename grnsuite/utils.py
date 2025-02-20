import yaml

def load_parameters(param_file='parameters.yaml'):
    """Load parameters from YAML file"""
    with open(param_file, 'r') as file:
        return yaml.safe_load(file)
