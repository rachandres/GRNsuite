import os

# Default filename mapping (users can modify this)
DEFAULT_FILENAME_MAPPING = [
    "date",
    "animal_id",
    "stimulus",
    "concentration",
    "location",
    "sensillum",
    "replicate"
]

def extract_metadata(filename, mapping=None, delimiter="-"):
    """
    Extracts metadata from a structured filename.
    
    Parameters:
        filename (str): The full filename (with or without extension).
        mapping (list): The ordered list of metadata keys (if None, uses defaults).
        delimiter (str): Character used to split filename parts (default "_").
    
    Returns:
        dict: Metadata dictionary with extracted values.
    """
    if mapping is None:
        mapping = DEFAULT_FILENAME_MAPPING  # Use default mapping if not provided

    # Remove file extension
    base_filename = os.path.splitext(filename)[0]
    
    # Split filename into parts
    parts = base_filename.split(delimiter)

    # Ensure we don't exceed available parts
    if len(parts) != len(mapping):
        raise ValueError(f"Expected {len(mapping)} parts in filename, but found {len(parts)}: {filename}")

    # Create metadata dictionary
    metadata = {key: value for key, value in zip(mapping, parts)}

    return metadata