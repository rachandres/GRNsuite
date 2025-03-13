import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from matplotlib.widgets import Slider, Button
from scipy.signal import butter, filtfilt
import pandas as pd
import os
from .utils import load_parameters
import json
from datetime import datetime

# ---------- PUBLIC API FUNCTIONS ----------

def process_recording(input_file, output_file, interactive=False):
    """
    Process a recording file through the complete pre-processingworkflow.
    
    Parameters:
        input_file (str): Path to input file
        output_file (str): Path to save processed data
        interactive (bool): Whether to use interactive contact selection
        
    Returns:
        str: Path to the processed output file
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load parameters
    params = load_parameters('parameters.yaml')
    
    # Step 1: Load data and extract metadata
    raw_data = _load_ephys_data(input_file)
    metadata = extract_metadata(input_file, params)
    
    # Step 2: Select contact interval
    if interactive:
        selected_signal = interactive_contact_selection(raw_data)
    else:
        selected_signal = automated_contact_selection(raw_data)
    
    # Step 3: Filter signal
    filtered_signal = filter_signal(selected_signal, params)
    
    # Step 4: Zoom to region of interest
    zoomed_signal, time_vector = zoom_to_region(filtered_signal, params)
    
    # Step 5: Normalize signal
    normalized_signal = normalize_signal(zoomed_signal)
    
    # Step 6: Save results
    save_results(normalized_signal, time_vector, output_file)
    
    # Step 7: Save metadata to the output directory
    if output_dir:
        save_metadata(input_file, output_dir, params)
    
    return output_file

def extract_metadata(input_file, params):
    """
    Extract metadata from the input file and parameters.
    
    Parameters:
        input_file (str): Path to input file
        params (dict): Parameters dictionary
        
    Returns:
        dict: Metadata dictionary
    """
    metadata = {
        'input_file': input_file,
        'experimenter': params.get('experimenter', ''),
        'analysis_date': params.get('analysis_date', datetime.now().strftime('%Y-%m-%d')),
        'notes': params.get('notes', ''),
        'sampling_rate': params['sampling_rate'],
        'offset_time': params['offset_time'],
        'analysis_length': params['analysis_length']
    }
    
    # Parse filename if parsing rules exist
    if 'filename_parsing' in params:
        parse_rules = params['filename_parsing']
        filename = os.path.basename(input_file)
        parts = filename.split(parse_rules['separator'])
        
        if len(parts) >= len(parse_rules['fields']):
            for field, value in zip(parse_rules['fields'], parts):
                metadata[field] = value
    
    return metadata

def filter_signal(signal, params):
    """
    Apply filtering to the signal.
    
    Parameters:
        signal (np.ndarray): Input signal
        params (dict): Parameters dictionary
        
    Returns:
        np.ndarray: Filtered signal
    """
    filter_low = params['filter_low']
    filter_high = params['filter_high']
    sampling_rate = params['sampling_rate']
    
    # Apply bandpass filter
    filtered_data = _apply_filter(signal, [filter_low, filter_high], 'bandpass', fs=sampling_rate)
    
    # Apply noise filter
    denoised_data = _apply_filter(filtered_data, None, 'noise', fs=sampling_rate)
    
    return denoised_data

def zoom_to_region(signal, params):
    """
    Zoom to region of interest in the signal.
    
    Parameters:
        signal (np.ndarray): Input signal
        params (dict): Parameters dictionary
        
    Returns:
        tuple: (zoomed_signal, time_vector)
    """
    offset_time = params['offset_time']
    analysis_length = params['analysis_length']
    sampling_rate = params['sampling_rate']
    
    return _zoom_data(signal, sampling_rate, offset_time, analysis_length)

def normalize_signal(signal):
    """
    Normalize the signal using MAD.
    
    Parameters:
        signal (np.ndarray): Input signal
        
    Returns:
        np.ndarray: Normalized signal
    """
    return _normalize_by_mad(signal)

def save_results(signal, time_vector, output_file):
    """
    Save processed results to file.
    
    Parameters:
        signal (np.ndarray): Processed signal
        time_vector (np.ndarray): Time vector
        output_file (str): Output file path
        
    Returns:
        str: Path to saved file
    """
    df = pd.DataFrame({
        'time': time_vector,
        'voltage': signal
    })
    df.to_csv(output_file, index=False)
    print(f"Successfully saved processed data to: {output_file}")
    
    return output_file

def save_metadata(input_file, output_dir, params):
    """
    Save metadata for a recording to a JSON file.
    
    Parameters
    ----------
    input_file : str
        Path to input data file
    output_dir : str
        Directory to save metadata
    params : dict
        Parameters from parameters.yaml
        
    Returns
    -------
    str
        Path to the created metadata file
    """
    try:
        # Create metadata dictionary
        metadata = {
            'input_file': input_file,
            'experimenter': params.get('experimenter', ''),
            'analysis_date': params.get('analysis_date', datetime.now().strftime('%Y-%m-%d')),
            'notes': params.get('notes', ''),
            'sampling_rate': params['sampling_rate'],
            'offset_time': params['offset_time'],
            'analysis_length': params['analysis_length'],
            'schmidt_t1': params['schmidt_t1'],
            'schmidt_t2': params['schmidt_t2']
        }
        
        # Parse filename if parsing rules exist
        if 'filename_parsing' in params:
            parse_rules = params['filename_parsing']
            filename = os.path.basename(input_file)
            parts = filename.split(parse_rules['separator'])
            
            if len(parts) >= len(parse_rules['fields']):
                for field, value in zip(parse_rules['fields'], parts):
                    metadata[field] = value
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'metadata.json')
        os.makedirs(output_dir, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Saved metadata to: {metadata_path}")
        return metadata_path
        
    except Exception as e:
        print(f"Error saving metadata: {str(e)}")
        raise


def interactive_contact_selection(raw_signal):
    """
    Interactive GUI for selecting the correct start time.

    Parameters:
        raw_signal (np.ndarray): Raw electrophysiology signal.

    Returns:
        np.ndarray: Extracted signal.
    """
    # load parameters
    params = load_parameters('parameters.yaml')
    recording_duration = params['recording_duration']
    sampling_rate = params['sampling_rate']

    num_samples = int(sampling_rate * recording_duration)
    artifact_index = _find_contact_artifact(raw_signal)

    # Initial selection
    start_idx = artifact_index
    end_idx = start_idx + num_samples

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.3)

    # Plot full trace
    ax.plot(raw_signal, label="Full Trace", alpha=0.5)
    signal_line, = ax.plot(range(start_idx, end_idx), raw_signal[start_idx:end_idx], 'r', label="Selected Window")
    ax.legend()
    ax.set_title("Adjust Start Time with Slider")

    # Add slider
    ax_slider = plt.axes([0.2, 0.15, 0.65, 0.03])
    slider = Slider(ax_slider, 'Start Time (samples)', 0, len(raw_signal) - num_samples, valinit=start_idx, valstep=1)

    def update(val):
        """Updates the plot when the slider is moved."""
        nonlocal start_idx, end_idx
        start_idx = int(slider.val)
        end_idx = start_idx + num_samples
        signal_line.set_xdata(range(start_idx, end_idx))
        signal_line.set_ydata(raw_signal[start_idx:end_idx])
        fig.canvas.draw_idle()  # Refresh plot dynamically

    slider.on_changed(update)

    # Add "Zoom" button
    ax_zoom = plt.axes([0.1, 0.02, 0.2, 0.05])
    button_zoom = Button(ax_zoom, "Zoom In")

    def zoom(event):
        """Zooms into the selected region."""
        ax.set_xlim(start_idx, end_idx)
        ax.set_ylim(min(raw_signal[start_idx:end_idx]), max(raw_signal[start_idx:end_idx]))
        fig.canvas.draw_idle()

    button_zoom.on_clicked(zoom)

    # Add "Confirm" button
    ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])
    button = Button(ax_button, "Confirm Start Time")

    def confirm(event):
        """Closes the window and saves the selected signal."""
        plt.close(fig)

    button.on_clicked(confirm)

    plt.show(block=True)  # Ensures the UI is interactive
    plt.pause(0.01)  # Allows GUI updates

    # Return the adjusted selection
    return raw_signal[start_idx:end_idx]


def automated_contact_selection(raw_signal):
    """
    Automatically select the signal segment beginning at the contact artifact.
    This is a non-interactive version of interactive_contact_selection.

    Parameters:
        raw_signal (np.ndarray): Raw electrophysiology signal.

    Returns:
        np.ndarray: Extracted signal starting at the contact artifact.
    """
    # Load parameters
    params = load_parameters('parameters.yaml')
    recording_duration = params['recording_duration']
    sampling_rate = params['sampling_rate']

    # Find contact artifact
    artifact_index = _find_contact_artifact(raw_signal)
    
    # Calculate window size and end index
    num_samples = int(sampling_rate * recording_duration)
    end_idx = artifact_index + num_samples
    
    print(f"Auto-selected signal from index {artifact_index} to {end_idx} (duration: {recording_duration}s)")
    
    # Return the selected segment
    return raw_signal[artifact_index:end_idx]


# ---------- PRIVATE HELPER FUNCTIONS ----------

def _load_ephys_data(filepath):
    """
    Loads electrophysiology data from a text file, skipping the first 6 rows.
    Cleans up extra spaces, tabs, and removes any non-numeric content.

    Parameters:
        filepath (str): Path to the data file.

    Returns:
        np.ndarray: Processed electrophysiology data as a NumPy array.
    """
    numeric_data = []

    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()[6:]  # Skip first 6 lines (metadata)

    for line in lines:
        # Remove tabs, Windows line endings, and any extra spaces
        line = line.strip().replace("\t", "").replace("\r", "")

        # Skip empty lines
        if not line:
            continue

        try:
            numeric_data.append(float(line))  # Convert valid numbers
        except ValueError:
            print(f"Skipping non-numeric line: {repr(line)}")  # Debugging info

    if not numeric_data:
        raise ValueError(f"No valid numeric data found in {filepath}. Please check file formatting.")

    return np.array(numeric_data)


def _find_contact_artifact(raw_data, min_peak_height=100):
    """
    Detects the contact artifact in the raw electrophysiology trace.

    Parameters:
        raw_data (np.ndarray): The raw electrophysiology signal.
        min_peak_height (float): Minimum peak height to detect artifact reduce if the artifact is small, increase if it trips before the contact artifact due to noise.

    Returns:
        int: Index of the detected artifact peak.
    """
    peaks, _ = signal.find_peaks(np.abs(raw_data), height=min_peak_height)
    if len(peaks) == 0:
        raise ValueError("No contact artifact detected. Adjust min_peak_height.")
    return peaks[0]  # First detected peak


def _apply_filter(signal_data, freq, filter_type, fs):
    """
    Applies a Butterworth filter to the given signal.

    Parameters:
        signal_data (np.ndarray): 1D or 2D array of the raw signal(s).
        freq (float or list): Cutoff frequency. Use a list [low, high] for bandpass/bandstop.
        filter_type (str): Type of filter ('bandpass', 'bandstop', 'low', 'high', 'noise').
        fs (int): Sampling frequency (Hz).

    Returns:
        np.ndarray: Filtered signal.
    """
    if filter_type == 'bandpass':
        b, a = butter(2, [freq[0] / (fs / 2), freq[1] / (fs / 2)], btype='bandpass')

    elif filter_type == 'low':
        b, a = butter(2, freq / (fs / 2), btype='low')

    elif filter_type == 'high':
        b, a = butter(2, freq / (fs / 2), btype='high')

    elif filter_type == 'bandstop':
        b, a = butter(2, [freq[0] / (fs / 2), freq[1] / (fs / 2)], btype='bandstop')

    elif filter_type == 'noise':  # Chain multiple bandstop filters to remove noise harmonics
        noise_freqs = [[49, 51], [99, 101], [149, 151], [199, 201], [299, 301], [665, 668], [24, 26]]
        filt_signal = signal_data.copy()
        for f in noise_freqs:
            b, a = butter(2, [f[0] / (fs / 2), f[1] / (fs / 2)], btype='bandstop')
            filt_signal = filtfilt(b, a, filt_signal)
        return filt_signal

    else:
        raise ValueError(f"Invalid filter type: {filter_type}. Choose 'bandpass', 'bandstop', 'low', 'high', or 'noise'.")

    # Apply filter to single or multi-row signals
    if signal_data.ndim == 1:
        return filtfilt(b, a, signal_data)
    elif signal_data.ndim == 2:
        return np.array([filtfilt(b, a, row) for row in signal_data])
    else:
        raise ValueError("Signal must be 1D or 2D.")


def _zoom_data(data, fs, offset_time, analysis_length):
    """
    Extract a portion of the data starting from offset_time and running for analysis_length duration.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input signal data
    fs : float
        Sampling frequency in Hz
    offset_time : float
        Start time in seconds
    analysis_length : float
        Duration to analyze in seconds
    
    Returns
    -------
    tuple
        (data_zoomed, current_time)
        - data_zoomed: numpy.ndarray, extracted portion of the signal
        - current_time: numpy.ndarray, time points in seconds corresponding to data_zoomed
    """
    # Calculate start and end indices (convert time to samples)
    start_idx = int(offset_time * fs)
    end_idx = int((offset_time + analysis_length) * fs)
    
    # Extract the zoomed portion of the data
    data_zoomed = data[start_idx:end_idx]
    
    # Create time array for the zoomed section (in seconds)
    current_time = np.arange(len(data_zoomed)) / fs + offset_time
    
    return data_zoomed, current_time


def _normalize_by_mad(signal):
    """
    Normalize signal using Median Absolute Deviation (MAD).
    
    Parameters:
    signal : array-like
        Input signal to normalize
        
    Returns:
    array-like
        Signal normalized by MAD
    """
    # Calculate MAD
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    # Return normalized signal
    return signal / (1.4826 * mad)