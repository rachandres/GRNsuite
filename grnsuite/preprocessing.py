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

def load_ephys_data(filepath):
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

def find_contact_artifact(raw_data, min_peak_height=100):
    """
    Detects the contact artifact in the raw electrophysiology trace.

    Parameters:
        raw_data (np.ndarray): The raw electrophysiology signal.
        min_peak_height (float): Minimum peak height to detect artifact.

    Returns:
        int: Index of the detected artifact peak.
    """
    peaks, _ = signal.find_peaks(np.abs(raw_data), height=min_peak_height)
    if len(peaks) == 0:
        raise ValueError("No contact artifact detected. Adjust min_peak_height.")
    return peaks[0]  # First detected peak

def interactive_contact_selection(raw_all, sampling_rate=30000, recording_duration=3.1):
    """
    Interactive GUI for selecting the correct start time.

    Parameters:
        raw_all (np.ndarray): Raw electrophysiology signal.
        sampling_rate (int): Sampling rate in Hz.
        recording_duration (float): Duration of recording in seconds.

    Returns:
        np.ndarray: Extracted signal.
    """
    num_samples = int(sampling_rate * recording_duration)
    artifact_index = find_contact_artifact(raw_all)

    # Initial selection
    start_idx = artifact_index
    end_idx = start_idx + num_samples

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.3)

    # Plot full trace
    ax.plot(raw_all, label="Full Trace", alpha=0.5)
    signal_line, = ax.plot(range(start_idx, end_idx), raw_all[start_idx:end_idx], 'r', label="Selected Window")
    ax.legend()
    ax.set_title("Adjust Start Time with Slider")

    # Add slider
    ax_slider = plt.axes([0.2, 0.15, 0.65, 0.03])
    slider = Slider(ax_slider, 'Start Time (samples)', 0, len(raw_all) - num_samples, valinit=start_idx, valstep=1)

    def update(val):
        """Updates the plot when the slider is moved."""
        nonlocal start_idx, end_idx
        start_idx = int(slider.val)
        end_idx = start_idx + num_samples
        signal_line.set_xdata(range(start_idx, end_idx))
        signal_line.set_ydata(raw_all[start_idx:end_idx])
        fig.canvas.draw_idle()  # Refresh plot dynamically

    slider.on_changed(update)

    # Add "Zoom" button
    ax_zoom = plt.axes([0.1, 0.02, 0.2, 0.05])
    button_zoom = Button(ax_zoom, "Zoom In")

    def zoom(event):
        """Zooms into the selected region."""
        ax.set_xlim(start_idx, end_idx)
        ax.set_ylim(min(raw_all[start_idx:end_idx]), max(raw_all[start_idx:end_idx]))
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
    return raw_all[start_idx:end_idx]

def ashfilt(signal, freq, filter_type, fs):
    """
    Applies a Butterworth filter to the given signal.

    Parameters:
        signal (np.ndarray): 1D or 2D array of the raw signal(s).
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
        filt_signal = signal.copy()
        for f in noise_freqs:
            b, a = butter(2, [f[0] / (fs / 2), f[1] / (fs / 2)], btype='bandstop')
            filt_signal = filtfilt(b, a, filt_signal)
        return filt_signal

    else:
        raise ValueError(f"Invalid filter type: {filter_type}. Choose 'bandpass', 'bandstop', 'low', 'high', or 'noise'.")

    # Apply filter to single or multi-row signals
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    elif signal.ndim == 2:
        return np.array([filtfilt(b, a, row) for row in signal])
    else:
        raise ValueError("Signal must be 1D or 2D.")

def zoom_data(data, fs, offset_time, analysis_length):
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
    import numpy as np
    
    # Calculate start and end indices (convert time to samples)
    start_idx = int(offset_time * fs)
    end_idx = int((offset_time + analysis_length) * fs)
    
    # Extract the zoomed portion of the data
    data_zoomed = data[start_idx:end_idx]
    
    # Create time array for the zoomed section (in seconds)
    current_time = np.arange(len(data_zoomed)) / fs + offset_time
    
    return data_zoomed, current_time

def load_and_process_data(input_file, output_file, sampling_rate=30000, filter_high=1000):
    """
    Load and process raw data file.
    
    Parameters:
        input_file (str): Path to input file
        output_file (str): Path to save processed data
        sampling_rate (int): Sampling rate in Hz
        filter_high (int): High-pass filter cutoff frequency
    """
    try:
        # Load parameters for timing windows
        params = load_parameters('parameters.yaml')
        offset_time = params['offset_time']
        analysis_length = params['analysis_length']
        
        print(f"Processing {input_file}")
        print(f"Using parameters: fs={sampling_rate}, offset={offset_time}, length={analysis_length}")
        
        # Create output directory if it's a directory path
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Load and process data
        raw_data = load_ephys_data(input_file)
        print(f"Loaded data with shape: {raw_data.shape}")
        
        # Process the data
        contact_idx = find_contact_artifact(raw_data)
        print(f"Found contact artifact at index: {contact_idx}")
        
        num_samples = int(sampling_rate * 3.1)
        selected_signal = raw_data[contact_idx:contact_idx + num_samples]
        
        filtered_data = ashfilt(selected_signal, [100, filter_high], 'bandpass', fs=sampling_rate)
        denoised_data = ashfilt(filtered_data, None, 'noise', fs=sampling_rate)
        data_zoomed, current_time = zoom_data(denoised_data, sampling_rate, offset_time, analysis_length)
        
        # Save processed data
        df = pd.DataFrame({
            'time': current_time,
            'voltage': data_zoomed
        })
        df.to_csv(output_file, index=False)
        print(f"Successfully saved processed data to: {output_file}")
        
        if not os.path.exists(output_file):
            raise RuntimeError(f"Failed to create output file: {output_file}")
            
        return output_file
        
    except Exception as e:
        print(f"Error in load_and_process_data: {str(e)}")
        raise

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