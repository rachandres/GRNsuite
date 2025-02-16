import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from matplotlib.widgets import Slider, Button
from scipy.signal import butter, filtfilt

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
