import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
import os
from .utils import load_parameters

def schmidt_trigger_auto(data_file, output_dir, t1=1.0, t2=1.25):
    """
    Detect spikes using Schmidt trigger with automatic threshold
    
    Parameters
    ----------
    data_file : str
        Path to processed data CSV file
    output_dir : str
        Directory to save results
    t1 : float
        Lower threshold multiplier
    t2 : float
        Upper threshold multiplier
    """
    # Load data
    data = pd.read_csv(data_file)
    voltage = data['voltage'].values
    current_time = data['time'].values
    
    # Calculate thresholds
    noise_std = np.std(voltage)
    threshold1 = t1 * noise_std
    threshold2 = t2 * noise_std

    # Create binary threshold exceedance arrays
    prelim1 = np.zeros(len(voltage), dtype=int)
    prelim2 = np.zeros(len(voltage), dtype=int)

    prelim1[voltage > threshold1] = 1  # Values exceeding lower threshold
    prelim2[voltage > threshold2] = 1  # Values exceeding upper threshold

    # Find threshold crossings
    on1 = np.where(np.diff(prelim1) == 1)[0] + 1  # Lower threshold crossings
    on2 = np.where(np.diff(prelim2) == 1)[0] + 1  # Upper threshold crossings
    off1 = np.where(np.diff(prelim1) == -1)[0]    # Lower threshold drop crossings
    off2 = np.where(np.diff(prelim2) == -1)[0]    # Upper threshold drop crossings

    # Remove early spikes before the first threshold crossing
    if len(on1) > 0:
        off1 = off1[off1 >= on1[0]]
        off2 = off2[off2 >= on1[0]]

    # Remove late spikes after the last lower threshold drop
    if len(off1) > 0:
        on1 = on1[on1 <= off1[-1]]
        on2 = on2[on2 <= off1[-1]]

    # Detect valid spikes
    mIndx, mVal = [], []
    if len(on1) > 0 and len(off1) > 0:
        flag = [np.any(np.isin(np.arange(on1[i], off1[i]), on2)) for i in range(len(on1))]
        flagOK = np.where(np.array(flag) == True)[0]

        for i in flagOK:
            segment = voltage[on1[i]:off1[i]]
            max_val = np.max(segment)
            max_idx = np.argmax(segment) + on1[i]  # Adjust index to match time

            mVal.append(max_val)
            mIndx.append(max_idx)

    # Save results
    spikes_df = pd.DataFrame({
        'spike_times': current_time[mIndx],
        'spike_values': mVal
    })
    output_file = os.path.join(output_dir, 'detected_spikes.csv')
    spikes_df.to_csv(output_file, index=False)
    
    return output_file


def adjust_threshold(data_zoomed, initial_threshold=None):
    """
    Interactive GUI for adjusting the threshold used for spike detection.
    
    Parameters:
        data_zoomed (numpy.ndarray): The segment of the signal to visualize.
        initial_threshold (float, optional): Initial threshold value. If None, calculates using schmidt_trigger_auto.

    Returns:
        float: The user-selected threshold.
    """
    # If no initial threshold provided, calculate using schmidt_trigger_auto
    if initial_threshold is None:
        _, _, thresholds = schmidt_trigger_auto(data_zoomed)
        initial_threshold = thresholds[1]  # Use upper threshold as starting point

    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.3)

    # Plot data segment
    ax.plot(data_zoomed, label="Data (Zoomed)", color="blue")
    threshold_line = ax.axhline(initial_threshold, color='red', linestyle="--", label="Threshold")
    ax.legend()
    ax.set_title("Adjust Threshold (Move the Red Line)")

    # Slider to adjust threshold
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Threshold', np.min(data_zoomed), np.max(data_zoomed), 
                   valinit=initial_threshold)

    def update(val):
        """Update the threshold line when slider moves."""
        threshold_line.set_ydata([slider.val, slider.val])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Confirm button
    ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])
    button = Button(ax_button, "Confirm")

    def confirm(event):
        """Closes the figure and saves the threshold."""
        plt.close(fig)

    button.on_clicked(confirm)

    plt.show(block=True)

    return slider.val

def detect_spikes_manual_threshold(current_seg, threshold):
    """
    Detects spikes using a manually adjusted threshold.

    Parameters
    ----------
    current_seg : numpy.ndarray
        The signal to process.
    threshold : float
        User-selected threshold for spike detection.

    Returns
    -------
    tuple
        - spike_indices (numpy.ndarray): Indices of detected spikes.
        - spike_values (numpy.ndarray): Peak values of detected spikes.
    """
    # Identify where the signal crosses the threshold
    above_thresh = np.zeros(len(current_seg), dtype=int)
    above_thresh[current_seg > threshold] = 1  # 1 where above threshold

    # Find crossings
    on_times = np.where(np.diff(above_thresh) == 1)[0] + 1  # Entering spike
    off_times = np.where(np.diff(above_thresh) == -1)[0]    # Leaving spike

    # Handle edge cases
    if len(on_times) == 0 or len(off_times) == 0:
        return np.array([]), np.array([])

    # Handle case where signal starts above threshold
    if off_times[0] < on_times[0]:
        on_times = np.insert(on_times, 0, 0)

    # Handle case where signal ends above threshold
    if on_times[-1] > off_times[-1]:
        off_times = np.append(off_times, len(current_seg)-1)

    # Ensure on_times and off_times are paired
    min_len = min(len(on_times), len(off_times))
    on_times = on_times[:min_len]
    off_times = off_times[:min_len]

    # Detect valid spikes
    spike_indices, spike_values = [], []
    for i in range(len(on_times)):
        segment = current_seg[on_times[i]:off_times[i]]
        max_val = np.max(segment)
        max_idx = np.argmax(segment) + on_times[i]  # Adjust index to match time

        spike_values.append(max_val)
        spike_indices.append(max_idx)

    return np.array(spike_indices), np.array(spike_values)

def extract_waveforms(data_file, spikes_file, output_dir, pre_samples=30, post_samples=30):
    """
    Extract waveforms around spike times.
    
    Parameters
    ----------
    data_file : str
        Path to processed data CSV file
    spikes_file : str
        Path to detected spikes CSV file
    output_dir : str
        Directory to save waveforms
    pre_samples : int
        Number of samples before spike peak
    post_samples : int
        Number of samples after spike peak
    """
    # Load data
    data = pd.read_csv(data_file)
    voltage = data['voltage'].values
    
    # Load spike times
    spikes = pd.read_csv(spikes_file)
    spike_indices = np.searchsorted(data['time'].values, spikes['spike_times'].values)
    
    # Extract waveforms
    n_spikes = len(spike_indices)
    n_samples = pre_samples + post_samples
    waveforms = np.zeros((n_spikes, n_samples))
    
    for i, idx in enumerate(spike_indices):
        start_idx = max(0, idx - pre_samples)
        end_idx = min(len(voltage), idx + post_samples)
        waveform = voltage[start_idx:end_idx]
        
        # Pad if necessary
        if len(waveform) < n_samples:
            if start_idx == 0:  # Pad start
                waveform = np.pad(waveform, (n_samples - len(waveform), 0))
            else:  # Pad end
                waveform = np.pad(waveform, (0, n_samples - len(waveform)))
        
        waveforms[i] = waveform
    
    # Save waveforms
    waveforms_df = pd.DataFrame(waveforms)
    waveforms_df.columns = [f'sample_{i}' for i in range(n_samples)]
    output_file = os.path.join(output_dir, 'waveforms.csv')
    waveforms_df.to_csv(output_file, index=False)
    
    return output_file

def plot_waveforms(waveforms):
    """
    Plots all extracted waveforms overlaid with semi-transparent thin lines.

    Parameters:
        waveforms (numpy.ndarray): Extracted spike waveforms, shape (num_spikes, window_samples).
    """
    plt.figure(figsize=(8, 5))

    for waveform in waveforms:
        plt.plot(waveform, color="blue", alpha=0.2, linewidth=0.5)  # Thin, semi-transparent lines

    plt.title("Extracted Spike Waveforms")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()

def detect_and_save_spikes(data_path, output_dir, param_file='parameters.yaml'):
    """Detect spikes and save results - non-interactive version"""
    # Load parameters
    params = load_parameters(param_file)
    t1 = params['schmidt_t1']
    t2 = params['schmidt_t2']
    fs = params['sampling_rate']
    
    # Load processed data
    data = pd.read_csv(data_path)
    data_zoomed = data['voltage'].values
    current_time = data['time'].values
    
    # Detect spikes using automatic thresholds
    spike_indices, spike_values, _ = schmidt_trigger_auto(
        data_path, output_dir, t1=t1, t2=t2
    )
    
    # Convert spike indices to times
    spike_times = current_time[spike_indices]
    
    # Extract waveforms
    waveforms = extract_waveforms(data_path, output_dir, pre_samples=30, post_samples=30)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save spike times and values
    spikes_df = pd.DataFrame({
        'spike_times': spike_times,
        'spike_values': spike_values
    })
    spikes_path = os.path.join(output_dir, 'detected_spikes.csv')
    spikes_df.to_csv(spikes_path, index=False)
    
    # Save waveforms
    waveforms_path = extract_waveforms(data_path, spikes_path, output_dir)
    
    return spikes_path, waveforms_path
