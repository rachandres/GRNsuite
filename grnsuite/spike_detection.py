import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
import os
from .utils import load_parameters

def _detect_spikes_schmidt_trigger(voltage, times, threshold_up, threshold_down):
    """
    Core Schmidt trigger implementation used by both auto and manual detection.
    
    Parameters
    ----------
    voltage : np.ndarray
        Voltage values of the signal
    times : np.ndarray
        Time points corresponding to voltage values
    threshold_up : float
        Upper threshold for spike detection
    threshold_down : float
        Lower threshold for hysteresis
        
    Returns
    -------
    tuple
        (spike_times, spike_values)
    """
    state = 0  # 0: below threshold, 1: above threshold
    spike_start = None
    spike_indices, spike_values = [], []
    
    for i in range(len(voltage)):
        v = voltage[i]
        
        if state == 0:  # Currently below threshold
            if v > threshold_up:  # Rising edge detected
                state = 1
                spike_start = i
        else:  # Currently above threshold
            if v < threshold_down:  # Falling edge detected
                state = 0
                if spike_start is not None:
                    # Find peak in this window
                    segment = voltage[spike_start:i]
                    max_val = np.max(segment)
                    max_idx = np.argmax(segment) + spike_start
                    
                    spike_values.append(max_val)
                    spike_indices.append(max_idx)
                    spike_start = None
    
    return times[spike_indices], np.array(spike_values)

def _save_spikes(spike_times, spike_values, output_dir):
    """
    Save detected spikes to CSV file.
    
    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times
    spike_values : np.ndarray
        Array of spike peak values
    output_dir : str
        Directory to save results
        
    Returns
    -------
    str
        Path to saved file
    """
    spikes_df = pd.DataFrame({
        'spike_times': spike_times,
        'spike_values': spike_values
    })
    output_file = os.path.join(output_dir, 'detected_spikes.csv')
    spikes_df.to_csv(output_file, index=False)
    return output_file

def schmidt_trigger_auto(data_file, output_dir):
    """
    Detect spikes using Schmidt trigger with automatic thresholds from parameters.
    
    Parameters
    ----------
    data_file : str
        Path to processed data CSV file
    output_dir : str
        Directory to save results
        
    Returns
    -------
    tuple
        (output_file, spike_times, spike_values)
    """
    # Load parameters and data
    params = load_parameters('parameters.yaml')
    data = pd.read_csv(data_file)
    voltage = data['voltage'].values
    times = data['time'].values
    
    # Calculate thresholds
    noise_std = np.std(voltage)
    threshold_up = params['schmidt_t2'] * noise_std
    threshold_down = params['schmidt_t1'] * noise_std
    
    # Detect spikes
    spike_times, spike_values = _detect_spikes_schmidt_trigger(
        voltage, times, threshold_up, threshold_down
    )
    
    # Save and return results
    output_file = _save_spikes(spike_times, spike_values, output_dir)
    return output_file, spike_times, spike_values

def adjust_threshold(voltage, initial_threshold=None):
    """
    Interactive GUI for adjusting spike detection threshold.
    
    Parameters
    ----------
    voltage : np.ndarray
        Voltage values to analyze
    initial_threshold : float, optional
        Initial threshold value
        
    Returns
    -------
    float
        User-selected threshold value
    """
    params = load_parameters('parameters.yaml')
    
    if initial_threshold is None:
        noise_std = np.std(voltage)
        initial_threshold = params['schmidt_t2'] * noise_std
        print(f"Initial threshold calculated: {initial_threshold}")
        print(f"Data range: min={np.min(voltage)}, max={np.max(voltage)}")
    
    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.3)

    # Plot data segment
    ax.plot(voltage, label="Data", color="blue")
    threshold_line = ax.axhline(initial_threshold, color='red', linestyle="--", label="Threshold")
    ax.legend()
    ax.set_title("Adjust Threshold (Move the Red Line)")

    # Slider to adjust threshold
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Threshold', np.min(voltage), np.max(voltage), 
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
        print(f"Selected threshold: {slider.val}")
        plt.close(fig)

    button.on_clicked(confirm)

    plt.show(block=True)
    
    final_threshold = slider.val
    print(f"Final threshold value: {final_threshold}")
    return final_threshold

def detect_spikes_manual_threshold(data_file, output_dir, threshold):
    """
    Detect spikes using manual threshold with hysteresis ratio from parameters.
    
    Parameters
    ----------
    data_file : str
        Path to processed data CSV file
    output_dir : str
        Directory to save results
    threshold : float
        User-selected threshold (upper threshold)
        
    Returns
    -------
    tuple
        (output_file, spike_times, spike_values)
    """
    # Load parameters and data
    params = load_parameters('parameters.yaml')
    data = pd.read_csv(data_file)
    voltage = data['voltage'].values
    times = data['time'].values
    
    # Calculate thresholds using same ratio as auto detection
    ratio = params['schmidt_t1'] / params['schmidt_t2']
    threshold_up = threshold
    threshold_down = threshold * ratio
    
    print(f"Using thresholds: upper={threshold_up:.4f}, lower={threshold_down:.4f}")
    
    # Detect spikes
    spike_times, spike_values = _detect_spikes_schmidt_trigger(
        voltage, times, threshold_up, threshold_down
    )
    
    # Save and return results
    output_file = _save_spikes(spike_times, spike_values, output_dir)
    return output_file, spike_times, spike_values

def extract_waveforms(data_file, spikes_file, output_dir, pre_peak_length=2, post_peak_length=2):
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
    pre_peak_length : int
        length of the waveform before the peak (ms), default is 2
    post_peak_length : int
        length of the waveform after the peak (ms), default is 2
    """
    # Load data
    data = pd.read_csv(data_file)
    voltage = data['voltage'].values

    # Load parameters
    params = load_parameters('parameters.yaml')
    fs = params['sampling_rate']
    pre_samples = int(pre_peak_length * fs / 1000)
    post_samples = int(post_peak_length * fs / 1000)
    
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
