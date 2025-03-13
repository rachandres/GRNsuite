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

def plot_waveforms(waveforms, pre_peak_ms=2.0, post_peak_ms=2.0, show_time_axis=True, figsize=(10, 8), output_dir=None):
    """
    Plots all extracted waveforms with two subplots:
    1. Raw waveforms overlay
    2. Average waveform with standard deviation

    Parameters:
        waveforms (numpy.ndarray): Extracted spike waveforms, shape (num_spikes, window_samples).
        pre_peak_ms (float): Milliseconds before the spike peak, used for time axis.
        post_peak_ms (float): Milliseconds after the spike peak, used for time axis.
        show_time_axis (bool): If True, x-axis is in milliseconds. If False, in samples.
        figsize (tuple): Figure size as (width, height).
        output_dir (str, optional): Directory to save the figure as PNG. If None, figure is not saved.
    
    Returns:
        tuple: (avg_waveform, std_waveform) - the average and standard deviation of waveforms
    """
    # Calculate average and standard deviation
    avg_waveform = np.mean(waveforms, axis=0)
    std_waveform = np.std(waveforms, axis=0)
    
    # Create x-axis values
    if show_time_axis:
        x_values = np.linspace(-pre_peak_ms, post_peak_ms, waveforms.shape[1])
        x_label = "Time (ms)"
    else:
        x_values = np.arange(waveforms.shape[1])
        x_label = "Samples"
    
    # Calculate y-limits based on all waveforms (with a small padding)
    y_min = np.min(waveforms) * 1.05  # Add 5% padding
    y_max = np.max(waveforms) * 1.05
        
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Subplot 1: Raw waveforms
    for waveform in waveforms:
        ax1.plot(x_values, waveform, color="blue", alpha=0.2, linewidth=0.5)
    
    ax1.set_title(f"Individual Spike Waveforms ({waveforms.shape[0]} spikes)")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(y_min, y_max)  # Set y-limits
    if show_time_axis:
        ax1.axvline(x=0, color='b', linestyle='--', alpha=0.7, label='Spike Peak')
        ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Average waveform with standard deviation
    ax2.plot(x_values, avg_waveform, 'r-', linewidth=2, label='Average Waveform')
    ax2.fill_between(x_values, 
                     avg_waveform - std_waveform, 
                     avg_waveform + std_waveform, 
                     color='r', alpha=0.2, label='± Standard Deviation')
    
    ax2.set_ylim(y_min, y_max)  # Use same y-limits as first subplot
    if show_time_axis:
        ax2.axvline(x=0, color='b', linestyle='--', alpha=0.7, label='Spike Peak')
    
    ax2.set_title("Average Waveform with Standard Deviation")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output directory is provided
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig_path = os.path.join(output_dir, 'waveforms_plot.png')
        plt.savefig(fig_path, dpi=300)
        print(f"Waveform figure saved to: {fig_path}")
    
    # Show the figure
    plt.show()
    
    return avg_waveform, std_waveform
