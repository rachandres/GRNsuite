import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import os
import matplotlib.pyplot as plt

def logarithmic_model(x, a, b, c):
    """Logarithmic model function for curve fitting.
    
    Parameters:
        x (np.ndarray): Independent variable
        a, b, c (float): Model parameters
        
    Returns:
        np.ndarray: Model values
    """
    return a * np.log(b * x + c)

def fit_log_model(spike_times, isis):
    """Fit a logarithmic model to the inter-spike intervals.
    
    Parameters:
        spike_times (np.ndarray): Spike times (x values)
        isis (np.ndarray): Inter-spike intervals (y values)
        
    Returns:
        tuple: (a, b, c) parameters of the fitted model
    """
    # Initial parameters
    p0 = [0.2, 0.6, 0.9]
    
    # Lower bounds
    bounds_lower = [0, 0, 0]
    # Upper bounds - high values for essentially no upper bound
    bounds_upper = [np.inf, np.inf, np.inf]
    
    # Weights - inverse of y values
    weights = 1.0 / isis
    
    # Curve fitting with robust method
    try:
        popt, _ = curve_fit(
            logarithmic_model, 
            spike_times, 
            isis, 
            p0=p0, 
            bounds=(bounds_lower, bounds_upper), 
            sigma=weights, 
            method='trf'
        )
        return popt
    except RuntimeError:
        # Default values if fitting fails
        print("Warning: Curve fitting failed. Using default parameters.")
        return [0.2, 0.6, 0.9]

def detect_end_of_burst(spike_times, isis, model_params, threshold=2.5):
    """Detect end of burst positions based on ISI outliers.
    
    Parameters:
        spike_times (np.ndarray): Spike times
        isis (np.ndarray): Inter-spike intervals
        model_params (tuple): (a, b, c) parameters from the fitted model
        threshold (float): Threshold for outlier detection
        
    Returns:
        tuple: (eob_times, non_eob_mask) End of burst times and mask for non-EOB spikes
    """
    # Calculate expected ISIs from the model
    a, b, c = model_params
    expected_isis = logarithmic_model(spike_times, a, b, c)
    
    # Identify outliers where actual ISI is much larger than expected
    outlier_mask = isis / expected_isis > threshold
    
    # Get EOB spike times
    eob_times = spike_times[outlier_mask]
    
    # Return EOB times and a mask for non-EOB spikes
    return eob_times, ~outlier_mask

def classify_grn2_spikes(eob_times, grn1_times, grnx_times, parameters=None):
    """Classify GRN2 spikes based on end of burst positions and various conditions.
    
    Parameters:
        eob_times (np.ndarray): End of burst spike times
        grn1_times (np.ndarray): GRN1 spike times (without EOB)
        grnx_times (np.ndarray): Unclassified spike times
        parameters (dict, optional): Time window parameters
        
    Returns:
        tuple: (grn2_times, updated_grn1_times, updated_grnx_times)
    """
    if parameters is None:
        parameters = {
            'time_window': 0.004,    # 4ms window
            'time_windowb': 0.010,   # 10ms window
            'burst_window': 0.015    # 15ms window
        }
    
    time_window = parameters['time_window']
    time_windowb = parameters['time_windowb']
    burst_window = parameters['burst_window']
    
    # Make copies to avoid modifying the originals
    grn1 = grn1_times.copy()
    grnx = grnx_times.copy() if grnx_times is not None else np.array([])
    grn2 = np.array([])
    
    # Process each EOB spike
    for spike_time in eob_times:
        # Check condition_a (GRN1 spike exists within 0.004s of EOB)
        condition_a = np.any((grn1 > spike_time - time_window) & (grn1 < spike_time))

        # Check condition_b (GRNx spike exists within 0.004s of EOB)
        condition_b = np.any((grnx > spike_time - time_window) & (grnx < spike_time + time_window))

        # Check condition_c
        condition_c = not (condition_a or condition_b)
        
        # Check condition_d (at least 1 spike preceding EOB within 0.015s)
        preceding_spikes = grn1[(grn1 < spike_time) & (grn1 > spike_time - burst_window)]
        condition_d = len(preceding_spikes) >= 1
        
        # Check condition_e (exact time match in GRNx_times)
        condition_e = np.any(grnx == spike_time)
        
        # Check condition_f (GRNx_times spike occurs less than 0.010s after EOB)
        after_eob_spike = grnx[(grnx > spike_time) & (grnx <= spike_time + time_windowb)]
        if len(after_eob_spike) > 0:
            condition_f = True
            after_eob_spike_time = after_eob_spike[0]  # First spike after EOB
        else:
            condition_f = False
            after_eob_spike_time = None
        
        # Apply conditions based on MATLAB code
        if condition_e and condition_d and not condition_a:
            # Move spike from GRNx to GRN2 and GRN1
            grn2 = np.append(grn2, spike_time)
            grn1 = np.append(grn1, spike_time)
            # Remove from GRNx
            grnx = grnx[grnx != spike_time]
            
        elif condition_a and not condition_b and condition_d and not condition_e:
            # Move spike from EOB to GRN2
            grn2 = np.append(grn2, spike_time)
            
        elif condition_c and condition_f and condition_d:
            # Move after_eob_spike from GRNx to GRN2 & GRN1
            grn2 = np.append(grn2, after_eob_spike_time)
            grn1 = np.append(grn1, after_eob_spike_time)
            # Remove from GRNx
            grnx = grnx[grnx != after_eob_spike_time]
            # Move EOB spike back to GRN1
            grn1 = np.append(grn1, spike_time)
            
        elif condition_b and not condition_a and condition_d:
            # Find and move spike from GRNx to GRN2
            moved_spike = grnx[(grnx > spike_time - time_window) & (grnx < spike_time + time_window)]
            if len(moved_spike) > 0:
                moved_spike = moved_spike[0]
                grn2 = np.append(grn2, moved_spike)
                # Remove from GRNx
                grnx = grnx[grnx != moved_spike]
                
        elif condition_a and condition_b and condition_d:
            # Move spike from EOB to GRN2
            grn2 = np.append(grn2, spike_time)
            
        elif condition_c and condition_d:
            # Move spike from EOB to GRN1 and GRN2
            grn1 = np.append(grn1, spike_time)
            grn2 = np.append(grn2, spike_time)
            
        elif not condition_d and not condition_b:
            # Move spike from EOB to GRNx
            grnx = np.append(grnx, spike_time)
    
    # Sort all arrays
    grn1 = np.sort(grn1)
    grn2 = np.sort(grn2)
    grnx = np.sort(grnx)
    
    return grn2, grn1, grnx

def detect_doublets(spike_times, isi_threshold=0.004):
    """Detect doublets (spikes with very short ISIs) as GRN3 spikes.
    
    Parameters:
        spike_times (np.ndarray): Spike times
        isi_threshold (float): ISI threshold for doublet detection (default: 4ms)
        
    Returns:
        np.ndarray: Indices of doublet spikes
    """
    if len(spike_times) < 2:
        return np.array([])
    
    # Calculate ISIs
    isis = np.diff(spike_times)
    
    # Find doublets
    doublet_mask = isis < isi_threshold
    
    # The doublet_mask corresponds to the first spike in each pair
    # We want the indices of the second spike in each pair
    doublet_indices = np.where(doublet_mask)[0] + 1
    
    return doublet_indices

def detect_bursts(spike_times, min_spikes=10):
    """Main function to detect bursts and classify spikes into GRN1, GRN2, and GRN3.
    
    Parameters:
        spike_times (np.ndarray): Array of spike times
        min_spikes (int): Minimum number of spikes required for burst detection
        
    Returns:
        dict: Dictionary containing classified spike times:
            'GRN1': Principal neuron spikes
            'GRN2': End of burst spikes
            'GRN3': Secondary neuron spikes/doublets
    """
    # Initialize with default classification
    result = {
        'GRN1': spike_times,
        'GRN2': np.array([]),
        'GRN3': np.array([])
    }
    
    # Basic validation
    if spike_times is None or len(spike_times) < min_spikes:
        print(f"Not enough spikes ({len(spike_times)}) to perform burst detection. Minimum required: {min_spikes}")
        return result
    
    # Sort spike times
    sorted_times = np.sort(spike_times)
    
    # 1. Initially, all spikes are GRN1, no GRNx
    grn1_times = sorted_times
    grnx_times = np.array([])
    
    # 2. Calculate inter-spike intervals
    if len(grn1_times) > 1:
        isis = np.diff(grn1_times)
        spike_positions = grn1_times[:-1]  # Positions correspond to first spike of each interval
        
        # 3. Fit logarithmic model to ISIs
        model_params = fit_log_model(spike_positions, isis)
        
        # 4. Detect end of burst positions
        eob_times, non_eob_mask = detect_end_of_burst(spike_positions, isis, model_params)
        
        # Get non-EOB spike times for further processing
        grn_new = spike_positions[non_eob_mask]
        
        # Save a copy for doublet detection
        peak_loc3 = grn_new.copy()
        
        # 5. Classify GRN2 spikes based on EOB positions
        grn2_times, grn1_updated, grnx_updated = classify_grn2_spikes(eob_times, grn_new, grnx_times)
        
        # 6. Calculate ISIs for doublet detection (without EOB positions)
        if len(peak_loc3) > 1:
            doublet_indices = detect_doublets(peak_loc3)
            grn3_from_doublets = peak_loc3[doublet_indices] if len(doublet_indices) > 0 else np.array([])
            
            # 7. Make GRN1 the remainder after removing GRN3 doublets
            grn1_final = np.setdiff1d(grn1_updated, grn3_from_doublets)
            
            # 8. Combine GRN3 doublets with any remaining unclassified spikes
            grn3_final = np.concatenate([grn3_from_doublets, grnx_updated])
            grn3_final = np.sort(grn3_final)
            
            result = {
                'GRN1': grn1_final,
                'GRN2': grn2_times,
                'GRN3': grn3_final
            }
        else:
            # If no spikes left for doublet detection, use the updated classifications
            result = {
                'GRN1': grn1_updated,
                'GRN2': grn2_times,
                'GRN3': grnx_updated
            }
    
    return result

def process_spike_file(spike_file_path, output_dir=None, column_name=None):
    """Process a spike times CSV file to detect bursts.
    
    Parameters:
        spike_file_path (str): Path to CSV file with spike times
        output_dir (str, optional): Directory to save results. If None, uses same directory as input file.
        column_name (str, optional): Name of the column containing spike times. If None, will try to detect.
        
    Returns:
        dict: Dictionary containing classified spike times
    """
    # Load spike CSV
    try:
        spikes_df = pd.read_csv(spike_file_path)
        print(f"Loaded spike file: {spike_file_path}")
        print(f"Columns in file: {spikes_df.columns.tolist()}")
        
        # Determine which column has spike times
        if column_name is not None:
            # Use the specified column name
            if column_name in spikes_df.columns:
                spike_times_column = column_name
            else:
                raise ValueError(f"Specified column '{column_name}' not found in the CSV file.")
        else:
            # Try common column names for spike times
            potential_columns = ['spike_times', 'time', 'times', 'spike_time']
            for col in potential_columns:
                if col in spikes_df.columns:
                    spike_times_column = col
                    print(f"Using column '{spike_times_column}' for spike times")
                    break
            else:
                # If no match found, use the first column that has numeric values
                numeric_cols = spikes_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    spike_times_column = numeric_cols[0]
                    print(f"No standard spike time column found. Using first numeric column: '{spike_times_column}'")
                else:
                    raise ValueError("No suitable numeric column found for spike times")
        
        # Extract spike times
        spike_times = spikes_df[spike_times_column].values
        print(f"Extracted {len(spike_times)} spike times from column '{spike_times_column}'")
        
        # Handle empty spike times
        if len(spike_times) == 0:
            print("Warning: No spike times found in the file")
            return {'GRN1': np.array([]), 'GRN2': np.array([]), 'GRN3': np.array([])}
        
        # Run burst detection
        burst_results = detect_bursts(spike_times)
        
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(spike_file_path)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output file path
        base_name = os.path.splitext(os.path.basename(spike_file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_burst_classification.csv")
        
        # Save results to CSV
        results_dict = {
            'neuron_type': [],
            'spike_times': []
        }
        
        for neuron_type, times in burst_results.items():
            for t in times:
                results_dict['neuron_type'].append(neuron_type)
                results_dict['spike_times'].append(t)
        
        results_df = pd.DataFrame(results_dict)
        results_df = results_df.sort_values('spike_times')
        results_df.to_csv(output_file, index=False)
        
        print(f"Burst classification saved to: {output_file}")
        print(f"Detected {len(burst_results['GRN1'])} GRN1 spikes, "
              f"{len(burst_results['GRN2'])} GRN2 spikes, "
              f"{len(burst_results['GRN3'])} GRN3 spikes")
        
        return burst_results
        
    except Exception as e:
        print(f"Error processing spike file: {e}")
        # Print a more detailed traceback
        import traceback
        traceback.print_exc()
        return {'GRN1': np.array([]), 'GRN2': np.array([]), 'GRN3': np.array([])}

def plot_burst_classification(signal_file, burst_results=None, spike_file=None, output_dir=None, 
                             time_range=None, figsize=(12, 6), save_fig=True, dpi=300):
    """Plot the signal with classified spike types (GRN1, GRN2, GRN3) highlighted.
    
    Parameters:
        signal_file (str): Path to the processed signal CSV file (e.g., interactive_processed.csv)
        burst_results (dict, optional): Dictionary with GRN1, GRN2, GRN3 spike times.
                                        If None, will try to load from spike_file
        spike_file (str, optional): Path to CSV file with burst classification results.
                                    Required if burst_results is None
        output_dir (str, optional): Directory to save the figure. If None, uses signal file's directory
        time_range (tuple, optional): (start_time, end_time) to zoom into a specific region
        figsize (tuple): Figure size in inches (width, height)
        save_fig (bool): Whether to save the figure to disk
        dpi (int): Resolution of saved figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    import matplotlib.pyplot as plt
    
    # Load the signal data
    try:
        signal_data = pd.read_csv(signal_file)
        
        # Determine column names for time and signal
        if 'time' in signal_data.columns:
            time_col = 'time'
        else:
            # Try to find time column - usually the first column
            numeric_cols = signal_data.select_dtypes(include=[np.number]).columns
            time_col = numeric_cols[0]
            
        # Find signal column - usually the second numeric column
        numeric_cols = signal_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            signal_col = numeric_cols[1] if numeric_cols[0] == time_col else numeric_cols[0]
        else:
            signal_col = numeric_cols[0]  # Use the same column if only one is available
            
        print(f"Using column '{time_col}' for time and '{signal_col}' for signal")
        
        # Extract time and signal data
        time = signal_data[time_col].values
        signal = signal_data[signal_col].values
        
        # Get burst classification results if not provided
        if burst_results is None:
            if spike_file is None:
                raise ValueError("Either burst_results or spike_file must be provided")
            
            # Try to find the burst classification file
            if 'burst_classification' not in spike_file:
                # Look for the classification file in the same directory
                base_dir = os.path.dirname(spike_file)
                base_name = os.path.splitext(os.path.basename(spike_file))[0]
                classification_file = os.path.join(base_dir, f"{base_name}_burst_classification.csv")
                
                if os.path.exists(classification_file):
                    spike_file = classification_file
                    print(f"Using burst classification file: {classification_file}")
            
            # Load classification data
            try:
                burst_df = pd.read_csv(spike_file)
                
                # Check if it's a burst classification file or a spike times file
                if 'neuron_type' in burst_df.columns and 'spike_times' in burst_df.columns:
                    # Already a classification file
                    grn1_times = burst_df[burst_df['neuron_type'] == 'GRN1']['spike_times'].values
                    grn2_times = burst_df[burst_df['neuron_type'] == 'GRN2']['spike_times'].values
                    grn3_times = burst_df[burst_df['neuron_type'] == 'GRN3']['spike_times'].values
                else:
                    # Regular spike file - run classification
                    print(f"No burst classification found. Running classification on {spike_file}")
                    if 'spike_times' in burst_df.columns:
                        spike_times = burst_df['spike_times'].values
                    else:
                        # Try to find spike times column
                        numeric_cols = burst_df.select_dtypes(include=[np.number]).columns
                        spike_times = burst_df[numeric_cols[0]].values
                    
                    # Run classification
                    burst_results = detect_bursts(spike_times)
                    grn1_times = burst_results['GRN1']
                    grn2_times = burst_results['GRN2']
                    grn3_times = burst_results['GRN3']
            except Exception as e:
                print(f"Error loading burst classification: {e}")
                raise
        else:
            # Use provided burst results
            grn1_times = burst_results['GRN1']
            grn2_times = burst_results['GRN2'] 
            grn3_times = burst_results['GRN3']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Apply time range if specified
        if time_range is not None:
            start_time, end_time = time_range
            mask = (time >= start_time) & (time <= end_time)
            plot_time = time[mask]
            plot_signal = signal[mask]
        else:
            plot_time = time
            plot_signal = signal
        
        # Plot signal
        ax.plot(plot_time, plot_signal, 'k-', alpha=0.7, label='Signal')
        
        # Find signal values at spike times for plotting markers
        def get_signal_values_at_times(spike_times):
            values = []
            for t in spike_times:
                # Find the closest time point
                idx = np.argmin(np.abs(time - t))
                values.append(signal[idx])
            return np.array(values)
        
        # Plot spikes
        for spike_type, spike_times, color, label in [
            ('GRN1', grn1_times, 'red', 'GRN1 - Principal'),
            ('GRN2', grn2_times, 'blue', 'GRN2 - End of Burst'),
            ('GRN3', grn3_times, 'green', 'GRN3 - Secondary')
        ]:
            # Filter spikes to the visible time range
            if time_range is not None:
                start_time, end_time = time_range
                visible_spikes = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
            else:
                visible_spikes = spike_times
            
            if len(visible_spikes) > 0:
                values = get_signal_values_at_times(visible_spikes)
                ax.scatter(visible_spikes, values, color=color, s=50, alpha=0.7, label=label)
                print(f"Plotted {len(visible_spikes)} {spike_type} spikes")
        
        # Set plot properties
        ax.set_title("Signal with Classified Spike Types")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            if output_dir is None:
                output_dir = os.path.dirname(signal_file)
            
            os.makedirs(output_dir, exist_ok=True)
            
            fig_path = os.path.join(output_dir, "burst_classification_plot.png")
            plt.savefig(fig_path, dpi=dpi)
            print(f"Saved figure to {fig_path}")
        
        return fig
        
    except Exception as e:
        print(f"Error plotting burst classification: {e}")
        import traceback
        traceback.print_exc()
        return None
