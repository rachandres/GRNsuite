import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN
import pandas as pd
import os

def reduce_dimensions(waveforms, n_components=3):
    """
    Reduce dimensionality of spike waveforms using SVD.
    
    Parameters:
        waveforms (np.ndarray): Matrix of spike waveforms (n_spikes x n_samples)
        n_components (int): Number of components to keep
        
    Returns:
        np.ndarray: Reduced dimensional representation of waveforms
    """
    # Standardize waveforms
    scaler = StandardScaler()
    waveforms_scaled = scaler.fit_transform(waveforms)
    
    # Apply SVD
    svd = TruncatedSVD(n_components=n_components)
    waveforms_reduced = svd.fit_transform(waveforms_scaled)
    
    # Calculate explained variance
    explained_var = np.sum(svd.explained_variance_ratio_) * 100
    print(f"Explained variance with {n_components} components: {explained_var:.2f}%")
    
    return waveforms_reduced

def cluster_spikes(reduced_waveforms, eps=0.5, min_samples=5):
    """
    Cluster reduced waveforms using DBSCAN.
    
    Parameters:
        reduced_waveforms (np.ndarray): SVD-reduced waveforms
        eps (float): DBSCAN epsilon parameter
        min_samples (int): DBSCAN min_samples parameter
        
    Returns:
        np.ndarray: Cluster labels (-1 for noise)
    """
    # Apply DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(reduced_waveforms)
    
    # Count clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Found {n_clusters} clusters")
    print(f"Number of noise points: {n_noise}")
    
    return labels

def sort_spikes(results_dir):
    """
    Sort spikes from a results directory and update spike times with unit labels.
    
    Parameters:
        results_dir (str): Path to results directory containing waveforms.csv and detected_spikes.csv
        
    Returns:
        str: Path to updated spike times file
    """
    # Load data
    waveforms_path = os.path.join(results_dir, 'waveforms.csv')
    spikes_path = os.path.join(results_dir, 'detected_spikes.csv')
    
    waveforms = pd.read_csv(waveforms_path).values
    spikes_df = pd.read_csv(spikes_path)
    
    # Reduce dimensions
    reduced_waveforms = reduce_dimensions(waveforms)
    
    # Cluster spikes
    labels = cluster_spikes(reduced_waveforms)
    
    # Create unit labels
    unit_labels = []
    for label in labels:
        if label == -1:
            unit_labels.append('unclassified_unit')
        else:
            unit_labels.append(f'unit_{label + 1}')
    
    # Add unit labels to spike times
    spikes_df['unit'] = unit_labels
    
    # Save updated spike times
    output_path = os.path.join(results_dir, 'sorted_spikes.csv')
    spikes_df.to_csv(output_path, index=False)
    
    # Save unit waveform averages
    unit_averages = {}
    for label in set(labels):
        if label != -1:
            mask = labels == label
            unit_averages[f'unit_{label + 1}'] = np.mean(waveforms[mask], axis=0)
    
    averages_df = pd.DataFrame(unit_averages)
    averages_path = os.path.join(results_dir, 'unit_averages.csv')
    averages_df.to_csv(averages_path, index=False)
    
    return output_path

def plot_sorting_results(results_dir):
    """
    Plot the results of spike sorting.
    
    Parameters:
        results_dir (str): Path to results directory
    """
    import matplotlib.pyplot as plt
    
    # Load sorted spikes and waveforms
    spikes_df = pd.read_csv(os.path.join(results_dir, 'sorted_spikes.csv'))
    waveforms = pd.read_csv(os.path.join(results_dir, 'waveforms.csv')).values
    
    # Plot waveform overlays by unit
    plt.figure(figsize=(15, 5))
    
    # Plot individual units
    unique_units = [u for u in spikes_df['unit'].unique() if u != 'unclassified_unit']
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_units)))
    
    for unit, color in zip(unique_units, colors):
        mask = spikes_df['unit'] == unit
        unit_waveforms = waveforms[mask]
        
        # Plot individual waveforms
        for waveform in unit_waveforms:
            plt.plot(waveform, color=color, alpha=0.1)
        
        # Plot mean waveform
        mean_waveform = np.mean(unit_waveforms, axis=0)
        plt.plot(mean_waveform, color=color, linewidth=2, label=unit)
    
    # Plot unclassified waveforms
    mask = spikes_df['unit'] == 'unclassified_unit'
    if np.any(mask):
        unclassified = waveforms[mask]
        for waveform in unclassified:
            plt.plot(waveform, 'k-', alpha=0.1)
    
    plt.title('Spike Waveforms by Unit')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Save plot
    plt.savefig(os.path.join(results_dir, 'unit_waveforms.png'))
    plt.close()
