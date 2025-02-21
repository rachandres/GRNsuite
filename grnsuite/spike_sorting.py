import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import os
import matplotlib.pyplot as plt

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

def estimate_clusters(reduced_waveforms, max_clusters=10):
    """
    Estimate optimal number of clusters using silhouette score.
    
    Parameters:
        reduced_waveforms (np.ndarray): SVD-reduced waveforms
        max_clusters (int): Maximum number of clusters to try
        
    Returns:
        int: Estimated optimal number of clusters
    """
    best_score = -1
    best_n = 2  # Start with minimum of 2 clusters
    
    for n in range(2, min(max_clusters + 1, len(reduced_waveforms))):
        kmeans = KMeans(n_clusters=n, random_state=42)
        labels = kmeans.fit_predict(reduced_waveforms)
        score = silhouette_score(reduced_waveforms, labels)
        print(f"Clusters: {n}, Silhouette Score: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_n = n
    
    print(f"\nOptimal number of clusters: {best_n}")
    return best_n

def cluster_spikes(reduced_waveforms, n_clusters=None):
    """
    Cluster reduced waveforms using KMeans.
    
    Parameters:
        reduced_waveforms (np.ndarray): SVD-reduced waveforms
        n_clusters (int, optional): Number of clusters. If None, estimates optimal number.
        
    Returns:
        np.ndarray: Cluster labels
    """
    if n_clusters is None:
        n_clusters = estimate_clusters(reduced_waveforms)
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced_waveforms)
    
    # Count clusters
    unique_labels = np.unique(labels)
    for label in unique_labels:
        print(f"Cluster {label} size: {np.sum(labels == label)}")
    
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
    # Load sorted spikes and waveforms
    spikes_df = pd.read_csv(os.path.join(results_dir, 'sorted_spikes.csv'))
    waveforms = pd.read_csv(os.path.join(results_dir, 'waveforms.csv')).values
    
    # Plot waveform overlays by unit
    plt.figure(figsize=(20, 6), dpi=300)
    
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
    
    # Save plot in both formats with high dpi
    base_path = os.path.join(results_dir, 'unit_waveforms')
    plt.savefig(base_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(base_path + '.svg', format='svg', bbox_inches='tight')
    plt.close()

def plot_clustering_summary(waveforms, reduced_waveforms, labels, save_path=None, display=False):
    """
    Create a three-panel figure showing waveform clusters analysis.
    
    Parameters
    ----------
    waveforms : ndarray
        Array of spike waveforms
    reduced_waveforms : ndarray
        Dimensionality-reduced waveforms
    labels : ndarray
        Cluster labels for each waveform
    save_path : str, optional
        Path to save the figure. If None, figure is displayed instead
    display : bool, optional
        If True, displays figure in interactive viewer
    """
    if display:
        import matplotlib
        matplotlib.use('TkAgg')
    
    # Increased figure size and added dpi
    fig = plt.figure(figsize=(20, 6), dpi=300)
    
    # Panel 1: Raw waveforms
    plt.subplot(131)
    for waveform in waveforms:
        plt.plot(waveform, color='blue', alpha=0.1)
    plt.title('Spike Waveforms')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    
    # Panel 2: Reduced components colored by cluster
    plt.subplot(132)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(reduced_waveforms[mask, 0], reduced_waveforms[mask, 1], 
                   c=[color], alpha=0.6, label=f'unit_{label + 1}')
    
    plt.title('Clusters in Component Space')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    
    # Panel 3: Mean waveforms by cluster
    plt.subplot(133)
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        cluster_waveforms = waveforms[mask]
        mean_waveform = np.mean(cluster_waveforms, axis=0)
        std_waveform = np.std(cluster_waveforms, axis=0)
        
        time_points = np.arange(len(mean_waveform))
        plt.plot(time_points, mean_waveform, color=color, 
                label=f'unit_{label + 1}', linewidth=2)
        plt.fill_between(time_points, 
                       mean_waveform - std_waveform,
                       mean_waveform + std_waveform,
                       color=color, alpha=0.2)
    
    plt.title('Mean Waveforms by Unit')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        svg_path = save_path.rsplit('.', 1)[0] + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    if display:
        plt.show()
    else:
        plt.close()

def plot_sorted_spikes(processed_data, spikes_df, save_path=None, display=False):
    """
    Plot processed data with spikes colored by unit.
    
    Parameters
    ----------
    processed_data : pandas.DataFrame
        DataFrame containing 'time' and 'voltage' columns
    spikes_df : pandas.DataFrame
        DataFrame containing spike times, values and unit labels
    save_path : str, optional
        Path to save the figure. If None, figure is displayed instead
    display : bool, optional
        If True, displays figure in interactive viewer
    """
    if display:
        import matplotlib
        matplotlib.use('TkAgg')
    
    # Increased figure size and added dpi
    plt.figure(figsize=(20, 8), dpi=300)
    
    # Plot full signal
    plt.plot(processed_data['time'], processed_data['voltage'], 'k-', alpha=0.5, label='Signal')
    
    # Plot spikes colored by unit
    unique_units = sorted(spikes_df['unit'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_units)))
    
    for unit, color in zip(unique_units, colors):
        unit_spikes = spikes_df[spikes_df['unit'] == unit]
        plt.scatter(unit_spikes['spike_times'], unit_spikes['spike_values'], 
                   c=[color], label=unit, s=100)
    
    plt.title('Spike Detection Results by Unit')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        svg_path = save_path.rsplit('.', 1)[0] + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    if display:
        plt.show()
    else:
        plt.close()
