# Filename: coincidence.py
# Created: 15-10-2025
# Description: All necessary functions for key sifting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import random

def load_time_tags(filename):
    """Load time tag data from CSV file"""
    df = pd.read_csv(filename)
    return df['time'].values

def build_coincidence_histogram(local_times, remote_times, sample_size=1000, expected_delay=0, bin_size=0.1, window_ns=100):
    """
    Build cross-correlation histogram for coincidence detection
    """
    # Select random sample from local data
    local_total = len(local_times)
    local_mid_idx = local_total // 2
    local_start = max(sample_size, local_mid_idx - sample_size)
    local_end = min(local_total - sample_size, local_mid_idx + sample_size)
    local_sample = local_times[local_start:local_end]
    
    # Find remote's corresponding time window
    local_middle_time = local_times[local_mid_idx]
    expected_remote_middle = local_middle_time + expected_delay
    remote_diffs = np.abs(remote_times - expected_remote_middle)
    remote_mid_idx = np.argmin(remote_diffs)
    remote_start = max(0, remote_mid_idx - sample_size)
    remote_end = min(len(remote_times), remote_mid_idx + sample_size)
    remote_sample = remote_times[remote_start:remote_end]
    
    # Create delay histogram bins
    delay_center = expected_delay
    delay_min = delay_center - window_ns
    delay_max = delay_center + window_ns
    bins = np.arange(delay_min, delay_max + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # build histogram
    histogram = np.zeros(len(bins)-1)
    chunk_size = 50
    local_chunks = [local_sample[i:i+chunk_size] for i in range(0, len(local_sample), chunk_size)]
    total_pairs = 0
    for i, local_chunk in enumerate(local_chunks):
        for local_time in local_chunk:
            time_delays = remote_sample - local_time
            valid_delays = time_delays[(time_delays >= delay_min) & (time_delays <= delay_max)]
            if len(valid_delays) > 0:
                counts, _ = np.histogram(valid_delays, bins=bins)
                histogram += counts
                total_pairs += len(valid_delays)    
    return bin_centers, histogram

def calculate_fwhm(x, y):
    """Calculate Full Width at Half Maximum with proper interpolation"""
    if len(y) == 0 or np.max(y) == 0:
        return None, None, None
    
    # Find peak
    peak_idx = np.argmax(y)
    peak_value = y[peak_idx]
    peak_position = x[peak_idx]
    
    # Find half maximum
    half_max = peak_value / 2
    
    # Find all points above half maximum
    above_half = y >= half_max
    indices = np.where(above_half)[0]
    if len(indices) == 0:
        return None, None, peak_position
    
    # Find the continuous region around the peak
    peak_region = []
    for idx in indices:
        if len(peak_region) == 0 or idx == peak_region[-1] + 1:
            peak_region.append(idx)
        elif abs(idx - peak_idx) < abs(peak_region[0] - peak_idx):
            peak_region = [idx]
    if len(peak_region) < 2:
        return None, None, peak_position
    left_idx = min(peak_region)
    right_idx = max(peak_region)
    
    # Interpolate for more precise FWHM
    def interpolate_half_max(idx, direction):
        if direction == 'left' and idx > 0:
            x1, x2 = x[idx-1], x[idx]
            y1, y2 = y[idx-1], y[idx]
            if y2 != y1:
                return x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
        elif direction == 'right' and idx < len(x) - 1:
            x1, x2 = x[idx], x[idx+1]
            y1, y2 = y[idx], y[idx+1]
            if y2 != y1:
                return x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
        return x[idx]
    left_x = interpolate_half_max(left_idx, 'left')
    right_x = interpolate_half_max(right_idx, 'right')
    fwhm = right_x - left_x
    return fwhm, half_max, peak_position

def main():
    # Parameters
    expected_delay = 0  # ns
    sample_size = 1000
    bin_size = 0.1  # ns
    window_ns = 100  # ns - window around expected delay
    
    # Load data
    local_times = load_time_tags('qkd_raw_data/alice_raw_timetags.csv')
    remote_times = load_time_tags('qkd_raw_data/bob_raw_timetags.csv')
    
    print(f"Loaded {len(local_times):,} local timestamps")
    print(f"Loaded {len(remote_times):,} remote timestamps")
    print(f"Local time range: {local_times[0]:.6f} to {local_times[-1]:.6f} ns")
    print(f"Remote time range: {remote_times[0]:.6f} to {remote_times[-1]:.6f} ns")
    
    # Perform coincidence detection
    bin_centers, counts = build_coincidence_histogram(
        local_times, remote_times, sample_size, expected_delay, bin_size, window_ns)
    
    # Calculate FWHM and other parameters
    fwhm, half_max, measured_delay = calculate_fwhm(bin_centers, counts)
    coincidence_window = fwhm / 2 if fwhm is not None else None
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot histogram
    plt.plot(bin_centers, counts, 'r-', linewidth=1.5, label='Coincidence counts')
    plt.fill_between(bin_centers, counts, alpha=0.3, color='red')
    
    # Mark peak and FWHM
    if fwhm is not None and measured_delay is not None:
        plt.axvline(measured_delay, color='blue', linestyle='--', linewidth=2,
                   label=f'Peak delay: {measured_delay:.2f} ns')
        plt.axhline(half_max, color='green', linestyle='--', alpha=0.7, linewidth=1,
                   label=f'Half maximum: {half_max:.1f}')
        
        # Mark FWHM boundaries
        left_bound = measured_delay - fwhm/2
        right_bound = measured_delay + fwhm/2
        plt.axvline(left_bound, color='orange', linestyle=':', alpha=0.8, linewidth=1)
        plt.axvline(right_bound, color='orange', linestyle=':', alpha=0.8, linewidth=1)
        
        # FWHM annotation
        plt.annotate(f'FWHM = {fwhm:.2f} ns', 
                    xy=(measured_delay, half_max * 1.1), 
                    xytext=(measured_delay, half_max * 1.3),
                    arrowprops=dict(arrowstyle='<->', color='orange', lw=1.5),
                    ha='center', fontsize=12, color='orange', weight='bold')
    
    plt.xlabel('Relative delay (ns)', fontsize=14)
    plt.ylabel('Coincidence counts', fontsize=14)
    plt.title('QKD Coincidence Detection Histogram', fontsize=16, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Set axis limits to zoom into the peak region - show absolute delay values
    if measured_delay is not None:
        zoom_window = max(10, fwhm * 3) if fwhm else 10  # Show 3x FWHM or minimum 10 ns
        plt.xlim(measured_delay - zoom_window, measured_delay + zoom_window)
        
        # Force display of absolute delay values instead of scientific notation
        from matplotlib.ticker import FuncFormatter
        def delay_formatter(x, pos):
            return f'{x:.1f}'
        plt.gca().xaxis.set_major_formatter(FuncFormatter(delay_formatter))
    
    # Save the plot
    plt.tight_layout()
    #plt.savefig('coincidence_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

def detect_coincidences(local_df, remote_df, measured_delay, coincidence_window):
    """
    Detect coincidence events using the measured delay and coincidence window.
    Returns only essential data: result and basis_match flag.
    Each coincidence gets a unique event_id for synchronization between nodes.
    """
    coincidences = []
    
    for i, local_row in local_df.iterrows():
        local_time = local_row['time']
        local_basis = local_row['basis']
        local_result = local_row['result']
        
        # Expected remote arrival time
        expected_remote_time = local_time + measured_delay
        
        # Find remote events within coincidence window
        time_diff = np.abs(remote_df['time'] - expected_remote_time)
        within_window = time_diff <= coincidence_window
        
        if within_window.any():
            # Take the closest remote event if multiple are within window
            closest_idx = remote_df[within_window].index[np.argmin(time_diff[within_window])]
            remote_row = remote_df.loc[closest_idx]
            
            coincidences.append({
                'event_id': i,  # Use the original local dataframe index
                'local_result': int(local_result),  # Ensure integer
                'basis_match': local_basis == remote_row['basis']
            })
    
    coincidences_df = pd.DataFrame(coincidences)
    # Set event_id as the index so both nodes have synchronized indices
    coincidences_df.set_index('event_id', inplace=True)
    
    return coincidences_df

def extract_matching_basis_events(coincidences_df):
    """
    Extract only the events where local and remote measured in the same basis.
    These are the events that can be used for key generation.
    """
    # Only keep events where basis match
    matching_events = coincidences_df[coincidences_df['basis_match'] == True].copy()
    
    # Reset index to create a sequential index starting from 0
    # This ensures both Alice and Bob have the same index scheme
    matching_events = matching_events.reset_index(drop=True)
    
    return matching_events

def calculate_qber_and_test_security(matching_events, test_fraction=0.5):
    """
    Calculate QBER using a random subset of matching basis events
    """
    if len(matching_events) == 0:
        return None, pd.DataFrame(), pd.DataFrame()

    n_test = int(len(matching_events) * test_fraction)
    if n_test == 0:
        n_test = min(1, len(matching_events))  # Test at least 1 event if possible
    
    # Randomly sample test events
    test_indices = np.random.choice(matching_events.index, size=n_test, replace=False)
    
    test_events = matching_events.loc[test_indices].copy()
    key_events = matching_events.drop(test_indices).copy()
    
    return test_indices, test_events, key_events

def save_keys(name, key_events):
    """Save key to files"""
    if len(key_events) == 0:
        print("No key events available")
        return ""
    
    local_key = ''.join(str(x) for x in key_events['local_result'].tolist())
    
    with open(name, 'w') as f:
        f.write(local_key)
    
    return local_key

if __name__ == "__main__":
    main()
