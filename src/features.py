import numpy as np
import pandas as pd

def engineer_features(delays_arr, packet_loss_arr, global_max_delay=1000.0):
    """
    Engineers statistical features from a lookback window numpy array.
    Focuses on detecting Buffer Bloat (congestion) and Packet Loss Bursts.
    """
    valid_delays = delays_arr[~np.isnan(delays_arr)]
    N = len(delays_arr)
    
    # --- 1. HISTORICAL / BASE STATS ---
    mean_delay = np.mean(valid_delays) if valid_delays.size > 0 else np.nan
    jitter = np.std(valid_delays, ddof=1) if valid_delays.size > 1 else 0.0
    max_delay = np.max(valid_delays) if valid_delays.size > 0 else np.nan
    q95 = np.quantile(valid_delays, 0.95) if valid_delays.size > 0 else np.nan
    
    # --- 2. PACKET LOSS BURST FEATURES ---
    hist_loss_count = np.sum(packet_loss_arr)
    
    # Time since last loss (extremely predictive for bursty drops)
    loss_indices = np.where(packet_loss_arr > 0)[0]
    if len(loss_indices) > 0:
        time_since_last_loss = N - 1 - loss_indices[-1]
    else:
        time_since_last_loss = N * 10 # Arbitrary large penalty if no loss seen
        
    # --- 3. RECENT / BUFFER FILLING FEATURES (LAST 5 & 10 SECONDS) ---
    recent_k = min(5, N)
    very_recent_k = min(3, N)
    
    recent_loss_count = np.sum(packet_loss_arr[-recent_k:]) if recent_k > 0 else 0
    
    recent_delays = delays_arr[-recent_k:]
    valid_recent = recent_delays[~np.isnan(recent_delays)]
    recent_mean = np.mean(valid_recent) if valid_recent.size > 0 else np.nan
    recent_max = np.max(valid_recent) if valid_recent.size > 0 else np.nan
    recent_jitter = np.std(valid_recent, ddof=1) if valid_recent.size > 1 else 0.0

    # Trend (slope) of the last 10 seconds to catch buffer bloating
    trend_k = min(10, N)
    trend_delays = delays_arr[-trend_k:]
    valid_trend_idx = np.where(~np.isnan(trend_delays))[0]
    if len(valid_trend_idx) > 1:
        x = valid_trend_idx
        y = trend_delays[valid_trend_idx]
        recent_slope = np.polyfit(x, y, 1)[0]
    else:
        recent_slope = 0.0

    # --- 4. RATIOS (Congestion indicators) ---
    ratio_recent_mean_to_global = recent_mean / mean_delay if mean_delay and mean_delay > 0 else 1.0
    ratio_recent_max_to_global = recent_max / max_delay if max_delay and max_delay > 0 else 1.0
    
    # Spike detection in the very last 3 seconds
    very_recent_delays = delays_arr[-very_recent_k:]
    valid_very_recent = very_recent_delays[~np.isnan(very_recent_delays)]
    if valid_very_recent.size > 0 and not pd.isna(q95):
        spikes_over_q95 = np.sum(valid_very_recent > q95)
    else:
        spikes_over_q95 = 0

    # --- 5. FOURIER (FFT) FEATURES ---
    # Fill NaNs with global max just for the FFT to avoid complex NaN propagation
    fft_delays = np.nan_to_num(delays_arr, nan=global_max_delay)
    
    # Calculate FFT (magnitude spectrum)
    # We remove the DC component (index 0) because it's just the mean and overshadows the rest
    if N > 2:
        fft_mag = np.abs(np.fft.rfft(fft_delays))[1:]
        fft_freqs = np.fft.rfftfreq(N, d=1.0)[1:] # 1 Hz sampling rate
        
        fft_energy = np.sum(fft_mag ** 2)
        
        # High-Frequency Energy (upper half of the spectrum)
        half_idx = len(fft_mag) // 2
        fft_high_freq_energy = np.sum(fft_mag[half_idx:] ** 2)
        
        # Dominant Frequency
        dom_idx = np.argmax(fft_mag)
        fft_dominant_freq = fft_freqs[dom_idx]
        
        # Spectral Centroid (weighted average of frequencies)
        if np.sum(fft_mag) > 0:
            fft_spectral_centroid = np.sum(fft_freqs * fft_mag) / np.sum(fft_mag)
        else:
            fft_spectral_centroid = 0.0
    else:
        fft_energy = 0.0
        fft_high_freq_energy = 0.0
        fft_dominant_freq = 0.0
        fft_spectral_centroid = 0.0

    features = {
        'mean': mean_delay,
        'jitter': jitter,
        'max': max_delay,
        'q95': q95,
        'hist_loss_count': hist_loss_count,
        'time_since_last_loss': time_since_last_loss,
        'recent_loss_count': recent_loss_count,
        'recent_mean': recent_mean,
        'recent_max': recent_max,
        'recent_jitter': recent_jitter,
        'recent_slope': recent_slope,
        'ratio_recent_mean_to_global': ratio_recent_mean_to_global,
        'ratio_recent_max_to_global': ratio_recent_max_to_global,
        'spikes_over_q95': spikes_over_q95,
        'fft_energy': fft_energy,
        'fft_high_freq_energy': fft_high_freq_energy,
        'fft_dominant_freq': fft_dominant_freq,
        'fft_spectral_centroid': fft_spectral_centroid
    }
    
    # Handle NaNs from all-loss windows
    for k, v in features.items():
        if pd.isna(v):
            if 'ratio' in k:
                features[k] = 1.0
            else:
                features[k] = global_max_delay
                
    return features
