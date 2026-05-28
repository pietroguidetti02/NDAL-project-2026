import numpy as np
import pandas as pd

def engineer_features(delays_arr, packet_loss_arr, global_max_delay=1000.0):
    """
    Engineers statistical features from a lookback window numpy array.
    """
    valid_delays = delays_arr[~np.isnan(delays_arr)]
    
    # Base stats
    mean_delay = np.mean(valid_delays) if valid_delays.size > 0 else np.nan
    jitter = np.std(valid_delays, ddof=1) if valid_delays.size > 1 else 0.0
    max_delay = np.max(valid_delays) if valid_delays.size > 0 else np.nan
    min_delay = np.min(valid_delays) if valid_delays.size > 0 else np.nan
    median_delay = np.median(valid_delays) if valid_delays.size > 0 else np.nan
    
    # Tail spikes
    q90 = np.quantile(valid_delays, 0.90) if valid_delays.size > 0 else np.nan
    q95 = np.quantile(valid_delays, 0.95) if valid_delays.size > 0 else np.nan
    q99 = np.quantile(valid_delays, 0.99) if valid_delays.size > 0 else np.nan
    
    # Trend / Slope (linear regression over time)
    if valid_delays.size > 1:
        x = np.arange(valid_delays.size)
        y = valid_delays
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0.0
        
    # Rate of delay changes (mean of absolute differences)
    if valid_delays.size > 1:
        delay_change_rate = np.mean(np.abs(np.diff(valid_delays)))
    else:
        delay_change_rate = 0.0
        
    # Recent delay delta
    if valid_delays.size > 1:
        recent_delta = valid_delays[-1] - valid_delays[0]
    else:
        recent_delta = 0.0
        
    # Historical packet loss counts
    hist_loss_count = np.sum(packet_loss_arr)
    
    features = {
        'mean': mean_delay,
        'jitter': jitter,
        'max': max_delay,
        'min': min_delay,
        'median': median_delay,
        'q90': q90,
        'q95': q95,
        'q99': q99,
        'trend_slope': slope,
        'delay_change_rate': delay_change_rate,
        'recent_delta': recent_delta,
        'hist_loss_count': hist_loss_count
    }
    
    if pd.isna(mean_delay):
        features['mean'] = global_max_delay
        features['max'] = global_max_delay
        features['min'] = global_max_delay
        features['median'] = global_max_delay
        features['q90'] = global_max_delay
        features['q95'] = global_max_delay
        features['q99'] = global_max_delay
        
    return features
