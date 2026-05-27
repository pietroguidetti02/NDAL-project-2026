import pandas as pd
import numpy as np

def engineer_features(window_df, global_max_delay=1000.0):
    """
    Engineers statistical features from a lookback window dataframe.
    """
    delays = window_df['delay_ms'].dropna()
    
    # Base stats
    mean_delay = delays.mean() if not delays.empty else np.nan
    jitter = delays.std() if len(delays) > 1 else 0.0
    max_delay = delays.max() if not delays.empty else np.nan
    min_delay = delays.min() if not delays.empty else np.nan
    median_delay = delays.median() if not delays.empty else np.nan
    
    # Tail spikes: Quantiles (90th, 95th, 99th)
    q90 = delays.quantile(0.90) if not delays.empty else np.nan
    q95 = delays.quantile(0.95) if not delays.empty else np.nan
    q99 = delays.quantile(0.99) if not delays.empty else np.nan
    
    # Trend / Slope (linear regression over time)
    if len(delays) > 1:
        x = np.arange(len(delays))
        y = delays.values
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0.0
        
    # Rate of delay changes (mean of absolute differences)
    if len(delays) > 1:
        delay_change_rate = np.abs(np.diff(delays)).mean()
    else:
        delay_change_rate = 0.0
        
    # Recent delay delta (over the entire lookback window)
    if len(delays) > 1:
        recent_delta = delays.iloc[-1] - delays.iloc[0]
    else:
        recent_delta = 0.0
        
    # Historical packet loss counts
    hist_loss_count = window_df['packet_loss'].sum()
    
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
    
    # Handle edge cases for 'Link Down' (where all delays are NaN due to complete packet loss)
    if pd.isna(mean_delay):
        features['mean'] = global_max_delay  # high default delay penalty
        features['max'] = global_max_delay
        features['min'] = global_max_delay
        features['median'] = global_max_delay
        features['q90'] = global_max_delay
        features['q95'] = global_max_delay
        features['q99'] = global_max_delay
        
    return features
