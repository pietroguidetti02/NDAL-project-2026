import pandas as pd
import numpy as np

def clean_data(df):
    """
    Creates a 'packet_loss' column based on delay_ms == -1, 
    and then replaces delay_ms = -1 with NaN to preserve true delay distribution.
    """
    df = df.copy()
    if 'delay_ms' in df.columns:
        df['packet_loss'] = (df['delay_ms'] == -1).astype(int)
        df['delay_ms'] = df['delay_ms'].replace(-1, np.nan)
    return df

def extract_sliding_windows(df, N, X):
    """
    Extracts sliding windows of size N and labels based on the subsequent X seconds.
    N: lookback window size (in seconds / rows)
    X: prediction window size (in seconds / rows)
    Assumes dataframe has 1 row per second granularity.
    """
    windows = []
    labels = []
    
    total_required = N + X
    
    for i in range(len(df) - total_required + 1):
        # The lookback window
        lookback_df = df.iloc[i : i+N]
        
        # The prediction window
        pred_df = df.iloc[i+N : i+N+X]
        
        # Label is 1 if there is any packet loss in the prediction window
        label = 1 if pred_df['packet_loss'].sum() > 0 else 0
        
        windows.append(lookback_df)
        labels.append(label)
        
    return windows, labels
