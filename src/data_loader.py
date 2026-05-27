import os
import pandas as pd
import yaml

def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_file_paths(base_dir, direction):
    # direction is like "A->B", we need to map it to "cpe_a-cpe_b"
    mapping = {
        "A->B": "cpe_a-cpe_b",
        "B->A": "cpe_b-cpe_a",
        "A->C": "cpe_a-cpe_c",
        "C->A": "cpe_c-cpe_a",
        "B->C": "cpe_b-cpe_c",
        "C->B": "cpe_c-cpe_b"
    }
    prefix = mapping.get(direction)
    if not prefix:
        return []
        
    paths = []
    for window in ['first_capture_window', 'second_capture_window']:
        for tunnel in ['fiber', 'mobile']:
            file_path = os.path.join(base_dir, window, f"{prefix}-{tunnel}.csv")
            if os.path.exists(file_path):
                paths.append((window, file_path))
    return paths

def load_and_split_data(config, base_dir="dataset"):
    """
    Loads CSV data according to config splits and returns dicts of DataFrames 
    for training and testing grouped by tunnel type.
    """
    train_dfs = {'fiber': [], 'mobile': []}
    test_dfs = {'fiber': [], 'mobile': []}
    
    splits = config.get("train_test_split", {})
    for direction, split_info in splits.items():
        train_pcts = split_info.get("train", [0, 0])
        test_pcts = split_info.get("test", [0, 0])
        
        file_paths = get_file_paths(base_dir, direction)
        
        for window, file_path in file_paths:
            tunnel_type = 'mobile' if 'mobile' in file_path else 'fiber'
            
            df = pd.read_csv(file_path)
            # Ensure time is datetime and sorted
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            
            window_idx = 0 if window == 'first_capture_window' else 1
            train_pct = train_pcts[window_idx] / 100.0
            test_pct = test_pcts[window_idx] / 100.0
            
            if train_pct > 0:
                train_size = int(len(df) * train_pct)
                train_dfs[tunnel_type].append(df.iloc[:train_size].copy())
            if test_pct > 0:
                # The test set comes AFTER the train set chronologically
                start_idx = int(len(df) * train_pct)
                end_idx = start_idx + int(len(df) * test_pct)
                test_dfs[tunnel_type].append(df.iloc[start_idx:end_idx].copy())
                
    return train_dfs, test_dfs
