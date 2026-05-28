import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Append current path to ensure imports work
sys.path.append(os.getcwd())

from src.data_loader import load_config, load_and_split_data
from src.preprocessor import clean_data
from main import extract_single_window
from joblib import Parallel, delayed

def main():
    print("Loading data for Mobile A->B...")
    config = {'train_test_split': {'A->B': {'train': [50, 0], 'test': [0, 0]}}}
    train_dfs_dict, _ = load_and_split_data(config)
    
    if not train_dfs_dict['mobile']:
        print("No mobile data found!")
        return
        
    df = clean_data(train_dfs_dict['mobile'][0])
    
    delays = df['delay_ms'].values
    losses = df['packet_loss'].values
    
    print(f"Extracting features from {len(delays)} rows...")
    # Extract features
    results = Parallel(n_jobs=-1)(
        delayed(extract_single_window)(j, delays, losses, 15, 5, 1000.0)
        for j in range(len(delays) - 20)
    )
    
    X_features = pd.DataFrame([r[0] for r in results])
    y_labels = pd.Series([r[1] for r in results])
    
    print(f"Target distribution:\n{y_labels.value_counts()}")
    
    # Filter to only rows where Target is well-represented if possible, or just plot
    X_features['Target'] = y_labels.map({0: 'No Loss', 1: 'Loss'})
    
    print("Generating distribution plots...")
    features_to_plot = ['mean', 'jitter', 'max', 'trend_slope', 'delay_change_rate', 'hist_loss_count']
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features_to_plot, 1):
        plt.subplot(2, 3, i)
        # Using KDE plot. Sometimes if a class has very low variance, it fails, so we catch errors
        try:
            sns.kdeplot(data=X_features, x=feature, hue='Target', common_norm=False, fill=True, alpha=0.5)
        except Exception as e:
            print(f"Failed to plot {feature}: {e}")
        plt.title(f'Distribution of {feature}')
        
    plt.tight_layout()
    output_path = os.path.join(os.getcwd(), 'feature_distributions_mobile.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main()
