import os
import pandas as pd
import numpy as np
import time

def debug_data(file_path, N=15, X=5):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    print(f"\n--- DEBUGGING {file_path} ---")
    df = pd.read_csv(file_path)
    
    print(f"Original shape: {df.shape}")
    num_losses = (df['delay_ms'] == -1).sum()
    print(f"Total -1 (packet losses) in raw data: {num_losses} out of {len(df)}")
    
    # Fast NumPy array extraction for debugging
    start_time = time.time()
    packet_loss_arr = (df['delay_ms'].values == -1).astype(int)
    
    total_required = N + X
    labels = []
    
    for i in range(len(packet_loss_arr) - total_required + 1):
        loss_in_pred = np.sum(packet_loss_arr[i+N : i+N+X])
        labels.append(1 if loss_in_pred > 0 else 0)
        
    print(f"NumPy extraction took: {time.time() - start_time:.4f} seconds (RAM safe!)")
    
    counts = pd.Series(labels).value_counts()
    print("\nLabel distribution in extracted windows:")
    print(counts)
    
    if 1 in counts:
        percentage = (counts[1] / len(labels)) * 100
        print(f"Percentage of Loss windows (label=1): {percentage:.4f}%")
        print("Model diagnosis: CLASSES ARE EXTREMELY IMBALANCED.")
        print("XGBoost and Neural Networks learn to ALWAYS predict 0 (No Loss) because doing so guarantees ~99.9% accuracy!")
        print("Solution: We need to set 'scale_pos_weight' in XGBoost and use oversampling/SMOTE or class weights for NN.")
    else:
        print("NO LOSS WINDOWS FOUND! The test split probably contains zero loss events.")

if __name__ == "__main__":
    debug_data("dataset/first_capture_window/cpe_a-cpe_b-fiber.csv")
    debug_data("dataset/first_capture_window/cpe_b-cpe_a-mobile.csv")
