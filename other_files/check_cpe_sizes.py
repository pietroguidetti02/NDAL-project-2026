import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
from src.data_loader import load_config, load_and_split_data

config = load_config('config/exp_federated.yaml')
train_dfs_dict, test_dfs_dict = load_and_split_data(config)
train_dfs = train_dfs_dict.get('mobile', [])

print(f"Total CSV chunks loaded: {len(train_dfs)}")

indices = np.array_split(range(len(train_dfs)), 3)
splits = [[train_dfs[i] for i in idx] for idx in indices]
clients_dfs = {'CPE_A': splits[0], 'CPE_B': splits[1], 'CPE_C': splits[2]}

for cid, dfs in clients_dfs.items():
    total_rows = sum(len(df) for df in dfs)
    print(f"{cid} ha {len(dfs)} chunks per un totale di {total_rows} righe raw.")
