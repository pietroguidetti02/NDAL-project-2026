import pandas as pd
import numpy as np
from src.data_loader import load_config, load_and_split_data
from main import process_dataset
import xgboost as xgb

config = load_config('config/exp1.yaml')
train_dfs_dict, test_dfs_dict = load_and_split_data(config)
train_dfs = train_dfs_dict.get('mobile', [])
test_dfs = test_dfs_dict.get('mobile', [])

X_train, y_train = process_dataset(train_dfs, 15, 5)
X_test, y_test = process_dataset(test_dfs, 15, 5)

print('Train labels:', np.unique(y_train, return_counts=True))
print('Test labels:', np.unique(y_test, return_counts=True))

scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
print("scale_weight:", scale_weight)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, scale_pos_weight=scale_weight)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
print('Max prob:', probs.max())
print('Mean prob:', probs.mean())
print('Min prob:', probs.min())

from sklearn.metrics import confusion_matrix
print("CM threshold 0.5:\n", confusion_matrix(y_test, (probs > 0.5).astype(int)))
print("CM threshold 0.05:\n", confusion_matrix(y_test, (probs > 0.05).astype(int)))
print("CM threshold 0.01:\n", confusion_matrix(y_test, (probs > 0.01).astype(int)))
