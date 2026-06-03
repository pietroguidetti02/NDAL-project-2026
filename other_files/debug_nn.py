import pandas as pd
import numpy as np
from src.data_loader import load_config, load_and_split_data
from main import process_dataset
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.models import train_nn, evaluate_model

config = load_config('config/exp1.yaml')
train_dfs_dict, test_dfs_dict = load_and_split_data(config)
train_dfs = train_dfs_dict.get('mobile', [])
test_dfs = test_dfs_dict.get('mobile', [])

X_train, y_train = process_dataset(train_dfs, 15, 5)
X_test, y_test = process_dataset(test_dfs, 15, 5)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

smote = SMOTE(random_state=42)
X_train_scaled_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print('Train labels resampled:', np.unique(y_train_resampled, return_counts=True))

nn_model = train_nn(X_train_scaled_resampled, y_train_resampled, params={'max_iter': 500, 'random_state': 42})
probs = nn_model.predict_proba(X_test_scaled)[:, 1]

print('Max prob:', probs.max())
print('Mean prob:', probs.mean())
print('Min prob:', probs.min())

from sklearn.metrics import confusion_matrix
print("CM threshold 0.5:\n", confusion_matrix(y_test, (probs > 0.5).astype(int)))
print("CM threshold 0.2:\n", confusion_matrix(y_test, (probs > 0.2).astype(int)))
print("CM threshold 0.1:\n", confusion_matrix(y_test, (probs > 0.1).astype(int)))
print("CM threshold 0.05:\n", confusion_matrix(y_test, (probs > 0.05).astype(int)))
print("CM threshold 0.01:\n", confusion_matrix(y_test, (probs > 0.01).astype(int)))
