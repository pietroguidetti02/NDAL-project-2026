import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

def train_xgboost(X_train, y_train, params=None):
    """
    Trains an XGBoost model.
    """
    model = xgb.XGBClassifier(**(params or {}))
    model.fit(X_train, y_train)
    return model

def train_nn(X_train, y_train, params=None):
    """
    Trains a Neural Network model (MLP).
    """
    model = MLPClassifier(**(params or {}))
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluates the given model and returns metrics.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        if probs.shape[1] > 1:
            preds = (probs[:, 1] >= threshold).astype(int)
        else:
            preds = (probs >= threshold).astype(int)
    else:
        preds_raw = model.predict(X_test)
        # If it's a Keras model, it returns probabilities of shape (samples, 1)
        if len(preds_raw.shape) == 2 and preds_raw.shape[1] == 1:
            preds = (preds_raw[:, 0] >= threshold).astype(int)
        else:
            preds = preds_raw
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'cm': confusion_matrix(y_test, preds, labels=[0, 1])
    }
    return metrics

def train_lstm(X_train_seq, y_train, params=None):
    """
    Trains an LSTM model on sequential raw data.
    X_train_seq shape: (samples, time_steps, features)
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    
    epochs = params.get('epochs', 15) if params else 15
    batch_size = params.get('batch_size', 128) if params else 128
    
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Calculate class weights
    import numpy as np
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    if num_pos > 0:
        weight_0 = 1.0
        weight_1 = num_neg / num_pos
        class_weight = {0: weight_0, 1: weight_1}
    else:
        class_weight = None
    
    # Train
    model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[es], class_weight=class_weight, verbose=1)
    
    return model
