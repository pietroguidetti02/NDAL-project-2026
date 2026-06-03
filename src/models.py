import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

def optimize_threshold_cv(model, X_train, y_train, cv=3):
    """
    Finds the optimal F1 threshold using out-of-fold predictions.
    """
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    print("      [CV] Computing out-of-fold predictions for threshold optimization...")
    try:
        probs = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')
        y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    except Exception as e:
        preds = cross_val_predict(model, X_train, y_train, cv=cv)
        y_prob = preds[:, 0] if (len(preds.shape) == 2 and preds.shape[1] == 1) else preds

    y_prob = np.nan_to_num(y_prob, nan=0.0)

    if np.sum(y_train) > 0:
        precision, recall, thresholds = precision_recall_curve(y_train, y_prob)
        num = 2 * (precision * recall)
        den = (precision + recall)
        f1_scores = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
        opt_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[opt_idx] if opt_idx < len(thresholds) else thresholds[-1]
        print(f"      [CV] Found optimal threshold: {optimal_threshold:.4f}")
        return float(optimal_threshold)
    return 0.5

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
    It calculates the optimal F1 threshold from the Precision-Recall curve
    and uses that threshold to compute the final confusion matrix and metrics.
    """
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        if probs.shape[1] > 1:
            y_prob = probs[:, 1]
        else:
            y_prob = probs[:, 0]
    else:
        preds_raw = model.predict(X_test)
        if len(preds_raw.shape) == 2 and preds_raw.shape[1] == 1:
            y_prob = preds_raw[:, 0]
        else:
            y_prob = preds_raw
            
    # Sanitize y_prob against NaNs (though we fixed the source, robustness is good)
    y_prob = np.nan_to_num(y_prob, nan=0.0)
            
    optimal_threshold = threshold
    print(f"Using Threshold: {optimal_threshold:.4f}")
    preds = (y_prob >= optimal_threshold).astype(int)

    metrics = {
        'optimal_threshold': float(optimal_threshold),
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'cm': confusion_matrix(y_test, preds, labels=[0, 1]),
        'y_true': np.array(y_test),
        'y_prob': np.array(y_prob)
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
    
    import numpy as np
    import tensorflow as tf
    
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    
    # MAGIC TRICK FOR IMBALANCED DL: Set initial bias so the model doesn't panic in epoch 1
    if num_pos > 0:
        initial_bias = np.log([num_pos / num_neg])
        output_bias = tf.keras.initializers.Constant(initial_bias)
    else:
        output_bias = 'zeros'
        
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid', bias_initializer=output_bias))
    
    # Use a slightly lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Calculate class weights safely
    if num_pos > 0:
        total = num_neg + num_pos
        weight_0 = (1 / num_neg) * (total / 2.0)
        weight_1 = (1 / num_pos) * (total / 2.0)
        # Cap weight_1 to avoid exploding gradients
        weight_1 = min(weight_1, 50.0) 
        class_weight = {0: weight_0, 1: weight_1}
    else:
        class_weight = None
    
    # Train
    verbose = params.get('verbose', 1) if params else 1
    model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[es], class_weight=class_weight, verbose=verbose)
    
    return model
