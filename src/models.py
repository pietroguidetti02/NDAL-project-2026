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
            preds = model.predict(X_test)
    else:
        preds = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'cm': confusion_matrix(y_test, preds, labels=[0, 1])
    }
    return metrics
