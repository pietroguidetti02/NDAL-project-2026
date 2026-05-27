import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay
import json

def plot_feature_importance(model, feature_names=None, output_dir=None):
    """
    Plots feature importance for a given model (e.g. XGBoost).
    """
    if isinstance(model, xgb.XGBClassifier):
        xgb.plot_importance(model)
        plt.title('Feature Importance')
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.show()
    else:
        print('Feature importance plotting not implemented for this model type.')

def plot_metrics(metrics_dict, model_name='Model', output_dir=None):
    """
    Helper to visualize evaluation metrics and confusion matrix.
    """
    print(f'\n--- Validation Metrics for {model_name} ---')
    print(f"Accuracy:  {metrics_dict.get('accuracy', 0):.4f}")
    print(f"Precision: {metrics_dict.get('precision', 0):.4f}")
    print(f"Recall:    {metrics_dict.get('recall', 0):.4f}")
    print(f"F1 Score:  {metrics_dict.get('f1', 0):.4f}")
    
    if output_dir:
        # Save metrics text
        metrics_to_save = {k: v for k, v in metrics_dict.items() if k != 'cm'}
        with open(os.path.join(output_dir, f'{model_name}_metrics.json'), 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
            
    if 'cm' in metrics_dict:
        disp = ConfusionMatrixDisplay(confusion_matrix=metrics_dict['cm'], display_labels=['No Loss', 'Loss'])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.grid(False) # Turn off grid for CM to look clean
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
        plt.show()
