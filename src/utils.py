import os
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
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
    if 'cm' in metrics_dict:
        print(f"Confusion Matrix:\n{metrics_dict['cm']}")

    
    if output_dir:
        # Save metrics text
        metrics_to_save = {}
        for k, v in metrics_dict.items():
            if k not in ['cm', 'y_true', 'y_pred', 'y_prob']:
                if isinstance(v, (np.floating, float)):
                    metrics_to_save[k] = float(v)
                elif isinstance(v, (np.integer, int)):
                    metrics_to_save[k] = int(v)
                else:
                    metrics_to_save[k] = v
                    
        with open(os.path.join(output_dir, f'{model_name}_metrics.json'), 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
            
    if 'cm' in metrics_dict:
        disp = ConfusionMatrixDisplay(confusion_matrix=metrics_dict['cm'], display_labels=['No Loss', 'Loss'])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.grid(False) # Turn off grid for CM to look clean
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        
        # Also plot and save individual ROC and PR curves
        if 'y_prob' in metrics_dict and 'y_true' in metrics_dict:
            fpr, tpr, _ = roc_curve(metrics_dict['y_true'], metrics_dict['y_prob'])
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(metrics_dict['y_true'], metrics_dict['y_prob'])
            pr_auc = average_precision_score(metrics_dict['y_true'], metrics_dict['y_prob'])
            
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f'{model_name} - ROC Curve')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, lw=2, label=f'PR AUC = {pr_auc:.3f}')
            plt.title(f'{model_name} - PR Curve')
            plt.legend()
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'{model_name}_roc_pr.png'))
            plt.show(block=False)
            plt.pause(1)
            plt.close()

def plot_model_comparison(metrics1, metrics2, model1_name='XGBoost', model2_name='NN', output_dir=None, prefix=''):
    """
    Plots a graphical comparison between two models based on their metrics.
    """
    def extract_per_class(cm):
        if cm is None:
            return [0,0], [0,0], [0,0]
        TN, FP = cm[0,0], cm[0,1]
        FN, TP = cm[1,0], cm[1,1]
        
        p0 = TN / (TN + FN) if (TN + FN) > 0 else 0
        p1 = TP / (TP + FP) if (TP + FP) > 0 else 0
        
        r0 = TN / (TN + FP) if (TN + FP) > 0 else 0
        r1 = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        f0 = 2 * (p0 * r0) / (p0 + r0) if (p0 + r0) > 0 else 0
        f1_class = 2 * (p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0
        
        return [p0, p1], [r0, r1], [f0, f1_class]
        
    p_m1, r_m1, f_m1 = extract_per_class(metrics1.get('cm'))
    p_m2, r_m2, f_m2 = extract_per_class(metrics2.get('cm'))
    
    # Global metrics
    m1_global = [metrics1.get('accuracy',0), metrics1.get('precision',0), metrics1.get('recall',0), metrics1.get('f1',0)]
    m2_global = [metrics2.get('accuracy',0), metrics2.get('precision',0), metrics2.get('recall',0), metrics2.get('f1',0)]
    
    label_names = ['No Loss', 'Loss']
    
    #---------------plots---------------#
    xM1 = np.arange(4)-0.1 
    xM2 = np.arange(4)+0.1 
    x2M1 = np.arange(2)-0.1 
    x2M2 = np.arange(2)+0.1 
    w = 0.2       
    
    # 1) global metrics
    plt.figure()
    plt.bar(xM1, m1_global, width=w, edgecolor='black', color='c', align='center', hatch='///', label=model1_name)
    plt.bar(xM2, m2_global, width=w, edgecolor='black', color='y', align='center', hatch='---', label=model2_name)
    plt.xticks(np.arange(4), ["Accuracy", "Precision", "Recall", "F1-Score"])
    plt.title(f'{model1_name} & {model2_name} performance metrics')
    plt.ylabel('Score')
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.legend()
    if output_dir: plt.savefig(os.path.join(output_dir, f'{prefix}_comparison_global_{model1_name}_vs_{model2_name}.png'))
    plt.show()

    # 2) precision
    plt.figure()
    plt.bar(x2M1, p_m1, width=w, edgecolor='black', color='c', align='center', hatch='///', label=model1_name)
    plt.bar(x2M2, p_m2, width=w, edgecolor='black', color='y', align='center', hatch='---', label=model2_name)
    plt.plot(x2M1, [metrics1.get('precision',0)]*2, color='c', linestyle='dashed')
    plt.plot(x2M1, [metrics2.get('precision',0)]*2, color='y', linestyle='dashed')
    plt.xticks(np.arange(2), label_names)
    plt.title(f'{prefix.capitalize()} - {model1_name} & {model2_name} precision')
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.grid()
    if output_dir: plt.savefig(os.path.join(output_dir, f'{prefix}_comparison_precision_{model1_name}_vs_{model2_name}.png'))
    plt.show()

    # 3) recall
    plt.figure()
    plt.bar(x2M1, r_m1, width=w, edgecolor='black', color='c', align='center', hatch='///', label=model1_name)
    plt.bar(x2M2, r_m2, width=w, edgecolor='black', color='y', align='center', hatch='---', label=model2_name)
    plt.plot(x2M1, [metrics1.get('recall',0)]*2, color='c', linestyle='dashed')
    plt.plot(x2M1, [metrics2.get('recall',0)]*2, color='y', linestyle='dashed')
    plt.xticks(np.arange(2), label_names)
    plt.title(f'{model1_name} & {model2_name} recall (global and per-class)')
    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.grid()
    if output_dir: plt.savefig(os.path.join(output_dir, f'{prefix}_comparison_recall_{model1_name}_vs_{model2_name}.png'))
    plt.show()

    # 4) f1-score
    plt.figure()
    plt.bar(x2M1, f_m1, width=w, edgecolor='black', color='c', align='center', hatch='///', label=model1_name)
    plt.bar(x2M2, f_m2, width=w, edgecolor='black', color='y', align='center', hatch='---', label=model2_name)
    plt.plot(x2M1, [metrics1.get('f1',0)]*2, color='c', linestyle='dashed')
    plt.plot(x2M1, [metrics2.get('f1',0)]*2, color='y', linestyle='dashed')
    plt.xticks(np.arange(2), label_names)
    plt.title(f'{prefix.capitalize()} - {model1_name} & {model2_name} F1-score')
    plt.xlabel('Class')
    plt.ylabel('F1-score')
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.grid()
    if output_dir: plt.savefig(os.path.join(output_dir, f'{prefix}_comparison_f1_{model1_name}_vs_{model2_name}.png'))
    plt.show()

def plot_model_comparison_3(metrics1, metrics2, metrics3, m1_name='XGBoost', m2_name='NN', m3_name='LSTM', output_dir=None, prefix=''):
    """
    Plots a graphical comparison between 3 models based on their metrics.
    """
    def extract_per_class(cm):
        if cm is None:
            return [0,0], [0,0], [0,0]
        TN, FP = cm[0,0], cm[0,1]
        FN, TP = cm[1,0], cm[1,1]
        p0 = TN / (TN + FN) if (TN + FN) > 0 else 0
        p1 = TP / (TP + FP) if (TP + FP) > 0 else 0
        r0 = TN / (TN + FP) if (TN + FP) > 0 else 0
        r1 = TP / (TP + FN) if (TP + FN) > 0 else 0
        f0 = 2 * (p0 * r0) / (p0 + r0) if (p0 + r0) > 0 else 0
        f1_class = 2 * (p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0
        return [p0, p1], [r0, r1], [f0, f1_class]
        
    p_m1, r_m1, f_m1 = extract_per_class(metrics1.get('cm'))
    p_m2, r_m2, f_m2 = extract_per_class(metrics2.get('cm'))
    p_m3, r_m3, f_m3 = extract_per_class(metrics3.get('cm'))
    
    m1_global = [metrics1.get('accuracy',0), metrics1.get('precision',0), metrics1.get('recall',0), metrics1.get('f1',0)]
    m2_global = [metrics2.get('accuracy',0), metrics2.get('precision',0), metrics2.get('recall',0), metrics2.get('f1',0)]
    m3_global = [metrics3.get('accuracy',0), metrics3.get('precision',0), metrics3.get('recall',0), metrics3.get('f1',0)]
    
    label_names = ['No Loss', 'Loss']
    
    x = np.arange(4)
    x2 = np.arange(2)
    w = 0.25       
    
    # 1) global metrics
    plt.figure(figsize=(10, 6))
    plt.bar(x - w, m1_global, width=w, edgecolor='black', color='c', align='center', hatch='///', label=m1_name)
    plt.bar(x, m2_global, width=w, edgecolor='black', color='y', align='center', hatch='---', label=m2_name)
    plt.bar(x + w, m3_global, width=w, edgecolor='black', color='m', align='center', hatch='\\\\\\', label=m3_name)
    plt.xticks(x, ["Accuracy", "Precision", "Recall", "F1-Score"])
    plt.title(f'{prefix.capitalize()} - Model Comparison (Global)')
    plt.ylabel('Score')
    plt.ylim([0.0, 1.05])
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    if output_dir: plt.savefig(os.path.join(output_dir, f'{prefix}_comparison_global_3models.png'))
    plt.show()

    # 2) precision per class
    plt.figure(figsize=(8, 6))
    plt.bar(x2 - w, p_m1, width=w, edgecolor='black', color='c', align='center', hatch='///', label=m1_name)
    plt.bar(x2, p_m2, width=w, edgecolor='black', color='y', align='center', hatch='---', label=m2_name)
    plt.bar(x2 + w, p_m3, width=w, edgecolor='black', color='m', align='center', hatch='\\\\\\', label=m3_name)
    plt.plot(x2, [metrics1.get('precision',0)]*2, color='c', linestyle='dashed')
    plt.plot(x2, [metrics2.get('precision',0)]*2, color='y', linestyle='dashed')
    plt.plot(x2, [metrics3.get('precision',0)]*2, color='m', linestyle='dashed')
    plt.xticks(x2, label_names)
    plt.title(f'{prefix.capitalize()} - Precision per class')
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    if output_dir: plt.savefig(os.path.join(output_dir, f'{prefix}_comparison_precision_3models.png'))
    plt.show()

    # 3) recall per class
    plt.figure(figsize=(8, 6))
    plt.bar(x2 - w, r_m1, width=w, edgecolor='black', color='c', align='center', hatch='///', label=m1_name)
    plt.bar(x2, r_m2, width=w, edgecolor='black', color='y', align='center', hatch='---', label=m2_name)
    plt.bar(x2 + w, r_m3, width=w, edgecolor='black', color='m', align='center', hatch='\\\\\\', label=m3_name)
    plt.plot(x2, [metrics1.get('recall',0)]*2, color='c', linestyle='dashed')
    plt.plot(x2, [metrics2.get('recall',0)]*2, color='y', linestyle='dashed')
    plt.plot(x2, [metrics3.get('recall',0)]*2, color='m', linestyle='dashed')
    plt.xticks(x2, label_names)
    plt.title(f'{prefix.capitalize()} - Recall per class')
    plt.xlabel('Class')
    plt.ylabel('Recall')
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    if output_dir: plt.savefig(os.path.join(output_dir, f'{prefix}_comparison_recall_3models.png'))
    plt.show()

    # 4) f1-score per class
    plt.figure(figsize=(8, 6))
    plt.bar(x2 - w, f_m1, width=w, edgecolor='black', color='c', align='center', hatch='///', label=m1_name)
    plt.bar(x2, f_m2, width=w, edgecolor='black', color='y', align='center', hatch='---', label=m2_name)
    plt.bar(x2 + w, f_m3, width=w, edgecolor='black', color='m', align='center', hatch='\\\\\\', label=m3_name)
    plt.xticks(x2, label_names)
    plt.title(f'{prefix.capitalize()} - F1-score per class')
    plt.xlabel('Class')
    plt.ylabel('F1-score')
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    if output_dir: plt.savefig(os.path.join(output_dir, f'{prefix}_comparison_f1_3models.png'))
    plt.show()

def plot_roc_pr_curves_2(metrics1, metrics2, m1_name='XGBoost', m2_name='NN', output_dir=None, prefix=''):
    """
    Plots ROC and PR Curves for two models.
    """
    plt.figure(figsize=(14, 6))
    
    # Plot ROC
    plt.subplot(1, 2, 1)
    for m, name, color in zip([metrics1, metrics2], [m1_name, m2_name], ['c', 'y']):
        if m is not None and 'y_prob' in m and 'y_true' in m:
            fpr, tpr, _ = roc_curve(m['y_true'], m['y_prob'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{prefix.capitalize()} - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Plot PR
    plt.subplot(1, 2, 2)
    for m, name, color in zip([metrics1, metrics2], [m1_name, m2_name], ['c', 'y']):
        if m is not None and 'y_prob' in m and 'y_true' in m:
            precision, recall, _ = precision_recall_curve(m['y_true'], m['y_prob'])
            pr_auc = average_precision_score(m['y_true'], m['y_prob'])
            plt.plot(recall, precision, color=color, lw=2, label=f'{name} (PR AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{prefix.capitalize()} - Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'{prefix}_roc_pr_curves.png'))
    plt.show()

def plot_roc_pr_curves_3(metrics1, metrics2, metrics3, m1_name='XGBoost', m2_name='NN', m3_name='LSTM', output_dir=None, prefix=''):
    """
    Plots ROC and PR Curves for three models, highlighting the chosen optimal point.
    """
    plt.figure(figsize=(16, 7))
    
    # colors and names
    all_metrics = [metrics1, metrics2, metrics3]
    names = [m1_name, m2_name, m3_name]
    colors = ['c', 'y', 'm']

    # 1. ROC CURVE
    plt.subplot(1, 2, 1)
    for m, name, color in zip(all_metrics, names, colors):
        if m is not None and 'y_prob' in m and 'y_true' in m:
            fpr, tpr, thresholds = roc_curve(m['y_true'], m['y_prob'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
            
            # Mark the point corresponding to the optimal_threshold
            opt_t = m.get('optimal_threshold', 0.5)
            # Find closest threshold index
            idx = np.argmin(np.abs(thresholds - opt_t))
            plt.plot(fpr[idx], tpr[idx], 'o', color=color, markersize=8, markeredgecolor='black')
            
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{prefix.capitalize()} - ROC Curve (dots = chosen threshold)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 2. PRECISION-RECALL CURVE
    plt.subplot(1, 2, 2)
    for m, name, color in zip(all_metrics, names, colors):
        if m is not None and 'y_prob' in m and 'y_true' in m:
            precision, recall, thresholds = precision_recall_curve(m['y_true'], m['y_prob'])
            pr_auc = average_precision_score(m['y_true'], m['y_prob'])
            plt.plot(recall, precision, color=color, lw=2, label=f'{name} (F1 Max = {m.get("f1",0):.3f})')
            
            # The chosen metrics already correspond to the max F1 point we calculated
            # Let's find it on the curve
            plt.plot(m.get('recall', 0), m.get('precision', 0), 'o', color=color, markersize=8, markeredgecolor='black')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{prefix.capitalize()} - PR Curve (dots = chosen threshold)')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'{prefix}_roc_pr_curves_3models.png'))
    plt.show()
