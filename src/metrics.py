# src/metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import json
import os


def compute_classification_metrics(y_true, y_pred, y_score=None, target_names=None):
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics['precision_per_class'] = precision_per_class.tolist()
    metrics['recall_per_class'] = recall_per_class.tolist()
    metrics['f1_per_class'] = f1_per_class.tolist()

    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    if y_score is not None:
        try:
            n_classes = len(np.unique(y_true))
            y_true_bin = label_binarize(y_true, classes=range(n_classes))

            if y_score.shape[1] == n_classes:
                metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, y_score, average='macro')
                metrics['roc_auc_weighted'] = roc_auc_score(y_true_bin, y_score, average='weighted')

                roc_auc_per_class = []
                for i in range(n_classes):
                    try:
                        auc_i = roc_auc_score(y_true_bin[:, i], y_score[:, i])
                        roc_auc_per_class.append(auc_i)
                    except:
                        roc_auc_per_class.append(np.nan)
                metrics['roc_auc_per_class'] = roc_auc_per_class
        except Exception as e:
            metrics['roc_auc_error'] = str(e)

    if target_names is not None:
        report_dict = classification_report(y_true, y_pred, target_names=target_names,
                                           output_dict=True, zero_division=0)
        metrics['classification_report'] = report_dict

    return metrics

def compute_roc_curves(y_true, y_score, n_classes):
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    roc_data = {}
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        roc_data[f'class_{i}'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc)
        }

    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    roc_data['micro_average'] = {
        'fpr': fpr_micro.tolist(),
        'tpr': tpr_micro.tolist(),
        'auc': float(roc_auc_micro)
    }

    all_fpr = np.unique(np.concatenate([roc_data[f'class_{i}']['fpr'] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, roc_data[f'class_{i}']['fpr'], roc_data[f'class_{i}']['tpr'])
    mean_tpr /= n_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)
    roc_data['macro_average'] = {
        'fpr': all_fpr.tolist(),
        'tpr': mean_tpr.tolist(),
        'auc': float(roc_auc_macro)
    }

    return roc_data

def create_metrics_dataframe(metrics_dict, model_name, feature_type):
    row = {
        'model': model_name,
        'features': feature_type,
        'accuracy': metrics_dict['accuracy'],
        'precision_macro': metrics_dict['precision_macro'],
        'recall_macro': metrics_dict['recall_macro'],
        'f1_macro': metrics_dict['f1_macro'],
        'precision_weighted': metrics_dict['precision_weighted'],
        'recall_weighted': metrics_dict['recall_weighted'],
        'f1_weighted': metrics_dict['f1_weighted']
    }

    if 'roc_auc_macro' in metrics_dict:
        row['roc_auc_macro'] = metrics_dict['roc_auc_macro']
        row['roc_auc_weighted'] = metrics_dict['roc_auc_weighted']

    return pd.DataFrame([row])

def save_metrics_to_json(metrics_dict, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    metrics_clean = convert_numpy(metrics_dict)

    with open(filepath, 'w') as f:
        json.dump(metrics_clean, f, indent=2)

    print(f"Metriche salvate in: {filepath}")

def print_metrics_summary(metrics_dict, model_name="Model"):
    print(f"METRICHE DI CLASSIFICAZIONE - {model_name}")
    print(f"Accuracy:           {metrics_dict['accuracy']:.4f}")
    print(f"\nMacro-Averaged:")
    print(f"  Precision:        {metrics_dict['precision_macro']:.4f}")
    print(f"  Recall:           {metrics_dict['recall_macro']:.4f}")
    print(f"  F1-Score:         {metrics_dict['f1_macro']:.4f}")
    print(f"\nWeighted-Averaged:")
    print(f"  Precision:        {metrics_dict['precision_weighted']:.4f}")
    print(f"  Recall:           {metrics_dict['recall_weighted']:.4f}")
    print(f"  F1-Score:         {metrics_dict['f1_weighted']:.4f}")

    if 'roc_auc_macro' in metrics_dict:
        print(f"\nROC AUC:")
        print(f"  Macro:            {metrics_dict['roc_auc_macro']:.4f}")
        print(f"  Weighted:         {metrics_dict['roc_auc_weighted']:.4f}")

def compare_models_metrics(metrics_list, save_path=None):
    dfs = []
    for model_name, feature_type, metrics_dict in metrics_list:
        df = create_metrics_dataframe(metrics_dict, model_name, feature_type)
        dfs.append(df)

    df_comparison = pd.concat(dfs, ignore_index=True)

    df_comparison = df_comparison.sort_values('accuracy', ascending=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_comparison.to_csv(save_path, index=False, float_format='%.4f')
        print(f"Tabella comparativa salvata in: {save_path}")

    return df_comparison

def calculate_confidence_intervals(y_true, y_pred, n_bootstrap=1000, confidence_level=0.95):
    n_samples = len(y_true)
    bootstrap_accuracies = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        acc = accuracy_score(y_true[indices], y_pred[indices])
        bootstrap_accuracies.append(acc)

    bootstrap_accuracies = np.array(bootstrap_accuracies)
    mean_acc = np.mean(bootstrap_accuracies)

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_accuracies, lower_percentile)
    upper_bound = np.percentile(bootstrap_accuracies, upper_percentile)

    return {
        'accuracy': float(mean_acc),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'confidence_level': confidence_level,
        'std': float(np.std(bootstrap_accuracies))
    }