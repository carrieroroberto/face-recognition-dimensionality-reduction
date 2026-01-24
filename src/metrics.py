import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

def compute_classification_metrics(y_true, y_pred, y_score=None, target_names=None):
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    if y_score is not None:
        try:
            n_classes = len(np.unique(y_true))
            y_true_bin = label_binarize(y_true, classes=range(n_classes))

            if y_score.shape[1] == n_classes:
                metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovr')
                metrics['roc_auc_weighted'] = roc_auc_score(y_true_bin, y_score, average='weighted', multi_class='ovr')
        except Exception as e:
            metrics['roc_auc_error'] = str(e)

    if target_names is not None:
        report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        metrics['classification_report'] = report_dict

    return metrics

def calculate_confidence_intervals(y_true, y_pred, n_bootstrap=1000, confidence_level=0.95):
    n_samples = len(y_true)
    bootstrap_accuracies = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        if len(indices) > 0:
            acc = accuracy_score(y_true[indices], y_pred[indices])
            bootstrap_accuracies.append(acc)

    if not bootstrap_accuracies:
        return {'accuracy': 0.0, 'lower_bound': 0.0, 'upper_bound': 0.0, 'std': 0.0}

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

def create_metrics_dataframe(metrics_dict, model_name, feature_type):
    row = {
        'model': model_name,
        'features': feature_type,
        'accuracy': metrics_dict['accuracy'],
        'precision_macro': metrics_dict['precision_macro'],
        'recall_macro': metrics_dict['recall_macro'],
        'f1_macro': metrics_dict['f1_macro'],
        'f1_weighted': metrics_dict['f1_weighted']
    }

    if 'roc_auc_macro' in metrics_dict:
        row['roc_auc_macro'] = metrics_dict['roc_auc_macro']

    if 'cv_score' in metrics_dict:
        row['cv_score'] = metrics_dict['cv_score']
    elif 'val_score' in metrics_dict:
        row['cv_score'] = metrics_dict['val_score']
    else:
        row['cv_score'] = np.nan

    if 'confidence_interval' in metrics_dict:
        ci = metrics_dict['confidence_interval']
        row['acc_lower'] = ci['lower_bound']
        row['acc_upper'] = ci['upper_bound']

    return pd.DataFrame([row])

def compare_models_metrics(metrics_list):
    dfs = []
    for model_name, feature_type, metrics_dict in metrics_list:
        df = create_metrics_dataframe(metrics_dict, model_name, feature_type)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df_comparison = pd.concat(dfs, ignore_index=True)
    df_comparison = df_comparison.sort_values('accuracy', ascending=False)

    return df_comparison

def compute_roc_curves(y_true, y_score, n_classes):
    from sklearn.preprocessing import label_binarize
    
    y_bin = label_binarize(y_true, classes=range(n_classes))
    
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack((1 - y_bin, y_bin))

    roc_data = {}

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data[i] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    roc_data['micro'] = {'fpr': fpr_micro, 'tpr': tpr_micro, 'auc': roc_auc_micro}

    all_fpr = np.unique(np.concatenate([roc_data[i]['fpr'] for i in range(n_classes)]))
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, roc_data[i]['fpr'], roc_data[i]['tpr'])
    
    mean_tpr /= n_classes
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    roc_data['macro'] = {'fpr': all_fpr, 'tpr': mean_tpr, 'auc': auc(all_fpr, mean_tpr)}

    return roc_data