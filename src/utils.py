import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from itertools import cycle
import config

def plot_dataset_stats(data_dict):
    plt.style.use('seaborn-v0_8-whitegrid')
    target_names = data_dict['target_names']
    y_train = data_dict['y_train']
    h, w = data_dict['image_shape']
    X_train = data_dict['X_train']

    _, ax = plt.subplots(figsize=(12, 6))
    unique, counts = np.unique(y_train, return_counts=True)
    
    names_subset = [target_names[i] for i in unique]
    
    sns.barplot(x=names_subset, y=counts, palette='viridis', ax=ax)
    ax.set_title("Class Distribution (Train Set)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Images", fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    
    save_path = os.path.join(config.OUTPUT_PATH, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    _, ax = plt.subplots(figsize=(5, 5))
    mean_face = np.mean(X_train, axis=0).reshape(h, w)
    ax.imshow(mean_face, cmap='gray')
    ax.set_title("Mean Face (Average)", fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    save_path = os.path.join(config.OUTPUT_PATH, "mean_face.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_eigenfaces(pca_model, h, w, n_top=12):
    plt.style.use('seaborn-v0_8-whitegrid')
    n_top = min(n_top, len(pca_model.components_))
    rows = (n_top + 3) // 4
    
    _, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i in range(n_top):
        eigenface = pca_model.components_[i].reshape(h, w)
        axes[i].imshow(eigenface, cmap='gray')
        axes[i].set_title(f"PC {i+1}", fontsize=10)
        axes[i].axis('off')
    
    for i in range(n_top, len(axes)):
        axes[i].axis('off')

    plt.suptitle("Top Principal Components (Eigenfaces)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(config.OUTPUT_PATH, "eigenfaces.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_scree_plot(pca_model):
    plt.style.use('seaborn-v0_8-whitegrid')
    _, ax = plt.subplots(figsize=(8, 6))
    
    cum_variance = np.cumsum(pca_model.explained_variance_ratio_)
    
    ax.plot(range(1, len(cum_variance) + 1), cum_variance, marker='o', linestyle='--', markersize=4)
    ax.set_xlabel("Number of Components", fontsize=11)
    ax.set_ylabel("Cumulative Variance Explained", fontsize=11)
    ax.set_title("Variance Analysis (Scree Plot)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(config.OUTPUT_PATH, "scree_plot.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_loss(train_loss, val_loss=None, title="Training Loss"):
    plt.style.use('seaborn-v0_8-whitegrid')
    _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_loss, label='Train Loss', color='tab:blue', linewidth=2)
    
    if val_loss is not None and len(val_loss) > 0:
        ax.plot(val_loss, label='Validation Loss', color='tab:orange', linestyle='--', linewidth=2)
        
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (MSE)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = "ae_loss.png" if "AE" in title or "Autoencoder" in title else "loss_curve.png"
    save_path = os.path.join(config.OUTPUT_PATH, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_reconstruction_comparison(original, pca_rec, ae_rec, h, w, n_images=5):
    plt.style.use('seaborn-v0_8-whitegrid')
    n_images = min(n_images, len(original))
    _, axes = plt.subplots(n_images, 3, figsize=(10, 3 * n_images))

    for i in range(n_images):
        axes[i, 0].imshow(original[i].reshape(h, w), cmap='gray')
        if i == 0: axes[i, 0].set_title("Original", fontsize=11, fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(pca_rec[i].reshape(h, w), cmap='gray')
        if i == 0: axes[i, 1].set_title("PCA", fontsize=11, fontweight='bold')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(ae_rec[i].reshape(h, w), cmap='gray')
        if i == 0: axes[i, 2].set_title("Autoencoder", fontsize=11, fontweight='bold')
        axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_PATH, "reconstruction_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, target_names, model_name):
    plt.style.use('seaborn-v0_8-whitegrid')
    cm = confusion_matrix(y_true, y_pred)
    
    _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=11)
    ax.set_xlabel('Predicted Class', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    safe_name = model_name.lower().replace(' ', '_')
    save_path = os.path.join(config.OUTPUT_PATH, f"cm_{safe_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_tsne_comparison(pca_features, ae_features, labels, target_names):
    plt.style.use('seaborn-v0_8-whitegrid')
    _, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, (name, feat) in enumerate([("PCA", pca_features), ("Autoencoder", ae_features)]):
        tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
        X_2d = tsne.fit_transform(feat)

        for j, label_idx in enumerate(unique_labels):
            idx = np.where(labels == label_idx)
            color = colors[j % len(colors)]
            axes[i].scatter(X_2d[idx, 0], X_2d[idx, 1], 
                            label=target_names[label_idx], 
                            color=color, alpha=0.7, s=30)

        axes[i].set_title(f"t-SNE: {name} Features", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("t-SNE 1", fontsize=10)
        axes[i].set_ylabel("t-SNE 2", fontsize=10)
        axes[i].grid(True, alpha=0.3)

    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title="Classes")
    
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_PATH, "tsne_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_multiclass_roc(roc_data, n_classes, target_names, title="ROC Curves"):
    plt.style.use('seaborn-v0_8-whitegrid')
    _, ax = plt.subplots(figsize=(10, 8))

    ax.plot(roc_data['micro']['fpr'], roc_data['micro']['tpr'],
            label=f"Micro-average ROC (area = {roc_data['micro']['auc']:.2f})",
            color='deeppink', linestyle=':', linewidth=3)

    ax.plot(roc_data['macro']['fpr'], roc_data['macro']['tpr'],
            label=f"Macro-average ROC (area = {roc_data['macro']['auc']:.2f})",
            color='navy', linestyle=':', linewidth=3)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    
    classes_to_plot = range(n_classes) if n_classes <= 10 else range(5)
    
    for i, color in zip(classes_to_plot, colors):
        ax.plot(roc_data[i]['fpr'], roc_data[i]['tpr'], color=color, lw=1.5, alpha=0.7,
                label=f"ROC {target_names[i]} (area = {roc_data[i]['auc']:.2f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc="lower right", fontsize=8)
    
    filename = f"{title.lower().replace(' ', '_')}.png"
    save_path = os.path.join(config.OUTPUT_PATH, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
def plot_similarity_distribution(scores_genuine, scores_impostor, method_name, threshold=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    _, ax = plt.subplots(figsize=(10, 8))

    ax.hist(scores_genuine, bins=40, alpha=0.6, label='Genuine (Same Person)',
            color='#27ae60', edgecolor='black', density=True)
    ax.hist(scores_impostor, bins=40, alpha=0.6, label='Impostor (Different Person)',
            color='#e74c3c', edgecolor='black', density=True)

    if threshold is not None:
        ax.axvline(threshold, color='blue', linestyle='--', linewidth=2,
                   label=f'Threshold: {threshold:.3f}')

    ax.set_xlabel('Cosine Similarity Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Similarity Distribution: {method_name}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    stats_text = f'Genuine: mean={scores_genuine.mean():.3f}, std={scores_genuine.std():.3f}\n'
    stats_text += f'Impostor: mean={scores_impostor.mean():.3f}, std={scores_impostor.std():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_PATH, f"similarity_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_verification_examples(images, pair_idx1, pair_idx2, scores, labels,
                               h, w, n_examples=5):
    plt.style.use('seaborn-v0_8-whitegrid')

    genuine_mask = labels == 1
    impostor_mask = labels == 0

    genuine_indices = np.where(genuine_mask)[0]
    genuine_scores = scores[genuine_mask]
    top_genuine_idx = genuine_indices[np.argsort(genuine_scores)[-n_examples:]]

    impostor_indices = np.where(impostor_mask)[0]
    impostor_scores = scores[impostor_mask]
    top_impostor_idx = impostor_indices[np.argsort(impostor_scores)[-n_examples:]]

    _, axes = plt.subplots(2, n_examples, figsize=(14, 6))

    for col, idx in enumerate(top_genuine_idx):
        i, j = pair_idx1[idx], pair_idx2[idx]
        score = scores[idx]
        img1 = images[i].reshape(h, w)
        img2 = images[j].reshape(h, w)
        combined = np.hstack([img1, img2])
        
        axes[0, col].imshow(combined, cmap='gray')
        axes[0, col].set_title(f'Genuine\nScore: {score:.3f}', fontsize=9, fontweight='bold', color='green')
        axes[0, col].axis('off')

    for col, idx in enumerate(top_impostor_idx):
        i, j = pair_idx1[idx], pair_idx2[idx]
        score = scores[idx]
        img1 = images[i].reshape(h, w)
        img2 = images[j].reshape(h, w)
        combined = np.hstack([img1, img2])
        
        axes[1, col].imshow(combined, cmap='gray')
        axes[1, col].set_title(f'Impostor\nScore: {score:.3f}', fontsize=9, fontweight='bold', color='red')
        axes[1, col].axis('off')

    plt.suptitle('Face Verification: Genuine vs Impostor Pairs', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_PATH, f"verification_examples.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_similarity_heatmap(features, labels, target_names, method_name,
                            max_classes=8):
    plt.style.use('seaborn-v0_8-whitegrid')

    unique_labels = np.unique(labels)[:max_classes]
    class_centers = []
    class_names = []

    for label in unique_labels:
        class_features = features[labels == label]
        class_centers.append(class_features.mean(axis=0))
        class_names.append(target_names[label])

    similarity_matrix = cosine_similarity(np.array(class_centers))

    _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Cosine Similarity'}, ax=ax,
                vmin=0, vmax=1, linewidths=0.5)

    ax.set_title(f'Inter-Person Similarity: {method_name}', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(config.OUTPUT_PATH, f"similarity_heatmap_{method_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
def plot_ablation_study(x_values, accuracy_scores, secondary_metric_scores, 
                        x_label, secondary_label, title):
    plt.style.use('seaborn-v0_8-whitegrid')
    _, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel(x_label)
    ax1.set_ylabel('SVM Test Accuracy', color='tab:blue')
    ax1.plot(x_values, accuracy_scores, marker='o', color='tab:blue', 
             linewidth=2, markersize=8, label='SVM Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel(secondary_label, color='tab:orange')
    ax2.plot(x_values, secondary_metric_scores, marker='s', color='tab:orange', 
             linewidth=2, markersize=8, linestyle='--', label=secondary_label)
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_PATH, f"{"pca" if "PCA" in title else "ae"}_ablation_study.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
def compute_dataset_statistics(X, y, target_names):
    unique, counts = np.unique(y, return_counts=True)

    stats = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(target_names),
        'pixel_min': X.min(),
        'pixel_max': X.max(),
        'pixel_mean': X.mean(),
        'pixel_std': X.std(),
        'class_distribution': dict(zip([target_names[i] for i in unique], counts)),
        'class_counts': counts,
        'min_samples_per_class': counts.min(),
        'max_samples_per_class': counts.max(),
        'mean_samples_per_class': counts.mean(),
        'std_samples_per_class': counts.std()
    }

    return stats

def print_dataset_statistics(stats):
    print("\nDATASET STATISTICS")
    print(f"Total samples: {stats['n_samples']}")
    print(f"Number of features: {stats['n_features']}")
    print(f"Number of classes: {stats['n_classes']}")

    print(f"\nPixel statistics")
    print(f"Range: [{stats['pixel_min']:.4f}, {stats['pixel_max']:.4f}]")
    print(f"Mean: {stats['pixel_mean']:.4f}")
    print(f"Std: {stats['pixel_std']:.4f}")

    print(f"\nClass distribution")
    print(f"Min samples per class: {stats['min_samples_per_class']}")
    print(f"Max samples per class: {stats['max_samples_per_class']}")
    print(f"Mean samples per class: {stats['mean_samples_per_class']:.1f}")
    print(f"Std: {stats['std_samples_per_class']:.1f}")
    
def print_metrics_summary(metrics_dict, model_name="Model"):
    print(f"\n--- METRICS SUMMARY: {model_name} ---")
    
    if 'best_params' in metrics_dict:
        print(f"Best Params: {metrics_dict['best_params']}")
        
    print(f"Accuracy: {metrics_dict['accuracy']:.4f}")

    if 'confidence_interval' in metrics_dict:
        ci = metrics_dict['confidence_interval']
        print(f"95% Confidence Interval: [{ci['lower_bound']:.4f}, {ci['upper_bound']:.4f}]")

    print(f"Macro F1: {metrics_dict['f1_macro']:.4f}")
    print(f"Weighted F1: {metrics_dict['f1_weighted']:.4f}")

    if 'cv_score' in metrics_dict:
        print(f"CV Score (Val): {metrics_dict['cv_score']:.4f}")
    
    if 'roc_auc_macro' in metrics_dict:
        print(f"ROC AUC (Macro): {metrics_dict['roc_auc_macro']:.4f}")

def export_metrics_to_csv(df_metrics):
    save_path = os.path.join(config.MODELS_PATH, "classification_metrics.csv")
    df_metrics.to_csv(save_path, index=False)
    
def save_model(model, name):
    if isinstance(model, torch.nn.Module):
        filename = f"{name}.pt"
        path = os.path.join(config.MODELS_PATH, filename)
        torch.save(model.state_dict(), path)
    else:
        filename = f"{name}.joblib"
        path = os.path.join(config.MODELS_PATH, filename)
        joblib.dump(model, path)