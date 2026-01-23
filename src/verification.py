# src/verification.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import roc_curve, auc
import config

def generate_pairs(features, labels, n_pairs=1000):
    np.random.seed(config.RANDOM_STATE)
    n_samples = features.shape[0]
    pair_indices_1 = []
    pair_indices_2 = []
    pair_features_1 = []
    pair_features_2 = []
    pair_labels = []

    unique_labels = np.unique(labels)
    while len(pair_labels) < n_pairs // 2:
        label = np.random.choice(unique_labels)
        idx = np.where(labels == label)[0]
        if len(idx) >= 2:
            i, j = np.random.choice(idx, 2, replace=False)
            pair_indices_1.append(i)
            pair_indices_2.append(j)
            pair_features_1.append(features[i])
            pair_features_2.append(features[j])
            pair_labels.append(1)
            
    while len(pair_labels) < n_pairs:
        i, j = np.random.choice(n_samples, 2, replace=False)
        if labels[i] != labels[j]:
            pair_indices_1.append(i)
            pair_indices_2.append(j)
            pair_features_1.append(features[i])
            pair_features_2.append(features[j])
            pair_labels.append(0)

    return (np.array(pair_features_1), np.array(pair_features_2),
            np.array(pair_labels), np.array(pair_indices_1), np.array(pair_indices_2))


def calculate_eer(fpr, tpr, thresholds):
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[idx]
    threshold = thresholds[idx]
    return eer, threshold


def plot_similarity_distribution(scores_genuine, scores_impostor, method_name, metric_name,
                                  threshold=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = 50
    ax.hist(scores_genuine, bins=bins, alpha=0.6, label='Genuine (Same Person)',
            color='#27ae60', edgecolor='black', density=True)
    ax.hist(scores_impostor, bins=bins, alpha=0.6, label='Impostor (Different Person)',
            color='#e74c3c', edgecolor='black', density=True)

    if threshold is not None:
        ax.axvline(threshold, color='blue', linestyle='--', linewidth=2,
                   label=f'Threshold: {threshold:.3f}')

    ax.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(f'Similarity Distribution: {method_name} + {metric_name}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    stats_text = f'Genuine: μ={scores_genuine.mean():.3f}, σ={scores_genuine.std():.3f}\n'
    stats_text += f'Impostor: μ={scores_impostor.mean():.3f}, σ={scores_impostor.std():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"Similarity distribution saved: {save_path}")

    plt.close()

def plot_verification_examples(images, pair_idx1, pair_idx2, scores, labels,
                                h, w, n_examples=5, save_path=None):
    genuine_mask = labels == 1
    impostor_mask = labels == 0

    genuine_indices = np.where(genuine_mask)[0]
    genuine_scores = scores[genuine_mask]
    top_genuine_idx = genuine_indices[np.argsort(genuine_scores)[-n_examples:]]

    impostor_indices = np.where(impostor_mask)[0]
    impostor_scores = scores[impostor_mask]
    top_impostor_idx = impostor_indices[np.argsort(impostor_scores)[-n_examples:]]

    fig, axes = plt.subplots(2, n_examples, figsize=(15, 6))

    for col, idx in enumerate(top_genuine_idx):
        i, j = pair_idx1[idx], pair_idx2[idx]
        score = scores[idx]

        img1 = images[i].reshape(h, w)
        img2 = images[j].reshape(h, w)
        combined = np.hstack([img1, img2])

        axes[0, col].imshow(combined, cmap='gray')
        axes[0, col].set_title(f'Genuine\nScore: {score:.3f}', fontsize=10, fontweight='bold', color='green')
        axes[0, col].axis('off')

    for col, idx in enumerate(top_impostor_idx):
        i, j = pair_idx1[idx], pair_idx2[idx]
        score = scores[idx]

        img1 = images[i].reshape(h, w)
        img2 = images[j].reshape(h, w)
        combined = np.hstack([img1, img2])

        axes[1, col].imshow(combined, cmap='gray')
        axes[1, col].set_title(f'Impostor\nScore: {score:.3f}', fontsize=10, fontweight='bold', color='red')
        axes[1, col].axis('off')

    plt.suptitle('Face Verification Examples: Genuine vs Impostor Pairs',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"Verification examples saved: {save_path}")

    plt.close()

def plot_similarity_heatmap(features, labels, target_names, method_name,
                             max_classes=10, save_path=None):
    unique_labels = np.unique(labels)[:max_classes]

    class_centers = []
    class_names = []

    for label in unique_labels:
        class_features = features[labels == label]
        center = class_features.mean(axis=0)
        class_centers.append(center)
        class_names.append(target_names[label])

    class_centers = np.array(class_centers)

    similarity_matrix = cosine_similarity(class_centers)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Cosine Similarity'}, ax=ax,
                vmin=0, vmax=1, linewidths=0.5)

    ax.set_title(f'Inter-Person Similarity Heatmap: {method_name}',
                 fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"Similarity heatmap saved: {save_path}")

    plt.close()

def run_verification_study(pca_features, ae_features, labels, images=None,
                            image_shape=None, target_names=None):
    print("FACE VERIFICATION STUDY")

    metrics = ['cosine', 'euclidean']
    feature_sets = {'PCA': pca_features, 'Autoencoder': ae_features}

    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))

    results = {}
    all_scores = {}

    for feat_name, feat_data in feature_sets.items():
        print(f"\n{feat_name} features:")

        f1, f2, y_true, idx1, idx2 = generate_pairs(feat_data, labels, n_pairs=1000)

        for metric in metrics:
            print(f"  - {metric.capitalize()} similarity...")

            if metric == 'cosine':
                scores = np.array([cosine_similarity(f1[i].reshape(1, -1),
                                                     f2[i].reshape(1, -1))[0][0]
                                  for i in range(len(y_true))])
            else:
                dists = np.array([euclidean_distances(f1[i].reshape(1, -1),
                                                       f2[i].reshape(1, -1))[0][0]
                                 for i in range(len(y_true))])
                scores = -dists

            fpr, tpr, thresholds = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            eer, best_th = calculate_eer(fpr, tpr, thresholds)

            label_str = f"{feat_name} + {metric.capitalize()}\n(AUC: {roc_auc:.3f}, EER: {eer:.3f})"
            ax_roc.plot(fpr, tpr, label=label_str, linewidth=2)

            results[f"{feat_name}_{metric}"] = {
                'auc': roc_auc,
                'eer': eer,
                'threshold': best_th
            }

            key = f"{feat_name}_{metric}"
            all_scores[key] = {
                'scores': scores,
                'labels': y_true,
                'idx1': idx1,
                'idx2': idx2,
                'threshold': best_th
            }

            print(f"    AUC: {roc_auc:.3f}, EER: {eer:.3f}, Threshold: {best_th:.3f}")

    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax_roc.set_xlabel('False Acceptance Rate (FAR)', fontsize=12, fontweight='bold')
    ax_roc.set_ylabel('True Acceptance Rate (TAR)', fontsize=12, fontweight='bold')
    ax_roc.set_title('Face Verification: ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=10)
    ax_roc.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_path = f"{config.OUTPUT_PATH}/verification_roc_curves.png"
    plt.savefig(roc_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    print(f"\nROC curves saved: {roc_path}")
    plt.close()

    print("\nGenerating similarity distributions...")
    best_method = max(results.items(), key=lambda x: x[1]['auc'])[0]
    feat_name, metric = best_method.split('_')

    scores_data = all_scores[best_method]
    genuine_scores = scores_data['scores'][scores_data['labels'] == 1]
    impostor_scores = scores_data['scores'][scores_data['labels'] == 0]

    dist_path = f"{config.OUTPUT_PATH}/verification_distribution_{best_method}.png"
    plot_similarity_distribution(
        genuine_scores, impostor_scores,
        feat_name, metric.capitalize(),
        threshold=scores_data['threshold'],
        save_path=dist_path
    )

    if images is not None and image_shape is not None:
        print("\nGenerating visual examples...")
        h, w = image_shape

        examples_path = f"{config.OUTPUT_PATH}/verification_examples_{best_method}.png"
        plot_verification_examples(
            images, scores_data['idx1'], scores_data['idx2'],
            scores_data['scores'], scores_data['labels'],
            h, w, n_examples=5, save_path=examples_path
        )

    if target_names is not None:
        print("\nGenerating similarity heatmaps...")

        for feat_name, feat_data in feature_sets.items():
            heatmap_path = f"{config.OUTPUT_PATH}/verification_heatmap_{feat_name.lower()}.png"
            plot_similarity_heatmap(
                feat_data, labels, target_names, feat_name,
                max_classes=8, save_path=heatmap_path
            )

    print("VERIFICATION STUDY COMPLETED")

    return results