# src/verification.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import config


def generate_pairs(features, labels, n_pairs=1000):
    np.random.seed(config.RANDOM_STATE)
    n_samples = features.shape[0]
    pair_indices_1, pair_indices_2 = [], []
    pair_features_1, pair_features_2 = [], []
    pair_labels = []

    unique_labels = np.unique(labels)

    # Coppie positive (stessa persona)
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

    # Coppie negative (persone diverse)
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
    return fpr[idx], thresholds[idx]


def plot_similarity_distribution(scores_genuine, scores_impostor, method_name,
                                  threshold=None, save_path=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE_MEDIUM)

    ax.hist(scores_genuine, bins=40, alpha=0.6, label='Genuine (Same Person)',
            color='#27ae60', edgecolor='black', density=True)
    ax.hist(scores_impostor, bins=40, alpha=0.6, label='Impostor (Different Person)',
            color='#e74c3c', edgecolor='black', density=True)

    if threshold is not None:
        ax.axvline(threshold, color='blue', linestyle='--', linewidth=2,
                   label=f'Threshold: {threshold:.3f}')

    ax.set_xlabel('Similarity Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Similarity Distribution: {method_name}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    stats_text = f'Genuine: mean={scores_genuine.mean():.3f}, std={scores_genuine.std():.3f}\n'
    stats_text += f'Impostor: mean={scores_impostor.mean():.3f}, std={scores_impostor.std():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_verification_examples(images, pair_idx1, pair_idx2, scores, labels,
                                h, w, n_examples=5, save_path=None):
    plt.style.use('seaborn-v0_8-whitegrid')

    genuine_mask = labels == 1
    impostor_mask = labels == 0

    genuine_indices = np.where(genuine_mask)[0]
    genuine_scores = scores[genuine_mask]
    top_genuine_idx = genuine_indices[np.argsort(genuine_scores)[-n_examples:]]

    impostor_indices = np.where(impostor_mask)[0]
    impostor_scores = scores[impostor_mask]
    top_impostor_idx = impostor_indices[np.argsort(impostor_scores)[-n_examples:]]

    fig, axes = plt.subplots(2, n_examples, figsize=(14, 6))

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
    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_similarity_heatmap(features, labels, target_names, method_name,
                             max_classes=8, save_path=None):
    plt.style.use('seaborn-v0_8-whitegrid')

    unique_labels = np.unique(labels)[:max_classes]
    class_centers = []
    class_names = []

    for label in unique_labels:
        class_features = features[labels == label]
        class_centers.append(class_features.mean(axis=0))
        class_names.append(target_names[label])

    similarity_matrix = cosine_similarity(np.array(class_centers))

    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE_MEDIUM)
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Cosine Similarity'}, ax=ax,
                vmin=0, vmax=1, linewidths=0.5)

    ax.set_title(f'Inter-Person Similarity: {method_name}', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def run_verification_study(pca_features, ae_features, labels, images=None,
                            image_shape=None, target_names=None):
    print("\nFACE VERIFICATION STUDY")
    print("-" * 40)

    feature_sets = {'PCA': pca_features, 'Autoencoder': ae_features}
    results = {}
    all_scores = {}

    for feat_name, feat_data in feature_sets.items():
        print(f"\n{feat_name} features:")
        f1, f2, y_true, idx1, idx2 = generate_pairs(feat_data, labels, n_pairs=1000)

        # Cosine similarity
        scores = np.array([cosine_similarity(f1[i].reshape(1, -1), f2[i].reshape(1, -1))[0][0]
                          for i in range(len(y_true))])

        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        eer, best_th = calculate_eer(fpr, tpr, thresholds)

        results[f"{feat_name}_cosine"] = {'auc': roc_auc, 'eer': eer, 'threshold': best_th}
        all_scores[feat_name] = {
            'scores': scores, 'labels': y_true,
            'idx1': idx1, 'idx2': idx2, 'threshold': best_th
        }

        print(f"  Cosine - AUC: {roc_auc:.4f}, EER: {eer:.4f}")

    # Visualizzazioni per il metodo migliore
    best_method = max(results.items(), key=lambda x: x[1]['auc'])[0]
    feat_name = best_method.split('_')[0]
    scores_data = all_scores[feat_name]

    genuine_scores = scores_data['scores'][scores_data['labels'] == 1]
    impostor_scores = scores_data['scores'][scores_data['labels'] == 0]

    print("\nGenerating visualizations...")

    # 1. Distribuzione similarita
    plot_similarity_distribution(
        genuine_scores, impostor_scores, feat_name,
        threshold=scores_data['threshold'],
        save_path=f"{config.OUTPUT_PATH}/verification_distribution.png"
    )

    # 2. Esempi visuali
    if images is not None and image_shape is not None:
        plot_verification_examples(
            images, scores_data['idx1'], scores_data['idx2'],
            scores_data['scores'], scores_data['labels'],
            image_shape[0], image_shape[1], n_examples=5,
            save_path=f"{config.OUTPUT_PATH}/verification_examples.png"
        )

    # 3. Heatmap similarita
    if target_names is not None:
        for feat_name, feat_data in feature_sets.items():
            plot_similarity_heatmap(
                feat_data, labels, target_names, feat_name,
                max_classes=7,
                save_path=f"{config.OUTPUT_PATH}/verification_heatmap_{feat_name.lower()}.png"
            )

    print("\nVerification study completed")
    return results
