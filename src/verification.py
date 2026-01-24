import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
from src.utils import (plot_similarity_distribution, plot_verification_examples, plot_similarity_heatmap)

def generate_pairs(features, labels, n_pairs=1000):
    n_samples = features.shape[0]
    pair_indices_1, pair_indices_2 = [], []
    pair_features_1, pair_features_2 = [], []
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

    return (np.array(pair_features_1), np.array(pair_features_2), np.array(pair_labels), np.array(pair_indices_1), np.array(pair_indices_2))

def calculate_eer(fpr, tpr, thresholds):
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    return fpr[idx], thresholds[idx]

def run_verification_study(pca_features, ae_features, labels, images, image_shape, target_names):
    feature_sets = {"PCA": pca_features, "Autoencoder": ae_features}
    results = {}
    all_scores = {}

    for feat_name, feat_data in feature_sets.items():
        print(f"{feat_name} features:")
        
        f1, f2, y_true, idx1, idx2 = generate_pairs(feat_data, labels, n_pairs=1000)

        scores = np.array([cosine_similarity(f1[i].reshape(1, -1), f2[i].reshape(1, -1))[0][0] for i in range(len(y_true))])

        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        eer, best_th = calculate_eer(fpr, tpr, thresholds)

        results[f"{feat_name}_cosine"] = {"auc": roc_auc, "eer": eer, "threshold": best_th}
        all_scores[feat_name] = {
            "scores": scores, "labels": y_true,
            "idx1": idx1, "idx2": idx2, "threshold": best_th
        }

        print(f"Cosine AUC: {roc_auc:.4f}, EER: {eer:.4f}")

    best_method = max(results.items(), key=lambda x: x[1]["auc"])[0]
    feat_name = best_method.split("_")[0]
    scores_data = all_scores[feat_name]

    genuine_scores = scores_data["scores"][scores_data["labels"] == 1]
    impostor_scores = scores_data["scores"][scores_data["labels"] == 0]

    plot_similarity_distribution(genuine_scores, impostor_scores, feat_name, threshold=scores_data["threshold"])

    if images is not None and image_shape is not None:
        plot_verification_examples(images, scores_data["idx1"], scores_data["idx2"], scores_data["scores"], scores_data["labels"], image_shape[0], image_shape[1], n_examples=5)

    if target_names is not None:
        for feat_name, feat_data in feature_sets.items():
            plot_similarity_heatmap(feat_data, labels, target_names, feat_name, max_classes=7)

    return results