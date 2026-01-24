"""
This module implements face verification evaluation.

Key concepts:
- Genuine pairs: Two images of the same person (should have high similarity)
- Impostor pairs: Two images of different people (should have low similarity)
- Equal Error Rate (EER): The point where false accept rate equals false reject rate
- ROC-AUC: Area under the ROC curve measuring verification performance
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
from src.utils import (plot_similarity_distribution, plot_verification_examples, plot_similarity_heatmap)


def generate_pairs(features, labels, n_pairs=1000):
    """
    Generate genuine and impostor face pairs for verification evaluation.

    Creates balanced pairs where half are genuine (same person) and half
    are impostor (different people) pairs.

    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Label array indicating person identity
        n_pairs: Total number of pairs to generate (half genuine, half impostor)

    Returns:
        tuple: (features_1, features_2, pair_labels, indices_1, indices_2)
            - features_1: Feature vectors for first images in pairs
            - features_2: Feature vectors for second images in pairs
            - pair_labels: Binary labels (1=genuine, 0=impostor)
            - indices_1: Original indices of first images
            - indices_2: Original indices of second images
    """
    n_samples = features.shape[0]
    pair_indices_1, pair_indices_2 = [], []
    pair_features_1, pair_features_2 = [], []
    pair_labels = []

    unique_labels = np.unique(labels)

    # Generate genuine pairs (same person)
    while len(pair_labels) < n_pairs // 2:
        # Randomly select a person
        label = np.random.choice(unique_labels)
        idx = np.where(labels == label)[0]

        # Need at least 2 images of this person
        if len(idx) >= 2:
            # Sample two different images of the same person
            i, j = np.random.choice(idx, 2, replace=False)
            pair_indices_1.append(i)
            pair_indices_2.append(j)
            pair_features_1.append(features[i])
            pair_features_2.append(features[j])
            pair_labels.append(1)  # Genuine pair

    # Generate impostor pairs (different people)
    while len(pair_labels) < n_pairs:
        # Randomly select two different samples
        i, j = np.random.choice(n_samples, 2, replace=False)

        # Ensure they belong to different people
        if labels[i] != labels[j]:
            pair_indices_1.append(i)
            pair_indices_2.append(j)
            pair_features_1.append(features[i])
            pair_features_2.append(features[j])
            pair_labels.append(0)  # Impostor pair

    return (np.array(pair_features_1), np.array(pair_features_2), np.array(pair_labels), np.array(pair_indices_1), np.array(pair_indices_2))


def calculate_eer(fpr, tpr, thresholds):
    """
    Calculate the Equal Error Rate (EER) from ROC curve data.

    EER is the point where the false positive rate equals the false negative rate.
    It provides a single metric to summarize verification performance.

    Args:
        fpr: False positive rates at different thresholds
        tpr: True positive rates at different thresholds
        thresholds: Threshold values corresponding to FPR/TPR points

    Returns:
        tuple: (eer, threshold) where eer is the equal error rate and
               threshold is the operating point that achieves this EER
    """
    # False negative rate is complement of true positive rate
    fnr = 1 - tpr

    # Find the point where FNR and FPR are closest (ideally equal)
    idx = np.nanargmin(np.absolute(fnr - fpr))

    return fpr[idx], thresholds[idx]


def run_verification_study(pca_features, ae_features, labels, images, image_shape, target_names):
    """
    Execute comprehensive face verification evaluation.

    Compares PCA and Autoencoder features for the verification task using
    cosine similarity as the matching metric.

    Args:
        pca_features: PCA-reduced feature array
        ae_features: Autoencoder-encoded feature array
        labels: Person identity labels
        images: Original images for visualization
        image_shape: Tuple of (height, width) for image reconstruction
        target_names: List of person names

    Returns:
        dict: Verification results for each method containing:
            - auc: Area under the ROC curve
            - eer: Equal error rate
            - threshold: Optimal decision threshold
    """
    # Define feature sets to evaluate
    feature_sets = {"PCA": pca_features, "Autoencoder": ae_features}
    results = {}
    all_scores = {}

    # Evaluate each feature extraction method
    for feat_name, feat_data in feature_sets.items():
        print(f"{feat_name} features:")

        # Generate evaluation pairs
        f1, f2, y_true, idx1, idx2 = generate_pairs(feat_data, labels, n_pairs=1000)

        # Compute pairwise cosine similarity scores
        scores = np.array([cosine_similarity(f1[i].reshape(1, -1), f2[i].reshape(1, -1))[0][0] for i in range(len(y_true))])

        # Compute ROC curve and metrics
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        eer, best_th = calculate_eer(fpr, tpr, thresholds)

        # Store results
        results[f"{feat_name}_cosine"] = {"auc": roc_auc, "eer": eer, "threshold": best_th}
        all_scores[feat_name] = {
            "scores": scores, "labels": y_true,
            "idx1": idx1, "idx2": idx2, "threshold": best_th
        }

        print(f"Cosine AUC: {roc_auc:.4f}, EER: {eer:.4f}")

    # Identify the best performing method for visualization
    best_method = max(results.items(), key=lambda x: x[1]["auc"])[0]
    feat_name = best_method.split("_")[0]
    scores_data = all_scores[feat_name]

    # Separate genuine and impostor scores for distribution plot
    genuine_scores = scores_data["scores"][scores_data["labels"] == 1]
    impostor_scores = scores_data["scores"][scores_data["labels"] == 0]

    # Generate similarity distribution visualization
    plot_similarity_distribution(genuine_scores, impostor_scores, feat_name, threshold=scores_data["threshold"])

    # Generate verification pair examples
    if images is not None and image_shape is not None:
        plot_verification_examples(images, scores_data["idx1"], scores_data["idx2"], scores_data["scores"], scores_data["labels"], image_shape[0], image_shape[1], n_examples=5)

    # Generate inter-class similarity heatmaps
    if target_names is not None:
        for feat_name, feat_data in feature_sets.items():
            plot_similarity_heatmap(feat_data, labels, target_names, feat_name, max_classes=7)

    return results