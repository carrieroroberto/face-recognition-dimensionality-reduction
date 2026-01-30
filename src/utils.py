"""
This module provides a comprehensive set of visualization functions for:
- Dataset exploration and statistics
- PCA analysis (eigenfaces, scree plots)
- Training progress monitoring
- Model comparison and evaluation
- Face verification analysis
- t-SNE dimensionality reduction visualization

All plots are saved to the configured output directory as PNG files.
"""

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
import sys


def plot_dataset_stats(data_dict):
    """
    Generate exploratory data analysis visualizations.

    Creates class distribution bar chart and mean face visualization
    to understand the dataset characteristics.

    Args:
        data_dict: Dictionary containing training data, labels, target names,
                   and image shape information
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    target_names = data_dict["target_names"]
    y_train = data_dict["y_train"]
    h, w = data_dict["image_shape"]
    X_train = data_dict["X_train"]

    # Plot class distribution
    _, ax = plt.subplots(figsize=(12, 6))
    unique, counts = np.unique(y_train, return_counts=True)

    names_subset = [target_names[i] for i in unique]

    sns.barplot(x=names_subset, y=counts, palette="viridis", ax=ax)
    ax.set_title("Class Distribution (Train Set)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Images", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(config.OUTPUT_PATH, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot mean face (average of all training images)
    _, ax = plt.subplots(figsize=(5, 5))
    mean_face = np.mean(X_train, axis=0).reshape(h, w)
    ax.imshow(mean_face, cmap="gray")
    ax.set_title("Mean Face", fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    save_path = os.path.join(config.OUTPUT_PATH, "mean_face.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_eigenfaces(pca_model, h, w, n_top=12):
    """
    Visualize the top principal components as eigenfaces.

    Eigenfaces represent the directions of maximum variance in face space
    and can be interpreted as "prototype" facial features.

    Args:
        pca_model: Fitted PCA model with components_ attribute
        h: Image height
        w: Image width
        n_top: Number of top components to display
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    n_top = min(n_top, len(pca_model.components_))
    rows = (n_top + 3) // 4  # Calculate grid rows needed

    _, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten()

    # Plot each eigenface
    for i in range(n_top):
        eigenface = pca_model.components_[i].reshape(h, w)
        axes[i].imshow(eigenface, cmap="gray")
        axes[i].set_title(f"PC {i+1}", fontsize=12, fontweight="bold")
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n_top, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Top Principal Components (Eigenfaces)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(config.OUTPUT_PATH, "eigenfaces.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scree_plot(pca_model):
    """
    Generate a scree plot showing cumulative variance explained.

    Helps determine the optimal number of components by visualizing
    how much variance is captured as components are added.

    Args:
        pca_model: Fitted PCA model with explained_variance_ratio_ attribute
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    _, ax = plt.subplots(figsize=(8, 6))

    # Compute cumulative variance
    cum_variance = np.cumsum(pca_model.explained_variance_ratio_)

    ax.plot(range(1, len(cum_variance) + 1), cum_variance, marker="o", linestyle="--", markersize=4)
    ax.set_xlabel("Number of Components", fontsize=11)
    ax.set_ylabel("Cumulative Variance Explained", fontsize=11)
    ax.set_title("Variance Analysis", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(config.OUTPUT_PATH, "scree_plot.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_loss(train_loss, val_loss=None, title="Training Loss"):
    """
    Plot training and validation loss curves over epochs.

    Useful for monitoring training progress and detecting overfitting.

    Args:
        train_loss: List of training loss values per epoch
        val_loss: Optional list of validation loss values
        title: Plot title
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_loss, label="Train Loss", color="tab:blue", linewidth=2)

    if val_loss is not None and len(val_loss) > 0:
        ax.plot(val_loss, label="Validation Loss", color="tab:orange", linestyle="--", linewidth=2)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (MSE)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Use appropriate filename based on title
    filename = "ae_loss.png" if "AE" in title or "Autoencoder" in title else "loss_curve.png"
    save_path = os.path.join(config.OUTPUT_PATH, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_reconstruction_comparison(original, pca_rec, ae_rec, h, w, n_images=5):
    """
    Compare original images with PCA and Autoencoder reconstructions.

    Side-by-side visualization to assess reconstruction quality
    of different dimensionality reduction methods.

    Args:
        original: Original image array
        pca_rec: PCA reconstructed images
        ae_rec: Autoencoder reconstructed images
        h: Image height
        w: Image width
        n_images: Number of images to display
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    n_images = min(n_images, len(original))
    _, axes = plt.subplots(n_images, 3, figsize=(10, 3 * n_images))

    for i in range(n_images):
        # Original image
        axes[i, 0].imshow(original[i].reshape(h, w), cmap="gray")
        if i == 0: axes[i, 0].set_title("Original", fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")

        # PCA reconstruction
        axes[i, 1].imshow(pca_rec[i].reshape(h, w), cmap="gray")
        if i == 0: axes[i, 1].set_title("PCA", fontsize=12, fontweight="bold")
        axes[i, 1].axis("off")

        # Autoencoder reconstruction
        axes[i, 2].imshow(ae_rec[i].reshape(h, w), cmap="gray")
        if i == 0: axes[i, 2].set_title("Autoencoder", fontsize=12, fontweight="bold")
        axes[i, 2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_PATH, "reconstruction_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, target_names, model_name):
    """
    Generate a confusion matrix heatmap for classification results.

    Shows the distribution of predictions across actual classes
    to identify misclassification patterns.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        target_names: List of class names
        model_name: Model identifier for the title
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    cm = confusion_matrix(y_true, y_pred)

    _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, ax=ax)

    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    # Create safe filename from model name
    safe_name = model_name.lower().replace(" ", "_")
    save_path = os.path.join(config.OUTPUT_PATH, f"cm_{safe_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tsne_comparison(pca_features, ae_features, labels, target_names):
    """
    Compare PCA and Autoencoder features using t-SNE visualization.

    Projects high-dimensional features to 2D for visual comparison
    of cluster separability between methods.

    Args:
        pca_features: PCA-reduced feature array
        ae_features: Autoencoder-reduced feature array
        labels: Class labels for coloring
        target_names: List of class names for legend
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    _, axes = plt.subplots(1, 2, figsize=(16, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    # Generate t-SNE for each feature type
    for i, (name, feat) in enumerate([("PCA", pca_features), ("Autoencoder", ae_features)]):
        # Fit t-SNE with PCA initialization for stability
        tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto")
        X_2d = tsne.fit_transform(feat)

        # Plot each class with different color
        for j, label_idx in enumerate(unique_labels):
            idx = np.where(labels == label_idx)
            color = colors[j % len(colors)]
            axes[i].scatter(X_2d[idx, 0], X_2d[idx, 1],
                            label=target_names[label_idx],
                            color=color, alpha=0.7, s=30)

        axes[i].set_title(f"t-SNE: {name} Features", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("t-SNE 1", fontsize=10)
        axes[i].set_ylabel("t-SNE 2", fontsize=10)
        axes[i].grid(True, alpha=0.3)

    # Add legend to second subplot
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, title="Classes")

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_PATH, "tsne_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_multiclass_roc(roc_data, n_classes, target_names, title="ROC Curves"):
    """
    Plot multi-class ROC curves with micro and macro averages.

    Visualizes the trade-off between true positive rate and false positive
    rate for each class, along with aggregate performance metrics.

    Args:
        roc_data: Dictionary containing FPR, TPR, and AUC for each class
        n_classes: Number of classes
        target_names: List of class names
        title: Plot title
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    _, ax = plt.subplots(figsize=(10, 8))

    # Plot micro-averaged ROC
    ax.plot(roc_data["micro"]["fpr"], roc_data["micro"]["tpr"],
            label=f"Micro-average ROC (area = {roc_data['micro']['auc']:.2f})",
            color="deeppink", linestyle=":", linewidth=3)

    # Plot macro-averaged ROC
    ax.plot(roc_data["macro"]["fpr"], roc_data["macro"]["tpr"],
            label=f"Macro-average ROC (area = {roc_data['macro']['auc']:.2f})",
            color="navy", linestyle=":", linewidth=3)

    # Define colors for per-class curves
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red", "purple"])

    # Limit number of per-class curves for readability
    classes_to_plot = range(n_classes) if n_classes <= 10 else range(5)

    # Plot per-class ROC curves
    for i, color in zip(classes_to_plot, colors):
        ax.plot(roc_data[i]["fpr"], roc_data[i]["tpr"], color=color, lw=1.5, alpha=0.7,
                label=f"ROC {target_names[i]} (area = {roc_data[i]['auc']:.2f})")

    # Add diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)

    # Create filename from title
    filename = f"{title.lower().replace(" ", "_")}.png"
    save_path = os.path.join(config.OUTPUT_PATH, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_similarity_distribution(scores_genuine, scores_impostor, method_name, threshold=None):
    """
    Plot distribution of similarity scores for genuine and impostor pairs.

    Visualizes the separation between same-person (genuine) and
    different-person (impostor) similarity scores.

    Args:
        scores_genuine: Similarity scores for same-person pairs
        scores_impostor: Similarity scores for different-person pairs
        method_name: Feature extraction method name
        threshold: Optional decision threshold to display
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    _, ax = plt.subplots(figsize=(10, 8))

    # Plot histograms for both distributions
    ax.hist(scores_genuine, bins=40, alpha=0.6, label="Genuine (Same Person)",
            color="#27ae60", edgecolor="black", density=True)
    ax.hist(scores_impostor, bins=40, alpha=0.6, label="Impostor (Different Person)",
            color="#e74c3c", edgecolor="black", density=True)

    # Add threshold line if provided
    if threshold is not None:
        ax.axvline(threshold, color="blue", linestyle="--", linewidth=2,
                   label=f"Threshold: {threshold:.3f}")

    ax.set_xlabel("Cosine Similarity Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Similarity Distribution: {method_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f"Genuine: mean={scores_genuine.mean():.3f}, std={scores_genuine.std():.3f}\n"
    stats_text += f"Impostor: mean={scores_impostor.mean():.3f}, std={scores_impostor.std():.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_PATH, f"similarity_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_verification_examples(images, pair_idx1, pair_idx2, scores, labels,
                               h, w, n_examples=5):
    """
    Display example face pairs for verification task.

    Shows top-scoring genuine pairs and impostor pairs to illustrate
    the verification challenge.

    Args:
        images: Array of face images
        pair_idx1: Indices of first images in pairs
        pair_idx2: Indices of second images in pairs
        scores: Similarity scores for each pair
        labels: Binary labels (1=genuine, 0=impostor)
        h: Image height
        w: Image width
        n_examples: Number of examples to display per category
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # Separate genuine and impostor pairs
    genuine_mask = labels == 1
    impostor_mask = labels == 0

    # Get top-scoring genuine pairs
    genuine_indices = np.where(genuine_mask)[0]
    genuine_scores = scores[genuine_mask]
    top_genuine_idx = genuine_indices[np.argsort(genuine_scores)[-n_examples:]]

    # Get top-scoring impostor pairs (hardest negatives)
    impostor_indices = np.where(impostor_mask)[0]
    impostor_scores = scores[impostor_mask]
    top_impostor_idx = impostor_indices[np.argsort(impostor_scores)[-n_examples:]]

    _, axes = plt.subplots(2, n_examples, figsize=(14, 6))

    # Display genuine pairs (top row)
    for col, idx in enumerate(top_genuine_idx):
        i, j = pair_idx1[idx], pair_idx2[idx]
        score = scores[idx]
        img1 = images[i].reshape(h, w)
        img2 = images[j].reshape(h, w)
        combined = np.hstack([img1, img2])

        axes[0, col].imshow(combined, cmap="gray")
        axes[0, col].set_title(f"Genuine\nScore: {score:.3f}", fontsize=12, fontweight="bold", color="green")
        axes[0, col].axis("off")

    # Display impostor pairs (bottom row)
    for col, idx in enumerate(top_impostor_idx):
        i, j = pair_idx1[idx], pair_idx2[idx]
        score = scores[idx]
        img1 = images[i].reshape(h, w)
        img2 = images[j].reshape(h, w)
        combined = np.hstack([img1, img2])

        axes[1, col].imshow(combined, cmap="gray")
        axes[1, col].set_title(f"Impostor\nScore: {score:.3f}", fontsize=12, fontweight="bold", color="red")
        axes[1, col].axis("off")

    plt.suptitle("Face Verification: Genuine vs Impostor Pairs", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_PATH, f"verification_examples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_similarity_heatmap(features, labels, target_names, method_name,
                            max_classes=8):
    """
    Generate inter-class similarity heatmap.

    Visualizes pairwise cosine similarity between class centroids
    to understand class separability in feature space.

    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Class labels
        target_names: List of class names
        method_name: Feature extraction method name
        max_classes: Maximum number of classes to display
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # Limit to max_classes for readability
    unique_labels = np.unique(labels)[:max_classes]
    class_centers = []
    class_names = []

    # Compute class centroids
    for label in unique_labels:
        class_features = features[labels == label]
        class_centers.append(class_features.mean(axis=0))
        class_names.append(target_names[label])

    # Compute pairwise similarity matrix
    similarity_matrix = cosine_similarity(np.array(class_centers))

    _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "Cosine Similarity"}, ax=ax,
                vmin=0, vmax=1, linewidths=0.5)

    ax.set_title(f"Inter-Person Similarity: {method_name}", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(config.OUTPUT_PATH, f"similarity_heatmap_{method_name.lower().replace(" ", "_")}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ablation_study(x_values, accuracy_scores, secondary_metric_scores, x_label, secondary_label, title):
    """
    Generate dual-axis ablation study plot.

    Shows the relationship between a hyperparameter (e.g., number of components)
    and both accuracy and a secondary metric (e.g., variance explained).

    Args:
        x_values: List of hyperparameter values tested
        accuracy_scores: Corresponding accuracy scores
        secondary_metric_scores: Corresponding secondary metric values
        x_label: Label for x-axis
        secondary_label: Label for secondary y-axis
        title: Plot title
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    _, ax1 = plt.subplots(figsize=(10, 6))

    # Primary axis: Accuracy
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("SVM Test Accuracy", color="tab:blue")
    ax1.plot(x_values, accuracy_scores, marker="o", color="tab:blue",
             linewidth=2, markersize=8, label="SVM Accuracy")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    # Secondary axis: Additional metric
    ax2 = ax1.twinx()
    ax2.set_ylabel(secondary_label, color="tab:orange")
    ax2.plot(x_values, secondary_metric_scores, marker="s", color="tab:orange",
             linewidth=2, markersize=8, linestyle="--", label=secondary_label)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    # Determine filename based on method
    save_path = os.path.join(config.OUTPUT_PATH, f"{"pca" if "PCA" in title else "ae"}_ablation_study.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_dataset_statistics(X, y, target_names):
    """
    Compute comprehensive statistics for the dataset.

    Calculates pixel statistics, class distribution metrics, and
    sample counts for dataset characterization.

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array
        target_names: List of class names

    Returns:
        dict: Dictionary containing various dataset statistics
    """
    unique, counts = np.unique(y, return_counts=True)

    stats = {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": len(target_names),
        "pixel_min": X.min(),
        "pixel_max": X.max(),
        "pixel_mean": X.mean(),
        "pixel_std": X.std(),
        "class_distribution": dict(zip([target_names[i] for i in unique], counts)),
        "class_counts": counts,
        "min_samples_per_class": counts.min(),
        "max_samples_per_class": counts.max(),
        "mean_samples_per_class": counts.mean(),
        "std_samples_per_class": counts.std()
    }

    return stats


def print_dataset_statistics(stats):
    """
    Display dataset statistics in formatted output.

    Args:
        stats: Dictionary of statistics from compute_dataset_statistics
    """
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
    """
    Display classification metrics summary in formatted output.

    Args:
        metrics_dict: Dictionary of computed metrics
        model_name: Identifier for the model being summarized
    """
    print(f"\n--- METRICS SUMMARY: {model_name} ---")

    if "best_params" in metrics_dict:
        print(f"Best Params: {metrics_dict['best_params']}")

    print(f"Accuracy: {metrics_dict['accuracy']:.4f}")

    if "confidence_interval" in metrics_dict:
        ci = metrics_dict["confidence_interval"]
        print(f"95% Confidence Interval: [{ci['lower_bound']:.4f}, {ci['upper_bound']:.4f}]")

    print(f"Macro F1: {metrics_dict['f1_macro']:.4f}")
    print(f"Weighted F1: {metrics_dict['f1_weighted']:.4f}")

    if "cv_score" in metrics_dict:
        print(f"CV Score (Val): {metrics_dict['cv_score']:.4f}")

    if "roc_auc_macro" in metrics_dict:
        print(f"ROC AUC (Macro): {metrics_dict['roc_auc_macro']:.4f}")


def export_metrics_to_csv(df_metrics, original=False):
    """
    Export metrics DataFrame to CSV file.

    Args:
        df_metrics: pandas DataFrame containing model comparison metrics
    """
    filename = "classification_metrics_original.csv" if original else "classification_metrics.csv"
    save_path = os.path.join(config.MODELS_PATH, filename)
    df_metrics.to_csv(save_path, index=False)


def save_model(model, name):
    """
    Save trained model to disk.

    Handles both PyTorch models (saved as .pt) and scikit-learn models
    (saved as .joblib).

    Args:
        model: Trained model instance
        name: Base filename for the saved model
    """
    if isinstance(model, torch.nn.Module):
        # Save PyTorch model state dict
        filename = f"{name}.pt"
        path = os.path.join(config.MODELS_PATH, filename)
        torch.save(model.state_dict(), path)
    else:
        # Save scikit-learn model using joblib
        filename = f"{name}.joblib"
        path = os.path.join(config.MODELS_PATH, filename)
        joblib.dump(model, path)


def log_output():
    """
    Redirect stdout and stderr to a log file.
    """
    log_file = open("results/log.txt", "w")
    terminal = sys.stdout
    class DualOutput:
        def __init__(self, terminal, logfile):
            self.terminal = terminal
            self.logfile = logfile
        def write(self, message):
            self.terminal.write(message)
            self.logfile.write(message)
        def flush(self):
            self.terminal.flush()
            self.logfile.flush()
    sys.stdout = sys.stderr = DualOutput(terminal, log_file)