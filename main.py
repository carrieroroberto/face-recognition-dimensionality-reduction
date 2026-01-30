"""
Project: Comparative Analysis of Linear and Non-Linear Dimensionality Reduction for Face Recognition and Verification.
Students: Roberto Carriero, Massimiliano Leone

This module orchestrates the complete machine learning workflow including:
- Data loading and preprocessing from the LFW dataset
- Exploratory data analysis with visualization
- Ablation study comparing PCA vs Autoencoder for dimensionality reduction
- Classification experiments using SVM and Neural Network classifiers
- Face verification study using cosine similarity
- Model persistence and results export
"""

import os
import joblib
import torch
import numpy as np
import pandas as pd
import config
from src.preprocessing import DataPreprocessor
from src.pca import run_pca_experiments
from src.autoencoder import run_autoencoder_experiments
from src.classification import (train_svm_with_grid_search, train_nn_with_grid_search)
from src.verification import run_verification_study
from src.metrics import (compute_classification_metrics, compare_models_metrics, compute_roc_curves, calculate_confidence_intervals)
from src.utils import (plot_dataset_stats, plot_reconstruction_comparison, plot_training_loss, plot_confusion_matrix, plot_tsne_comparison, plot_multiclass_roc, save_model, compute_dataset_statistics, print_dataset_statistics, print_metrics_summary, export_metrics_to_csv, log_output)
import warnings
warnings.filterwarnings("ignore")
log_output()

def main():
    """
    Execute the complete face recognition pipeline.

    The pipeline consists of 8 sequential phases:
    1. Data loading and preprocessing
    2. Exploratory data analysis
    3. Ablation study (PCA vs Autoencoder)
    4. Classification with hyperparameter tuning
    5. Model comparison and evaluation
    6. Face verification study
    7. t-SNE visualization
    8. Model saving
    """
    # =========================================================================
    # PHASE 1: DATA LOADING AND PREPROCESSING
    # Load the LFW dataset and apply preprocessing transformations
    # =========================================================================
    print("\n1. DATA LOADING AND PREPROCESSING")
    preprocessor = DataPreprocessor()
    X, y, _, target_names = preprocessor.load_dataset()

    # Compute and display dataset statistics for initial analysis
    stats = compute_dataset_statistics(X, y, target_names)
    print_dataset_statistics(stats)

    # Apply preprocessing: normalization and train/test split
    data = preprocessor.preprocess_data(X, y)

    # Extract preprocessed data splits
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    X_train_raw = data["X_train_raw"]  # Non-standardized data for visualization
    X_test_raw = data["X_test_raw"]

    # Retrieve dataset metadata
    data_info = preprocessor.get_data_info()
    n_classes = data_info["n_classes"]
    h, w = data_info["image_shape"]
    n_features = data_info["n_features"]
    print(f"Image shape: {h}x{w}, Total features: {n_features}, Classes: {n_classes}")

    # =========================================================================
    # PHASE 2: EXPLORATORY DATA ANALYSIS
    # Generate visualizations to understand data distribution and characteristics
    # =========================================================================
    print("\n2. EXPLORATORY DATA ANALYSIS")
    eda_data = {
        "X_train": X_train_raw,
        "y_train": y_train,
        "target_names": target_names,
        "image_shape": (h, w)
    }

    plot_dataset_stats(eda_data)
    print("Generating Analytics Plots...")

    # =========================================================================
    # PHASE 3: ABLATION STUDY - DIMENSIONALITY REDUCTION COMPARISON
    # Compare PCA (linear) vs Autoencoder (non-linear) methods
    # =========================================================================
    print("\n3. ABLATION STUDY (PCA VS AUTOENCODER)")
    components_range = config.COMPONENTS_RANGE
    print(f"Testing different numbero of components/latent dimensions: {components_range}")

    # Run PCA experiments across different component counts
    print("\nRunning PCA via SVD Experiments")
    pca_results = run_pca_experiments(
        n_components_list=components_range,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        h=h, w=w
    )

    # Select the best PCA configuration based on SVM accuracy
    best_pca_n = max(pca_results, key=lambda k: pca_results[k]["svm_test_acc"])
    print(f"\nBest PCA: {best_pca_n} components (Acc: {pca_results[best_pca_n]['svm_test_acc']:.4f})")

    # Extract the best PCA model and transformed features
    final_pca_results = pca_results[best_pca_n]
    X_train_pca = final_pca_results["X_train_pca"]
    X_test_pca = final_pca_results["X_test_pca"]
    pca_model = final_pca_results["pca_model"]

    # Run Autoencoder experiments across different latent dimensions
    print("\nRunning Autoencoder Experiments")
    ae_results = run_autoencoder_experiments(
        latent_dims_list=components_range,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        h=h, w=w
    )

    # Select the best Autoencoder configuration based on test accuracy
    best_ae_dim = max(ae_results, key=lambda k: ae_results[k]["test_acc"])
    print(f"\nBest Autoencoder: {best_ae_dim} dimensions (Acc: {ae_results[best_ae_dim]['test_acc']:.4f})")

    # Extract the best Autoencoder model and encoded features
    final_ae_results = ae_results[best_ae_dim]
    X_train_ae = final_ae_results["X_train_feat"]
    X_test_ae = final_ae_results["X_test_feat"]
    ae_model = final_ae_results["model"]

    # Plot training history if available
    if "history" in final_ae_results:
        hist = final_ae_results["history"]
        plot_training_loss(hist["train_loss"], hist.get("val_loss"), title=f"AE Training Loss (Latent {best_ae_dim})")

    # Generate side-by-side reconstruction comparison
    print("\nGenerating Reconstruction Comparison Plot...")
    X_test_pca_rec = pca_model.inverse_transform(X_test_pca)
    X_test_ae_rec = ae_model.reconstruct(X_test)

    plot_reconstruction_comparison(
        original=X_test,
        pca_rec=X_test_pca_rec,
        ae_rec=X_test_ae_rec,
        h=h, w=w, n_images=5
    )

    # =========================================================================
    # PHASE 4: CLASSIFICATION WITH HYPERPARAMETER TUNING
    # Train SVM and Neural Network classifiers on reduced features
    # =========================================================================
    print("\n4. CLASSIFICATION (SVM vs NEURAL NETWORK)")
    all_results = []

    # Define feature configurations to evaluate
    feature_configs = [
        ("PCA", best_pca_n, X_train_pca, X_test_pca),
        ("Autoencoder", best_ae_dim, X_train_ae, X_test_ae)
    ]

    # Iterate over each feature extraction method
    for feat_name, feat_dim, X_tr, X_te in feature_configs:
        print(f"\nTraining Classifiers on {feat_name} features (Dim: {feat_dim})")

        # --- SVM Classifier with Grid Search ---
        print(f"Running SVM Grid Search for {feat_name}...")
        svm_params = {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        }

        best_svm, svm_cv_score, best_svm_params = train_svm_with_grid_search(X_train=X_tr, y_train=y_train, param_grid=svm_params)

        # Generate predictions and probability scores
        y_pred_svm = best_svm.predict(X_te)
        y_score_svm = best_svm.predict_proba(X_te)

        # Compute comprehensive evaluation metrics
        metrics_svm = compute_classification_metrics(y_test, y_pred_svm, y_score_svm, target_names)
        metrics_svm["cv_score"] = svm_cv_score
        metrics_svm["best_params"] = best_svm_params
        metrics_svm["confidence_interval"] = calculate_confidence_intervals(y_test, y_pred_svm)

        print_metrics_summary(metrics_svm, f"SVM on {feat_name}")
        save_model(best_svm, f"svm_{feat_name.lower()}")
        all_results.append(("SVM", f"{feat_name}", metrics_svm))

        # --- Neural Network Classifier with Grid Search ---
        print(f"\nRunning Neural Network Grid Search for {feat_name}...")
        nn_params = {
            "hidden_layers": [(128, 64), (256, 128)],
            "lr": [1e-2, 1e-3],
            "dropout_rate": [0.1, 0.2],
            "weight_decay": [1e-3, 1e-4],
            "epochs": [50, 100]
        }

        best_nn, nn_cv_score, best_nn_params = train_nn_with_grid_search(X_train=X_tr, y_train=y_train, param_grid=nn_params)

        # Generate predictions and probability scores
        y_pred_nn = best_nn.predict(X_te)
        y_score_nn = best_nn.predict_proba(X_te)

        # Generate ROC curves for both classifiers
        print(f"Generating SVM ROC curves for {feat_name}...")
        roc_data_svm = compute_roc_curves(y_test, y_score_svm, n_classes)
        plot_multiclass_roc(roc_data_svm, n_classes, target_names, title=f"SVM ROC {feat_name}")

        print(f"Generating NN ROC curves for {feat_name}...")
        roc_data_nn = compute_roc_curves(y_test, y_score_nn, n_classes)
        plot_multiclass_roc(roc_data_nn, n_classes, target_names, title=f"NN ROC {feat_name}")

        # Compute comprehensive evaluation metrics
        metrics_nn = compute_classification_metrics(y_test, y_pred_nn, y_score_nn, target_names)
        metrics_nn["val_score"] = nn_cv_score
        metrics_nn["best_params"] = best_nn_params
        metrics_nn["confidence_interval"] = calculate_confidence_intervals(y_test, y_pred_nn)

        print_metrics_summary(metrics_nn, f"NN on {feat_name}")
        save_model(best_nn, f"nn_{feat_name.lower()}")
        all_results.append(("NN", f"{feat_name}", metrics_nn))

        # Generate confusion matrix for the best performing model
        best_model_name = "SVM" if metrics_svm["accuracy"] > metrics_nn["accuracy"] else "NN"
        best_pred = y_pred_svm if best_model_name == "SVM" else y_pred_nn
        plot_confusion_matrix(y_test, best_pred, target_names, f"{best_model_name}_{feat_name}")

    # =========================================================================
    # PHASE 5: MODEL COMPARISON AND RESULTS EXPORT
    # Compare all models and export metrics to CSV
    # =========================================================================
    print("\n5. FINAL MODEL COMPARISON")
    df_comparison = compare_models_metrics(all_results)

    print("Comparison Table (Sorted by Accuracy):")
    print(df_comparison.to_string(index=False))
    export_metrics_to_csv(df_comparison)

    # =========================================================================
    # PHASE 6: FACE VERIFICATION STUDY
    # Evaluate feature representations for identity verification task
    # =========================================================================
    print("\n6. FACE VERIFICATION STUDY")
    verification_results = run_verification_study(
        pca_features=X_test_pca,
        ae_features=X_test_ae,
        labels=y_test,
        images=X_test_raw,
        image_shape=(h, w),
        target_names=target_names
    )

    print("\nVerification Performance (AUC & EER):")
    for method, res in verification_results.items():
        print(f"{method}: AUC={res['auc']:.4f}, EER={res['eer']:.4f}")

    # =========================================================================
    # PHASE 7: t-SNE VISUALIZATION
    # Generate 2D embeddings to visualize feature space separability
    # =========================================================================
    print("\n7. t-SNE VISUALIZATION")
    print("Generating t-SNE graph...")

    # Sample subset for efficient t-SNE computation
    n_tsne = min(1000, len(X_train_pca))
    idx_tsne = np.random.choice(len(X_train_pca), n_tsne, replace=False)

    plot_tsne_comparison(
        X_train_pca[idx_tsne],
        X_train_ae[idx_tsne],
        y_train[idx_tsne],
        target_names
    )

    # =========================================================================
    # PHASE 8: MODEL PERSISTENCE
    # Save trained models for future inference
    # =========================================================================
    print("\n8. SAVING TRAINED MODELS")
    joblib.dump(pca_model, os.path.join(config.MODELS_PATH, f"best_pca_{best_pca_n}.joblib"))
    torch.save(ae_model.state_dict(), os.path.join(config.MODELS_PATH, f"best_autoencoder_{best_ae_dim}.pt"))

    print("Saving best PCA and Autoencoder models...")

    print("\nPipeline completed successfully.\n")
    
    # =========================================================================
    # EXTRA: CLASSIFICATION BEFORE DIMENSIONALITY REDUCTION
    # Compare SVM and NN (best tuned models) on original high-dimensional data
    # =========================================================================
    # SVM
    print("\nEXTRA: CLASSIFICATION ON ORIGINAL HIGH-DIMENSIONAL DATA")
    best_svm.fit(X_train, y_train)
    y_pred_orig_svm = best_svm.predict(X_test)
    y_score_orig_svm = best_svm.predict_proba(X_test)
    metrics_orig_svm = compute_classification_metrics(y_test, y_pred_orig_svm, y_score_orig_svm, target_names)
    print_metrics_summary(metrics_orig_svm, "SVM on original data")
    
    # Neural Network
    best_nn.fit(X_train, y_train)
    y_pred_orig_nn = best_nn.predict(X_test)
    y_score_orig_nn = best_nn.predict_proba(X_test)
    metrics_orig_nn = compute_classification_metrics(y_test, y_pred_orig_nn, y_score_orig_nn, target_names)
    print_metrics_summary(metrics_orig_nn, "NN on original data")
    
    # Export original data metrics to CSV
    df = pd.DataFrame([
        {"model": "SVM_original", **metrics_orig_svm},
        {"model": "NN_original", **metrics_orig_nn}
    ])
    export_metrics_to_csv(df, original=True)


if __name__ == "__main__":
    main()