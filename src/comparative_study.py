# src/comparative_study.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import config
from src.pca import SVD_PCA
from src.autoencoder import ImprovedFaceAutoencoder
from src.classification import train_svm_with_grid_search

def compare_pca_vs_autoencoder_ablation(X_train, X_test, y_train, y_test,
                                         component_range=None, verbose=True):
    if component_range is None:
        component_range = config.PCA_COMPONENTS_RANGE

    if verbose:
        print("STUDIO COMPARATIVO: PCA vs AUTOENCODER")
        print(f"Dimensioni da testare: {component_range}")
        print(f"Classificatore: SVM con Grid Search")

    results = []

    for n_comp in component_range:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Testing con {n_comp} componenti/dimensioni latenti")
            print(f"{'='*70}")

        if verbose:
            print(f"\n[1/2] PCA con {n_comp} componenti...")

        pca = SVD_PCA(n_components=n_comp)
        pca.fit(X_train)

        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        variance_explained_pca = np.sum(pca.explained_variance_ratio_)

        X_test_pca_reconstructed = pca.inverse_transform(X_test_pca)
        reconstruction_error_pca = np.mean((X_test - X_test_pca_reconstructed) ** 2)

        svm_pca, svm_pca_cv_score, svm_pca_params = train_svm_with_grid_search(
            X_train_pca, y_train, verbose=False
        )

        y_pred_pca = svm_pca.predict(X_test_pca)
        acc_pca = accuracy_score(y_test, y_pred_pca)
        prec_pca, rec_pca, f1_pca, _ = precision_recall_fscore_support(
            y_test, y_pred_pca, average='weighted', zero_division=0
        )

        if verbose:
            print(f"  Varianza spiegata: {variance_explained_pca*100:.2f}%")
            print(f"  Reconstruction MSE: {reconstruction_error_pca:.6f}")
            print(f"  SVM CV Score: {svm_pca_cv_score:.4f}")
            print(f"  SVM Test Accuracy: {acc_pca:.4f}")

        if verbose:
            print(f"\n[2/2] Autoencoder con {n_comp} dimensioni latenti...")

        ae = ImprovedFaceAutoencoder(
            input_dim=X_train.shape[1],
            latent_dim=n_comp,
            hidden_layers=config.AE_HIDDEN_LAYERS,
            use_batch_norm=config.AE_USE_BATCH_NORM,
            dropout_rate=config.AE_DROPOUT_RATE
        )

        from sklearn.model_selection import train_test_split
        X_train_ae, X_val_ae, y_train_ae, y_val_ae = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train,
            random_state=config.RANDOM_STATE
        )

        ae, ae_history = ae.fit(
            X_train_ae, X_val=X_val_ae,
            epochs=config.AE_EPOCHS,
            batch_size=config.AE_BATCH_SIZE,
            lr=config.AE_LEARNING_RATE,
            weight_decay=config.AE_WEIGHT_DECAY,
            patience=config.AE_PATIENCE,
            verbose=False
        )

        X_train_ae_features = ae.transform(X_train)
        X_test_ae_features = ae.transform(X_test)

        reconstruction_error_ae = ae.compute_reconstruction_error(X_test)

        variance_original = np.var(X_test)
        variance_retained_ae = 1 - (reconstruction_error_ae / variance_original)
        variance_retained_ae = max(0, min(1, variance_retained_ae))  # Clamp [0,1]

        svm_ae, svm_ae_cv_score, svm_ae_params = train_svm_with_grid_search(
            X_train_ae_features, y_train, verbose=False
        )

        y_pred_ae = svm_ae.predict(X_test_ae_features)
        acc_ae = accuracy_score(y_test, y_pred_ae)
        prec_ae, rec_ae, f1_ae, _ = precision_recall_fscore_support(
            y_test, y_pred_ae, average='weighted', zero_division=0
        )

        if verbose:
            print(f"  Reconstruction MSE: {reconstruction_error_ae:.6f}")
            print(f"  Variance Retained (approx): {variance_retained_ae*100:.2f}%")
            print(f"  SVM CV Score: {svm_ae_cv_score:.4f}")
            print(f"  SVM Test Accuracy: {acc_ae:.4f}")

        results.append({
            'n_components': n_comp,
            'method': 'PCA',
            'variance_explained': variance_explained_pca,
            'reconstruction_mse': reconstruction_error_pca,
            'svm_cv_score': svm_pca_cv_score,
            'svm_test_accuracy': acc_pca,
            'svm_precision': prec_pca,
            'svm_recall': rec_pca,
            'svm_f1': f1_pca,
            'svm_best_params': str(svm_pca_params)
        })

        results.append({
            'n_components': n_comp,
            'method': 'Autoencoder',
            'variance_explained': variance_retained_ae,
            'reconstruction_mse': reconstruction_error_ae,
            'svm_cv_score': svm_ae_cv_score,
            'svm_test_accuracy': acc_ae,
            'svm_precision': prec_ae,
            'svm_recall': rec_ae,
            'svm_f1': f1_ae,
            'svm_best_params': str(svm_ae_params)
        })

    df_results = pd.DataFrame(results)

    if verbose:
        print("\n" + "="*70)
        print("RIEPILOGO RISULTATI COMPARATIVI")
        print("="*70)
        print(df_results.to_string(index=False))
        print("="*70 + "\n")

    return df_results

def analyze_classifier_comparison(X_train_pca, X_test_pca, X_train_ae, X_test_ae,
                                   y_train, y_test, n_classes, verbose=True):
    from src.classification import train_svm_with_grid_search, train_nn_with_grid_search

    if verbose:
        print("CONFRONTO CLASSIFICATORI: SVM vs NEURAL NETWORK")

    results = []

    for feat_name, X_tr, X_te in [('PCA', X_train_pca, X_test_pca),
                                    ('Autoencoder', X_train_ae, X_test_ae)]:

        if verbose:
            print(f"\n--- Features: {feat_name} ---")

        if verbose:
            print("\nTraining SVM...")
        svm_model, svm_cv, svm_params = train_svm_with_grid_search(
            X_tr, y_train, verbose=verbose
        )
        y_pred_svm = svm_model.predict(X_te)
        acc_svm = accuracy_score(y_test, y_pred_svm)
        prec_svm, rec_svm, f1_svm, _ = precision_recall_fscore_support(
            y_test, y_pred_svm, average='weighted', zero_division=0
        )

        results.append({
            'features': feat_name,
            'classifier': 'SVM',
            'cv_score': svm_cv,
            'test_accuracy': acc_svm,
            'precision': prec_svm,
            'recall': rec_svm,
            'f1': f1_svm,
            'best_params': str(svm_params)
        })

        if verbose:
            print("\nTraining Neural Network...")
        nn_model, nn_params, nn_val_score = train_nn_with_grid_search(
            X_tr, y_train, n_classes, verbose=verbose
        )
        y_pred_nn = nn_model.predict(X_te)
        acc_nn = accuracy_score(y_test, y_pred_nn)
        prec_nn, rec_nn, f1_nn, _ = precision_recall_fscore_support(
            y_test, y_pred_nn, average='weighted', zero_division=0
        )

        results.append({
            'features': feat_name,
            'classifier': 'Neural Network',
            'cv_score': nn_val_score,
            'test_accuracy': acc_nn,
            'precision': prec_nn,
            'recall': rec_nn,
            'f1': f1_nn,
            'best_params': str(nn_params)
        })

    df_results = pd.DataFrame(results)

    if verbose:
        print("\n" + "="*70)
        print("RIEPILOGO CONFRONTO CLASSIFICATORI")
        print("="*70)
        print(df_results.to_string(index=False))
        print("="*70 + "\n")

    return df_results

def summary_statistics(df_comparative):
    df_pca = df_comparative[df_comparative['method'] == 'PCA']
    df_ae = df_comparative[df_comparative['method'] == 'Autoencoder']

    stats = {
        'pca': {
            'best_accuracy': df_pca['svm_test_accuracy'].max(),
            'best_n_components': df_pca.loc[df_pca['svm_test_accuracy'].idxmax(), 'n_components'],
            'mean_accuracy': df_pca['svm_test_accuracy'].mean(),
            'std_accuracy': df_pca['svm_test_accuracy'].std(),
            'best_f1': df_pca['svm_f1'].max(),
            'mean_variance_explained': df_pca['variance_explained'].mean()
        },
        'autoencoder': {
            'best_accuracy': df_ae['svm_test_accuracy'].max(),
            'best_n_components': df_ae.loc[df_ae['svm_test_accuracy'].idxmax(), 'n_components'],
            'mean_accuracy': df_ae['svm_test_accuracy'].mean(),
            'std_accuracy': df_ae['svm_test_accuracy'].std(),
            'best_f1': df_ae['svm_f1'].max(),
            'mean_reconstruction_mse': df_ae['reconstruction_mse'].mean()
        }
    }

    return stats