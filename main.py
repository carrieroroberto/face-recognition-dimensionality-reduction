# main.py
import numpy as np
import pandas as pd
import joblib
import torch
import warnings
warnings.filterwarnings('ignore')

import config
from src.preprocessing import DataPreprocessor, compute_dataset_statistics, print_dataset_statistics
from src.pca import run_pca_experiments
from src.autoencoder import FaceAutoencoder
from src.classification import train_svm_with_grid_search, train_nn_with_grid_search
from src.verification import run_verification_study
from src.metrics import (
    compute_classification_metrics, save_metrics_to_json,
    print_metrics_summary, compare_models_metrics, calculate_confidence_intervals
)
from src.utils import (
    plot_dataset_stats, plot_training_loss, plot_reconstruction_comparison,
    plot_confusion_matrix, plot_tsne_comparison, save_model
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main():
    # 1. DATA LOADING
    print("\n1. DATA LOADING E PREPROCESSING")

    preprocessor = DataPreprocessor()
    X, y, images, target_names = preprocessor.load_dataset()

    stats = compute_dataset_statistics(X, y, target_names)
    print_dataset_statistics(stats)

    data = preprocessor.preprocess_data(X, y, test_size=config.TEST_SIZE)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    X_train_raw = data['X_train_raw']
    X_test_raw = data['X_test_raw']

    info = preprocessor.get_data_info()
    n_classes = info['n_classes']
    h, w = info['image_shape']
    n_features = info['n_features']

    print(f"\nPreprocessing completato:")
    print(f"- Train: {X_train.shape[0]} samples")
    print(f"- Test: {X_test.shape[0]} samples")
    print(f"- Features: {n_features}")
    print(f"- Classes: {n_classes}")
    print(f"- Image shape: {h}x{w}")

    # 2. EDA
    print("\n2. EXPLORATORY DATA ANALYSIS")

    data_dict = {
        'X_train': X_train_raw,
        'y_train': y_train,
        'target_names': target_names,
        'image_shape': (h, w)
    }
    plot_dataset_stats(data_dict)
    print("EDA completata")

    # 3. PCA ABLATION
    print("\n3. PCA via SVD - ABLATION STUDY")

    print(f"Range componenti: {config.COMPONENTS_RANGE}")

    pca_results = run_pca_experiments(
        X_train, X_test, y_train, y_test, h, w,
        n_components_list=config.COMPONENTS_RANGE,
        use_grid_search=False, verbose=True
    )

    best_pca_n = max(pca_results, key=lambda k: pca_results[k]['svm_test_acc'])
    print(f"\nMiglior PCA: {best_pca_n} componenti (Acc: {pca_results[best_pca_n]['svm_test_acc']:.4f})")

    final_pca_n = best_pca_n
    X_train_pca = pca_results[final_pca_n]['X_train_pca']
    X_test_pca = pca_results[final_pca_n]['X_test_pca']
    pca_model = pca_results[final_pca_n]['pca_model']

    # 4. AUTOENCODER ABLATION
    print("\n4. AUTOENCODER - ABLATION STUDY")

    print(f"Range latent dim: {config.COMPONENTS_RANGE}")

    ae_results = {}

    for latent_dim in config.COMPONENTS_RANGE:
        print(f"\n--- Latent dim: {latent_dim} ---")

        ae = FaceAutoencoder(
            input_dim=n_features,
            latent_dim=latent_dim,
            hidden_layers=config.AE_HIDDEN_LAYERS
        )

        X_tr_ae, X_val_ae, _, _ = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=config.RANDOM_STATE
        )

        ae, ae_history = ae.fit(
            X_tr_ae, X_val=X_val_ae,
            epochs=config.AE_EPOCHS,
            batch_size=config.AE_BATCH_SIZE,
            lr=config.AE_LEARNING_RATE,
            weight_decay=config.AE_WEIGHT_DECAY,
            patience=config.AE_PATIENCE,
            verbose=True
        )

        X_train_ae_feat = ae.transform(X_train)
        X_test_ae_feat = ae.transform(X_test)
        reconstruction_mse = ae.compute_reconstruction_error(X_test)

        svm_quick = SVC(kernel='linear', C=1.0, random_state=config.RANDOM_STATE)
        svm_quick.fit(X_train_ae_feat, y_train)
        acc_quick = np.mean(svm_quick.predict(X_test_ae_feat) == y_test)

        print(f"  Reconstruction MSE: {reconstruction_mse:.6f}")
        print(f"  Quick SVM Accuracy: {acc_quick:.4f}")

        ae_results[latent_dim] = {
            'model': ae, 'history': ae_history,
            'X_train_feat': X_train_ae_feat, 'X_test_feat': X_test_ae_feat,
            'reconstruction_mse': reconstruction_mse, 'svm_quick_acc': acc_quick
        }

        if latent_dim == config.COMPONENTS_FINAL:
            plot_training_loss(ae_history['train_loss'])

    best_ae_dim = max(ae_results, key=lambda k: ae_results[k]['svm_quick_acc'])
    print(f"\nMiglior AE: {best_ae_dim} dim (Acc: {ae_results[best_ae_dim]['svm_quick_acc']:.4f})")

    final_ae_dim = best_ae_dim
    X_train_ae = ae_results[final_ae_dim]['X_train_feat']
    X_test_ae = ae_results[final_ae_dim]['X_test_feat']
    ae_model = ae_results[final_ae_dim]['model']

    # Reconstruction comparison
    X_test_pca_rec = pca_model.inverse_transform(X_test_pca)
    X_test_ae_rec = ae_model.reconstruct(X_test)
    plot_reconstruction_comparison(X_test, X_test_pca_rec, X_test_ae_rec, h, w, n_images=5)

    # 5. CLASSIFICATION
    print("\n5. CLASSIFICAZIONE - SVM E NEURAL NETWORK")

    all_results = []

    feature_configs = [
        ('PCA', final_pca_n, X_train_pca, X_test_pca),
        ('Autoencoder', final_ae_dim, X_train_ae, X_test_ae)
    ]

    for feat_name, feat_dim, X_tr, X_te in feature_configs:
        print(f"\n--- {feat_name} ({feat_dim} dim) ---")

        # SVM
        print("\nSVM:")
        svm_model, svm_cv_score, svm_params = train_svm_with_grid_search(
            X_tr, y_train, param_grid=config.SVM_PARAM_GRID, verbose=True
        )

        y_pred_svm = svm_model.predict(X_te)
        y_score_svm = svm_model.predict_proba(X_te)

        metrics_svm = compute_classification_metrics(y_test, y_pred_svm, y_score_svm, target_names)
        metrics_svm['cv_score'] = svm_cv_score
        metrics_svm['best_params'] = svm_params
        metrics_svm['confidence_interval'] = calculate_confidence_intervals(y_test, y_pred_svm)

        print_metrics_summary(metrics_svm, f"SVM on {feat_name}")
        save_metrics_to_json(metrics_svm, f"{config.METRICS_PATH}/svm_{feat_name.lower()}.json")
        plot_confusion_matrix(y_test, y_pred_svm, target_names, f"SVM_{feat_name}")
        save_model(svm_model, f"svm_{feat_name.lower()}")

        all_results.append(('SVM', f'{feat_name}_{feat_dim}', metrics_svm))

        # Neural Network
        print("\nNeural Network:")
        X_tr_nn, X_val_nn, y_tr_nn, y_val_nn = train_test_split(
            X_tr, y_train, test_size=0.2, stratify=y_train, random_state=config.RANDOM_STATE
        )

        nn_model, nn_params, nn_val_score = train_nn_with_grid_search(
            X_tr_nn, y_tr_nn, n_classes, X_val=X_val_nn, y_val=y_val_nn,
            param_grid=config.NN_PARAM_GRID, verbose=True
        )

        y_pred_nn = nn_model.predict(X_te)
        y_score_nn = nn_model.predict_proba(X_te)

        metrics_nn = compute_classification_metrics(y_test, y_pred_nn, y_score_nn, target_names)
        metrics_nn['val_score'] = nn_val_score
        metrics_nn['best_params'] = nn_params
        metrics_nn['confidence_interval'] = calculate_confidence_intervals(y_test, y_pred_nn)

        print_metrics_summary(metrics_nn, f"NN on {feat_name}")
        save_metrics_to_json(metrics_nn, f"{config.METRICS_PATH}/nn_{feat_name.lower()}.json")
        plot_confusion_matrix(y_test, y_pred_nn, target_names, f"NN_{feat_name}")
        save_model(nn_model, f"nn_{feat_name.lower()}")

        all_results.append(('NN', f'{feat_name}_{feat_dim}', metrics_nn))

    # 6. COMPARISON
    print("\n6. CONFRONTO MODELLI")

    df_comparison = compare_models_metrics(all_results, save_path=f"{config.METRICS_PATH}/comparison.csv")
    print("\nTabella comparativa:")
    print(df_comparison.to_string(index=False))

    best_idx = df_comparison['accuracy'].idxmax()
    best = df_comparison.iloc[best_idx]
    print(f"\nMiglior modello: {best['model']} su {best['features']}")
    print(f"- Accuracy: {best['accuracy']:.4f}, F1: {best['f1_macro']:.4f}")

    # 7. FACE VERIFICATION
    print("\n7. FACE VERIFICATION")

    verification_results = run_verification_study(
        X_test_pca, X_test_ae, y_test,
        images=X_test_raw, image_shape=(h, w), target_names=target_names
    )

    df_verif = pd.DataFrame.from_dict(verification_results, orient='index')
    df_verif.to_csv(f"{config.METRICS_PATH}/verification.csv")
    print("\nRisultati verification:")
    print(df_verif.to_string())

    # 8. t-SNE
    print("\n8. t-SNE VISUALIZATION")

    n_tsne = min(500, len(X_train_pca))
    idx_tsne = np.random.choice(len(X_train_pca), n_tsne, replace=False)
    plot_tsne_comparison(X_train_pca[idx_tsne], X_train_ae[idx_tsne], y_train[idx_tsne], target_names)
    print("t-SNE completata")

    # 9. SAVE MODELS
    print("\n9. SALVATAGGIO MODELLI")

    joblib.dump(pca_model, f"{config.MODELS_PATH}/pca_{final_pca_n}.joblib")
    print(f"PCA salvato: pca_{final_pca_n}.joblib")

    torch.save(ae_model.state_dict(), f"{config.MODELS_PATH}/autoencoder_{final_ae_dim}.pt")
    print(f"Autoencoder salvato: autoencoder_{final_ae_dim}.pt")

    # RIEPILOGO
    print("\nRIEPILOGO FINALE")

    print(f"\nDATASET:")
    print(f"- {n_classes} classi, {len(y)} immagini totali")
    print(f"- Train/Test: {len(y_train)}/{len(y_test)}")
    print(f"- Features: {n_features} ({h}x{w})")

    print(f"\nRIDUZIONE DIMENSIONALE:")
    print(f"- PCA: {final_pca_n} componenti ({pca_results[final_pca_n]['variance_explained']*100:.1f}% varianza)")
    print(f"- AE: {final_ae_dim} latent dim (MSE: {ae_results[final_ae_dim]['reconstruction_mse']:.6f})")

    print(f"\nCLASSIFICAZIONE:")
    for model, features, metrics in all_results:
        print(f"  {model} on {features}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

    print(f"\nVERIFICATION:")
    for method, res in verification_results.items():
        print(f"- {method}: AUC={res['auc']:.4f}, EER={res['eer']:.4f}")

    print("\nTutti i risultati sono stati salvati.")


if __name__ == "__main__":
    main()