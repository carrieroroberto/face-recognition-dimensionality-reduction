import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import (
    load_and_preprocess_data, plot_dataset_stats, plot_training_loss, plot_reconstruction_comparison,
    plot_confusion_matrix, plot_roc_curves, plot_tsne_comparison, save_model
)
from src.pca import run_pca_experiments
from src.autoencoder import FaceAutoencoder
from src.classification import train_svm, NeuralNetwork
from src.verification import run_verification_study
import config

# =========================================
# MAIN
# =========================================
def main():
    print("\n=== CARICAMENTO E PREPROCESSING DATI ===")
    data = load_and_preprocess_data()
    plot_dataset_stats(data)
    h, w = data['image_shape']

    # =========================================
    # PCA modulare
    # =========================================
    print("\n=== PCA SVD: Esperimenti con diversi n_components ===")
    n_components_list = [5, 10, 20, 50, 100, 150]
    pca_results = run_pca_experiments(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test'],
        h, w, n_components_list
    )

    # Seleziona il migliore per SVM
    best_n = max(pca_results, key=lambda k: pca_results[k]['svm_acc'])
    print(f"Numero di componenti PCA scelto: {best_n}")

    # Feature PCA standardizzate migliori
    X_train_pca_scaled = pca_results[best_n]['X_train_scaled']
    X_test_pca_scaled = pca_results[best_n]['X_test_scaled']
    best_pca_model = pca_results[best_n]['pca_model']
    X_test_pca_reduced = best_pca_model.transform(data['X_test'])
    X_test_pca_rec = best_pca_model.inverse_transform(X_test_pca_reduced)

    # =========================================
    # AUTOENCODER
    # =========================================
    print("\n=== AUTOENCODER ===")
    input_dim = data['X_train'].shape[1]
    ae_model = FaceAutoencoder(input_dim=input_dim, latent_dim=best_n)
    ae_model, loss_history = ae_model.fit(data['X_train'], epochs=200, batch_size=32, lr=1e-3)
    plot_training_loss(loss_history)

    # Trasformazioni con AE
    X_train_ae = ae_model.transform(data['X_train'])
    X_test_ae = ae_model.transform(data['X_test'])
    X_test_ae_rec = ae_model.reconstruct(data['X_test'])

    plot_reconstruction_comparison(data['X_test'], X_test_pca_rec, X_test_ae_rec, h, w)

    # =========================================
    # STANDARDIZZAZIONE FEATURE PER CLASSIFICAZIONE
    # =========================================
    scaler_ae = StandardScaler().fit(X_train_ae)
    X_train_ae_scaled = scaler_ae.transform(X_train_ae)
    X_test_ae_scaled = scaler_ae.transform(X_test_ae)

    feature_sets = {
        "PCA": (X_train_pca_scaled, X_test_pca_scaled),
        "Autoencoder": (X_train_ae_scaled, X_test_ae_scaled)
    }

    results = []
    n_classes = len(data['target_names'])

    # =========================================
    # CLASSIFICAZIONE
    # =========================================
    for feat_name, (X_tr, X_te) in feature_sets.items():
        print(f"\n=== CLASSIFICAZIONE: {feat_name} ===")

        # --- SVM ---
        svm_model = train_svm(X_tr, data['y_train'])
        y_pred = svm_model.predict(X_te)
        y_score = svm_model.predict_proba(X_te)
        acc = np.round(np.mean(y_pred == data['y_test']), 4)
        results.append({"Feature": feat_name, "Classifier": "SVM", "Accuracy": acc})
        plot_confusion_matrix(data['y_test'], y_pred, data['target_names'], f"SVM {feat_name}")
        plot_roc_curves(data['y_test'], y_score, n_classes, f"SVM {feat_name}")
        save_model(svm_model, f"svm_{feat_name.lower()}")

        # --- NN ---
        nn_model = NeuralNetwork(input_dim=X_tr.shape[1], n_classes=n_classes)
        nn_model.fit(X_tr, data['y_train'], epochs=200, batch_size=32, lr=1e-3)
        y_pred = nn_model.predict(X_te)
        y_score = nn_model.predict_proba(X_te)
        acc = np.round(np.mean(y_pred == data['y_test']), 4)
        results.append({"Feature": feat_name, "Classifier": "NN", "Accuracy": acc})

        # Visualizzazioni
        plot_confusion_matrix(data['y_test'], y_pred, data['target_names'], f"NN {feat_name}")
        plot_roc_curves(data['y_test'], y_score, n_classes, f"NN {feat_name}")
        save_model(nn_model, f"nn_{feat_name.lower()}")

    # =========================================
    # Salvataggio tabella comparativa
    # =========================================
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(config.OUTPUT_PATH, "classification_metrics.csv"), index=False)
    print("\n=== TABELLA METRICHE CLASSIFICAZIONE ===")
    print(df_results)

    # =========================================
    # FACE VERIFICATION
    # =========================================
    print("\n=== FACE VERIFICATION ===")
    verification_metrics = run_verification_study(X_test_pca_scaled, X_test_ae_scaled, data['y_test'])
    df_verif = pd.DataFrame.from_dict(verification_metrics, orient='index')
    df_verif.to_csv(os.path.join(config.OUTPUT_PATH, "verification_metrics.csv"))
    print("\n=== VERIFICATION METRICS ===")
    print(df_verif)

    # =========================================
    # t-SNE per visualizzazione features
    # =========================================
    print("\n=== t-SNE VISUALIZATION ===")
    plot_tsne_comparison(X_train_pca_scaled, X_train_ae_scaled, data['y_train'], data['target_names'])

if __name__ == "__main__":
    main()