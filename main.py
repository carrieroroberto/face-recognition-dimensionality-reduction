# main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import config
from src.preprocessing import (
    DataPreprocessor, compute_dataset_statistics,
    print_dataset_statistics
)
from src.pca import run_pca_experiments
from src.autoencoder import ImprovedFaceAutoencoder
from src.classification import train_svm_with_grid_search, train_nn_with_grid_search
from src.verification import run_verification_study
from src.metrics import (
    compute_classification_metrics, compute_roc_curves,
    save_metrics_to_json, print_metrics_summary,
    compare_models_metrics, calculate_confidence_intervals
)
from src.comparative_study import compare_pca_vs_autoencoder_ablation
from src.utils import (
    plot_dataset_stats, plot_eigenfaces, plot_scree_plot,
    plot_training_loss, plot_reconstruction_comparison,
    plot_confusion_matrix, plot_roc_curves, plot_tsne_comparison,
    save_model
)

def main():
    config.print_config()
    
    print("1. DATA LOADING E PREPROCESSING")
    
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

    print(f"Preprocessing completato:")
    print(f"  - Train: {X_train.shape[0]} samples")
    print(f"  - Test: {X_test.shape[0]} samples")
    print(f"  - Features: {n_features}")
    print(f"  - Classes: {n_classes}")
    print(f"  - Image shape: {h}x{w}")

    print("2. EXPLORATORY DATA ANALYSIS")

    data_dict_for_plots = {
        'X_train': X_train_raw,
        'y_train': y_train,
        'target_names': target_names,
        'image_shape': (h, w)
    }
    plot_dataset_stats(data_dict_for_plots)

    print("EDA completata e grafici salvati")

    print("3. PCA via SVD - ABLATION STUDY")

    print("Testando PCA con diversi numeri di componenti...")
    print(f"Range: {config.PCA_COMPONENTS_RANGE}\n")

    pca_results = run_pca_experiments(
        X_train, X_test, y_train, y_test,
        h, w,
        n_components_list=config.PCA_COMPONENTS_RANGE,
        use_grid_search=False,
        verbose=True
    )

    best_pca_n = max(pca_results, key=lambda k: pca_results[k]['svm_test_acc'])
    print(f"\nMiglior PCA: {best_pca_n} componenti "
          f"(Accuracy: {pca_results[best_pca_n]['svm_test_acc']:.4f})")

    final_pca_n = config.N_COMPONENTS_PCA_FINAL
    print(f"Usando {final_pca_n} componenti per il modello finale\n")

    X_train_pca_final = pca_results[final_pca_n]['X_train_pca']
    X_test_pca_final = pca_results[final_pca_n]['X_test_pca']
    pca_model_final = pca_results[final_pca_n]['pca_model']

    print("4. AUTOENCODER - ABLATION STUDY")

    print("Testando Autoencoder con diverse dimensioni latenti...")
    print(f"Range: {config.AE_LATENT_DIM_RANGE}\n")

    ae_results = {}

    for latent_dim in config.AE_LATENT_DIM_RANGE:
        print(f"\n--- Autoencoder con latent_dim={latent_dim} ---")

        ae = ImprovedFaceAutoencoder(
            input_dim=n_features,
            latent_dim=latent_dim,
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
            verbose=True
        )

        X_train_ae_feat = ae.transform(X_train)
        X_test_ae_feat = ae.transform(X_test)

        reconstruction_mse = ae.compute_reconstruction_error(X_test)

        from sklearn.svm import SVC
        svm_quick = SVC(kernel='linear', C=1.0, random_state=config.RANDOM_STATE)
        svm_quick.fit(X_train_ae_feat, y_train)
        y_pred_quick = svm_quick.predict(X_test_ae_feat)
        acc_quick = np.mean(y_pred_quick == y_test)

        print(f"Reconstruction MSE: {reconstruction_mse:.6f}")
        print(f"Quick SVM Accuracy: {acc_quick:.4f}")

        ae_results[latent_dim] = {
            'model': ae,
            'history': ae_history,
            'X_train_feat': X_train_ae_feat,
            'X_test_feat': X_test_ae_feat,
            'reconstruction_mse': reconstruction_mse,
            'svm_quick_acc': acc_quick
        }

        if latent_dim == config.LATENT_DIM_AE_FINAL:
            plot_training_loss(ae_history['train_loss'])

    best_ae_dim = max(ae_results, key=lambda k: ae_results[k]['svm_quick_acc'])
    print(f"\nMiglior AE: {best_ae_dim} dim latenti "
          f"(Accuracy: {ae_results[best_ae_dim]['svm_quick_acc']:.4f})")

    final_ae_dim = config.LATENT_DIM_AE_FINAL
    print(f"Usando {final_ae_dim} dim latenti per il modello finale\n")

    X_train_ae_final = ae_results[final_ae_dim]['X_train_feat']
    X_test_ae_final = ae_results[final_ae_dim]['X_test_feat']
    ae_model_final = ae_results[final_ae_dim]['model']

    X_test_pca_rec = pca_model_final.inverse_transform(X_test_pca_final)
    X_test_ae_rec = ae_model_final.reconstruct(X_test)
    plot_reconstruction_comparison(X_test, X_test_pca_rec, X_test_ae_rec, h, w, n_images=5)

    print("5. STUDIO COMPARATIVO PCA vs AUTOENCODER")
    print("Confronto dettagliato con tutte le dimensioni...\n")

    comparison_range = [dim for dim in config.PCA_COMPONENTS_RANGE
                       if dim in config.AE_LATENT_DIM_RANGE or dim <= max(config.AE_LATENT_DIM_RANGE)]

    print("Studio comparativo completato (se abilitato)")

    print("6. CLASSIFICAZIONE - SVM E NEURAL NETWORK")

    all_results = []

    feature_configs = [
        ('PCA', final_pca_n, X_train_pca_final, X_test_pca_final),
        ('Autoencoder', final_ae_dim, X_train_ae_final, X_test_ae_final)
    ]

    for feat_name, feat_dim, X_tr, X_te in feature_configs:
        print(f"Features: {feat_name} ({feat_dim} dimensioni)")

        print(f"Training SVM su {feat_name} features...")

        svm_model, svm_cv_score, svm_params = train_svm_with_grid_search(
            X_tr, y_train,
            param_grid=config.SVM_PARAM_GRID,
            cv_folds=config.SVM_CV_FOLDS,
            verbose=True
        )

        y_pred_svm = svm_model.predict(X_te)
        y_score_svm = svm_model.predict_proba(X_te)

        metrics_svm = compute_classification_metrics(
            y_test, y_pred_svm, y_score_svm, target_names
        )
        metrics_svm['cv_score'] = svm_cv_score
        metrics_svm['best_params'] = svm_params

        ci_svm = calculate_confidence_intervals(y_test, y_pred_svm)
        metrics_svm['confidence_interval'] = ci_svm

        print_metrics_summary(metrics_svm, f"SVM on {feat_name}")

        save_metrics_to_json(
            metrics_svm,
            f"{config.METRICS_PATH}/svm_{feat_name.lower()}_{feat_dim}.json"
        )

        plot_confusion_matrix(y_test, y_pred_svm, target_names, f"SVM_{feat_name}")
        plot_roc_curves(y_test, y_score_svm, n_classes, f"SVM_{feat_name}")

        save_model(svm_model, f"svm_{feat_name.lower()}_{feat_dim}")

        all_results.append((f'SVM', f'{feat_name}_{feat_dim}', metrics_svm))

        print(f"\nTraining Neural Network su {feat_name} features...")

        from sklearn.model_selection import train_test_split
        X_tr_nn, X_val_nn, y_tr_nn, y_val_nn = train_test_split(
            X_tr, y_train, test_size=0.2, stratify=y_train,
            random_state=config.RANDOM_STATE
        )

        nn_model, nn_params, nn_val_score = train_nn_with_grid_search(
            X_tr_nn, y_tr_nn, n_classes,
            X_val=X_val_nn, y_val=y_val_nn,
            param_grid=config.NN_PARAM_GRID,
            verbose=True
        )

        y_pred_nn = nn_model.predict(X_te)
        y_score_nn = nn_model.predict_proba(X_te)

        metrics_nn = compute_classification_metrics(
            y_test, y_pred_nn, y_score_nn, target_names
        )
        metrics_nn['val_score'] = nn_val_score
        metrics_nn['best_params'] = nn_params

        ci_nn = calculate_confidence_intervals(y_test, y_pred_nn)
        metrics_nn['confidence_interval'] = ci_nn

        print_metrics_summary(metrics_nn, f"Neural Network on {feat_name}")

        save_metrics_to_json(
            metrics_nn,
            f"{config.METRICS_PATH}/nn_{feat_name.lower()}_{feat_dim}.json"
        )

        plot_confusion_matrix(y_test, y_pred_nn, target_names, f"NN_{feat_name}")
        plot_roc_curves(y_test, y_score_nn, n_classes, f"NN_{feat_name}")

        save_model(nn_model, f"nn_{feat_name.lower()}_{feat_dim}")
        
        all_results.append((f'NN', f'{feat_name}_{feat_dim}', metrics_nn))

    print("7. CONFRONTO FINALE TRA TUTTI I MODELLI")

    df_comparison = compare_models_metrics(
        all_results,
        save_path=f"{config.METRICS_PATH}/final_comparison.csv"
    )

    print("\nTABELLA COMPARATIVA FINALE:")
    print(df_comparison.to_string(index=False))

    best_idx = df_comparison['accuracy'].idxmax()
    best_model = df_comparison.iloc[best_idx]
    print(f"\nMIGLIOR MODELLO:")
    print(f"{best_model['model']} su {best_model['features']}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"F1 (macro): {best_model['f1_macro']:.4f}")

    print("8. FACE VERIFICATION")

    print("Generazione coppie per face verification...")
    print(f"Metriche di similarità: {config.VERIFICATION_METRICS}\n")

    verification_results = run_verification_study(
        X_test_pca_final, X_test_ae_final, y_test,
        images=X_test_raw,
        image_shape=(h, w),
        target_names=target_names
    )

    df_verif = pd.DataFrame.from_dict(verification_results, orient='index')
    df_verif.to_csv(f"{config.METRICS_PATH}/verification_results.csv")

    print("\nRISULTATI FACE VERIFICATION:")
    print(df_verif.to_string())

    print("\nFace verification completata")

    print("9. t-SNE VISUALIZATION")

    print("Generazione visualizzazione t-SNE...")

    n_samples_tsne = min(500, len(X_train_pca_final))
    indices_tsne = np.random.choice(len(X_train_pca_final), n_samples_tsne, replace=False)

    plot_tsne_comparison(
        X_train_pca_final[indices_tsne],
        X_train_ae_final[indices_tsne],
        y_train[indices_tsne],
        target_names
    )

    print("t-SNE visualization completata")

    print("10. SALVATAGGIO MODELLI")

    import joblib
    joblib.dump(pca_model_final, f"{config.MODELS_PATH}/pca_final_{final_pca_n}.joblib")
    print(f"PCA model salvato")

    import torch
    torch.save(ae_model_final.state_dict(),
               f"{config.MODELS_PATH}/autoencoder_final_{final_ae_dim}.pt")
    print(f"\nAutoencoder model salvato")

    joblib.dump(preprocessor.scaler, f"{config.MODELS_PATH}/scaler.joblib")
    print(f"\nPreprocessor scaler salvato")

    print("\nRIEPILOGO FINALE")

    print("\nDATASET")
    print(f"- Samples: {n_classes} classes, {len(y)} total images")
    print(f"- Train/Test: {len(y_train)}/{len(y_test)} split")
    print(f"- Features: {n_features} ({h}x{w} images)")

    print(f"\nRIDUZIONE DIMENSIONALE")
    print(f"- PCA: {final_pca_n} componenti "
          f"({pca_results[final_pca_n]['variance_explained']*100:.1f}% varianza)")
    print(f"- Autoencoder: {final_ae_dim} dim latenti "
          f"(MSE: {ae_results[final_ae_dim]['reconstruction_mse']:.6f})")

    print(f"\nCLASSIFICAZIONE")
    for model, features, metrics in all_results:
        print(f"- {model} on {features}: "
              f"Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1_macro']:.4f}")

    print(f"\nFACE VERIFICATION")
    for method, results in verification_results.items():
        print(f"- {method}: AUC={results['auc']:.4f}, EER={results['eer']:.4f}")

    print("\nTutti i risultati sono stati salvati.")

if __name__ == "__main__":
    main()