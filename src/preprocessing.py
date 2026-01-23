# src/preprocessing.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
import config

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.data_info = {}

    def load_dataset(self):
        print("Caricamento dataset LFW...")
        lfw_people = fetch_lfw_people(
            min_faces_per_person=config.MIN_FACES_PER_PERSON,
            resize=config.RESIZE_FACTOR,
            color=False
        )

        X = lfw_people.data
        y = lfw_people.target
        target_names = lfw_people.target_names
        images = lfw_people.images
        n_samples, h, w = images.shape

        self.data_info = {
            'n_samples': n_samples,
            'n_features': X.shape[1],
            'n_classes': len(target_names),
            'image_shape': (h, w),
            'target_names': target_names
        }

        print(f"Dataset caricato: {n_samples} campioni, {len(target_names)} classi")
        print(f"Forma immagini: {h}x{w} = {X.shape[1]} features")

        return X, y, images, target_names

    def preprocess_data(self, X, y, test_size=0.25, random_state=None):
        if random_state is None:
            random_state = config.RANDOM_STATE

        print("\nPreprocessing dati...")
        X_normalized = X / 255.0

        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        print(f"Train set: {X_train.shape[0]} campioni")
        print(f"Test set: {X_test.shape[0]} campioni")

        print("\nStandardizzazione (z-score)...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Train - Media: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.4f}")
        print(f"Test  - Media: {X_test_scaled.mean():.6f}, Std: {X_test_scaled.std():.4f}")

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_raw': X_train,
            'X_test_raw': X_test
        }

    def inverse_transform(self, X_scaled):
        if self.scaler is None:
            raise ValueError("Scaler non ancora fittato. Esegui preprocess_data prima.")
        return self.scaler.inverse_transform(X_scaled)

    def get_data_info(self):
        return self.data_info


def compute_dataset_statistics(X, y, target_names):
    unique, counts = np.unique(y, return_counts=True)

    stats = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(target_names),
        'pixel_min': X.min(),
        'pixel_max': X.max(),
        'pixel_mean': X.mean(),
        'pixel_std': X.std(),
        'class_distribution': dict(zip([target_names[i] for i in unique], counts)),
        'class_counts': counts,
        'min_samples_per_class': counts.min(),
        'max_samples_per_class': counts.max(),
        'mean_samples_per_class': counts.mean(),
        'std_samples_per_class': counts.std()
    }

    return stats

def print_dataset_statistics(stats):
    print("\nSTATISTICHE DESCRITTIVE DEL DATASET")
    print(f"Numero totale campioni: {stats['n_samples']}")
    print(f"Numero features: {stats['n_features']}")
    print(f"Numero classi: {stats['n_classes']}")
    print(f"\nStatistiche pixel:")
    print(f"  Range: [{stats['pixel_min']:.4f}, {stats['pixel_max']:.4f}]")
    print(f"  Media: {stats['pixel_mean']:.4f}")
    print(f"  Std Dev: {stats['pixel_std']:.4f}")
    print(f"\nDistribuzione classi:")
    print(f"  Min campioni per classe: {stats['min_samples_per_class']}")
    print(f"  Max campioni per classe: {stats['max_samples_per_class']}")
    print(f"  Media campioni per classe: {stats['mean_samples_per_class']:.1f}")
    print(f"  Std Dev: {stats['std_samples_per_class']:.1f}")