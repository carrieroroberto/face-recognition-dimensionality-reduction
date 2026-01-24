from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
import config

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.data_info = {}

    def load_dataset(self, min_faces=config.MIN_FACES_PER_PERSON, resize=config.RESIZE_RATIO):
        print(f"Loading LFW dataset (min_faces={min_faces}, resize={resize})...")
        
        lfw_people = fetch_lfw_people(
            min_faces_per_person=min_faces,
            resize=resize,
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

        print(f"Dataset loaded: {n_samples} samples, {len(target_names)} classes")
        print(f"Image shape: {h}x{w} = {X.shape[1]} features")

        return X, y, images, target_names

    def preprocess_data(self, X, y, test_size=config.TEST_SIZE):
        print("\nPreprocessing data...")
        
        X_normalized = X / 255.0

        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test_size, stratify=y)

        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        print("Applying z-score standardization...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
        print(f"Test mean: {X_test_scaled.mean():.4f}, std: {X_test_scaled.std():.4f}")

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
            raise ValueError("Scaler not fitted. Run preprocess_data first.")
        return self.scaler.inverse_transform(X_scaled)

    def get_data_info(self):
        return self.data_info