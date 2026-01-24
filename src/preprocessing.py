"""
This module handles loading and preprocessing of the Labeled Faces in the Wild (LFW)
dataset.

It provides functionality for:
- Loading face images with configurable filtering criteria
- Normalizing pixel values to [0, 1] range
- Applying z-score standardization for zero mean and unit variance
- Splitting data into training and test sets with stratified sampling
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
import config


class DataPreprocessor:
    """
    Data preprocessing pipeline for face recognition tasks.

    Encapsulates all preprocessing steps including dataset loading,
    normalization, standardization, and train/test splitting.

    Attributes:
        scaler: StandardScaler instance for z-score normalization
        data_info: Dictionary containing dataset metadata
    """

    def __init__(self):
        """Initialize the preprocessor with empty state."""
        self.scaler = None
        self.data_info = {}

    def load_dataset(self, min_faces=config.MIN_FACES_PER_PERSON, resize=config.RESIZE_RATIO):
        """
        Load the LFW face dataset with specified parameters.

        Fetches the Labeled Faces in the Wild dataset from scikit-learn,
        filtering to include only individuals with a minimum number of images.

        Args:
            min_faces: Minimum number of images required per person
            resize: Scaling factor for image dimensions (0.5 = half size)

        Returns:
            tuple: (X, y, images, target_names) containing flattened features,
                   labels, image arrays, and class name mapping
        """
        print(f"Loading LFW dataset (min_faces={min_faces}, resize={resize})...")

        # Fetch dataset with specified criteria
        lfw_people = fetch_lfw_people(
            min_faces_per_person=min_faces,
            resize=resize,
            color=False  # Grayscale images
        )

        # Extract data components
        X = lfw_people.data  # Flattened pixel arrays
        y = lfw_people.target  # Integer class labels
        target_names = lfw_people.target_names  # Person names
        images = lfw_people.images  # 2D image arrays

        n_samples, h, w = images.shape

        # Store dataset metadata for later use
        self.data_info = {
            "n_samples": n_samples,
            "n_features": X.shape[1],
            "n_classes": len(target_names),
            "image_shape": (h, w),
            "target_names": target_names
        }

        print(f"Dataset loaded: {n_samples} samples, {len(target_names)} classes")
        print(f"Image shape: {h}x{w} = {X.shape[1]} features")

        return X, y, images, target_names

    def preprocess_data(self, X, y, test_size=config.TEST_SIZE):
        """
        Apply preprocessing transformations to the data.

        Performs the following steps:
        1. Normalize pixel values to [0, 1] range
        2. Split data into training and test sets (stratified)
        3. Apply z-score standardization (fit on train, transform both)

        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            test_size: Proportion of data to use for testing

        Returns:
            dict: Dictionary containing preprocessed train/test splits
                  with both standardized and raw versions
        """
        print("\nPreprocessing data...")

        # Normalize pixel values from [0, 255] to [0, 1]
        X_normalized = X / 255.0

        # Stratified train/test split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test_size, stratify=y)

        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Apply z-score standardization: (x - mean) / std
        print("Applying z-score standardization...")

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)  # Fit on training data
        X_test_scaled = self.scaler.transform(X_test)  # Transform test data

        # Verify standardization results
        print(f"Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
        print(f"Test mean: {X_test_scaled.mean():.4f}, std: {X_test_scaled.std():.4f}")

        return {
            "X_train": X_train_scaled,  # Standardized training features
            "X_test": X_test_scaled,  # Standardized test features
            "y_train": y_train,
            "y_test": y_test,
            "X_train_raw": X_train,  # Non-standardized for visualization
            "X_test_raw": X_test
        }

    def inverse_transform(self, X_scaled):
        """
        Reverse the z-score standardization.

        Args:
            X_scaled: Standardized data array

        Returns:
            numpy.ndarray: Data in original scale

        Raises:
            ValueError: If scaler has not been fitted yet
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Run preprocess_data first.")
        return self.scaler.inverse_transform(X_scaled)

    def get_data_info(self):
        """
        Retrieve stored dataset metadata.

        Returns:
            dict: Dataset information including dimensions and class count
        """
        return self.data_info