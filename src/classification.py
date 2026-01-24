from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
import config

def train_svm_with_grid_search(X_train, y_train, param_grid, cv_folds=config.CV_FOLDS):
    print("\n--- SVM GRID SEARCH STARTING ---")
    print(f"Hyperparameters grid: {param_grid}")
    print(f"Cross-Validation: {cv_folds} folds")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True)
    svm = SVC(probability=True, class_weight='balanced')

    grid = GridSearchCV(
        estimator=svm, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=True
    )
    
    grid.fit(X_train, y_train)

    print("\n--- SVM GRID SEARCH COMPLETED ---")
    print(f"Best Params: {grid.best_params_}")
    print(f"Best CV Score: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_score_, grid.best_params_

class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers=(64,), epochs=200, lr=0.01, 
                 weight_decay=0.001, dropout_rate=0.2, batch_size=32, verbose=True):
        
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.verbose = verbose
        
        self.model_ = None
        self.classes_ = None
        self.history_ = {}

    def _build_model(self, input_dim, n_classes):
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))
        model = nn.Sequential(*layers)
        
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        return model

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = self._build_model(n_features, n_classes)

        class_counts = np.bincount(y)
        safe_counts = np.maximum(class_counts, 1)
        weights = 1.0 / safe_counts
        weights = weights / weights.sum() * n_classes
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.history_ = {'loss': [], 'acc': []}
        self.model_.train()

        for epoch in range(self.epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model_(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            
            avg_loss = total_loss / len(dataset)
            acc = correct / total
            self.history_['loss'].append(avg_loss)
            self.history_['acc'].append(acc)

            if self.verbose and (epoch == 0 or (epoch + 1) % 20 == 0):
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")
        
        return self

    def predict(self, X):     
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(torch.FloatTensor(X))
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(torch.FloatTensor(X))
            probs = torch.softmax(outputs, dim=1)
        return probs.numpy()

def train_nn_with_grid_search(X_train, y_train, param_grid, cv_folds=config.CV_FOLDS):
    print("\n--- NEURAL NET GRID SEARCH STARTING ---")
    print(f"Hyperparameters Grid: {param_grid}")
    print(f"Cross-Validation: {cv_folds} Folds")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True)
    model = NeuralNetwork()
    
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1, 
        verbose=True
    )
    
    grid.fit(X_train, y_train)

    print("\n--- NEURAL NET GRID SEARCH COMPLETED ---")
    print(f"Best Params: {grid.best_params_}")
    print(f"Best CV Score: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_score_, grid.best_params_