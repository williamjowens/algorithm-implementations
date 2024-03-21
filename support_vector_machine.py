import numpy as np
import matplotlib.pyplot as plt

# RBF Support Vector Machine class
class RBF_SVM:
    def __init__(self, C=1.0, gamma='scale', tol=1e-3, max_iter=-1, kernel_cache_size=500):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.kernel_cache_size = kernel_cache_size
        self.alpha = None
        self.b = None
        self.X = None
        self.y = None
        self.n_sv = None
        self.kernel_cache = {}

    def _compute_gamma(self, X):
        if self.gamma == 'scale':
            return 1.0 / (X.shape[1] * np.var(X))
        elif self.gamma == 'auto':
            return 1.0 / X.shape[1]
        else:
            return self.gamma

    def kernel(self, X1, X2):
        gamma = self._compute_gamma(self.X)
        key = tuple(X1) + tuple(X2)
        if key not in self.kernel_cache:
            result = np.exp(-gamma * np.sum((X1 - X2) ** 2))
            if len(self.kernel_cache) < self.kernel_cache_size:
                self.kernel_cache[key] = result
        else:
            result = self.kernel_cache[key]
        return result

    def _objective(self, alpha):
        return 0.5 * np.dot(alpha, np.dot(self.Q, alpha)) - np.sum(alpha)

    def _constraint(self, alpha):
        return np.dot(alpha, self.y)

    def _compute_bounds(self, C, alpha_i, alpha_j, y_i, y_j):
        if y_i != y_j:
            L = max(0, alpha_j - alpha_i)
            H = min(C, C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_i + alpha_j - C)
            H = min(C, alpha_i + alpha_j)
        return L, H

    def _update_alpha(self, i, j):
        E_i = np.dot(self.alpha * self.y, self.K[:, i]) + self.b - self.y[i]
        E_j = np.dot(self.alpha * self.y, self.K[:, j]) + self.b - self.y[j]

        alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
        L, H = self._compute_bounds(self.C, self.alpha[i], self.alpha[j], self.y[i], self.y[j])

        eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
        if eta >= 0:
            return 0

        self.alpha[j] -= self.y[j] * (E_i - E_j) / eta
        self.alpha[j] = np.clip(self.alpha[j], L, H)

        if abs(self.alpha[j] - alpha_j_old) < 1e-5:
            return 0

        self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

        b1 = self.b - E_i - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
        b2 = self.b - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]

        if 0 < self.alpha[i] < self.C:
            self.b = b1
        elif 0 < self.alpha[j] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        return 1

    def _examine_example(self, i):
        E_i = np.dot(self.alpha * self.y, self.K[:, i]) + self.b - self.y[i]

        if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (self.y[i] * E_i > self.tol and self.alpha[i] > 0):
            non_bound_indices = np.nonzero((self.alpha > 0) & (self.alpha < self.C))[0]
            if len(non_bound_indices) > 1:
                j = non_bound_indices[np.argmax(np.abs(E_i - np.dot(self.alpha * self.y, self.K[:, non_bound_indices]) - self.b + self.y[non_bound_indices]))]
                if self._update_alpha(i, j):
                    return 1
            
            for j in np.roll(np.where((self.alpha > 0) & (self.alpha < self.C))[0], np.random.choice(np.arange(self.m))):
                if self._update_alpha(i, j):
                    return 1
            
            for j in np.roll(np.arange(self.m), np.random.choice(np.arange(self.m))):
                if self._update_alpha(i, j):
                    return 1
        
        return 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X = X
        self.y = y
        self.m = n_samples

        # Initialize alpha with random values between 0 and C
        self.alpha = np.random.uniform(0, self.C, size=n_samples)
        self.b = 0

        self.K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            self.K[:, i] = np.array([self.kernel(X[i], x) for x in X])

        self.Q = self.y[:, np.newaxis] * self.K * self.y

        n_iter = 0
        examine_all = True
        num_changed = 0
        while (n_iter < self.max_iter and num_changed > 0) or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(n_samples):
                    num_changed += self._examine_example(i)
            else:
                non_bound_indices = np.nonzero((self.alpha > 0) & (self.alpha < self.C))[0]
                for i in non_bound_indices:
                    num_changed += self._examine_example(i)
            
            n_iter += 1
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        self.n_sv = np.sum(self.alpha > 0)

    def predict(self, X):
        y_pred = np.where(np.array([np.sum(self.alpha * self.y * np.array([self.kernel(x, x_train) for x_train in self.X])) for x in X]) + self.b >= 0, 1, 0)
        return y_pred
    

##################
# Implementation #
##################
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # Load the breast cancer dataset
    X, y = load_breast_cancer(return_X_y=True)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the custom RBF SVM classifier
    svm = RBF_SVM(C=1.0, gamma='scale', tol=1e-3, max_iter=1000, kernel_cache_size=500)

    # Train the classifier
    svm.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = svm.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    roc_auc = roc_auc_score(y_test, y_pred)

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")