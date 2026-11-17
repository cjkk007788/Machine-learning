import numpy as np
import matplotlib.pyplot as plt
class GaussianGLM:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape
        self.beta = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            error = X @ self.beta + self.bias - y
            self.loss_history.append(0.5 * np.mean(error ** 2))
            self.beta -= self.learning_rate * (X.T @ error) / n_samples
            self.bias -= self.learning_rate * error.mean()
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.beta + self.bias