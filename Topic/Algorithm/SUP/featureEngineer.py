import numpy as np

def raw_features(X):
    # [1, x1, x2]
    return np.column_stack([np.ones(len(X)), X])

def minimal_poly(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([np.ones(len(X)), x1**2, x2**2])
        
def poly_features(X):
    # [1, x1, x2, x1², x2², x1*x2]
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([np.ones(len(X)), x1, x2, x1**2, x2**2, x1 * x2])
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class FeatureEngineer:
    def __init__(self, feature_fn, learning_rate=1.0, epochs=5000):
        self.feature_fn = feature_fn
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
        self.loss_history = []

    def mapping_function(self, X):
        return self.feature_fn(np.asarray(X, dtype=float))

    def explicit_kernel(self, X, Y):
        return self.mapping_function(X) @ self.mapping_function(Y).T

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        phi = self.mapping_function(X)      # 한 번만 계산
        n, f = phi.shape

        self.theta = np.zeros(f)
        self.loss_history = []
        eps = 1e-12

        for _ in range(self.epochs):
            p = sigmoid(phi @ self.theta)
            grad = phi.T @ (y - p) / n
            self.theta = self.theta + self.learning_rate * grad

            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        return sigmoid(self.mapping_function(X) @ self.theta)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(float)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)