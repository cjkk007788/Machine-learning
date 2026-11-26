import numpy as np

def poly_kernel(X, Z, C = 0 , d = 2):

    return (X @ Z.T + C)**d

def make_poly_kernel(C= 0,d=2):
    
    return lambda X, Z: poly_kernel(X, Z, C, d)


class Kernel_trick:
    def __init__(self, kernel_fn, learning_rate=1.0, epochs=500):
        self.kernel_fn = kernel_fn
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = None
        self.X_train = None
        self.loss_history = []

    def kernel_matrix(self,X,Z):
        return self.kernel_fn(X,Z)


    def fit(self,X,y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n = X.shape[0]

        self.X_train = X

        K = self.kernel_matrix(X,X)
        
        self.beta = np.zeros(n)
        self.loss_history = []

        for _ in range(self.epochs):
            error = y - K @ self.beta
            self.beta = self.beta + self.learning_rate*error
            self.loss_history.append(0.5 * np.mean(error ** 2))

        return self

    def predict(self,X_new):
        X_new = np.asarray(X_new, dtype=float)
        K_new = self.kernel_matrix(X_new, self.X_train)
        return K_new @ self.beta

