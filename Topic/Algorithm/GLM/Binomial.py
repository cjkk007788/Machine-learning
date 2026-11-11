import numpy as np

class BinomialGlM:

    def __init__(self,learning_rate = 0.05 ,n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.beta = None
        self.bias = None

    def sigmoid(self,Z):
        z = np.clip(z, -500, 500)
        return 1.0 / 1.0 + np.exp(-z)

    def fit(self,X,y,n_trials):

        #x - (N_samples x n_features)
        #x - > matrix

        n_samples, n_features = X.shape

        #beta = vector

        self.beta = np.zeros(n_features)
        self.bias = 0.0

        for i in range(n_trials):
            
            Z = np.dot(X,self.beta) + self.bias
            #x 들에 대한 확률 분포
            p = self.sigmoid(Z)

            mu = n_trials * p
            error = mu - y
            
            d_beta = np.dot(X.T, error) / n_samples
            d_bias = np.sum(error) / n_samples


            self.beta -= d_beta*self.learning_rate
            self.bias -= d_bias*self.learning_rate
    def predict_proba(self, X):
        # 성공 확률 p 반환
        eta = np.dot(X, self.beta) + self.bias
        return self._sigmoid(eta)

    def predict_count(self, X, n_trials):
        # 예상 성공 횟수 반환
        return self.predict_proba(X) * n_trials

