import numpy as np

class GaussianGLM:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.beta = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 가중치(beta)와 절편(bias) 초기화
        self.beta = np.zeros(n_features)
        self.bias = 0.0

        # 경사하강법(Gradient Descent)
        for i in range(self.n_iterations):
            # 1. 선형 예측 (Identity Link: mu = X*beta + bias)
            mu = np.dot(X, self.beta) + self.bias

            # 2. 오차 계산
            error = mu - y

            # 3. 그래디언트 계산
            # 가우시안 GLM의 손실함수 미분값
            d_beta = np.dot(X.T, error) / n_samples
            d_bias = np.sum(error) / n_samples

            # 4. 가중치 업데이트
            self.beta -= self.learning_rate * d_beta
            self.bias -= self.learning_rate * d_bias

    def predict(self, X):
        # 항등 링크를 사용하므로 예측값은 선형 결합 결과 그대로 반환
        return np.dot(X, self.beta) + self.bias