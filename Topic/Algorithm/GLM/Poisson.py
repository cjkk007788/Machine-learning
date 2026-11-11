import numpy as np

class PoissonGLM:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.beta = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.beta = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            # 1. 선형 결합 (eta = X*beta + bias)
            eta = np.dot(X, self.beta) + self.bias

            # 2. 역 링크 함수 적용 (Inverse Log Link -> Exponential -> 평균 lambda)
            # 오버플로 방지를 위해 np.clip 적용
            eta = np.clip(eta, -500, 500)
            lam = np.exp(eta)

            # 3. 오차 계산 (예측 평균 - 실제 카운트)
            error = lam - y

            # 4. 그래디언트 계산
            d_beta = np.dot(X.T, error) / n_samples
            d_bias = np.sum(error) / n_samples

            # 5. 가중치 업데이트
            self.beta -= self.learning_rate * d_beta
            self.bias -= self.learning_rate * d_bias

    def predict(self, X):
        # 예측된 평균 발생 횟수 (lambda) 반환
        eta = np.dot(X, self.beta) + self.bias
        eta = np.clip(eta, -500, 500)
        return np.exp(eta)