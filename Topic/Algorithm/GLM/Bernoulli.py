import numpy as np

class BernoulliGLM:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.beta = None
        self.bias = None

    def _sigmoid(self, z):
        # 수치적 안정성을 위해 np.clip 적용
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 가중치(beta)와 절편(bias) 초기화
        self.beta = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iterations):
            # 1. 선형 결합 (eta = X*beta + bias)
            eta = np.dot(X, self.beta) + self.bias

            # 2. 역 링크 함수 적용 (Inverse Logit Link -> Sigmoid -> 확률 p)
            p = self._sigmoid(eta)

            # 3. 오차 계산 (p - y)
            error = p - y

            # 4. 그래디언트 계산 (Binary Cross-Entropy 미분 결과)
            d_beta = np.dot(X.T, error) / n_samples
            d_bias = np.sum(error) / n_samples

            # 5. 가중치 업데이트
            self.beta -= self.learning_rate * d_beta
            self.bias -= self.learning_rate * d_bias

    def predict_proba(self, X):
        # 확률 p = P(y=1|X) 반환
        eta = np.dot(X, self.beta) + self.bias
        return self._sigmoid(eta)

    def predict(self, X, threshold=0.5):
        # 0.5 기준으로 0 또는 1 이분 예측
        return (self.predict_proba(X) >= threshold).astype(int)





