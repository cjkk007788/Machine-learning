class MultiGaussian:
    def __init__(self, X):
        self.X = np.asarray(X, dtype=float)      # 리스트, 튜플 모두 OK
        self.N, self.D = self.X.shape
        self.mu = self.mean()
        self.sigma = self.cov()

    def mean(self):
        return np.mean(self.X, axis=0)

    def cov(self):
        dev = self.X - self.mu
        return dev.T @ dev / self.N

    def pdf(self, x):
        x = np.asarray(x, dtype=float)           # 리스트 -> 배열
        dev = x - self.mu
        inverse = np.linalg.inv(self.sigma)
        norm_const = np.sqrt((2 * np.pi) ** self.D * np.linalg.det(self.sigma))
        return np.exp(-0.5 * dev @ inverse @ dev) / norm_const