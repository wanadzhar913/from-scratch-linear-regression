import numpy as np

class LinearRegression:

    def __init__(
            self,
            lr: float = 0.001,
            n_iters: int = 1000,
            l1: float = 0.0,
            l2: float = 0.0
    ) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.l1 = l1
        self.l2 = l2
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = 2 * (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = 2 * (1/n_samples) * np.sum(y_pred - y)

            # Ridge (L2) Regularization  - penalizes big weights by shrinking them
            if self.l2 > 0:
                dw += 2 * self.l2 * self.weights

            # Lasso (L1) Regularization - Encourages sparsity (weights -> exactly 0)
            if self.l1 > 0:
                dw += self.l1 * np.sign(self.weights)

            # Elastic Net (if each of `self.l1` & `self.l2` > 0)

            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
