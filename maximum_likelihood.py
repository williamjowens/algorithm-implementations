import numpy as np

class MaximumLikelihood:
    def __init__(self, max_iterations=100, tolerance=1e-6, verbose=False):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.mu = None
        self.sigma = None
        self.regularization = 1e-6

    def fit(self, X):
        if len(X.shape) != 2:
            raise ValueError("Input data X must be a 2D array.")

        n_samples, n_features = X.shape

        # Initialize parameters
        self.mu = np.mean(X, axis=0)
        self.sigma = np.cov(X, rowvar=False) + self.regularization * np.eye(n_features)
        prev_mu = np.copy(self.mu)

        for iteration in range(self.max_iterations):
            # Expectation step
            likelihood = self._gaussian_likelihood(X)

            # Maximization step
            self.mu = np.average(X, axis=0, weights=likelihood)
            weighted_diff = (X - self.mu).T * likelihood
            self.sigma = weighted_diff.dot((X - self.mu)) / np.sum(likelihood) + self.regularization * np.eye(n_features)

            # Check for convergence
            mu_change = np.max(np.abs(self.mu - prev_mu))
            if mu_change < self.tolerance:
                if self.verbose:
                    print(f"Convergence reached at iteration {iteration+1}.")
                break

            prev_mu = np.copy(self.mu)

            if self.verbose:
                print(f"Iteration {iteration+1}: mu_change = {mu_change}")

    def _gaussian_likelihood(self, X):
        n_samples, n_features = X.shape
        diff = X - self.mu
        inv_sigma = np.linalg.inv(self.sigma)
        exponent = np.sum(diff.dot(inv_sigma) * diff, axis=1)
        
        log_likelihood = -0.5 * (exponent + n_features * np.log(2 * np.pi) + np.log(np.linalg.det(self.sigma)))
        return np.exp(log_likelihood)


if __name__ == '__main__':
    np.random.seed(0)
    X_train = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=1000)

    ml = MaximumLikelihood(verbose=True)
    ml.fit(X_train)

    print("Estimated mean:")
    print(ml.mu)
    print("Estimated covariance matrix:")
    print(ml.sigma)