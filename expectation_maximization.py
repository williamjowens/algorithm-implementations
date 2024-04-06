import numpy as np
import matplotlib.pyplot as plt

class EMAlgorithm:
    def __init__(self, max_iter=100, tol=1e-3):
        self.max_iter = max_iter
        self.tol = tol
        self.parameters = None

    def fit(self, X):
        self._initialize_parameters(X)

        for i in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)

            # M-step
            self._m_step(X, responsibilities)

            # Check for convergence
            if self._converged(X, responsibilities):
                break

            # Visualization
            self._visualize(X, responsibilities, i)

    def _initialize_parameters(self, X):
        pass

    def _e_step(self, X):
        pass

    def _m_step(self, X, responsibilities):
        pass

    def _converged(self, X, responsibilities):
        pass

    def _visualize(self, X, responsibilities, iteration):
        pass

class GaussianMixtureEM(EMAlgorithm):
    def __init__(self, n_components, max_iter=100, tol=1e-3, reg_covar=1e-6):
        super().__init__(max_iter, tol)
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.weights = None
        self.means = None
        self.covariances = None

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(n_features)] * self.n_components)

    def _e_step(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self._gaussian_pdf(X, self.means[k], self.covariances[k])
        responsibilities *= self.weights
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X, responsibilities):
        total_resp = np.sum(responsibilities, axis=0)
        self.weights = total_resp / X.shape[0]
        self.means = np.dot(responsibilities.T, X) / total_resp.reshape(-1, 1)
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / total_resp[k]
            self.covariances[k].flat[::X.shape[1] + 1] += self.reg_covar

    def _converged(self, X, responsibilities):
        log_likelihood = self._compute_log_likelihood(X, responsibilities)
        if hasattr(self, 'prev_log_likelihood'):
            if abs(log_likelihood - self.prev_log_likelihood) < self.tol:
                return True
        self.prev_log_likelihood = log_likelihood
        return False

    def _compute_log_likelihood(self, X, responsibilities):
        log_likelihood = 0
        for k in range(self.n_components):
            log_likelihood += np.sum(responsibilities[:, k] * np.log(
                self.weights[k] * self._gaussian_pdf(X, self.means[k], self.covariances[k])))
        return log_likelihood

    def _gaussian_pdf(self, X, mean, cov):
        n_features = X.shape[1]
        det_cov = np.linalg.det(cov)
        inv_cov = np.linalg.pinv(cov)
        diff = X - mean
        exponent = -0.5 * np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        return (2 * np.pi) ** (-n_features / 2) * det_cov ** (-0.5) * np.exp(exponent)

    def _visualize(self, X, responsibilities, iteration):
        plt.clf()
        for k in range(self.n_components):
            plt.scatter(X[:, 0], X[:, 1], c=responsibilities[:, k], alpha=0.5, cmap='viridis')
        plt.title(f"Iteration {iteration}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

def main():
    # Generate sample data from a mixture of Gaussian distributions
    np.random.seed(42)
    n_samples = 1000
    n_features = 2
    n_components = 3

    true_weights = np.array([0.3, 0.4, 0.3])
    true_means = np.array([[0, 0], [5, 5], [10, 10]])
    true_covariances = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 3]]])

    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        component = np.random.choice(n_components, p=true_weights)
        X[i] = np.random.multivariate_normal(true_means[component], true_covariances[component])

    # Initialize and fit the EM algorithm
    em = GaussianMixtureEM(n_components=n_components, max_iter=100, tol=1e-3)
    em.fit(X)

    print("Estimated weights:", em.weights)
    print("Estimated means:")
    print(em.means)
    print("Estimated covariances:")
    print(em.covariances)
    print("True weights:", true_weights)
    print("True means:")
    print(true_means)
    print("True covariances:")
    print(true_covariances)

    plt.show()

if __name__ == "__main__":
    main()