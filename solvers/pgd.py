from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse


def prox_enet(x, l1_ratio, alpha):
    prox = np.sign(x) * np.maximum(0, np.abs(x) - l1_ratio * alpha)
    prox /= (1 + (1 - l1_ratio) * alpha)
    return prox


class Solver(BaseSolver):
    name = 'pgd'  # proximal gradient descent, optionally accelerated
    stopping_strategy = "callback"

    parameters = {'use_acceleration': [False]}
    references = [
        'I. Daubechies, M. Defrise and C. De Mol, '
        '"An iterative thresholding algorithm for linear inverse problems '
        'with a sparsity constraint", Comm. Pure Appl. Math., '
        'vol. 57, pp. 1413-1457, no. 11, Wiley Online Library (2004)',
        'A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding '
        'algorithm for linear inverse problems", SIAM J. Imaging Sci., '
        'vol. 2, no. 1, pp. 183-202 (2009)'
    ]

    def skip(self, X, y, l1_ratio, lmbda, fit_intercept):
        # XXX - not implemented but not too complicated to implement
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, l1_ratio, lmbda, fit_intercept):
        self.X, self.y, self.lmbda, self.l1_ratio = X, y, lmbda, l1_ratio
        self.fit_intercept = fit_intercept

    def run(self, callback):
        L = self.compute_lipschitz_constant()

        n_samples, n_features = self.X.shape
        w = np.zeros(n_features)
        if self.use_acceleration:
            z = np.zeros(n_features)

        t_new = 1
        while callback(w):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                w_old = w.copy()
                z -= self.X.T @ (self.X @ z - self.y) / (L * n_samples)
                w = prox_enet(z, self.l1_ratio, self.lmbda / L)
                z = w + (t_old - 1.) / t_new * (w - w_old)
            else:
                w -= self.X.T @ (self.X @ w - self.y) / (L * n_samples)
                w = prox_enet(w, self.l1_ratio, self.lmbda / L)

        self.w = w

    def get_result(self):
        return self.w

    def compute_lipschitz_constant(self):
        n_samples = self.X.shape[0]
        if not sparse.issparse(self.X):
            L = np.linalg.norm(self.X, ord=2) ** 2
        else:
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2
        return L / n_samples
