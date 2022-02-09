from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from numba import njit


if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


@njit
def st(x, mu):
    if x > mu:
        return x - mu
    if x < - mu:
        return x + mu
    return 0.


@njit
def prox_1d(x, l1_ratio, alpha):
    prox = st(x, l1_ratio * alpha)
    prox /= (1 + (1 - l1_ratio) * alpha)
    return prox


class Solver(BaseSolver):
    name = "cd"

    install_cmd = 'conda'
    requirements = ['numba']
    references = [
        'H. Zou, T.J. Hastie,'
        '"Regularization and Variable Selection via the Elastic Net", '
        'Journal of the Royal Statistical Society, vol.67, pp. 301-320 (2005)',
        'J. Friedman, T. J. Hastie, H. Höfling and R. Tibshirani, '
        '"Pathwise coordinate optimization", Ann. Appl. Stat., vol 1, no. 2, '
        'pp. 302-332 (2007)'
    ]

    def skip(self, X, y, l1_ratio, lmbda, fit_intercept):
        # fit intercept is not implemented
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, l1_ratio, lmbda, fit_intercept):
        self.y = y
        self.l1_ratio, self.lmbda = l1_ratio, lmbda

        if sparse.issparse(X):
            self.X = X
        else:
            # use Fortran order to compute gradient on contiguous columns
            self.X = np.asfortranarray(X)

        # Make sure we cache the numba compilation.
        self.run(1)

    def run(self, n_iter):
        n_samples = self.X.shape[0]
        if sparse.issparse(self.X):
            L = np.array((self.X.multiply(self.X)).sum(axis=0)).squeeze()
            L /= n_samples
            self.w = self.sparse_cd(
                self.X.data, self.X.indices, self.X.indptr, self.y,
                self.l1_ratio, self.lmbda, L, n_iter
            )
        else:
            L = (self.X ** 2).sum(axis=0) / n_samples
            self.w = self.cd(self.X, self.y, self.l1_ratio,
                             self.lmbda, L, n_iter)

    @staticmethod
    @njit
    def cd(X, y, l1_ratio, alpha, L, n_iter):
        n_samples, n_features = X.shape
        R = np.copy(y)
        w = np.zeros(n_features)
        for _ in range(n_iter):
            for j in range(n_features):
                if L[j] == 0.:
                    continue
                old = w[j]
                w[j] = prox_1d(w[j] + X[:, j] @ R / (L[j] * n_samples),
                               l1_ratio, alpha / L[j])
                diff = old - w[j]
                if diff != 0:
                    R += diff * X[:, j]
        return w

    @staticmethod
    @njit
    def sparse_cd(X_data, X_indices, X_indptr, y, l1_ratio, alpha, L, n_iter):
        n_features = len(X_indptr) - 1
        n_samples = len(y)
        w = np.zeros(n_features)
        R = np.copy(y)
        for _ in range(n_iter):
            for j in range(n_features):
                if L[j] == 0.:
                    continue
                old = w[j]
                start, end = X_indptr[j:j+2]
                scal = 0.
                for ind in range(start, end):
                    scal += X_data[ind] * R[X_indices[ind]]
                w[j] = prox_1d(
                    w[j] + scal / (L[j] * n_samples), l1_ratio, alpha / L[j])
                diff = old - w[j]
                if diff != 0:
                    for ind in range(start, end):
                        R[X_indices[ind]] += diff * X_data[ind]
        return w

    def get_result(self):
        return self.w
