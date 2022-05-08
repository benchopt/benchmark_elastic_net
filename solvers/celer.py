from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import warnings
    import numpy as np
    from celer import ElasticNet
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'celer'

    install_cmd = 'conda'
    requirements = ['pip:celer']
    references = [
        'M. Massias, A. Gramfort and J. Salmon, ICML, '
        '"Celer: a Fast Solver for the Lasso with Dual Extrapolation", '
        'vol. 80, pp. 3321-3330 (2018)'
    ]

    def set_objective(self, X, y, l1_ratio, lmbda, fit_intercept):
        self.X, self.y, self.l1_ratio, self.lmbda = X, y, l1_ratio, lmbda
        self.fit_intercept = fit_intercept

        self.enet = ElasticNet(alpha=self.lmbda,
                               l1_ratio=self.l1_ratio,
                               fit_intercept=fit_intercept, tol=0)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.X.shape[1] + self.fit_intercept])
        else:
            self.enet.max_iter = n_iter
            self.enet.fit(self.X, self.y)

            coef = self.enet.coef_.flatten()
            if self.fit_intercept:
                coef = np.r_[coef, self.enet.intercept_]
            self.coef = coef

    @staticmethod
    def get_next(previous):
        "Linear growth for n_iter."
        return previous + 1

    def get_result(self):
        return self.coef

