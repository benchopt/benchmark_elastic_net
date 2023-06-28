import warnings
from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from skglm import ElasticNet
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'skglm'

    install_cmd = 'conda'
    requirements = [
        'pip:skglm'
    ]

    def set_objective(self, X, y, l1_ratio, lmbda, fit_intercept):
        self.X, self.y, self.l1_ratio, self.lmbda = X, y, l1_ratio, lmbda
        self.fit_intercept = fit_intercept

        self.clf = ElasticNet(alpha=self.lmbda,
                              l1_ratio=self.l1_ratio,
                              fit_intercept=fit_intercept, tol=0)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.run(1)

    def get_next(self, stop_val):
        return stop_val + 1

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros(self.X.shape[1] + self.fit_intercept)
        else:
            self.clf.max_iter = n_iter
            self.clf.fit(self.X, self.y)
            coef = self.clf.coef_.flatten()
            if self.fit_intercept:
                coef = np.r_[coef, self.clf.intercept_]
            self.coef = coef

    def get_result(self):
        return self.coef
