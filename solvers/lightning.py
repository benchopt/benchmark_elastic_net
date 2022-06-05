import warnings
from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from lightning.regression import SDCARegressor
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'lightning'

    install_cmd = 'conda'
    requirements = ['sklearn-contrib-lightning']
    references = [
        'M. Blondel, K. Seki and K. Uehara, '
        '"Block coordinate descent algorithms for large-scale sparse '
        'multiclass classification" '
        'Mach. Learn., vol. 93, no. 1, pp. 31-52 (2013)'
    ]

    def set_objective(self, X, y, l1_ratio, lmbda, fit_intercept):
        self.X, self.y, self.l1_ratio, self.lmbda = X, y, l1_ratio, lmbda
        self.fit_intercept = fit_intercept

        self.clf = SDCARegressor(alpha=self.lmbda,
                                 l1_ratio=self.l1_ratio, tol=0)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.run(1)

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros(self.X.shape[1] + self.fit_intercept)
        else:
            self.clf.max_iter = n_iter
            X, y = self.X, self.y

            if self.fit_intercept:
                X_cols_mean = X.mean(axis=0)
                y_mean = y.mean()

                self.clf.fit(X - X_cols_mean, y - y_mean)
                self.clf.intercept_ = y_mean - self.clf.coef_.flatten().T @ X_cols_mean
            else:
                self.clf.fit(self.X, self.y)

            coef = self.clf.coef_.flatten()
            if self.fit_intercept:
                coef = np.r_[coef, self.clf.intercept_]
            self.coef = coef

    def get_result(self):
        return self.coef
