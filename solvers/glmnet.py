from benchopt import BaseSolver, safe_import_context
from benchopt.runner import INFINITY
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri, packages
    from benchopt.helpers.r_lang import import_rpackages

    # Setup the system to allow rpy2 running
    numpy2ri.activate()
    import_rpackages('glmnet')


class Solver(BaseSolver):
    name = "glmnet"

    install_cmd = 'conda'
    requirements = ['r-base', 'rpy2', 'r-glmnet', 'r-matrix']
    references = [
        'J. Friedman, T. J. Hastie and R. Tibshirani, "Regularization paths '
        'for generalized linear models via coordinate descent", '
        'J. Stat. Softw., vol. 33, no. 1, pp. 1-22, NIH Public Access (2010)'
    ]
    support_sparse = True

    stopping_criterion = SufficientProgressCriterion(
        patience=7, eps=1e-38, strategy='tolerance')

    def set_objective(self, X, y, lmbda, l1_ratio, fit_intercept):
        if sparse.issparse(X):
            r_Matrix = packages.importr("Matrix")
            X = X.tocoo()
            self.X = r_Matrix.sparseMatrix(
                i=robjects.IntVector(X.row + 1),
                j=robjects.IntVector(X.col + 1),
                x=robjects.FloatVector(X.data),
                dims=robjects.IntVector(X.shape)
            )
        else:
            self.X = X
        self.y, self.lmbda, self.l1_ratio = y, lmbda, l1_ratio
        self.fit_intercept = fit_intercept

        self.glmnet = robjects.r['glmnet']

    def run(self, tol):
        maxit = 0 if tol == INFINITY else 1_000_000
        fit_dict = {"lambda": self.lmbda,
                    "alpha": self.l1_ratio}

        glmnet_fit = self.glmnet(self.X, self.y, intercept=self.fit_intercept,
                                 standardize=False, maxit=maxit,
                                 thresh=tol ** 1.8, **fit_dict)
        results = dict(zip(glmnet_fit.names, list(glmnet_fit)))
        as_matrix = robjects.r['as']
        coefs = np.array(as_matrix(results["beta"], "matrix"))
        beta = coefs.flatten()
        self.w = np.r_[beta, results["a0"]] if self.fit_intercept else beta

    def get_result(self):
        return self.w
