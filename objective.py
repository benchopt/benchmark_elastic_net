import numpy as np
from numpy.linalg import norm

from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Enet"

    parameters = {
        'l1_ratio': [0.9, 0.5],
        'reg': [0.1, 0.01],
        'fit_intercept': [False]
    }

    def __init__(self, l1_ratio=.5, reg=1.0, fit_intercept=False):
        self.l1_ratio = l1_ratio
        self.reg = reg
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.n_samples, self.n_features = self.X.shape
        self.lmbda_max = np.max(np.abs(X.T @ y)) / len(y)
        self.lmbda = self.reg * self.lmbda_max

    def compute(self, beta):
        # compute residuals
        a = self.l1_ratio * self.lmb
        b = (1 - self.L1_ratio) * self.lmbd

        if self.fit_intercept:
            beta, intercept = beta[:self.n_features], beta[self.n_features:]
        diff = self.y - self.X @ beta

        if self.fit_intercept:
            diff -= intercept

        # compute primal objective
        p_obj = (1. / (2 * self.n_samples) * diff @ diff
                 + a * abs(beta).sum() + b * (beta ** 2).sum())

        # Compute dual with Lasso/Enet equivalence coming from
        # Mind the duality gap: safer rules for the Lasso - Appendix A.4
        scaling = max(1, norm(self.X.T @ (- diff) + b * beta, ord=np.inf) / a)
        d_obj = (norm(self.y) ** 2 / (2 * self.n_samples)
                 - norm(self.y - diff / scaling) / (2 * self.n_samples)
                 - b * norm(beta / scaling) ** 2 / 2)
        # scaled_beta = -np.sqrt(
        #     (1-self.l1_ratio) * self.lmbda * self.n_samples) * beta
        # diff = np.hstack([diff, scaled_beta])
        # theta = diff / (self.lmbda * self.l1_ratio * self.n_samples)
        # theta /= norm(self.X.T @ theta[:self.n_samples] +
        #               np.sqrt((self.lmbda * self.l1_ratio * self.n_samples)) *
        #               theta[self.n_samples:], ord=np.inf)

        # d_obj = self.lmbda * self.l1_ratio * (self.y @ theta[:self.n_samples])
        # d_obj -= (self.lmbda * self.l1_ratio) ** 2 * \
        #     self.n_samples / 2 * (theta ** 2).sum()

        return dict(value=p_obj,
                    support_size=(beta != 0).sum(),
                    duality_gap=p_obj - d_obj,)

    def to_dict(self):
        return dict(X=self.X, y=self.y, l1_ratio=self.l1_ratio,
                    lmbda=self.lmbda, fit_intercept=self.fit_intercept)
