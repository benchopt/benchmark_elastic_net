# Author: Mathurin Massias <mathurin.massias@gmail.com>

import numpy as np
from numpy.linalg import norm

from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Enet"

    parameters = {
        'l1_ratio': [0.9, 0.5],
        'reg': [0.1, 0.01],
        'fit_intercept': [False, True]
    }

    def __init__(self, l1_ratio=.5, reg=1.0, fit_intercept=False):
        self.l1_ratio = l1_ratio
        self.reg = reg
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.n_samples, self.n_features = self.X.shape
        y_cen = y - np.mean(y) if self.fit_intercept else y
        self.lmbda_max = np.max(np.abs(X.T @ y_cen)) / (len(y) * self.l1_ratio)
        self.lmbda = self.reg * self.lmbda_max

    def get_one_solution(self):
        return np.zeros([self.X.shape[1] + self.fit_intercept])

    def compute(self, beta):
        # compute residuals
        a = self.l1_ratio * self.lmbda
        b = (1 - self.l1_ratio) * self.lmbda
        X, y = self.X, self.y
        n_samples, n_features = X.shape

        if self.fit_intercept:
            beta, intercept = beta[:n_features], beta[n_features:]
        diff = y - X @ beta

        if self.fit_intercept:
            diff -= intercept

        # compute primal objective
        p_obj = (1. / (2 * n_samples) * diff @ diff
                 + a * abs(beta).sum() + b * (beta ** 2).sum() / 2)

        # Compute dual with Lasso/Enet equivalence coming from
        # Mind the duality gap: safer rules for the Lasso - Appendix A.4
        theta = - diff
        scaling = max(1, norm(X.T @ theta + b * n_samples * beta,
                              ord=np.inf) / (a * n_samples))
        d_obj = (norm(y) ** 2 / 2
                 - norm(y + theta / scaling) ** 2 / 2
                 - b * n_samples * norm(beta / scaling) ** 2 / 2) / n_samples

        return dict(value=p_obj,
                    support_size=(beta != 0).sum(),
                    duality_gap=p_obj - d_obj,)

    def to_dict(self):
        return dict(X=self.X, y=self.y, l1_ratio=self.l1_ratio,
                    lmbda=self.lmbda, fit_intercept=self.fit_intercept)
