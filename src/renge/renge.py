import jax
import jax.numpy as jnp
import numpy as np
import optuna
import pandas as pd
import tqdm
from jax import grad, jit
from jax.interpreters import xla
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import truncnorm
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate
from statsmodels.stats.multitest import fdrcorrection


class RengeSub(BaseEstimator):
    def __init__(self, C=1.0, A_l1_ratio=0.3, n_k=5, maxiter=3000):
        self.C = C
        self.A_l1_ratio = A_l1_ratio
        self.n_k = n_k
        self.maxiter = maxiter

    @jax.partial(jit, static_argnums=(0,))
    def w(self, t, k, alpha, beta, gamma):
        return 1 / (1 + jnp.exp(-(alpha + beta * t - gamma * k)))

    @jax.partial(jit, static_argnums=(0, 2, 3))
    def loss_sq(self, params, X, E):
        A, b, w_params = params[:self.n_gene ** 2], params[self.n_gene ** 2: self.n_gene ** 2 + len(
            self.t) * self.n_gene], params[self.n_gene ** 2 + len(self.t) * self.n_gene:]
        alpha = w_params[:-2]
        beta, gamma = w_params[-2:]
        A = jnp.reshape(A, newshape=((self.n_gene, self.n_gene)))
        X_t = np.where(X[:, -len(self.t):] == 1)[1]

        loss_sq_val = 0

        for j in range(len(self.t)):
            n_cell_j = (X_t == j).sum()
            if n_cell_j == 0:
                continue

            X_j = X[X_t == j, :-len(self.t)].T  # gene x cell
            E_j = E[X_t == j, :].T

            pred_j = jnp.tile(b[self.n_gene * j:self.n_gene * (j + 1)], (X_j.shape[1], 1)).T  # gene x cell
            M = jnp.where(X_j < 0, 0, 1)
            X_a = X_j.copy()
            X_a[0, X_a.sum(axis=0) == 0] = -1  # for control cell, use w value of gene 0
            w_idx = jnp.array(np.where(X_a.T < 0)[1])

            for k in range(self.n_k):
                pred_j_l = A @ X_j
                for l in range(k):
                    pred_j_l = pred_j_l * M  # no regulation to gene i when gene i is knocked out
                    pred_j_l = A @ pred_j_l

                pred_j += self.w(self.t[j], k, alpha, beta, gamma)[w_idx] * pred_j_l

            # ignore self repression (KO effect)
            mask = X_j.copy()  # gene x cell
            mask[X_j < 0] = 0
            mask[X_j == 0] = 1

            loss_sq_val += jnp.sum(((E_j - pred_j) * mask) ** 2) / n_cell_j

        return loss_sq_val

    @jax.partial(jit, static_argnums=(0,))
    def loss_regularize(self, params):

        A, b, _ = params[:self.n_gene ** 2], params[self.n_gene ** 2: self.n_gene ** 2 + len(
            self.t) * self.n_gene], params[self.n_gene ** 2 + len(self.t) * self.n_gene:]
        A = jnp.reshape(A, newshape=((self.n_gene, self.n_gene)))

        A_l1 = self.C * self.A_l1_ratio * self.n_k * jnp.sum(jnp.abs(A))
        A_l2 = 0
        for k in range(self.n_k):
            A_l2 += jnp.sum((jnp.linalg.matrix_power(A, k + 1)) ** 2)
        A_l2 = self.C * (1 - self.A_l1_ratio) * A_l2

        loss_regularize_val = A_l1 + A_l2

        return loss_regularize_val

    def loss(self, params, X, E):
        loss_sq = self.loss_sq(params, X, E)
        loss_regularize = self.loss_regularize(params)
        return loss_sq + loss_regularize

    def grad_wrap_loss(self, params, *args):
        g_sq = np.array(grad(self.loss_sq)(params, args[0], args[1]), dtype=np.float64)
        g_sq.setflags(write=True)
        g_reg = np.array(grad(self.loss_regularize)(params), dtype=np.float64)
        g_reg.setflags(write=True)

        return g_sq + g_reg

    def linear_regression(self, X, E):
        lm = LinearRegression(fit_intercept=False).fit(X, E)
        A_lr = lm.coef_[:, :-len(self.t)]
        b_lr = lm.coef_[:, -len(self.t):]

        return A_lr, b_lr

    def elastic_net(self, X, E):
        # X : sample (cell) x (gene + sample id)
        # E : sample (cell) x (gene)

        reg = ElasticNet(l1_ratio=0.5, alpha=0.0005, max_iter=10000, fit_intercept=False)
        reg.fit(X, E)

        A_elast = reg.coef_[:, :-len(self.t)]
        b_elast = reg.coef_[:, -len(self.t):]

        return A_elast, b_elast

    def fit(self, X, E):
        self.n_gene = E.shape[1]
        self.t = X.columns[self.n_gene:]
        gene_names = E.columns
        X = X.values
        E = E.values

        ### initiialize parameters
        # A_lr, b_lr = self.linear_regression(X, E)
        A_lr, b_lr = self.elastic_net(X, E)
        max_param_A = np.max(np.abs(A_lr.flatten())) + 1e-5
        max_param_b = np.max(np.abs(b_lr.flatten()))

        self.params = np.zeros(self.n_gene ** 2 + self.n_gene * len(self.t) + self.n_gene + 2)
        # self.params[:self.n_gene**2] = np.random.permutation(A_lr.flatten())
        # self.params[:self.n_gene**2] = A_lr.flatten()
        self.params[:self.n_gene ** 2] = truncnorm.rvs(a=-2 * max_param_A, b=2 * max_param_A, loc=0,
                                                       scale=A_lr.flatten().std(ddof=1) + 1e-5, size=self.n_gene ** 2)
        self.params[self.n_gene ** 2:self.n_gene ** 2 + self.n_gene * len(self.t)] = b_lr.flatten()

        A_mask = np.zeros(A_lr.shape)
        for i in range(len(A_mask)):
            A_mask[i, i] = 1
        A_mask = A_mask.flatten()

        ### define parameter range (bound)
        bounds = []

        for i in range(len(self.params)):
            if i < self.n_gene ** 2:
                # A
                if A_mask[i] == 1:
                    bounds.append((0, 0))
                else:
                    bounds.append((-2 * max_param_A, 2 * max_param_A))
            elif i < self.n_gene ** 2 + len(self.t) * self.n_gene:
                # b
                bounds.append((-2 * max_param_b, 2 * max_param_b))
            elif i < self.n_gene ** 2 + len(self.t) * self.n_gene + self.n_gene:
                # alpha
                bounds.append((-10, 10))
            else:
                # beta, gamma
                bounds.append((1e-10, 5))

        ### optimize parameters
        res = minimize(fun=self.loss, args=(X, E), x0=self.params, jac=self.grad_wrap_loss,
                       method='L-BFGS-B', bounds=bounds, options={'maxiter': self.maxiter})
        # print(res.message)
        # print(res.nit)
        self.params = res.x
        self.A = pd.DataFrame(res.x[:self.n_gene ** 2].reshape((self.n_gene, self.n_gene)))
        self.A.columns = gene_names
        self.A.index = gene_names
        # self.b = res.x[self.n_gene ** 2:self.n_gene ** 2 + self.n_gene * len(self.t)].reshape(
        #    newshape=(self.n_gene, len(self.t)))

        return self

    @jax.partial(jit, static_argnums=(0, 1))
    def calc_pred_j(self, X_list, A, b, alpha, beta, gamma):
        pred_j_list = []

        for j in range(len(self.t)):
            X = X_list[j].T  # gene x cell
            pred_j = jnp.tile(b[self.n_gene * j:self.n_gene * (j + 1)], (X.shape[1], 1)).T  # gene x cell
            M = jnp.where(X < 0, 0, 1)
            X_a = X.copy()
            X_a[0, X_a.sum(axis=0) == 0] = -1  # for control cell, use w value of gene 0
            w_idx = jnp.array(np.where(X_a.T < 0)[1])

            for k in range(self.n_k):
                pred_j_l = A @ X
                for l in range(k):
                    pred_j_l = pred_j_l * M  # no regulation to gene i when gene i is knocked out
                    pred_j_l = A @ pred_j_l

                pred_j += self.w(self.t[j], k, alpha, beta, gamma)[w_idx] * pred_j_l
            pred_j_list.append(pred_j)

        return pred_j_list

    def predict(self, X):
        # X : cell x (gene + t)
        A, b, w_params = self.params[:self.n_gene ** 2], self.params[self.n_gene ** 2: self.n_gene ** 2 + len(
            self.t) * self.n_gene], self.params[self.n_gene ** 2 + len(self.t) * self.n_gene:]
        alpha = w_params[:-2]
        beta, gamma = w_params[-2:]
        A = np.reshape(A, newshape=((self.n_gene, self.n_gene)))

        pred = np.zeros((X.shape[0], X.shape[1] - len(self.t)))  # cell x gene

        X_t = np.where(X[:, -len(self.t):] == 1)[1]
        X_list = []
        for j in range(len(self.t)):
            X_list.append(X[X_t == j, :-len(self.t)])

        pred_j_list = self.calc_pred_j(X_list, A, b, alpha, beta, gamma)
        for j in range(len(self.t)):
            pred_j = pred_j_list[j]
            pred[X_t == j] = pred_j.T

        return pred

    def score(self, X, E):
        X = X.values
        E = E.values
        # ignore self repression (KO effect)
        mask = X[:, :-len(self.t)].copy()  # gene x cell
        mask[X[:, :-len(self.t)] < 0] = 0
        mask[X[:, :-len(self.t)] == 0] = 1

        E_pred = self.predict(X)
        res_sum_sq = jnp.sum(((E - E_pred) * mask) ** 2)
        tot_sum_sq = jnp.sum(((E - E.mean(axis=0)) * mask) ** 2)
        return 1 - res_sum_sq / tot_sum_sq


class Renge():
    def fit(self, C=1.0, A_l1_ratio=0.5, n_k=5):
        if not hasattr(self, "X_prep"):
            raise ValueError("no attribute named X_prep. run preprocess_X() first")
        self.reg = RengeSub(C=C, A_l1_ratio=A_l1_ratio, n_k=n_k)
        self.reg.fit(self.X_prep, self.E)
        self.A = self.reg.A

    def predict(self, X_pred):
        if not hasattr(self, "reg"):
            raise ValueError("no attribute named reg. run estimate_hyperparams_and_fit() or fit() first")

        T = X_pred.values[:, -1]
        t = self.X_prep.columns[self.E.shape[1]:]
        T_one_hot = np.zeros((X_pred.shape[0], len(t)))
        for i in range(T_one_hot.shape[0]):
            idx = np.where(t == T[i])[0]
            T_one_hot[i, idx] = 1
        T_one_hot = pd.DataFrame(T_one_hot, index=X_pred.index)
        X_T = pd.concat([X_pred.iloc[:, :-1], T_one_hot], axis=1)
        X_T.columns = list(X_T.columns[:-len(t)]) + list(t)
        pred = self.reg.predict(X_T.values)
        return pred

    def bayes_cov_col(self, Y, X, cols, lm):
        """
        Copyright (c) 2016 asncd
        Released under the MIT license
        https://github.com/asncd/MIMOSCA/blob/master/LICENSE

        @Y    = Expression matrix, cells x x genes, expecting pandas dataframe
        @X    = Covariate matrix, cells x covariates, expecting pandas dataframe
        @cols = The subset of columns that the EM should be performed over, expecting list
        @lm   = linear model object
        """

        # EM iterateit
        Yhat = pd.DataFrame(lm.predict(X))
        Yhat.index = Y.index
        Yhat.columns = Y.columns
        SSE_all = np.square(Y.subtract(Yhat))
        X_adjust = X.copy()

        df_SSE = []
        df_logit = []

        for curcov in cols:

            if (X[curcov] > 0).values.sum() == 0:
                continue
            else:
                curcells = X[X[curcov] > 0].index

            if len(curcells) > 2:
                X_notcur = X.copy()
                X_notcur[curcov] = [0] * len(X_notcur)

                X_sub = X_notcur.loc[curcells]

                Y_sub = Y.loc[curcells]

                GENE_var = 2.0 * Y_sub.var(axis=0)
                vargenes = GENE_var[GENE_var > 0].index

                Yhat_notcur = pd.DataFrame(lm.predict(X_sub))
                Yhat_notcur.index = Y_sub.index
                Yhat_notcur.columns = Y_sub.columns

                SSE_notcur = np.square(Y_sub.subtract(Yhat_notcur))
                SSE = SSE_all.loc[curcells].subtract(SSE_notcur)
                SSE_sum = SSE.sum(axis=1)

                SSE_transform = SSE.div(GENE_var + 0.5)[vargenes].sum(axis=1)
                logitify = np.divide(1.0, 1.0 + np.exp(SSE_transform))  # sum))

                df_SSE.append(SSE_sum)
                df_logit.append(logitify)

                X_adjust[curcov].loc[curcells] = logitify

        return X_adjust

    def adjust_X(self, X, E, t):
        lm = LinearRegression(fit_intercept=False).fit(X, E)
        cols = list(X.columns[:-len(t)])
        X_adjust = self.bayes_cov_col(E, X, cols, lm)
        return X_adjust

    def one_hot_t(self, X):
        T = X.values[:, -1]
        t = np.sort(list(set(T)))
        T_one_hot = np.zeros((X.shape[0], len(t)))
        for i in range(T_one_hot.shape[0]):
            idx = np.where(t == T[i])[0]
            T_one_hot[i, idx] = 1
        T_one_hot = pd.DataFrame(T_one_hot, index=X.index)
        X_T = pd.concat([X.iloc[:, :-1], T_one_hot], axis=1)
        X_T.columns = list(X_T.columns[:-len(t)]) + list(t)
        return X_T, t

    def preprocess_X(self, X, E):
        self.E = E
        X, t = self.one_hot_t(X)
        X = self.adjust_X(X, E, t)

        X_col = X.columns
        X_ind = X.index
        X = X.values
        E = E.values

        E_ctrl = E[X[:, :-len(t)].sum(axis=1) == 0, :]
        E_ctrl[E_ctrl == 0] = np.nan
        C_nonzero_mean = np.nanmean(E_ctrl, axis=0)
        C_nonzero_mean[np.isnan(C_nonzero_mean)] = 0  # expression is 0 in all cells
        X[:, :-len(t)] = -X[:, :-len(t)] * C_nonzero_mean

        self.X_prep = pd.DataFrame(X, index=X_ind, columns=X_col)
        self.n_gene = E.shape[1]
        self.t = self.X_prep.columns[self.n_gene:]

    def estimate_hyperparams_and_fit(self, X, E, n_trials=10, max_n_k=5, min_n_k=1, cv_group='gene'):
        if len(X.index) != len(E.index):
            raise ValueError("cell number did not match between X and E")
        if len(X.columns[:-1]) != len(E.columns):
            raise ValueError("gene number did not match between X and E")
        if (X.index != E.index).any():
            raise ValueError("cell label did not match between X and E")
        if (X.columns[:-1] != E.columns).any():
            raise ValueError("gene name did not match between X and E")

        self.preprocess_X(X, E)
        X_t = np.where(self.X_prep.iloc[:, -len(self.t):] == 1)[1]

        groups = np.zeros(X.shape[0])
        KO_idx = np.where(X.iloc[:, :-1] == 1)[0]  # not control cells

        if cv_group == 'gene':
            # which gene is KO?
            groups[KO_idx] = (np.where(X.iloc[:, :-1] == 1)[1] + 1)
        elif cv_group == 'gene_and_time':
            # which gene is KO and which sampling time?
            groups[KO_idx] = (np.where(X.iloc[:, :-1] == 1)[1] + 1) + (X_t[KO_idx]) * max(np.where(X == 1)[1] + 1)
        else:
            raise ValueError("'cv_group' must be 'gene' or 'gene_and_time'")

        id_max = max(groups)
        CT_idx = np.where(X.iloc[:, :-1].sum(axis=1) == 0)[0]
        for i in range(len(CT_idx)):
            groups[CT_idx[i]] = id_max + 1 + X_t[CT_idx[i]]

        # print('1: '+str(len(set(groups))))

        uni_groups = np.unique(groups.copy())
        idx_perm = np.arange(len(uni_groups), dtype=np.int64)
        np.random.shuffle(idx_perm)

        for i in range(len(groups)):
            groups[i] = idx_perm[np.where(np.sort(uni_groups) == groups[i])[0]]

        # print(len(set(groups)))
        gss = GroupKFold(n_splits=5)

        def objective(trial):
            C = trial.suggest_loguniform('C', 1e-10, 1)
            A_l1_ratio = trial.suggest_loguniform('A_l1_ratio', 1e-10, 1)
            n_k = trial.suggest_int("n_k", min_n_k, max_n_k, log=False)

            reg = RengeSub(C=C, A_l1_ratio=A_l1_ratio, n_k=n_k)
            cv_results = cross_validate(reg, self.X_prep, E, cv=gss, groups=groups, n_jobs=5)
            return cv_results['test_score'].mean()

        sampler = optuna.samplers.TPESampler(n_startup_trials=n_trials // 3, multivariate=True)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        self.trial = study.best_trial

        self.fit(C=self.trial.params['C'], A_l1_ratio=self.trial.params['A_l1_ratio'], n_k=self.trial.params['n_k'])

        return self.A

    def calc_qval(self, n_boot=30):
        if not hasattr(self, "A"):
            raise ValueError("no attribute named A. run fit() first")

        A_boot_list = []
        X_t = np.where(self.X_prep.iloc[:, -len(self.t):] == 1)[1]
        for i in tqdm.tqdm(range(n_boot)):
            E_boot = self.E.values.copy()
            # resample E
            for j in range(len(self.t)):
                for k in range(self.n_gene):
                    # sample j, gene k KO cells
                    if self.X_prep.values[:, k].sum() == 0:
                        continue
                    cell_bool = (self.X_prep.values[:, k] < 0) & (X_t == j)
                    cell_idx = np.where(cell_bool == True)[0]
                    # print(np.where(self.X_prep.iloc[cell_idx, :] < 0)[1])
                    # print(np.where(self.X_prep.iloc[cell_idx, -len(self.t):] == 1)[1])
                    boot_cell_idx = np.random.choice(cell_idx, size=len(cell_idx), replace=True)
                    E_boot[cell_idx, :] = self.E.values[boot_cell_idx, :]

            E_boot = pd.DataFrame(E_boot, index=self.E.index, columns=self.E.columns)
            self.E_boot = E_boot
            reg = RengeSub(C=self.trial.params['C'], A_l1_ratio=self.trial.params['A_l1_ratio'],
                           n_k=self.trial.params['n_k'])
            reg.fit(self.X_prep, E_boot)
            A_boot = reg.A.values
            A_boot_list.append(A_boot)
            xla._xla_callable.cache_clear()

        A_boot_list = np.array(A_boot_list)
        self.A_boot_list = A_boot_list
        A_pval = np.zeros(self.A.shape)

        for i in range(len(self.A)):
            for j in range(len(self.A)):
                boot_std = A_boot_list[:, i, j].std(ddof=1)
                if self.A.iloc[i, j] > 0:
                    # A_pval[i, j] = 2*(1-norm.cdf(x=A[i, j]/boot_std, loc=0, scale=1))
                    A_pval[i, j] = norm.logsf(x=self.A.iloc[i, j] / boot_std, loc=0, scale=1) + np.log(2)
                elif self.A.iloc[i, j] < 0:
                    # A_pval[i, j] = 2*norm.cdf(x=A[i, j]/boot_std, loc=0, scale=1)
                    A_pval[i, j] = norm.logcdf(x=self.A.iloc[i, j] / boot_std, loc=0, scale=1) + np.log(2)
                else:
                    A_pval[i, j] = 0

        A_pval = np.exp(A_pval)
        A_rejected, A_qval = fdrcorrection(A_pval.flatten(), alpha=0.05, method='indep', is_sorted=False)
        A_qval = np.reshape(A_qval, newshape=(A_pval.shape))
        self.A_qval = pd.DataFrame(A_qval, index=self.A.index, columns=self.A.columns)

        return self.A_qval
