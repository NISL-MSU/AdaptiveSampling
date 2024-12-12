import warnings
import numpy as np
from sklearn.neighbors import KDTree
from AdaptiveSampling.estimators.EstimatorInterface import EstimatorInterface


class ASPINN(EstimatorInterface):
    """Use a GP as a surrogate model using a potential epistemic uncertainty metric"""

    def __init__(self, **kwargs):
        self.name = 'ASPINN'
        self.epsi = kwargs.get('epsi', 0.25)
        self.length = kwargs.get('length', 0.1)
        if 'epsi' not in kwargs:
            warnings.warn("ASPINN: 'epsi' not provided, defaulting to 0.2", UserWarning)
        if 'length' not in kwargs:
            warnings.warn("ASPINN: 'length' not provided, defaulting to 0.1", UserWarning)

        self.mu, self.K, self.x = None, None, None

    def set_mean(self, x, mu):
        self.x = x
        self.mu = mu

    def set_cov_matrix(self, diag):
        n = len(diag)
        # self.K = np.diag(diag)
        self.K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                noise = (diag[i] if i == j else (np.sqrt(diag[i]) * np.sqrt(diag[j])))
                self.K[i, j] = self.K[j, i] = self.RBF(self.x[i], self.x[j]) * (1 + noise)

    def RBF(self, a, b):
        return np.exp(-0.5 * ((a - b) / self.length) ** 2)

    def cov_obs_pred(self, x):
        """Covariance between the observed data point and the prediction points"""
        k_star = np.zeros(len(self.x))
        for i, xi in enumerate(self.x):
            k_star[i] = self.RBF(x, xi)
        return k_star

    def predictive_mean(self, x):
        k_star = self.cov_obs_pred(x)
        mu_star = k_star[np.argmax(k_star)] * self.mu[np.argmax(k_star)]
        return mu_star

    def predictive_cov(self, x):
        k_star = self.cov_obs_pred(x)[:, np.newaxis]
        # Covariance of x with itself
        K_star = self.K[np.argmax(k_star), np.argmax(k_star)]

        return K_star  # - np.dot(np.dot(k_star.T, np.linalg.inv(self.K)), k_star)

    def posterior_update(self, xB, K=None):
        """A single observation at position x is made"""
        if K is None:
            K = self.K.copy()
        # Calculate the covariance matrix Ks between training x-values and prediction x-values
        Ks = np.zeros((len(self.x), 1))
        rbf_xb = np.argmax(self.cov_obs_pred(xB)[:, np.newaxis])
        for i in range(len(self.x)):
            k = K[i, rbf_xb]
            Ks[i] = k
        # Calculate the covariance matrix by evaluating the covariance function at the training data x-values
        noise = 0.1 ** 2
        KB = K[rbf_xb, rbf_xb] + noise
        KBInv = 1 / KB

        return np.clip(K - np.dot(Ks, np.dot(KBInv, np.transpose(Ks))), 0, None)

    def estimate_uncertainty(self, **kwargs):
        """Use potential epistemic uncertainty"""
        xt, yu, yl = kwargs['xt'], kwargs['yu'], kwargs['yl']
        Xtr, Ytr, x_PI_unique = kwargs['Xtr'], kwargs['Ytr'], kwargs['x_PI_unique']
        ep_unc, ep_unc_not_obs = np.zeros(len(xt)), np.zeros(len(xt))
        # Create a KDTree from the dataset
        tree = None
        if len(Xtr) > 0:
            if Xtr.ndim == 1:
                tree = KDTree(Xtr.reshape(-1, 1), leaf_size=2)
            else:
                tree = KDTree(Xtr, leaf_size=2)

        x_uniq = x_PI_unique[0]
        y_u_unique, y_l_unique = x_PI_unique[1]

        for iv, v in enumerate(xt):
            # Query the tree to find indices of samples within distance epsilon from v
            if len(Xtr) > 0:
                inds = tree.query_radius(np.array(v).reshape(-1, 1), r=self.epsi)[0]
                Xsubset = Xtr[inds]
                Ysubset = Ytr[inds]

                # PIs of training samples were pre-computed, just find them
                Ysubset_u, Ysubset_l = Ysubset.copy(), Ysubset.copy()
                for ii, xs in enumerate(Xsubset):
                    pos = np.where(x_uniq == xs)[0][0]
                    Ysubset_u[ii], Ysubset_l[ii] = y_u_unique[pos], y_l_unique[pos]

                inds = np.where((Ysubset <= Ysubset_u) & (Ysubset >= Ysubset_l))[0]
                Ysubset, Ysubset_u, Ysubset_l = Ysubset[inds], Ysubset_u[inds], Ysubset_l[inds]
            else:
                Ysubset, Ysubset_u, Ysubset_l =  Ytr, Ytr, Ytr
            if len(Ysubset) == 0:
                ep_unc[iv] = abs(yu[iv] - yl[iv])
                ep_unc_not_obs[iv] = abs(yu[iv] - yl[iv])
            else:
                # Calculate minimum distance between a training sample and the predicted upper bound
                d_u = np.min(Ysubset_u - Ysubset)
                # Calculate minimum distance between a training sample and the predicted upper bound
                d_l = np.min(Ysubset - Ysubset_l)
                ep_unc[iv] = d_u + d_l
                ep_unc_not_obs[iv] = 0
        return ep_unc

    def sample_points(self, n_samples, **kwargs):
        """Selects a batch of sampling points that would reduce the epistemic uncertainty"""
        Xtest, Ypred, epistemic_unc = kwargs['Xtest'], kwargs['Ypred'], kwargs['epistemic_unc']

        # Update GP
        self.set_mean(x=Xtest, mu=Ypred)
        self.set_cov_matrix(diag=epistemic_unc)

        x_sampled, K, new_K = [], self.K.copy(), None

        x_pool = self.x.copy()
        while len(x_sampled) < n_samples:
            max_diff, x_add, best_K = 0, None, None
            for x in x_pool:
                new_K = self.posterior_update(xB=x, K=K)
                diff = np.sum(np.sqrt(np.diag(K)) - np.sqrt(np.diag(new_K)))
                if diff > max_diff:
                    max_diff = diff
                    x_add = x
                    best_K = new_K
            # Sample selected point
            x_sampled.append(x_add)
            K = best_K.copy()
        return x_sampled
