import numpy as np
from AdaptiveSampling.estimators.EstimatorInterface import EstimatorInterface


class MCDropout(EstimatorInterface):
    """Use MCDropout to estimate epistemic uncertainty"""

    def __init__(self):
        self.name = 'MCDropout'

    def estimate_uncertainty(self, **kwargs):
        ypred = kwargs['ypred']

        # Calculate variance of predictions as estimation of the epistemic uncertainty
        return np.std(ypred, axis=1)

    def sample_points(self, n_samples, **kwargs):
        """Selects a batch of sampling points that would reduce the epistemic uncertainty"""
        Xtest, Ypred, epistemic_unc = kwargs['Xtest'], kwargs['Ypred'], kwargs['epistemic_unc']

        probabilities = epistemic_unc / np.sum(epistemic_unc)
        sampled_indices = np.random.choice(len(epistemic_unc), size=len(probabilities), p=probabilities)
        unique_sampled_indices, counts = np.unique(sampled_indices, return_counts=True)
        sampled_probs = counts / np.sum(counts)
        index_prob_dict = dict(zip(unique_sampled_indices, sampled_probs))
        top_k_indices = sorted(index_prob_dict, key=index_prob_dict.get, reverse=True)[:n_samples]

        return Xtest[top_k_indices]
