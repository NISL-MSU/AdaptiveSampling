import torch
import numpy as np
from AdaptiveSampling.estimators.gpytorch_utils import *
from AdaptiveSampling.estimators.EstimatorInterface import EstimatorInterface


class GP(EstimatorInterface):
    """Use a classic Gaussian Process to estimate epistemic uncertainty"""

    def __init__(self):
        self.name = 'GP'
        self.model, self.likelihood = None, None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, X, y, epochs=1000):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        train_losses = []
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y.shape[1]) \
            .to(self.device)
        self.model = MultitaskGPModel(X, y, self.likelihood).to(self.device)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(epochs):
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.mean().backward()
            optimizer.step()
            train_losses.append(loss.mean().item())
        return train_losses

    def estimate_uncertainty(self, **kwargs):
        xt, yu, yl = kwargs['xt'], kwargs['yu'], kwargs['yl']
        Xtr, Ytr, x_PI_unique = kwargs['Xtr'], kwargs['Ytr'], kwargs['x_PI_unique']

        # Train GP
        _ = self.train(Xtr, Ytr)
        # Evaluate GP
        self.model.eval()
        self.likelihood.eval()
        if xt.ndim == 1:
            xt = xt.reshape(-1, 1)
        xt = torch.tensor(xt, dtype=torch.float32).to(self.device)
        preds = self.model(xt)
        # Calculate variance of predictions as estimation of the epistemic uncertainty
        return preds.variance.sqrt().detach().cpu().numpy().flatten()

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
