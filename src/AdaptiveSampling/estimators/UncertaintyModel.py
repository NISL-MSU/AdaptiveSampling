import sys
from AdaptiveSampling.estimators.GP import GP
from AdaptiveSampling.estimators.ASPINN import ASPINN
from AdaptiveSampling.estimators.MCDropout import MCDropout


class UncertaintyModel:

    def __init__(self, method: str = 'ASPINN', **kwargs):
        """
        Model used for epistemic uncertainty quantification
        :param method: Options: ['ASPINN', 'MCDropout', 'GP']. Default: 'ASPINN'
        :param kwargs: Arguments needed for each method
        """
        self.method = method

        if self.method == 'ASPINN':
            self.strategy = ASPINN(**kwargs)
        elif self.method == 'MCDropout':
            self.strategy = MCDropout()
        elif self.method == 'GP':
            self.strategy = GP()
        else:
            sys.exit("The only available methods are ['ASPINN', 'MCDropout', 'GP'].")

    def estimate_uncertainty(self, **kwargs):
        return self.strategy.estimate_uncertainty(**kwargs)

    def sample_points(self, n_samples, **kwargs):
        return self.strategy.sample_points(n_samples, **kwargs)
