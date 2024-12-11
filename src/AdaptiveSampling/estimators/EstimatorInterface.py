import abc


class EstimatorInterface(abc.ABC):
    """Interface containing the methods used for uncertainty estimation"""
    @abc.abstractmethod
    def estimate_uncertainty(self, **kwargs):
        pass

    @abc.abstractmethod
    def sample_points(self, X_current, Y_current, n_samples):
        pass

