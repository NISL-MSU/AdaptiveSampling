import sys
from AdaptiveSampling.utils import *
from PredictionIntervals.Trainer.TrainNN import Trainer
from AdaptiveSampling.estimators.UncertaintyModel import UncertaintyModel


class Sampler:
    """Class used to sample additional data points when the generating equations are not known"""

    def __init__(self, data: tuple, grid: np.array, n_extra_samples: int = 10, method: str = 'ASPINN',
                 problem: str = 'Test'):
        """
        Adaptive Sampling with Prediction-Interval Neural Networks
        :param data: Tuple containing dataset of the type (np.array(X), np.array(Y))
        :param grid: Define the input domain and all possible values that the variables can take. E.g., np.linspace(-5, -5, 100)
        :param n_extra_samples: Numer of additional data points that are sampled at each iteration
        :param method: Name of the uncertainty model method. Options: ['ASPINN', 'MCDropout', 'GP'].
        :param problem: Name of the problem/dataset. If not provided, results are saved in results/Test
        """
        self.data, self.grid = data, grid
        self.method, self.estimator = method, None
        self.n_extra_samples = n_extra_samples
        self.problem = problem

        os.makedirs('results/' + self.problem + '/' + self.method, exist_ok=True)  # Create results directory

    def set_estimator(self, estimator):
        """Set the uncertainty estimator object"""
        self.estimator = estimator
        self.method = self.estimator.name

    def run(self, **kwargs):
        """
        :return x_sampled: An np.array with the positions of the next recommended sampling points
        """
        if self.method == '':
            sys.exit("Attempted to execute the adaptive learning process without defining an uncertainty estimator (Use set_estimator())")

        # Initialize uncertainty model
        if self.method == 'nflows':
            unc_model = None
        else:
            unc_model = UncertaintyModel(method=self.method, **kwargs)
        np.random.seed(7)
        X, Y = self.data[0], self.data[1]

        # Separate training and validation sets
        indices = np.arange(len(Y))
        np.random.shuffle(indices)
        training_inds, val_inds = indices[:int(len(Y) * 0.8)], indices[int(len(Y) * 0.8):]
        Xtrain, Ytrain = X[training_inds], Y[training_inds]
        Xval, Yval = X[val_inds], Y[val_inds]

        # Train
        model = Trainer(X=Xtrain, Y=Ytrain, Xval=Xval, Yval=Yval)
        model.train(printProcess=False, epochs=3000, batch_size=16, eta_=0.01, plotCurves=False)

        ypred, y_u, y_l, ypred_raw = model.evaluate(Xeval=self.grid, normData=True, returnRaw=True, MC_samples=50)
        x_unique = np.unique(Xtrain)
        _, y_unique_u, y_unique_l, _ = model.evaluate(Xeval=x_unique, normData=True, returnRaw=True, MC_samples=50)
        PI_unique = [y_unique_u, y_unique_l]

        # Estimate uncertainty
        epistemic_unc = unc_model.estimate_uncertainty(xt=self.grid, yu=y_u, yl=y_l, Xtr=Xtrain, Ytr=Ytrain,
                                                       x_PI_unique=[x_unique, PI_unique], ypred=ypred_raw)
        # Update uncertainty model and sample next data points
        x_sampled = np.array(unc_model.sample_points(n_samples=self.n_extra_samples, Xtest=self.grid, 
                                                     Ypred=ypred.copy(), epistemic_unc=epistemic_unc))
        return x_sampled
