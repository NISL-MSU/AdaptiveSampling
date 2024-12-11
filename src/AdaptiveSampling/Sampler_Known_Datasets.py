import os.path
import sys
from AdaptiveSampling.utils import *
from PredictionIntervals.Trainer.TrainNN import Trainer
from AdaptiveSampling.Datasets.GenerateDatasets import GenerateDatasets
from AdaptiveSampling.estimators.UncertaintyModel import UncertaintyModel


class SamplerKD:
    """Class used to sample additional data points through multiple iterations when the full dataset and the equations
    that were used to generate the datasets are known"""

    def __init__(self, dataset: str = 'cos', method: str = 'ASPINN', n_extra_samples: int = 10, n_iterations: int = 10,
                 multiple_seeds: bool = False, evaluate: bool = False):
        """
        Adaptive Sampling with Prediction-Interval Neural Networks
        :param dataset: Name of the dataset. Options: ['cos', 'hetero', 'cosexp']
        :param method: Name of the uncertainty model method. Options: ['ASPINN', 'MCDropout', 'GP'].
        :param n_extra_samples: Numer of additional data points that are sampled at each iteration
        :param n_iterations: Total number of iterations
        :param multiple_seeds: If True, repeat the experiments 10 times using multiple seeds
        :param evaluate: If True, train a separate PI-generation NN that will calculate performance metrics
        """
        self.dataset = dataset
        self.method = method
        if method not in ['ASPINN', 'MCDropout', 'GP', 'nflows']:
            sys.exit("The only available methods are ['ASPINN', 'MCDropout', 'GP'].")
        self.n_extra_samples = n_extra_samples
        self.n_iterations = n_iterations
        self.multiple_seeds = multiple_seeds
        self.evaluate = evaluate
        if method == 'ASPINN' or method == 'nflows':
            self.evaluate = True

        if self.multiple_seeds:
            self.seeds = [7, 12, 53, 768, 63, 327, 512, 568, 25, 700]
        else:
            self.seeds = [7]

        os.makedirs(os.path.join(get_project_root(), 'AdaptiveSampling/results/' + self.dataset + '/' + self.method),
                    exist_ok=True)  # Create results directory
        self.NNmethod = 'DualAQD'
        self.batch = 16
        self.epochs = 4500
        if method == 'MCDropout' and not self.evaluate:
            self.NNmethod = 'MCDropout'
            self.batch = 64
            self.epochs = 1500

    def run(self, **kwargs):

        ypred, y_u, y_l, ypred_raw, x_unique, PI_unique = [], [], [], [], [], []
        for seed in self.seeds:
            PI_cov = []
            # Initialize uncertainty model
            if self.method == 'nflows':
                unc_model = None
            else:
                unc_model = UncertaintyModel(method=self.method, **kwargs)
            np.random.seed(seed)

            # Generate data
            loader = GenerateDatasets(name=self.dataset, seed=seed)
            X, Y, gen_fun = loader.X, loader.Y, loader.gen_fun
            xtest, YU_test, YL_test = loader.grid, loader.YU_grid, loader.YL_grid
            X_full, Y_full, YU_full, YL_full = loader.X_full, loader.Y_full, loader.YU_full, loader.YL_full

            # Separate training and validation sets
            indices = np.arange(len(Y))
            np.random.shuffle(indices)
            training_inds, val_inds = indices[:int(len(Y) * 0.8)], indices[int(len(Y) * 0.8):]
            Xtrain, Ytrain = X[training_inds], Y[training_inds]
            Xval, Yval = X[val_inds], Y[val_inds]

            # Train (in the beginning all compared methods use the same data, so let's train just one NN for all)
            modelName = self.dataset + '_ASPINN_Seed' + str(seed) + '_Iteration0'
            model = Trainer(X=Xtrain, Y=Ytrain, Xval=Xval, Yval=Yval, architecture=loader.architecture,
                            method=self.NNmethod, modelName=modelName)
            if self.evaluate or self.method == 'MCDropout' or self.method == 'ASPINN':
                # Train a model only if it doesn't exist
                if not os.path.exists(model.f + '/weights-NN-temp_MCDropout'):
                    model.train(printProcess=False, epochs=self.epochs, batch_size=self.batch, eta_=0.002, plotCurves=False)

                ypred, y_u, y_l, ypred_raw = model.evaluate(Xeval=xtest, normData=True, returnRaw=True, MC_samples=50)
                x_unique = np.unique(Xtrain)
                _, y_unique_u, y_unique_l, _ = model.evaluate(Xeval=x_unique, normData=True, returnRaw=True, MC_samples=50)
                PI_unique = [y_unique_u, y_unique_l]

            # If the AS process has already been executed, load its results
            filename = f"src/AdaptiveSampling/results/{self.dataset}/{self.method}/Seed{seed}_Iteration0_args.pkl"
            if self.evaluate and os.path.exists(filename):
                with open(filename, 'rb') as fp:
                    load_data = pickle.load(fp)
                epistemic_unc = load_data['ep_unc']
            else:
                # Calculate initial uncertainty
                epistemic_unc = unc_model.estimate_uncertainty(xt=xtest, yu=y_u, yl=y_l, Xtr=Xtrain, Ytr=Ytrain,
                                                               x_PI_unique=[x_unique, PI_unique], ypred=ypred_raw)
                save_arguments(dataset=self.dataset, met=self.method, s=seed, iteration=0,
                               xt=xtest, yu=y_u, yl=y_l, Xtr=Xtrain, Ytr=Ytrain, Xv=Xval, Yv=Yval, ep_unc=epistemic_unc,
                               Xorig=X_full, P1=YU_full, P2=YL_full, sampled=None, total_sampled=None)
            if len(YU_test) == len(y_u):
                pi_cov = np.mean(np.abs(YU_test - y_u) + np.abs(YL_test - y_l))
                print("PI comparison. PI_delta = ", pi_cov)
            plot_state(xt=xtest, yu=y_u, yl=y_l, Xtr=Xtrain, Ytr=Ytrain, Xv=Xval, Yv=Yval, ep_unc=epistemic_unc,
                       Xorig=X_full, P1=YU_full, P2=YL_full)
            plt.savefig(os.path.join(get_project_root(), 'AdaptiveSampling/results/' + self.dataset + '/' +
                                     self.method + '/Seed' + str(seed) + '_Iteration0.jpg'), dpi=400)
            plt.close('all')

            Xtrain2, Ytrain2 = Xtrain.copy(), Ytrain.copy()
            Xval2, Yval2 = Xval.copy(), Yval.copy()
            ypred2, y_u2, y_l2 = ypred.copy(), y_u.copy(), y_l.copy()
            epistemic_unc2 = epistemic_unc

            if self.evaluate and self.method != 'ASPINN':
                n_rep = 5
            else:
                n_rep = 1
            all_X_sampled, all_Y_sampled = [], []
            while n_rep <= self.n_iterations:
                print("*******************")
                print("Iteration ", n_rep)
                print("*******************")
                # If the AS process has already been executed, load its results
                filename = f"src/AdaptiveSampling/results/{self.dataset}/{self.method}/Seed{seed}_Iteration{n_rep}_args.pkl"
                x_unique2, PI_unique2, xtr_sampled, ytr_sampled, x_sampled, y_sampled = [], [], [], [], [], []
                if self.evaluate and self.method != 'ASPINN':
                    if os.path.exists(filename):
                        with open(filename, 'rb') as fp:
                            load_data = pickle.load(fp)
                        y_u2, y_l2 = load_data['yu'], load_data['yl']
                        Xtrain2, Ytrain2, Xval2, Yval2 = load_data['Xtr'], load_data['Ytr'], load_data['Xv'], load_data['Yv']
                        epistemic_unc2, X_full = load_data['ep_unc'], load_data['Xorig']
                        YU_full, YL_full = load_data['P1'], load_data['P2']
                        xtr_sampled, ytr_sampled = load_data['sampled']
                        all_X_sampled, all_Y_sampled = load_data['total_sampled']
                    else:
                        sys.exit("The uncertainty metrics should be calculated first. Run the method using evaluate=False")
                else:
                    # Update uncertainty model and sample next data points
                    x_sampled = np.array(unc_model.sample_points(n_samples=self.n_extra_samples, Xtest=xtest, Ypred=ypred2,
                                                                 epistemic_unc=epistemic_unc2))
                    y_sampled = np.array(gen_fun(x_sampled)[0])
                    # Update dataset. Separate training and validation sets
                    indices = np.arange(len(x_sampled))
                    np.random.shuffle(indices)
                    training_inds, val_inds = indices[:int(len(x_sampled) * 0.8)], indices[int(len(x_sampled) * 0.8):]
                    Xtrain2 = np.concatenate((Xtrain2, x_sampled[training_inds]))
                    Ytrain2 = np.concatenate((Ytrain2, y_sampled[training_inds]))
                    Xval2 = np.concatenate((Xval2, x_sampled[val_inds]))
                    Yval2 = np.concatenate((Yval2, y_sampled[val_inds]))

                model.set_data(X=Xtrain2, Y=Ytrain2, Xval=Xval2, Yval=Yval2)
                model.set_modelName(self.dataset + '_' + self.method + '_Seed' + str(seed) + '_Iteration' + str(n_rep))

                if self.evaluate or self.method == 'MCDropout' or self.method == 'ASPINN':
                    model.train(printProcess=False, epochs=self.epochs, batch_size=self.batch,
                                eta_=0.002, plotCurves=False, scratch=False)
                    # Calculate updated uncertainties
                    ypred2, y_u2, y_l2, ypred_raw = model.evaluate(Xeval=xtest, normData=True, returnRaw=True, MC_samples=50)
                    x_unique2 = np.unique(Xtrain2)
                    _, y_unique_u, y_unique_l, _ = model.evaluate(Xeval=x_unique2, normData=True, returnRaw=True,
                                                                  MC_samples=50)
                    PI_unique2 = [y_unique_u, y_unique_l]

                if not os.path.exists(filename) or self.method == 'ASPINN':
                    epistemic_unc2 = unc_model.estimate_uncertainty(xt=xtest, yu=y_u2, yl=y_l2, Xtr=Xtrain2, Ytr=Ytrain2,
                                                                    x_PI_unique=[x_unique2, PI_unique2], ypred=ypred_raw)
                    all_X_sampled = np.concatenate((all_X_sampled, x_sampled))
                    all_Y_sampled = np.concatenate((all_Y_sampled, y_sampled))
                    xtr_sampled, ytr_sampled = x_sampled[training_inds], y_sampled[training_inds]
                    save_arguments(dataset=self.dataset, met=self.method, s=seed, iteration=n_rep,
                                   xt=xtest, yu=y_u2, yl=y_l2, Xtr=Xtrain2, Ytr=Ytrain2, Xv=Xval2, Yv=Yval2,
                                   ep_unc=epistemic_unc2, Xorig=X_full, P1=YU_full, P2=YL_full,
                                   sampled=[xtr_sampled, ytr_sampled],
                                   total_sampled=[all_X_sampled, all_Y_sampled])
                if len(YU_test) == len(y_u2):
                    pi_cov = np.mean(np.abs(YU_test - y_u2) + np.abs(YL_test - y_l2))
                    PI_cov.append(pi_cov)
                    print("PI comparison. PI_delta = ", pi_cov)
                plot_state(xt=xtest, yu=y_u2, yl=y_l2, Xtr=Xtrain2, Ytr=Ytrain2, Xv=Xval2, Yv=Yval2,
                           ep_unc=epistemic_unc2, sampled=[xtr_sampled, ytr_sampled],
                           total_sampled=[all_X_sampled, all_Y_sampled], Xorig=X_full, P1=YU_full, P2=YL_full)
                plt.savefig(os.path.join(get_project_root(),
                                         'AdaptiveSampling/results/' + self.dataset + '/' + self.method + '/Seed' +
                                         str(seed) + '_Iteration' + str(n_rep) + '.jpg'), dpi=400)
                plt.close('all')

                if self.evaluate and self.method != 'ASPINN':
                    n_rep += 5
                else:
                    n_rep += 1
            np.save(os.path.join(get_project_root(), 'AdaptiveSampling/results/' + self.dataset + '/' + self.method +
                                 '/Seed' + str(seed) + '_PI_delta_history.npy'), np.array(PI_cov))
