import sys
from AdaptiveSampling.utils import *
from PredictionIntervals.Trainer.TrainNN import Trainer
from AdaptiveSampling.Datasets.GenerateDatasets import GenerateDatasets
from AdaptiveSampling.estimators.UncertaintyModel import UncertaintyModel


class SamplerMZ:
    """Class used to sample additional data points through multiple iterations when the full dataset and the equations
    that were used to generate the datasets are known"""

    def __init__(self, method: str = 'ASPINN', n_extra_samples: int = 10, n_iterations: int = 10,
                 multiple_seeds: bool = False, evaluate: bool = False):
        """
        Adaptive Sampling with Prediction-Interval Neural Networks
        :param method: Name of the uncertainty model method. Options: ['ASPINN', 'MCDropout', 'GP'].
        :param n_extra_samples: Numer of additional data points that are sampled at each iteration
        :param n_iterations: Total number of iterations
        :param multiple_seeds: If True, repeat the experiments 10 times using multiple seeds
        :param evaluate: If True, train a separate PI-generation NN that will calculate performance metrics
        """
        self.dataset = 'MZ'
        self.method = method
        if method not in ['ASPINN', 'MCDropout', 'GP']:
            sys.exit("The only available methods are ['ASPINN', 'MCDropout', 'GP'].")
        self.n_extra_samples = n_extra_samples
        self.n_iterations = n_iterations
        self.multiple_seeds = multiple_seeds
        self.evaluate = evaluate
        self.n_cells = 1

        if self.multiple_seeds:
            self.seeds = [7, 12, 53, 768, 63]
        else:
            self.seeds = [7]

        os.makedirs(os.path.join(get_project_root(), 'AdaptiveSampling/results/' + self.dataset + '/' + self.method),
                    exist_ok=True)  # Create results directory
        self.NNmethod = 'DualAQD'
        self.batch = 4
        self.epochs = 500

    def run(self, **kwargs):

        for seed in self.seeds:
            # Initialize uncertainty model
            unc_model = UncertaintyModel(method=self.method, **kwargs)
            np.random.seed(seed)
            prec_values = np.random.choice([75, 90, 105, 120, 135, 150], self.n_iterations)

            # Generate data
            loader = GenerateDatasets(name=self.dataset, seed=seed)
            X, Y, gen_fun = loader.X, loader.Y, loader.gen_fun
            xtest, YU_test, YL_test = loader.grid, loader.YU_grid, loader.YL_grid
            X_full, Y_full, YU_full, YL_full = loader.X_full, loader.Y_full, loader.YU_full, loader.YL_full
            n_cells = int(len(X) / 5)
            # Isolate the aspect and VH/prec vectors, which are invariable
            aspect = X[:n_cells, 1]
            VH_withoutP = X[:n_cells, 2] / X[:n_cells, 0]

            # Separate training and validation sets
            indices = np.arange(len(Y))
            np.random.shuffle(indices)
            training_inds, val_inds = indices[:int(len(Y) * 0.8)], indices[int(len(Y) * 0.8):]
            Xtrain, Ytrain = X[training_inds], Y[training_inds]
            Xval, Yval = X[val_inds], Y[val_inds]

            # Train
            model = Trainer(X=Xtrain, Y=Ytrain, Xval=Xval, Yval=Yval,
                            architecture=loader.architecture, method=self.NNmethod)
            Xtrain2, Ytrain2 = Xtrain.copy(), Ytrain.copy()
            Xval2, Yval2 = Xval.copy(), Yval.copy()
            n_rep = 0
            all_X_sampled, all_Y_sampled = np.empty((0, 4)), np.empty(0)
            PI_cov = []
            while n_rep < self.n_iterations:
                print("*******************")
                print("Iteration ", n_rep)
                print("*******************")
                print("Precipition value this year: ", prec_values[n_rep])
                # Train
                model.set_modelName(self.dataset + '_' + self.method + '_Seed' + str(seed) + '_Iteration' + str(n_rep))
                model.set_data(X=Xtrain2, Y=Ytrain2, Xval=Xval2, Yval=Yval2)
                # if self.evaluate or self.method == 'MCDropout' or self.method == 'ASPINN':
                model.train(printProcess=False, epochs=self.epochs, batch_size=self.batch,
                            eta_=0.001, plotCurves=False, scratch=False)

                # Read precipitation value of current year (current iteration)
                current_prec = prec_values[n_rep]
                current_prec = np.repeat(current_prec, n_cells)
                # Merge test data
                VH = current_prec * VH_withoutP
                passive_variables = np.concatenate((current_prec[:, None], aspect[:, None], VH[:, None]), axis=1)
                X_test = np.zeros((len(passive_variables) * len(xtest), 4))
                y_unique_u, y_unique_l = np.zeros(len(xtest)), np.zeros(len(xtest))
                YU_unique, YL_unique = np.zeros(len(xtest)), np.zeros(len(xtest))
                X_obs, Y_obs = [], []
                Y_pred = np.zeros(len(passive_variables) * len(xtest))
                Y_real = np.zeros(len(passive_variables) * len(xtest))
                if self.method == 'MCDropout':
                    ypred_raw = np.zeros((len(xtest), 50))
                else:
                    ypred_raw = np.zeros((len(xtest), 3, 50))
                YU_test, YL_test = np.zeros(len(passive_variables) * len(xtest)), np.zeros(len(passive_variables) * len(xtest))
                for n, N_rate in enumerate(xtest):
                    X_test[n*len(passive_variables):(n+1)*len(passive_variables), 0:3] = passive_variables
                    X_test[n*len(passive_variables):(n+1)*len(passive_variables), 3] = N_rate
                    X_currentNr = X_test[n*len(passive_variables):(n+1)*len(passive_variables), :]

                    # Find previous data with the same precipitation and Nr values
                    selected_indices = np.where((Xtrain2[:, -1] == N_rate) & (Xtrain2[:, 0] == prec_values[n_rep]))[0]
                    X_selected, Y_selected = Xtrain2[selected_indices, -1], Ytrain2[selected_indices]

                    # Predict upper and lower bounds per current Nr
                    ypred_Nr, y_u_Nr, y_l_Nr, ypred_raw_Nr = model.evaluate(Xeval=X_currentNr, normData=True,
                                                                            returnRaw=True, MC_samples=50)
                    Ytest_Nr, YU_test_Nr, YL_test_Nr, _ = gen_fun(X_currentNr)
                    y_unique_u[n], y_unique_l[n] = np.max(y_u_Nr), np.min(y_l_Nr)
                    if self.method == 'MCDropout':
                        ypred_raw[n] = ypred_raw_Nr[np.argmax(np.std(ypred_raw_Nr[:, 2, :], axis=1)), 2, :]
                    YU_unique[n], YL_unique[n] = np.max(YU_test_Nr), np.min(YL_test_Nr)
                    YU_test[n*len(passive_variables):(n+1)*len(passive_variables)] = np.max(YU_test_Nr)
                    YL_test[n*len(passive_variables):(n+1)*len(passive_variables)] = np.min(YL_test_Nr)
                    Y_pred[n*len(passive_variables):(n+1)*len(passive_variables)] = ypred_Nr
                    Y_real[n*len(passive_variables):(n+1)*len(passive_variables)] = gen_fun(X_currentNr)[0]

                    X_obs.extend(X_selected)
                    Y_obs.extend(Y_selected)

                # Evaluate model using current data
                X_obs, Y_obs = np.array(X_obs), np.array(Y_obs)
                PI_unique2 = [y_unique_u, y_unique_l]
                pi_cov = np.mean(np.abs(YU_unique - y_unique_u) + np.abs(YL_unique - y_unique_l))
                PI_cov.append(pi_cov)
                print("PI comparison. PI_delta = ", pi_cov)

                # If the AS process has already been executed, load its results
                filename = f"src/AdaptiveSampling/results/{self.dataset}/{self.method}/Seed{seed}_Iteration{n_rep}_args.pkl"
                if self.evaluate and self.method != 'ASPINN':
                    if os.path.exists(filename):
                        with open(filename, 'rb') as fp:
                            load_data = pickle.load(fp)
                        Xtrain2, Ytrain2, Xval2, Yval2 = load_data['Xtr'], load_data['Ytr'], load_data['yu'], load_data[
                            'yl']
                        epistemic_unc2, X_full = load_data['ep_unc'], load_data['Xorig']
                        YU_full, YL_full = load_data['P1'], load_data['P2']
                        xtr_sampled, ytr_sampled = load_data['sampled']
                        all_X_sampled, all_Y_sampled = load_data['total_sampled']
                    else:
                        sys.exit("The uncertainty metrics should be calculated first. Run the method using evaluate=False")
                else:
                    # Update uncertainty model and sample next data points
                    epistemic_unc2 = unc_model.estimate_uncertainty(xt=xtest, yu=y_unique_u, yl=y_unique_l, Xtr=X_obs,
                                                                    Ytr=Y_obs,
                                                                    x_PI_unique=[xtest, PI_unique2],
                                                                    ypred=ypred_raw)
                    N_sampled = np.array(unc_model.sample_points(n_samples=self.n_extra_samples, Xtest=xtest, Ypred=Y_pred,
                                                                 epistemic_unc=epistemic_unc2))
                    print("N rates chosen: ", N_sampled)

                    # Apply the sampled N rates onto the field
                    N_rates = np.random.choice(N_sampled, n_cells)
                    x_sampled = np.concatenate((passive_variables, N_rates[:, None]), axis=1)

                    # Evaluate the effect of the sampled points during the harvest season
                    y_sampled = np.array(gen_fun(x_sampled)[0])

                    # Update dataset. Separate training and validation sets
                    indices = np.arange(len(x_sampled))
                    np.random.shuffle(indices)
                    training_inds, val_inds = indices[:int(len(x_sampled) * 0.8)], indices[int(len(x_sampled) * 0.8):]
                    Xtrain2 = np.concatenate((Xtrain2, x_sampled[training_inds, :]))
                    Ytrain2 = np.concatenate((Ytrain2, y_sampled[training_inds]))
                    Xval2 = np.concatenate((Xval2, x_sampled[val_inds, :]))
                    Yval2 = np.concatenate((Yval2, y_sampled[val_inds]))

                    all_X_sampled = np.concatenate((all_X_sampled, x_sampled))
                    all_Y_sampled = np.concatenate((all_Y_sampled, y_sampled))
                    xtr_sampled, ytr_sampled = x_sampled[training_inds], y_sampled[training_inds]
                    save_arguments(dataset=self.dataset, met=self.method, s=seed, iteration=n_rep,
                                   xt=xtest, yu=y_unique_u, yl=y_unique_l, Xtr=Xtrain2, Ytr=Ytrain2, Xv=Xval2, Yv=Yval2,
                                   ep_unc=epistemic_unc2, Xorig=X_full, P1=YU_full, P2=YL_full,
                                   sampled=[xtr_sampled, ytr_sampled],
                                   total_sampled=[all_X_sampled, all_Y_sampled])

                # plot_state(xt=xtest, yu=y_u2, yl=y_l2, Xtr=Xtrain2, Ytr=Ytrain2, Xv=Xval2, Yv=Yval2,
                #            ep_unc=epistemic_unc2, sampled=[xtr_sampled, ytr_sampled],
                #            total_sampled=[all_X_sampled, all_Y_sampled], Xorig=X_full, P1=YU_full, P2=YL_full)
                plt.figure()
                plt.scatter(X_obs, Y_obs)
                plt.scatter(xtest, YU_unique, label='YU_unique')
                plt.scatter(xtest, YL_unique, label='YL_unique')
                plt.scatter(xtest, y_unique_u, label='yu_unique')
                plt.scatter(xtest, y_unique_l, label='yl_unique')
                plt.legend()
                plt.savefig(os.path.join(get_project_root(),
                                         'AdaptiveSampling/results/' + self.dataset + '/' + self.method + '/Seed' +
                                         str(seed) + '_Iteration' + str(n_rep) + '.jpg'), dpi=400)
                plt.close('all')

                n_rep += 1
            np.save(os.path.join(get_project_root(), 'AdaptiveSampling/results/' + self.dataset + '/' + self.method +
                                 '/Seed' + str(seed) + '_PI_delta_history.npy'), np.array(PI_cov))
