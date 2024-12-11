import os
from src.AdaptiveSampling.estimators.first_draftGP import *
from utils import *
import matplotlib.pyplot as plt
from PredictionIntervals.Trainer.TrainNN import Trainer


if __name__ == '__main__':

    seeds = [7, 12, 53, 768, 63, 327, 512, 568, 25, 700]
    method = 'Ours'
    os.makedirs('results/' + method, exist_ok=True)

    # Method params
    epsi = 0.25

    for ns, seed in enumerate(seeds):
        np.random.seed(seed)
        X, Y, Xorig, P1, P2 = create_sin_data(n=2000, plot=False)  # Plot the dataset

        # Separate 90% for training and 10% for validation
        indices = np.arange(len(Y))
        np.random.shuffle(indices)
        training_inds, val_inds = indices[:int(len(Y) * 0.8)], indices[int(len(Y) * 0.8):]
        Xtrain, Ytrain = X[training_inds], Y[training_inds]
        Xval, Yval = X[val_inds], Y[val_inds]

        # Train
        model = Trainer(X=Xtrain, Y=Ytrain, Xval=Xval, Yval=Yval)
        model.train(printProcess=False, epochs=3000, batch_size=16, eta_=0.01, plotCurves=False)

        xtest = np.linspace(-5, 5, num=100)
        ypred, y_u, y_l, y_raw = model.evaluate(Xeval=xtest, normData=True, returnRaw=True, MC_samples=50)
        x_unique = np.unique(Xtrain)
        _, y_unique_u, y_unique_l, _ = model.evaluate(Xeval=x_unique, normData=True, returnRaw=True, MC_samples=50)
        PI_unique = [y_unique_u, y_unique_l]

        # Calculate ideal 95% PI
        gauss = (2 + 2 * np.cos(1.2 * xtest))
        orig = 10 + 5 * np.cos(xtest + 2)
        YU, YL = orig + 1.96 * gauss, orig - 1.96 * gauss
        epistemic_unc = epist_unc_Wdistance(xtest, y_u, y_l, Xtrain, Ytrain, [x_unique, PI_unique], epsi=epsi)
        plot_state(xt=xtest, yu=y_u, yl=y_l, Xtr=Xtrain, Ytr=Ytrain, Xv=Xval, Yv=Yval, ep_unc=epistemic_unc[0],
                   Xorig=Xorig, P1=P1, P2=P2)
        save_arguments(met=method, s=seed, iteration=0,
                       xt=xtest, yu=y_u, yl=y_l, Xtr=Xtrain, Ytr=Ytrain, Xv=Xval, Yv=Yval, ep_unc=epistemic_unc[0],
                       Xorig=Xorig, P1=P1, P2=P2, sampled=None, total_sampled=None)
        # plt.savefig("images/InitPIs.jpg", dpi=400)
        print("PI comparison. PI_delta = ", np.mean(np.abs(YU - y_u) + np.abs(YL - y_l)))

        Xtrain2, Ytrain2 = Xtrain.copy(), Ytrain.copy()
        Xval2, Yval2 = Xval.copy(), Yval.copy()
        ypred2, y_u2, y_l2 = ypred.copy(), y_u.copy(), y_l.copy()
        epistemic_unc2 = epistemic_unc

        n_rep = 0
        n_extra_samples = 10
        all_X_sampled, all_Y_sampled = [], []
        PI_cov = []
        while n_rep < 50:
            print("*******************")
            print("Iteration ", n_rep + 1)
            print("*******************")

            with open("images/unc_" + str(n_rep + 1) + ".pickle", 'wb') as handle:
                pickle.dump(epistemic_unc2, handle)
            # Create GP
            gp = GP(length=0.1)
            gp.set_mean(xtest, ypred2)
            gp.set_cov_matrix(diag=epistemic_unc2[0])
            # Sample next data points
            x_sampled = np.array(gp.select_samples(batch_size=n_extra_samples))
            y_sampled = synth_func(x_sampled)
            # Update dataset. Separate 90% for training and 10% for validation
            indices = np.arange(len(x_sampled))
            np.random.shuffle(indices)
            training_inds, val_inds = indices[:int(len(x_sampled) * 0.8)], indices[int(len(x_sampled) * 0.8):]
            Xtrain2 = np.concatenate((Xtrain2, x_sampled[training_inds]))
            Ytrain2 = np.concatenate((Ytrain2, y_sampled[training_inds]))
            Xval2 = np.concatenate((Xval2, x_sampled[val_inds]))
            Yval2 = np.concatenate((Yval2, y_sampled[val_inds]))
            model.set_data(X=Xtrain2, Y=Ytrain2, Xval=Xval2, Yval=Yval2)
            model.train(printProcess=False, epochs=4500, batch_size=16, eta_=0.002, plotCurves=False, scratch=False)

            # Calculate updated uncertainties
            ypred2, y_u2, y_l2, _ = model.evaluate(Xeval=xtest, normData=True, returnRaw=True, MC_samples=50)
            x_unique2 = np.unique(Xtrain2)
            _, y_unique_u, y_unique_l, _ = model.evaluate(Xeval=x_unique2, normData=True, returnRaw=True, MC_samples=50)
            PI_unique2 = [y_unique_u, y_unique_l]
            epistemic_unc2 = epist_unc_Wdistance(xtest, y_u2, y_l2, Xtrain2, Ytrain2, [x_unique2, PI_unique2], epsi=epsi)
            all_X_sampled = np.concatenate((all_X_sampled, x_sampled))
            all_Y_sampled = np.concatenate((all_Y_sampled, y_sampled))
            plot_state(xt=xtest, yu=y_u2, yl=y_l2, Xtr=Xtrain2, Ytr=Ytrain2, Xv=Xval2, Yv=Yval2, ep_unc=epistemic_unc2[0],
                       sampled=[x_sampled[training_inds], y_sampled[training_inds]],
                       total_sampled=[all_X_sampled, all_Y_sampled], Xorig=Xorig, P1=P1, P2=P2)
            save_arguments(met=method, s=seed, iteration=n_rep,
                           xt=xtest, yu=y_u2, yl=y_l2, Xtr=Xtrain2, Ytr=Ytrain2, Xv=Xval2, Yv=Yval2,
                           ep_unc=epistemic_unc2[0], Xorig=Xorig, P1=P1, P2=P2,
                           sampled=[x_sampled[training_inds], y_sampled[training_inds]],
                           total_sampled=[all_X_sampled, all_Y_sampled])
            plt.savefig("images/Seed" + str(ns) + "_Iteration" + str(n_rep + 1) + ".jpg", dpi=400)
            pi_cov = np.mean(np.abs(YU - y_u2) + np.abs(YL - y_l2))
            PI_cov.append(pi_cov)
            print("PI comparison. PI_delta = ", pi_cov)
            n_rep += 1
        np.save("images/Seed" + str(ns) + "_PI_delta_history.npy", np.array(PI_cov))
