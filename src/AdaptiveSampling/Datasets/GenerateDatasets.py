import sys
import numpy as np
import matplotlib.pyplot as plt


class GenerateDatasets:
    """Class used to generate datasets reported in the paper"""

    def __init__(self, name: str = 'cos', seed: int = 7, plot: bool = False):
        """
        :param name: Options: ['cos', 'hetero', 'cosexp']
        """
        self.X, self.Y, self.architecture = np.zeros(0), np.zeros(0), ''
        self.name = name
        self.X_full, self.Y_full, self.YU_full, self.YL_full, self.grid = None, None, None, None, None
        self.YU_grid, self.YL_grid, self.Y_target = None, None, None
        self.seed = seed
        self.plot = plot
        self.gen_fun = None

        if hasattr(self, f'{name}'):
            method = getattr(self, f'{name}')
            method()
        else:
            sys.exit('The provided dataset name does not exist')

        if self.plot:
            plt.figure(figsize=(4, 3))
            plt.fill_between(self.X_full, self.YU_full, self.YL_full, color='gray', alpha=0.5, linewidth=0,
                             label='95% PIs\nfrom $\\varepsilon(\mathbf{x})$')
            plt.scatter(self.X, self.Y, s=10)
            plt.plot(self.X_full, self.Y_target, 'r', label='$f(\mathbf{x})$')
            plt.xlabel('$\mathbf{x}$', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylabel('$y$', fontsize=16)
            # plt.legend(framealpha=0.4, loc='lower left', fontsize=16, handletextpad=0.1, labelspacing=0.1)
            plt.tight_layout()

    def cos(self, n=200):
        """Sinusoidal function with sinusoidal Gaussian heteroskedastic noise (varying PI width)"""

        def cos_fun(inp):
            gauss = (2 + 2 * np.cos(1.2 * inp))
            orig = 10 + 5 * np.cos(inp + 2)
            return orig + gauss * np.random.normal(size=len(inp)), orig + 1.96 * gauss, orig - 1.96 * gauss, orig

        self.gen_fun, self.architecture = cos_fun, 'shallow'
        np.random.seed(self.seed)
        # Define input domain and calculate the upper and lower bounds at each possible position
        self.grid = np.linspace(-5, 5, 100)
        _, self.YU_grid, self.YL_grid, _ = cos_fun(self.grid)

        # Select a few random samples from certain ranges
        X = np.array([np.random.choice(self.grid) for _ in range(n)])
        Xorig = X.copy()
        mask = ((X >= -4) & (X < 1)) | (X > 3)
        mask2 = (X >= 1) & (X <= 3)
        mask3 = (X < -4) & (X >= -5)
        selected_indices = np.random.choice(np.where(mask2)[0], size=1, replace=False)
        selected_indices2 = np.random.choice(np.where(mask3)[0], size=3, replace=False)

        # Initial "incomplete" dataset
        Yorig, _, _, _ = cos_fun(Xorig)
        # self.X = np.concatenate([X[mask], X[selected_indices], X[selected_indices2]])
        # self.Y = np.concatenate([Yorig[mask], Yorig[selected_indices], Yorig[selected_indices2]])
        self.X = X
        self.Y = Yorig

        # Generate a "complete" version of the dataset uniformly sampled across the domain
        self.X_full = np.linspace(-5, 5, num=100000)
        self.Y_full, self.YU_full, self.YL_full, self.Y_target = cos_fun(self.X_full)

    def hetero(self, n=200):
        """Hetero function introduced by Depeweg, 2018"""

        def hetero_fun(inp):
            gauss = 3 * np.abs(np.cos(inp / 2))
            orig = 7 * np.sin(inp)
            return orig + gauss * np.random.normal(size=len(inp)), orig + 1.96 * gauss, orig - 1.96 * gauss, orig

        self.gen_fun, self.architecture = hetero_fun, 'shallow'
        np.random.seed(self.seed)
        # Define input domain and calculate the upper and lower bounds at each possible position
        self.grid = np.linspace(-4.5, 4.5, 300)
        _, self.YU_grid, self.YL_grid, _ = hetero_fun(self.grid)

        # Initial "incomplete" dataset
        comp = np.random.choice(3, n)
        means = [-4, 0, 4]
        stds = [2 / 5, 9 / 10, 2 / 5]
        self.X = np.zeros(n)
        for s in range(n):
            sampled_x = -np.infty
            while sampled_x <= -4.5 or sampled_x >= 4.5:
                sampled_x = np.random.normal(means[comp[s]], stds[comp[s]])
            self.X[s] = sampled_x
        self.Y, _, _, _ = hetero_fun(self.X)

        # Generate a "complete" version of the dataset uniformly sampled across the domain
        self.X_full = np.linspace(-4.5, 4.5, num=5000)
        self.Y_full, self.YU_full, self.YL_full, self.Y_target = hetero_fun(self.X_full)

    def cosexp(self, n=2000):
        """Sinusoidal-exponential function with sinusoidal Gaussian heteroskedastic noise (varying PI width)"""

        def cosexp_fun(inp):
            gauss = (1 - 0.01 * inp ** 2) * 0.5
            orig = np.cos(inp ** 2 / 5)
            return orig + gauss * np.random.normal(size=len(inp)), orig + 1.96 * gauss, orig - 1.96 * gauss, orig

        self.gen_fun, self.architecture = cosexp_fun, 'deep'
        np.random.seed(self.seed)
        # Define input domain and calculate the upper and lower bounds at each possible position
        self.grid = np.linspace(-10, 10, 500)
        _, self.YU_grid, self.YL_grid, _ = cosexp_fun(self.grid)

        # Select a few random samples from certain ranges
        X = np.array([np.random.choice(self.grid) for _ in range(n)])
        Xorig = X.copy()
        mask = ((X >= -10) & (X < -8)) | ((X >= -5) & (X < -2)) | ((X >= 3) & (X < 6)) | ((X >= 7) & (X <= 10))
        mask2 = (X >= -8) & (X <= -5)
        mask3 = (X >= -2) & (X < 3)
        mask4 = (X >= -6) & (X < 7)
        selected_indices = np.random.choice(np.where(mask2)[0], size=1, replace=False)
        selected_indices2 = np.random.choice(np.where(mask3)[0], size=10, replace=False)
        selected_indices3 = np.random.choice(np.where(mask4)[0], size=3, replace=False)

        # Initial "incomplete" dataset
        Yorig, _, _, _ = cosexp_fun(Xorig)
        self.X = np.concatenate([X[mask], X[selected_indices], X[selected_indices2], X[selected_indices3]])
        self.Y = np.concatenate([Yorig[mask], Yorig[selected_indices], Yorig[selected_indices2], Yorig[selected_indices3]])

        # Generate a "complete" version of the dataset uniformly sampled across the domain
        self.X_full = np.linspace(-10, 10, num=5000)
        self.Y_full, self.YU_full, self.YL_full, self.Y_target = cosexp_fun(self.X_full)

    def MZ(self):
        """Hetero function introduced by Depeweg, 2018"""

        def MZ_fun(inp):
            """Equation used to simulate the behavior of the MZ"""
            gauss = (inp[:, 0] + inp[:, 3]) / 1500
            orig = inp[:, 0] / 1.5 + 15 * (inp[:, 1] / np.pi + 1) * np.tanh(inp[:, 3] / (30 * inp[:, 2] + 20))
            return orig + gauss * np.random.normal(size=len(inp)), orig + 1.96 * gauss, orig - 1.96 * gauss, orig

        self.gen_fun, self.architecture = MZ_fun, 'deeper'
        np.random.seed(self.seed)
        # Define input domain and calculate the upper and lower bounds at each possible position
        self.grid = np.arange(0, 180, 30)  # Complete set of possible N rates
        n_cells, years = 1, 50

        # We consider the MZ contains 100 cells. Let's populate them with variable values
        # prec_values = np.array([75, 90, 105, 120, 150])  # Prec. in mm. for each year
        prec_values = np.random.choice([75, 90, 105, 120, 135, 150], years)
        prec_values = np.repeat(prec_values, n_cells)
        aspect = np.random.uniform(np.pi/4, np.pi/2, size=years)  # Aspect values for the 100 cells, they're invariant throughout the years
        # aspect = np.tile(aspect, years)
        VH = prec_values / 150 * np.tile(np.random.uniform(0.5, 1, size=n_cells), years)

        # Let's define the N-rates used in each year
        # N_rates = np.zeros(n_cells * 5)
        # # 1st year. N-rates: [0, 30, 150]
        # N_rates[0:n_cells] = np.random.choice([0], n_cells)
        # # 2nd year. N-rates: [30, 120, 150]
        # N_rates[n_cells:n_cells * 2] = np.random.choice([150], n_cells)
        # # 3rd year. N-rates: [0, 60, 120]
        # N_rates[n_cells * 2:n_cells * 3] = np.random.choice([60], n_cells)
        # # 4th year. N-rates: [30, 60, 120]
        # N_rates[n_cells * 3:n_cells * 4] = np.random.choice([120], n_cells)
        # # 5th year. N-rates: [0, 60, 90]
        # N_rates[n_cells * 4:] = np.random.choice([90], n_cells)
        N_rates = np.random.choice(self.grid, n_cells * years)

        # Ensemble the dataset
        self.X = np.concatenate((prec_values[:, None], aspect[:, None], VH[:, None], N_rates[:, None]), axis=1)
        self.Y, _, _, _ = MZ_fun(self.X)


if __name__ == '__main__':
    loader = GenerateDatasets(name='cosexp', plot=True, seed=12)
