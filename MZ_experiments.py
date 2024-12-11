import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from AdaptiveSampling.Sampler_MZ import SamplerMZ

if __name__ == '__main__':
    # Input arguments
    n_extra_samples = 1
    n_iterations = 50
    multiple_seeds = True
    evaluate = False
    method = 'ASPINN'
    if method == 'ASPINN':
        hyperparameters = {'epsi': 0.25, 'length': 0.1}
    else:
        hyperparameters = {}

    sampler = SamplerMZ(method=method,
                        n_extra_samples=n_extra_samples,
                        n_iterations=n_iterations,
                        multiple_seeds=multiple_seeds,
                        evaluate=evaluate)
    sampler.run(**hyperparameters)
